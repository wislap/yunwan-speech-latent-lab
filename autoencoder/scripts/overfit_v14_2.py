"""V14.2 single-sample overfit test.

dim=128, strides=[2,4,4,4,4]=512, 5 blocks, 129M params.

Usage:
  python -u -m autoencoder.scripts.overfit_v14_2
"""

import math
import time
import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np

from autoencoder.models.autoencoder import WavVAE
from autoencoder.losses_stft import MultiResolutionSTFTLoss
from autoencoder.losses import MultiScaleMelLoss


def compute_snr(x_hat, x):
    noise = x_hat - x
    sig_pow = (x ** 2).mean()
    noise_pow = (noise ** 2).mean()
    if noise_pow < 1e-10:
        return 100.0
    return 10.0 * math.log10(sig_pow.item() / noise_pow.item())


def get_lr(step, warmup_steps, peak_lr, total_steps):
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Config ───────────────────────────────────────────────
    latent_dim = 128
    strides = (2, 4, 4, 4, 4)
    encoder_channels = (128, 256, 512, 512, 1024, 1024)
    total_stride = 512

    steps = 5000
    peak_lr = 3e-4
    warmup_steps = 300
    grad_clip = 0.5
    stft_weight = 0.5
    log_every = 100
    save_every = 1000

    # ── Load audio ───────────────────────────────────────────
    audio_path = "/root/autodl-tmp/project/data/LJSpeech-1.1/wavs/LJ001-0001.wav"
    data, sr = sf.read(audio_path)
    audio = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)

    segment_length = total_stride * 86  # = 44032, ~2 seconds
    if audio.shape[-1] < segment_length:
        audio = F.pad(audio, (0, segment_length - audio.shape[-1]))
    elif audio.shape[-1] > segment_length:
        audio = audio[:, :segment_length]

    full_audio = audio.unsqueeze(0).to(device)
    print(f"Audio: {full_audio.shape}, sr={sr}, duration={full_audio.shape[-1]/sr:.2f}s")

    # ── Model ────────────────────────────────────────────────
    model = WavVAE(
        latent_dim=latent_dim,
        encoder_channels=encoder_channels,
        strides=strides,
        dilations=(1, 3, 9),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"Params: {n_params:,} (enc={n_enc:,}, dec={n_dec:,})")
    print(f"Stride: {model.total_stride}, latent frames: {full_audio.shape[-1]//model.total_stride}")

    # ── Loss ─────────────────────────────────────────────────
    mrstft = MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512],
        win_sizes=[512, 1024, 2048], sr=sr,
    ).to(device)
    mel_fn = MultiScaleMelLoss(sr=sr).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.8, 0.99), weight_decay=0.01)

    # ── Train ────────────────────────────────────────────────
    print(f"\nV14.2 overfit: {steps} steps, dim={latent_dim}, stride={total_stride}")
    print("=" * 80)

    model.train()
    t_start = time.time()

    for step in range(1, steps + 1):
        lr = get_lr(step, warmup_steps, peak_lr, steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.set_step(step)
        optimizer.zero_grad()

        out = model(full_audio)
        x_hat = out["x_hat"]
        reg_loss = out["reg_loss"]

        l1_loss = F.l1_loss(x_hat, full_audio)
        stft_loss = mrstft(x_hat, full_audio)
        mel_loss = mel_fn(x_hat, full_audio)
        total_loss = l1_loss + stft_weight * stft_loss + mel_loss + reg_loss

        if not torch.isfinite(total_loss):
            print(f"  [step {step}] NaN! Skipping.")
            optimizer.zero_grad()
            continue

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                model.eval()
                out_eval = model(full_audio)
                snr = compute_snr(out_eval["x_hat"], full_audio)
                ratio = out_eval["x_hat"].std().item() / full_audio.std().item()
                enc_std = out_eval["std"].mean().item()
                model.train()

            elapsed = time.time() - t_start
            print(
                f"  [{step:5d}/{steps}] "
                f"loss={total_loss.item():.4f} "
                f"(l1={l1_loss.item():.4f} stft={stft_loss.item():.4f} "
                f"mel={mel_loss.item():.4f} reg={reg_loss.item():.6f}) "
                f"| SNR={snr:+.2f}dB ratio={ratio:.3f} enc_std={enc_std:.4f} "
                f"| gnorm={grad_norm:.2f} lr={lr:.2e} "
                f"| {elapsed:.1f}s"
            )

        if step % save_every == 0:
            with torch.no_grad():
                model.eval()
                out_save = model(full_audio)
                model.train()
            save_dir = "/root/autodl-tmp/project/outputs/v14_2_overfit"
            import os
            os.makedirs(save_dir, exist_ok=True)
            sf.write(f"{save_dir}/recon_step{step:05d}.wav", out_save["x_hat"][0, 0].cpu().numpy(), sr)
            if step == save_every:
                sf.write(f"{save_dir}/original.wav", full_audio[0, 0].cpu().numpy(), sr)
            print(f"  → Saved to {save_dir}/recon_step{step:05d}.wav")

    # ── Final ────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        out_final = model(full_audio)
        final_snr = compute_snr(out_final["x_hat"], full_audio)
        final_ratio = out_final["x_hat"].std().item() / full_audio.std().item()

    print("=" * 80)
    print(f"V14.2 overfit complete in {time.time()-t_start:.1f}s")
    print(f"  SNR: {final_snr:+.2f} dB")
    print(f"  Ratio: {final_ratio:.3f}")
    print(f"  Latent: {out_final['z'].shape}, std={out_final['z'].std().item():.4f}")
    print(f"  Encoder std: {out_final['std'].mean().item():.4f}")


if __name__ == "__main__":
    main()
