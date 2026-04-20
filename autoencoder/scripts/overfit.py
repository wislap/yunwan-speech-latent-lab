"""V14 Wav-VAE single-sample overfit test.

No Hydra dependency. Loads one audio file and trains the model to reconstruct it.
This is the fastest way to verify the architecture can learn.

Usage:
  python -u scripts/overfit_v14.py
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load single audio file ───────────────────────────────
    audio_path = "/root/autodl-tmp/project/data/LJSpeech-1.1/wavs/LJ001-0001.wav"
    data, sr = sf.read(audio_path)
    audio = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)  # [1, T]

    # Crop to segment_length (must be divisible by total_stride=441)
    segment_length = 441 * 100  # = 44100, exactly 2 seconds
    if audio.shape[-1] < segment_length:
        audio = F.pad(audio, (0, segment_length - audio.shape[-1]))
    elif audio.shape[-1] > segment_length:
        audio = audio[:, :segment_length]

    # [1, 1, T] — keep full audio for eval, crop for training
    full_audio = audio.unsqueeze(0).to(device)
    print(f"Audio shape: {full_audio.shape}, sr={sr}, duration={full_audio.shape[-1]/sr:.2f}s")

    # ── Model ────────────────────────────────────────────────
    model = WavVAE(
        latent_dim=256,
        encoder_channels=(128, 256, 512, 1024),
        strides=(7, 7, 9),
        dilations=(1, 3, 9),
        kl_weight=1e-4,
        free_bits=0.25,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"Model params: {n_params:,} (enc={n_enc:,}, dec={n_dec:,})")
    print(f"Total stride: {model.total_stride}")

    # ── Loss functions ───────────────────────────────────────
    mrstft = MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_sizes=[512, 1024, 2048],
        sr=sr,
    ).to(device)

    mel_loss_fn = MultiScaleMelLoss(sr=sr).to(device)

    # ── Optimizer ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.8, 0.99), weight_decay=0.01)

    # ── Training loop ────────────────────────────────────────
    steps = 10000
    log_every = 100
    save_every = 2000
    grad_accum = 1  # 3-block was stable without accumulation

    print(f"\nStarting overfit test: {steps} steps")
    print("=" * 70)

    model.train()
    t_start = time.time()

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_l1 = 0.0
        accum_stft = 0.0
        accum_mel = 0.0
        accum_kl = 0.0

        for _ in range(grad_accum):
            # Random crop from the full audio (data augmentation for single sample)
            full_T = full_audio.shape[-1]
            if full_T > segment_length:
                start = torch.randint(0, full_T - segment_length + 1, (1,)).item()
                audio = full_audio[:, :, start:start + segment_length]
            else:
                audio = full_audio

            out = model(audio)
            x_hat = out["x_hat"]
            kl_loss = out["kl_loss"]

            l1_loss = F.l1_loss(x_hat, audio)
            stft_loss = mrstft(x_hat, audio)
            mel_loss = mel_loss_fn(x_hat, audio)

            total_loss = (l1_loss + stft_loss + mel_loss + kl_loss) / grad_accum

            if not torch.isfinite(total_loss):
                print(f"  [step {step}] NaN/Inf loss! Skipping.")
                optimizer.zero_grad()
                break
            total_loss.backward()

            accum_loss += total_loss.item()
            accum_l1 += l1_loss.item() / grad_accum
            accum_stft += stft_loss.item() / grad_accum
            accum_mel += mel_loss.item() / grad_accum
            accum_kl += kl_loss.item() / grad_accum
        else:
            # Only step if all accum passes succeeded
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                model.eval()
                out_eval = model(full_audio)
                snr = compute_snr(out_eval["x_hat"], full_audio)
                ratio = out_eval["x_hat"].std().item() / full_audio.std().item()
                model.train()

            elapsed = time.time() - t_start
            print(
                f"  [{step:4d}/{steps}] "
                f"loss={accum_loss:.4f} "
                f"(l1={accum_l1:.4f} stft={accum_stft:.4f} "
                f"mel={accum_mel:.4f} kl={accum_kl:.6f}) "
                f"| SNR={snr:+.2f}dB ratio={ratio:.3f} "
                f"| gnorm={grad_norm:.2f} "
                f"| {elapsed:.1f}s"
            )

        if step % save_every == 0:
            # Save reconstructed audio
            with torch.no_grad():
                model.eval()
                out_save = model(full_audio)
                model.train()
            save_dir = "/root/autodl-tmp/project/outputs/v14_overfit"
            import os, soundfile as sf_save
            os.makedirs(save_dir, exist_ok=True)
            sf_save.write(
                f"{save_dir}/recon_step{step:04d}.wav",
                out_save["x_hat"][0, 0].cpu().numpy(), sr,
            )
            sf_save.write(
                f"{save_dir}/original.wav",
                full_audio[0, 0].cpu().numpy(), sr,
            )
            print(f"  → Saved audio to {save_dir}/recon_step{step:04d}.wav")

    # ── Final evaluation ─────────────────────────────────────
    model.eval()
    with torch.no_grad():
        out_final = model(full_audio)
        final_snr = compute_snr(out_final["x_hat"], full_audio)
        final_ratio = out_final["x_hat"].std().item() / full_audio.std().item()

    total_time = time.time() - t_start
    print("=" * 70)
    print(f"Overfit test complete in {total_time:.1f}s")
    print(f"  Final SNR: {final_snr:+.2f} dB")
    print(f"  Final ratio: {final_ratio:.3f}")
    print(f"  Latent shape: {out_final['z'].shape}")
    print(f"  Latent std: {out_final['z'].std().item():.4f}")
    print(f"  Latent mean: {out_final['z'].mean().item():.4f}")
    print(f"  Encoder std (mean of std): {out_final['std'].mean().item():.4f}")


if __name__ == "__main__":
    main()
