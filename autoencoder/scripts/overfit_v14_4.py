"""V14.4 quick overfit + sine sweep test.

Validates: 1) training stability without shortcuts, 2) frequency response.

Usage:
  python -u -m autoencoder.scripts.overfit_v14_4
"""

import math, time, torch, torch.nn.functional as F
import soundfile as sf, numpy as np
from autoencoder.models.autoencoder import WavVAE
from autoencoder.losses_stft import MultiResolutionSTFTLoss
from autoencoder.losses import MultiScaleMelLoss


def compute_snr(x_hat, x):
    n = ((x_hat - x) ** 2).mean().item()
    s = (x ** 2).mean().item()
    return 100.0 if n < 1e-10 else 10 * math.log10(s / n)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = 22050
    stride = 512
    steps = 2000

    # Load audio
    data, _ = sf.read("/root/autodl-tmp/project/data/LJSpeech-1.1/wavs/LJ001-0001.wav")
    seg_len = stride * 86
    audio = torch.from_numpy(data.astype(np.float32))[:seg_len]
    audio = audio.unsqueeze(0).unsqueeze(0).to(device)
    print(f"Audio: {audio.shape}, std={audio.std():.4f}")

    # Model
    model = WavVAE(
        latent_dim=128,
        encoder_channels=(128, 256, 512, 512, 1024, 1024),
        strides=(2, 4, 4, 4, 4), dilations=(1, 3, 9),
    ).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    mrstft = MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512],
        win_sizes=[512, 1024, 2048], sr=sr,
    ).to(device)
    mel_fn = MultiScaleMelLoss(sr=sr).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.8, 0.99))

    # Train
    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        # LR warmup
        lr = min(step / 300, 1.0) * 3e-4 * 0.5 * (1 + math.cos(math.pi * step / steps))
        for pg in opt.param_groups: pg["lr"] = lr

        model.set_step(step)
        opt.zero_grad()
        out = model(audio)
        l1 = F.l1_loss(out["x_hat"], audio)
        l2 = F.mse_loss(out["x_hat"], audio)
        stft = mrstft(out["x_hat"], audio)
        mel = mel_fn(out["x_hat"], audio)
        loss = 10 * l1 + 5 * l2 + 0.5 * stft + 0.1 * mel + out["reg_loss"]

        if not torch.isfinite(loss):
            print(f"  [{step}] NaN! Skipping.")
            opt.zero_grad()
            continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 200 == 0 or step == 1:
            with torch.no_grad():
                model.eval()
                snr = compute_snr(model(audio)["x_hat"].float(), audio.float())
                model.train()
            print(f"  [{step:4d}] loss={loss.item():.4f} SNR={snr:+.2f}dB gnorm={gn:.1f} lr={lr:.1e} | {time.time()-t0:.0f}s")

    # Sine sweep
    print(f"\n=== Sine Sweep (V14.4, no shortcuts) ===")
    model.eval()
    T = stride * 86
    t = torch.linspace(0, T / sr, T)
    print(f"{'Freq':>6} {'SNR':>8} {'Ratio':>7}")
    for freq in [200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 7000, 8000]:
        sine = (0.3 * torch.sin(2 * math.pi * freq * t)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            recon = model(sine)["x_hat"].float()
        snr = compute_snr(recon, sine.float())
        ratio = recon.std().item() / sine.float().std().item()
        print(f"{freq:>6} {snr:>8.2f} {ratio:>7.3f}")


if __name__ == "__main__":
    main()
