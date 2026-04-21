"""V14.2 detailed analysis: frequency band reconstruction + sine sweep response.

Usage:
  python -u -m autoencoder.scripts.analyze_v14_2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import os
import numpy as np
import soundfile as sf
from torch.nn.utils import weight_norm
from pathlib import Path


# ── Load model (V14.2 uses parameter-free shortcuts) ─────────

def load_v14_2_model(ckpt_path, device):
    from autoencoder.models.autoencoder import WavVAE
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch", "?")
    best_snr = ckpt.get("best_snr", "?")
    print(f"Checkpoint: epoch={epoch}, snr={best_snr}")

    model = WavVAE(
        latent_dim=128,
        encoder_channels=(128, 256, 512, 512, 1024, 1024),
        strides=(2, 4, 4, 4, 4),
        dilations=(1, 3, 9),
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def compute_snr(x_hat, x):
    noise_pow = ((x_hat - x) ** 2).mean().item()
    sig_pow = (x ** 2).mean().item()
    if noise_pow < 1e-10:
        return 100.0
    return 10.0 * math.log10(sig_pow / max(noise_pow, 1e-10))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = 22050
    stride = 512

    ckpt_path = "outputs/models/wav_vae_v14.2_full_remote_main/best.pt"
    if not Path(ckpt_path).exists():
        # Try latest
        ckpt_path = "outputs/models/wav_vae_v14.2_full_remote_main/latest.pt"
    model = load_v14_2_model(ckpt_path, device)

    out_dir = Path("outputs/analysis_v14_2")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # Part 1: Sine sweep frequency response
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Part 1: Sine Sweep Frequency Response")
    print("=" * 70)

    duration = 1.0  # seconds
    T = int(duration * sr)
    T = (T // stride) * stride  # align
    t = torch.linspace(0, duration, T)

    # Test frequencies from 50 Hz to Nyquist
    test_freqs = [50, 100, 200, 300, 500, 800, 1000, 1500, 2000, 3000,
                  4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]
    
    freq_snrs = []
    freq_ratios = []
    freq_phase_errs = []

    print(f"\n{'Freq (Hz)':>10} {'SNR (dB)':>10} {'Ratio':>8} {'Phase Err':>10}")
    print("-" * 45)

    for freq in test_freqs:
        if freq > sr / 2:
            break
        # Generate pure sine
        sine = (0.3 * torch.sin(2 * math.pi * freq * t)).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(sine)
        recon = out["x_hat"].float()
        sine_f = sine.float()

        snr = compute_snr(recon, sine_f)
        ratio = recon.std().item() / sine_f.std().item()
        freq_snrs.append(snr)
        freq_ratios.append(ratio)

        # Phase error: cross-correlation to find delay, then compute residual phase
        r = recon.squeeze().cpu().numpy()
        s = sine_f.squeeze().cpu().numpy()
        # Normalize
        r_n = r / (np.abs(r).max() + 1e-8)
        s_n = s / (np.abs(s).max() + 1e-8)
        corr = np.correlate(r_n[:2048], s_n[:2048], mode='full')
        delay = np.argmax(corr) - 2047
        phase_err_deg = delay / sr * freq * 360 % 360
        if phase_err_deg > 180:
            phase_err_deg -= 360
        freq_phase_errs.append(phase_err_deg)

        print(f"{freq:>10} {snr:>10.2f} {ratio:>8.3f} {phase_err_deg:>10.1f}°")

    # Save sine sweep audio for listening
    for freq in [200, 1000, 4000, 8000]:
        if freq > sr / 2:
            continue
        sine = (0.3 * torch.sin(2 * math.pi * freq * t)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            recon = model(sine)["x_hat"]
        sf.write(str(out_dir / f"sine_{freq}hz_original.wav"), sine.squeeze().cpu().numpy(), sr)
        sf.write(str(out_dir / f"sine_{freq}hz_recon.wav"), recon.squeeze().cpu().float().numpy(), sr)

    # ═══════════════════════════════════════════════════════════
    # Part 2: Band-wise reconstruction quality on real speech
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Part 2: Band-wise Reconstruction Quality (real speech)")
    print("=" * 70)

    with open("data/ljspeech_train_splits.json") as f:
        items = json.load(f)

    # Collect spectrograms
    n_eval = 100
    band_edges = [0, 300, 1000, 2000, 4000, 6000, 8000, 11025]
    band_names = ["0-300", "300-1k", "1k-2k", "2k-4k", "4k-6k", "6k-8k", "8k-11k"]
    n_bands = len(band_names)

    band_snrs = [[] for _ in range(n_bands)]
    band_energies_orig = [[] for _ in range(n_bands)]
    band_energies_recon = [[] for _ in range(n_bands)]
    overall_snrs = []

    fft_size = 2048
    hop_size = 512
    window = torch.hann_window(fft_size).to(device)

    for i, item in enumerate(items[:n_eval]):
        data, file_sr = sf.read(item["audio_path"])
        audio = torch.from_numpy(data.astype(np.float32))
        T_aligned = (audio.shape[0] // stride) * stride
        if T_aligned < stride * 2:
            continue
        audio = audio[:T_aligned].unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            recon = model(audio)["x_hat"].float()
        audio_f = audio.float()

        overall_snrs.append(compute_snr(recon, audio_f))

        # STFT
        orig_stft = torch.stft(audio_f.squeeze(1), fft_size, hop_size, fft_size,
                                window=window, return_complex=True)
        recon_stft = torch.stft(recon.squeeze(1), fft_size, hop_size, fft_size,
                                 window=window, return_complex=True)
        orig_mag = orig_stft.abs().squeeze(0)   # [F, T]
        recon_mag = recon_stft.abs().squeeze(0)

        hz_per_bin = sr / fft_size
        for b in range(n_bands):
            lo_bin = int(band_edges[b] / hz_per_bin)
            hi_bin = min(int(band_edges[b + 1] / hz_per_bin), orig_mag.shape[0])

            orig_band = orig_mag[lo_bin:hi_bin, :]
            recon_band = recon_mag[lo_bin:hi_bin, :]

            orig_energy = (orig_band ** 2).mean().item()
            recon_energy = (recon_band ** 2).mean().item()
            band_energies_orig[b].append(orig_energy)
            band_energies_recon[b].append(recon_energy)

            diff_energy = ((orig_band - recon_band) ** 2).mean().item()
            if diff_energy < 1e-10:
                band_snrs[b].append(100.0)
            else:
                band_snrs[b].append(10 * math.log10(orig_energy / max(diff_energy, 1e-10)))

    print(f"\nOverall SNR ({len(overall_snrs)} samples): "
          f"mean={np.mean(overall_snrs):.2f}, median={np.median(overall_snrs):.2f}")

    print(f"\n{'Band':>10} {'SNR (dB)':>10} {'Orig E':>10} {'Recon E':>10} {'E Ratio':>10}")
    print("-" * 55)
    for b in range(n_bands):
        snr_mean = np.mean(band_snrs[b])
        orig_e = np.mean(band_energies_orig[b])
        recon_e = np.mean(band_energies_recon[b])
        e_ratio = recon_e / max(orig_e, 1e-10)
        print(f"{band_names[b]:>10} {snr_mean:>10.2f} {orig_e:>10.6f} {recon_e:>10.6f} {e_ratio:>10.3f}")

    # ═══════════════════════════════════════════════════════════
    # Part 3: Latent space quick stats
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Part 3: Latent Space Quick Stats")
    print("=" * 70)

    all_z = []
    for i, item in enumerate(items[:200]):
        data, file_sr = sf.read(item["audio_path"])
        audio = torch.from_numpy(data.astype(np.float32))
        T_aligned = (audio.shape[0] // stride) * stride
        if T_aligned < stride:
            continue
        audio = audio[:T_aligned].unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            z = model(audio)["z"].squeeze(0).cpu()
        all_z.append(z.mean(dim=-1).numpy())

    Z = np.stack(all_z)
    print(f"  z shape: {Z.shape}")
    print(f"  z global: mean={Z.mean():.4f}, std={Z.std():.4f}")
    dim_stds = Z.std(axis=0)
    print(f"  per-dim std: mean={dim_stds.mean():.4f}, min={dim_stds.min():.4f}, max={dim_stds.max():.4f}")
    print(f"  active dims (std>0.1): {(dim_stds > 0.1).sum()}/128")
    print(f"  low dims (std<0.05): {(dim_stds < 0.05).sum()}/128")

    # PCA
    from numpy.linalg import svd
    Z_c = Z - Z.mean(axis=0)
    _, sigma, _ = svd(Z_c, full_matrices=False)
    ev = (sigma ** 2) / (sigma ** 2).sum()
    cumvar = np.cumsum(ev)
    eff_rank = np.exp(-np.sum(ev * np.log(ev + 1e-10)))
    print(f"  effective rank: {eff_rank:.1f}")
    for k in [1, 5, 10, 20, 50, 100, 128]:
        if k <= len(cumvar):
            print(f"    {k:3d} dims: {cumvar[k-1]*100:.1f}%")

    # Save results
    np.savez(
        str(out_dir / "analysis.npz"),
        freq_hz=np.array(test_freqs[:len(freq_snrs)]),
        freq_snrs=np.array(freq_snrs),
        freq_ratios=np.array(freq_ratios),
        freq_phase_errs=np.array(freq_phase_errs),
        band_names=band_names,
        band_snrs_mean=np.array([np.mean(b) for b in band_snrs]),
        overall_snrs=np.array(overall_snrs),
        Z=Z, sigma=sigma, explained_var=ev,
    )
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
