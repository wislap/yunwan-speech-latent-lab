"""PCA-shaped latent Gaussian noise robustness test.

Noise model:
  eps_i(t) ~ N(0, variance_scale * lambda_i)
  noise(t) = sum_i eps_i(t) * u_i
  z_noisy = z + noise

where u_i are frame-level PCA unit directions and lambda_i are PCA eigenvalues
from the real latent distribution.

This tests natural latent-distribution-shaped perturbations, not arbitrary
channel-isotropic noise. It writes CSV/PNG only and never saves audio.

Usage:
  python -m autoencoder.scripts.latent_pca_noise_robustness \
    --out docs/v16.3/figures/pca_noise_robustness \
    --pca docs/v16.3/figures/latent_pca_directions/v16_3/frame_p95_pca_directions.npz \
    --run V16.3=outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt:384:256:2,4,4,8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from scipy.signal import stft

from autoencoder.models.autoencoder import WavVAE


BANDS = (
    (0, 500),
    (500, 1500),
    (1500, 3500),
    (3500, 5000),
    (5000, 7000),
    (7000, 9000),
    (9000, 11025),
)


@dataclass
class RunSpec:
    name: str
    ckpt: Path
    latent_dim: int
    stride: int
    strides: list[int]


def parse_run(value: str) -> RunSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be NAME=CKPT:LATENT_DIM:TOTAL_STRIDE:STRIDES")
    name, rest = value.split("=", 1)
    parts = rest.rsplit(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--run must be NAME=CKPT:LATENT_DIM:TOTAL_STRIDE:STRIDES")
    ckpt, latent_dim, stride, strides = parts
    return RunSpec(
        name=name,
        ckpt=Path(ckpt),
        latent_dim=int(latent_dim),
        stride=int(stride),
        strides=[int(x) for x in strides.split(",")],
    )


def item_path(item: dict) -> str:
    return str(item.get("audio_path") or item.get("path") or item.get("wav_path") or item.get("file"))


def load_model(spec: RunSpec, device: torch.device) -> tuple[WavVAE, dict]:
    ckpt = torch.load(spec.ckpt, map_location="cpu", weights_only=False)
    model = WavVAE(
        latent_dim=spec.latent_dim,
        encoder_channels=[128, 256, 512, 1024, 1024],
        strides=spec.strides,
        dilations=[1, 3, 9],
        sample_latent=False,
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def load_audio(path: str, stride: int, segment_length: int) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if segment_length > 0:
        audio = audio[:segment_length]
    n = (len(audio) // stride) * stride
    return audio[:n], int(sr)


def snr_db(ref: np.ndarray, pred: np.ndarray) -> float:
    err = pred - ref
    return 10.0 * math.log10((float(np.mean(ref * ref)) + 1e-12) / (float(np.mean(err * err)) + 1e-12))


def band_snr_and_phase(ref: np.ndarray, pred: np.ndarray, sr: int) -> dict[str, float]:
    n_fft = 2048
    hop = 512
    if len(ref) < n_fft:
        return {}
    freqs, _, ref_stft = stft(ref, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    _, _, pred_stft = stft(pred, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    err_stft = pred_stft - ref_stft
    ref_mag = np.abs(ref_stft)
    pred_mag = np.abs(pred_stft)
    phase_delta = np.angle(pred_stft * np.conj(ref_stft))
    phase_vec = np.exp(1j * phase_delta)
    phase_weight = ref_mag * pred_mag

    out = {}
    for lo, hi in BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            continue
        ref_energy = float(np.mean(np.abs(ref_stft[mask]) ** 2))
        err_energy = float(np.mean(np.abs(err_stft[mask]) ** 2))
        key = f"{lo}_{hi}"
        out[f"snr_{key}"] = 10.0 * math.log10((ref_energy + 1e-12) / (err_energy + 1e-12))
        w = phase_weight[mask]
        out[f"phase_coh_{key}"] = float(np.abs(np.sum(phase_vec[mask] * w)) / (np.sum(w) + 1e-12))
    hf_keys = ["3500_5000", "5000_7000", "7000_9000"]
    out["snr_3500_9000_mean"] = float(np.mean([out[f"snr_{k}"] for k in hf_keys]))
    out["phase_coh_3500_9000_mean"] = float(np.mean([out[f"phase_coh_{k}"] for k in hf_keys]))
    return out


def decode_with_noise(
    model: WavVAE,
    x: torch.Tensor,
    components: torch.Tensor,
    score_std: torch.Tensor,
    variance_scale: float,
    mode: str,
    rng: torch.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    with torch.no_grad():
        z, _, _ = model.encode(x)
        clean = model.decode(z, output_length=x.shape[-1])
        if variance_scale == 0.0:
            noisy = clean
            noise_rel_rms = 0.0
        else:
            if mode == "pca":
                k = components.shape[0]
                eps = torch.randn((z.shape[0], k, z.shape[-1]), device=z.device, generator=rng)
                eps = eps * (math.sqrt(variance_scale) * score_std.view(1, k, 1))
                noise = torch.einsum("bkt,kc->bct", eps, components)
            elif mode == "channel_iso":
                z_std = z.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
                noise = torch.randn(z.shape, device=z.device, generator=rng) * (math.sqrt(variance_scale) * z_std)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            noisy = model.decode(z + noise, output_length=x.shape[-1])
            noise_rel_rms = float(noise.pow(2).mean().sqrt().item() / (z.pow(2).mean().sqrt().item() + 1e-12))
    return (
        clean.squeeze().detach().cpu().float().numpy(),
        noisy.squeeze().detach().cpu().float().numpy(),
        noise_rel_rms,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(rows: list[dict], out_dir: Path) -> None:
    modes = sorted({r["mode"] for r in rows})
    scales = sorted({float(r["variance_scale"]) for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)
    metrics = [
        ("delta_snr", "SNR drop (dB)"),
        ("delta_snr_3500_9000_mean", "3.5-9k SNR drop (dB)"),
        ("delta_phase_coh_3500_9000_mean", "3.5-9k coherence drop"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        for mode in modes:
            means = []
            p25 = []
            p75 = []
            for scale in scales:
                vals = [float(r[metric]) for r in rows if r["mode"] == mode and float(r["variance_scale"]) == scale]
                means.append(float(np.mean(vals)))
                p25.append(float(np.percentile(vals, 25)))
                p75.append(float(np.percentile(vals, 75)))
            ax.plot(scales, means, marker="o", label=mode)
            ax.fill_between(scales, p25, p75, alpha=0.12)
        ax.set_xscale("log")
        ax.set_xlabel("Noise variance scale")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].legend()
    fig.suptitle("Latent Noise Robustness")
    fig.tight_layout()
    fig.savefig(out_dir / "noise_robustness_curves.png")
    plt.close(fig)


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for mode in sorted({r["mode"] for r in rows}):
        for scale in sorted({float(r["variance_scale"]) for r in rows if r["mode"] == mode}):
            rr = [r for r in rows if r["mode"] == mode and float(r["variance_scale"]) == scale]
            out.append({
                "mode": mode,
                "variance_scale": scale,
                "n": len(rr),
                "delta_snr_mean": float(np.mean([float(r["delta_snr"]) for r in rr])),
                "delta_snr_median": float(np.median([float(r["delta_snr"]) for r in rr])),
                "delta_hf_snr_mean": float(np.mean([float(r["delta_snr_3500_9000_mean"]) for r in rr])),
                "delta_phase_coh_mean": float(np.mean([float(r["delta_phase_coh_3500_9000_mean"]) for r in rr])),
                "noise_rel_rms_mean": float(np.mean([float(r["noise_rel_rms"]) for r in rr])),
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=parse_run, required=True)
    parser.add_argument("--pca", type=Path, required=True)
    parser.add_argument("--data", default="data/ljspeech_train_splits.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-eval", type=int, default=24)
    parser.add_argument("--segment-length", type=int, default=65536)
    parser.add_argument("--variance-scales", default="0,0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--modes", default="pca,channel_iso")
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    variance_scales = [float(x) for x in args.variance_scales.split(",")]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    with open(args.data) as f:
        items = json.load(f)
    random.Random(args.seed).shuffle(items)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, ckpt = load_model(args.run, device)

    pca = np.load(args.pca)
    components = torch.from_numpy(pca["components"].astype(np.float32)).to(device)
    score_std = torch.from_numpy(pca["pca_score_std"].astype(np.float32)).to(device)

    rows = []
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    used = 0
    for item in items:
        if used >= args.n_eval:
            break
        audio, sr = load_audio(item_path(item), args.run.stride, args.segment_length)
        if len(audio) < args.run.stride * 2:
            continue
        x = torch.from_numpy(audio).view(1, 1, -1).to(device)
        clean_np = None
        clean_metrics = None
        for mode in modes:
            for scale in variance_scales:
                clean, noisy, noise_rel_rms = decode_with_noise(
                    model, x, components, score_std, scale, mode, generator
                )
                if clean_np is None:
                    clean_np = clean
                    clean_metrics = band_snr_and_phase(audio, clean_np, sr)
                    clean_metrics["snr"] = snr_db(audio, clean_np)
                noisy_metrics = band_snr_and_phase(audio, noisy, sr)
                noisy_metrics["snr"] = snr_db(audio, noisy)
                row = {
                    "run": args.run.name,
                    "checkpoint_epoch": ckpt.get("epoch"),
                    "checkpoint_best_snr": ckpt.get("best_snr"),
                    "sample": Path(item_path(item)).stem,
                    "mode": mode,
                    "variance_scale": scale,
                    "noise_rel_rms": noise_rel_rms,
                    "clean_snr": clean_metrics["snr"],
                    "noisy_snr": noisy_metrics["snr"],
                    "delta_snr": clean_metrics["snr"] - noisy_metrics["snr"],
                    "clean_snr_3500_9000_mean": clean_metrics["snr_3500_9000_mean"],
                    "noisy_snr_3500_9000_mean": noisy_metrics["snr_3500_9000_mean"],
                    "delta_snr_3500_9000_mean": clean_metrics["snr_3500_9000_mean"] - noisy_metrics["snr_3500_9000_mean"],
                    "clean_phase_coh_3500_9000_mean": clean_metrics["phase_coh_3500_9000_mean"],
                    "noisy_phase_coh_3500_9000_mean": noisy_metrics["phase_coh_3500_9000_mean"],
                    "delta_phase_coh_3500_9000_mean": clean_metrics["phase_coh_3500_9000_mean"] - noisy_metrics["phase_coh_3500_9000_mean"],
                }
                rows.append(row)
        used += 1

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    summary = summarize(rows)
    write_csv(args.out / "per_sample_noise_metrics.csv", rows)
    write_csv(args.out / "noise_summary.csv", summary)
    plot_curves(rows, args.out)
    with (args.out / "metadata.json").open("w") as f:
        json.dump({
            "run": args.run.name,
            "checkpoint": str(args.run.ckpt),
            "checkpoint_epoch": ckpt.get("epoch"),
            "checkpoint_best_snr": ckpt.get("best_snr"),
            "pca": str(args.pca),
            "n_eval": used,
            "variance_scales": variance_scales,
            "modes": modes,
            "noise_definition": "PCA eps_i~N(0, variance_scale*eigenvalue_i), then rotate to latent channel space",
        }, f, indent=2)
    print(f"Wrote {args.out}; rows={len(rows)}")


if __name__ == "__main__":
    main()
