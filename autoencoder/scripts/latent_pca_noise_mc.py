"""Monte Carlo PCA-shaped latent noise robustness test.

For each sample and variance scale, repeat stochastic latent perturbations many
times, then report mean/quantile curves and density plots. This reduces the
single-draw noise of latent Gaussian robustness estimates.

Usage:
  python -m autoencoder.scripts.latent_pca_noise_mc \
    --out docs/v16.3/figures/pca_noise_mc \
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
    ckpt, latent_dim, stride, strides = rest.rsplit(":", 3)
    return RunSpec(name, Path(ckpt), int(latent_dim), int(stride), [int(x) for x in strides.split(",")])


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


def hf_metrics(ref: np.ndarray, pred: np.ndarray, sr: int) -> dict[str, float]:
    n_fft = 2048
    hop = 512
    freqs, _, ref_stft = stft(ref, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    _, _, pred_stft = stft(pred, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    mask = (freqs >= 3500) & (freqs < 9000)
    err = pred_stft - ref_stft
    ref_e = float(np.mean(np.abs(ref_stft[mask]) ** 2))
    err_e = float(np.mean(np.abs(err[mask]) ** 2))
    ref_mag = np.abs(ref_stft[mask])
    pred_mag = np.abs(pred_stft[mask])
    phase_delta = np.angle(pred_stft[mask] * np.conj(ref_stft[mask]))
    w = ref_mag * pred_mag
    coh = float(np.abs(np.sum(np.exp(1j * phase_delta) * w)) / (np.sum(w) + 1e-12))
    return {
        "hf_snr_3500_9000": 10.0 * math.log10((ref_e + 1e-12) / (err_e + 1e-12)),
        "hf_phase_coh_3500_9000": coh,
    }


def make_noise(
    mode: str,
    z: torch.Tensor,
    components: torch.Tensor,
    score_std: torch.Tensor,
    variance_scale: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    if variance_scale == 0.0:
        return torch.zeros_like(z), 0.0
    if mode == "pca":
        k = components.shape[0]
        eps = torch.randn((z.shape[0], k, z.shape[-1]), device=z.device, generator=generator)
        eps = eps * (math.sqrt(variance_scale) * score_std.view(1, k, 1))
        noise = torch.einsum("bkt,kc->bct", eps, components)
    elif mode == "channel_iso":
        z_std = z.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
        noise = torch.randn(z.shape, device=z.device, generator=generator) * (math.sqrt(variance_scale) * z_std)
    else:
        raise ValueError(mode)
    rel = float(noise.pow(2).mean().sqrt().item() / (z.pow(2).mean().sqrt().item() + 1e-12))
    return noise, rel


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for mode in sorted({r["mode"] for r in rows}):
        for scale in sorted({float(r["variance_scale"]) for r in rows if r["mode"] == mode}):
            rr = [r for r in rows if r["mode"] == mode and float(r["variance_scale"]) == scale]
            for metric in ["delta_snr", "delta_hf_snr", "delta_hf_phase_coh"]:
                vals = np.asarray([float(r[metric]) for r in rr])
                out.append({
                    "mode": mode,
                    "variance_scale": scale,
                    "metric": metric,
                    "n": len(vals),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "p05": float(np.percentile(vals, 5)),
                    "p25": float(np.percentile(vals, 25)),
                    "p50": float(np.percentile(vals, 50)),
                    "p75": float(np.percentile(vals, 75)),
                    "p95": float(np.percentile(vals, 95)),
                })
    return out


def sample_summary(rows: list[dict]) -> list[dict]:
    out = []
    for sample in sorted({r["sample"] for r in rows}):
        for mode in sorted({r["mode"] for r in rows}):
            rr = [r for r in rows if r["sample"] == sample and r["mode"] == mode and float(r["variance_scale"]) > 0]
            if not rr:
                continue
            out.append({
                "sample": sample,
                "mode": mode,
                "mean_delta_snr": float(np.mean([float(r["delta_snr"]) for r in rr])),
                "mean_delta_hf_snr": float(np.mean([float(r["delta_hf_snr"]) for r in rr])),
                "max_delta_snr": float(np.max([float(r["delta_snr"]) for r in rr])),
            })
    return out


def plot_mean_ci(summary: list[dict], out_dir: Path) -> None:
    metrics = [
        ("delta_snr", "SNR drop (dB)"),
        ("delta_hf_snr", "3.5-9k SNR drop (dB)"),
        ("delta_hf_phase_coh", "3.5-9k coherence drop"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
    for ax, (metric, ylabel) in zip(axes, metrics):
        for mode in sorted({r["mode"] for r in summary}):
            rr = sorted([r for r in summary if r["mode"] == mode and r["metric"] == metric], key=lambda x: x["variance_scale"])
            x = np.asarray([r["variance_scale"] for r in rr])
            y = np.asarray([r["mean"] for r in rr])
            lo = np.asarray([r["p05"] for r in rr])
            hi = np.asarray([r["p95"] for r in rr])
            ax.plot(x, y, marker="o", label=mode)
            ax.fill_between(x, lo, hi, alpha=0.12)
        ax.set_xscale("symlog", linthresh=0.002)
        ax.set_xlabel("Variance scale")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].legend()
    fig.suptitle("Monte Carlo Latent Noise Robustness")
    fig.tight_layout()
    fig.savefig(out_dir / "mc_mean_p05_p95_curves.png")
    plt.close(fig)


def plot_violin(rows: list[dict], out_dir: Path) -> None:
    scales = sorted({float(r["variance_scale"]) for r in rows if float(r["variance_scale"]) > 0})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150, sharey=True)
    for ax, mode in zip(axes, ["pca", "channel_iso"]):
        data = [[float(r["delta_snr"]) for r in rows if r["mode"] == mode and float(r["variance_scale"]) == s] for s in scales]
        ax.violinplot(data, showmeans=True, showextrema=True)
        ax.set_xticks(np.arange(1, len(scales) + 1))
        ax.set_xticklabels([str(s) for s in scales], rotation=30)
        ax.set_title(mode)
        ax.set_xlabel("Variance scale")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("SNR drop (dB)")
    fig.suptitle("SNR Drop Density")
    fig.tight_layout()
    fig.savefig(out_dir / "snr_drop_violin.png")
    plt.close(fig)


def plot_heatmap(rows: list[dict], out_dir: Path) -> None:
    samples = sorted({r["sample"] for r in rows})
    scales = sorted({float(r["variance_scale"]) for r in rows if float(r["variance_scale"]) > 0})
    for mode in ["pca", "channel_iso"]:
        mat = np.zeros((len(samples), len(scales)), dtype=np.float64)
        for i, sample in enumerate(samples):
            for j, scale in enumerate(scales):
                vals = [float(r["delta_snr"]) for r in rows if r["mode"] == mode and r["sample"] == sample and float(r["variance_scale"]) == scale]
                mat[i, j] = np.mean(vals)
        fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
        im = ax.imshow(mat, aspect="auto", interpolation="nearest")
        ax.set_yticks(np.arange(len(samples)))
        ax.set_yticklabels(samples, fontsize=7)
        ax.set_xticks(np.arange(len(scales)))
        ax.set_xticklabels([str(s) for s in scales], rotation=30)
        ax.set_title(f"{mode}: per-sample mean SNR drop")
        ax.set_xlabel("Variance scale")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("SNR drop (dB)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{mode}_per_sample_snr_drop_heatmap.png")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=parse_run, required=True)
    parser.add_argument("--pca", type=Path, required=True)
    parser.add_argument("--data", default="data/ljspeech_train_splits.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-eval", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--segment-length", type=int, default=65536)
    parser.add_argument("--variance-scales", default="0,0.002,0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--modes", default="pca,channel_iso")
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.data) as f:
        items = json.load(f)
    rng = random.Random(args.seed)
    rng.shuffle(items)
    scales = [float(x) for x in args.variance_scales.split(",")]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, ckpt = load_model(args.run, device)
    pca = np.load(args.pca)
    components = torch.from_numpy(pca["components"].astype(np.float32)).to(device)
    score_std = torch.from_numpy(pca["pca_score_std"].astype(np.float32)).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    rows = []
    used = 0
    for item in items:
        if used >= args.n_eval:
            break
        audio, sr = load_audio(item_path(item), args.run.stride, args.segment_length)
        if len(audio) < args.run.stride * 2:
            continue
        x = torch.from_numpy(audio).view(1, 1, -1).to(device)
        with torch.no_grad():
            z, _, _ = model.encode(x)
            clean = model.decode(z, output_length=x.shape[-1]).squeeze().detach().cpu().float().numpy()
        clean_snr = snr_db(audio, clean)
        clean_hf = hf_metrics(audio, clean, sr)
        sample = Path(item_path(item)).stem
        for mode in modes:
            for scale in scales:
                repeat_count = 1 if scale == 0.0 else args.repeats
                for rep in range(repeat_count):
                    noise, rel = make_noise(mode, z, components, score_std, scale, generator)
                    with torch.no_grad():
                        noisy = model.decode(z + noise, output_length=x.shape[-1]).squeeze().detach().cpu().float().numpy()
                    noisy_snr = snr_db(audio, noisy)
                    noisy_hf = hf_metrics(audio, noisy, sr)
                    rows.append({
                        "run": args.run.name,
                        "checkpoint_epoch": ckpt.get("epoch"),
                        "checkpoint_best_snr": ckpt.get("best_snr"),
                        "sample": sample,
                        "mode": mode,
                        "variance_scale": scale,
                        "repeat": rep,
                        "noise_rel_rms": rel,
                        "clean_snr": clean_snr,
                        "noisy_snr": noisy_snr,
                        "delta_snr": clean_snr - noisy_snr,
                        "clean_hf_snr": clean_hf["hf_snr_3500_9000"],
                        "noisy_hf_snr": noisy_hf["hf_snr_3500_9000"],
                        "delta_hf_snr": clean_hf["hf_snr_3500_9000"] - noisy_hf["hf_snr_3500_9000"],
                        "clean_hf_phase_coh": clean_hf["hf_phase_coh_3500_9000"],
                        "noisy_hf_phase_coh": noisy_hf["hf_phase_coh_3500_9000"],
                        "delta_hf_phase_coh": clean_hf["hf_phase_coh_3500_9000"] - noisy_hf["hf_phase_coh_3500_9000"],
                    })
        used += 1

    if device.type == "cuda":
        torch.cuda.empty_cache()

    summary = summarize(rows)
    sample_rows = sample_summary(rows)
    write_csv(args.out / "mc_per_trial_metrics.csv", rows)
    write_csv(args.out / "mc_summary.csv", summary)
    write_csv(args.out / "mc_per_sample_sensitivity.csv", sample_rows)
    plot_mean_ci(summary, args.out)
    plot_violin(rows, args.out)
    plot_heatmap(rows, args.out)
    with (args.out / "metadata.json").open("w") as f:
        json.dump({
            "run": args.run.name,
            "checkpoint": str(args.run.ckpt),
            "checkpoint_epoch": ckpt.get("epoch"),
            "checkpoint_best_snr": ckpt.get("best_snr"),
            "pca": str(args.pca),
            "n_eval": used,
            "repeats": args.repeats,
            "variance_scales": scales,
            "modes": modes,
            "seed": args.seed,
        }, f, indent=2)
    print(f"Wrote {args.out}; samples={used}; rows={len(rows)}")


if __name__ == "__main__":
    main()
