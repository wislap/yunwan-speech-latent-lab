"""Analyze latent effective rank for WavVAE checkpoints.

This follows the older V14 analysis convention but reports two PCA views:

- frame-level rank: PCA over individual latent frames, best for DiT sequence use.
- utterance-mean rank: PCA over per-sample temporal means, comparable to older
  scripts that used z.mean(dim=-1).

Usage:
  python -m autoencoder.scripts.analyze_effective_rank \
    --out docs/v16.2/figures/effective_rank \
    --run V16.1=outputs/models/wav_vae_v16.1_hf_time_remote_main/best.pt:256:128:2,4,4,4 \
    --run V16.3=outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt:384:256:2,4,4,8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from autoencoder.models.autoencoder import WavVAE


@dataclass
class RunSpec:
    name: str
    ckpt: Path
    latent_dim: int
    stride: int
    strides: list[int]


def parse_run(value: str) -> RunSpec:
    # NAME=CKPT:LATENT_DIM:TOTAL_STRIDE:STRIDES_CSV
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be NAME=CKPT:LATENT_DIM:TOTAL_STRIDE:STRIDES")
    name, rest = value.split("=", 1)
    parts = rest.rsplit(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--run must be NAME=CKPT:LATENT_DIM:TOTAL_STRIDE:STRIDES")
    ckpt, latent_dim, total_stride, strides = parts
    return RunSpec(
        name=name,
        ckpt=Path(ckpt),
        latent_dim=int(latent_dim),
        stride=int(total_stride),
        strides=[int(x) for x in strides.split(",")],
    )


def effective_rank(values: np.ndarray) -> dict[str, float]:
    values = values.astype(np.float64)
    values = values - values.mean(axis=0, keepdims=True)
    _, sigma, _ = np.linalg.svd(values, full_matrices=False)
    power = sigma ** 2
    ev = power / (power.sum() + 1e-30)
    cum = np.cumsum(ev)
    entropy_rank = float(np.exp(-np.sum(ev * np.log(ev + 1e-30))))
    participation = float((power.sum() ** 2) / (np.sum(power ** 2) + 1e-30))
    out = {
        "entropy_rank": entropy_rank,
        "participation_rank": participation,
        "rank_90": float(np.searchsorted(cum, 0.90) + 1),
        "rank_95": float(np.searchsorted(cum, 0.95) + 1),
        "rank_99": float(np.searchsorted(cum, 0.99) + 1),
        "top1_var": float(ev[0]),
        "top5_var": float(cum[min(4, len(cum) - 1)]),
        "top10_var": float(cum[min(9, len(cum) - 1)]),
        "top20_var": float(cum[min(19, len(cum) - 1)]),
    }
    for i in range(min(16, len(sigma))):
        out[f"sigma_{i + 1}"] = float(sigma[i])
    return out


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
    model = model.to(device).eval()
    return model, ckpt


def analyze_run(spec: RunSpec, items: list[dict], args: argparse.Namespace, device: torch.device) -> tuple[dict, np.ndarray, np.ndarray]:
    model, ckpt = load_model(spec, device)
    frame_chunks = []
    mean_chunks = []
    std_chunks = []
    snrs = []

    for item in items[: args.n_eval]:
        data, sr = sf.read(item["audio_path"])
        if data.ndim == 2:
            data = data.mean(axis=1)
        audio = torch.from_numpy(data.astype(np.float32))
        if args.segment_length > 0:
            audio = audio[: args.segment_length]
        aligned = (audio.shape[0] // spec.stride) * spec.stride
        if aligned < spec.stride * 2:
            continue
        audio = audio[:aligned].view(1, 1, -1).to(device)

        with torch.no_grad():
            out = model(audio)
        z = out["z"].squeeze(0).detach().cpu().numpy()  # [C, T]
        std = out["std"].squeeze(0).detach().cpu().numpy()
        x_hat = out["x_hat"].float()
        err = ((x_hat - audio.float()) ** 2).mean().item()
        sig = (audio.float() ** 2).mean().item()
        snrs.append(10.0 * math.log10((sig + 1e-12) / (err + 1e-12)))

        mean_chunks.append(z.mean(axis=1))
        std_chunks.append(std.mean(axis=1))
        frames = z.T
        if args.frame_subsample > 1:
            frames = frames[:: args.frame_subsample]
        frame_chunks.append(frames)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    frame_z = np.concatenate(frame_chunks, axis=0)
    mean_z = np.stack(mean_chunks, axis=0)
    std_z = np.stack(std_chunks, axis=0)
    dim_std_frame = frame_z.std(axis=0)
    dim_std_mean = mean_z.std(axis=0)

    summary = {
        "run": spec.name,
        "checkpoint": str(spec.ckpt),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_best_snr": ckpt.get("best_snr"),
        "total_stride": spec.stride,
        "latent_dim": spec.latent_dim,
        "strides": ",".join(str(x) for x in spec.strides),
        "n_samples": len(mean_z),
        "n_frames": len(frame_z),
        "eval_snr_mean": float(np.mean(snrs)),
        "eval_snr_median": float(np.median(snrs)),
        "z_frame_std_mean": float(dim_std_frame.mean()),
        "z_frame_std_min": float(dim_std_frame.min()),
        "z_frame_std_max": float(dim_std_frame.max()),
        "dead_dims_frame_std_lt_0_01": int((dim_std_frame < 0.01).sum()),
        "low_dims_frame_std_lt_0_05": int((dim_std_frame < 0.05).sum()),
        "active_dims_frame_std_gt_0_1": int((dim_std_frame > 0.1).sum()),
        "z_mean_std_mean": float(dim_std_mean.mean()),
        "encoder_std_mean": float(std_z.mean()),
    }
    for k, v in effective_rank(frame_z).items():
        summary[f"frame_{k}"] = v
    for k, v in effective_rank(mean_z).items():
        summary[f"mean_{k}"] = v
    return summary, frame_z, mean_z


def write_summary(rows: list[dict], out_dir: Path) -> None:
    keys = sorted({k for row in rows for k in row})
    with (out_dir / "effective_rank_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict], out_dir: Path) -> None:
    names = [r["run"] for r in rows]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    width = 0.35
    axes[0].bar(x - width / 2, [r["frame_entropy_rank"] for r in rows], width, label="entropy rank")
    axes[0].bar(x + width / 2, [r["frame_rank_99"] for r in rows], width, label="99% variance dims")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("dims")
    axes[0].set_title("Frame-level latent rank")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(x - width / 2, [r["mean_entropy_rank"] for r in rows], width, label="entropy rank")
    axes[1].bar(x + width / 2, [r["mean_rank_99"] for r in rows], width, label="99% variance dims")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("dims")
    axes[1].set_title("Utterance-mean latent rank")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "effective_rank_summary.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=parse_run, action="append", required=True)
    parser.add_argument("--data", default="data/ljspeech_train_splits.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-eval", type=int, default=64)
    parser.add_argument("--segment-length", type=int, default=65536)
    parser.add_argument("--frame-subsample", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-arrays", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.data) as f:
        items = json.load(f)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    rows = []
    for spec in args.run:
        summary, frame_z, mean_z = analyze_run(spec, items, args, device)
        rows.append(summary)
        if args.save_arrays:
            np.savez_compressed(args.out / f"{spec.name}_rank_arrays.npz", frame_z=frame_z, mean_z=mean_z)
        print(f"{spec.name}: frame_rank={summary['frame_entropy_rank']:.1f}, frame_99={summary['frame_rank_99']:.0f}, mean_rank={summary['mean_entropy_rank']:.1f}, mean_99={summary['mean_rank_99']:.0f}")

    write_summary(rows, args.out)
    plot_summary(rows, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
