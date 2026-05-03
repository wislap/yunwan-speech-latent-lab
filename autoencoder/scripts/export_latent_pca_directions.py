"""Export latent PCA unit directions for robustness tests.

The exported directions are frame-level PCA components in latent channel space.
Each component is a unit vector. We keep all components needed to reach a target
cumulative explained variance, usually 95%.

Sign convention:
  PCA signs are mathematically arbitrary. For reproducibility, each component is
  flipped so the largest-absolute loading is positive.

Usage:
  python -m autoencoder.scripts.export_latent_pca_directions \
    --out docs/v16.3/figures/latent_pca_directions/v16_3 \
    --run V16.3=outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt:384:256:2,4,4,8
"""

from __future__ import annotations

import argparse
import csv
import json
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


def collect_latent_frames(spec: RunSpec, items: list[dict], args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, dict]:
    model, ckpt = load_model(spec, device)
    chunks = []
    used = 0

    for item in items[: args.n_eval]:
        audio, sr = sf.read(item_path(item), always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if args.segment_length > 0:
            audio = audio[: args.segment_length]
        n = (len(audio) // spec.stride) * spec.stride
        if n < spec.stride * 2:
            continue
        audio = audio[:n]
        x = torch.from_numpy(audio).view(1, 1, -1).to(device)
        with torch.no_grad():
            z, _, _ = model.encode(x)
        frames = z.squeeze(0).detach().cpu().float().numpy().T
        if args.frame_subsample > 1:
            frames = frames[:: args.frame_subsample]
        chunks.append(frames)
        used += 1

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not chunks:
        raise RuntimeError("No latent frames collected")
    meta = {
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_best_snr": ckpt.get("best_snr"),
        "n_samples": used,
    }
    return np.concatenate(chunks, axis=0).astype(np.float64), meta


def pca_directions(frames: np.ndarray, variance_target: float) -> dict[str, np.ndarray | int | float]:
    mean = frames.mean(axis=0)
    centered = frames - mean
    cov = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order].T

    for i in range(eigvecs.shape[0]):
        j = int(np.argmax(np.abs(eigvecs[i])))
        if eigvecs[i, j] < 0:
            eigvecs[i] *= -1.0
        eigvecs[i] /= np.linalg.norm(eigvecs[i]) + 1e-30

    explained = eigvals / (eigvals.sum() + 1e-30)
    cumulative = np.cumsum(explained)
    k = int(np.searchsorted(cumulative, variance_target) + 1)
    scores_std = np.sqrt(np.maximum(eigvals, 0.0))
    return {
        "mean": mean,
        "eigvals": eigvals,
        "explained": explained,
        "cumulative": cumulative,
        "components": eigvecs,
        "pca_scores_std": scores_std,
        "k": k,
        "variance_target": variance_target,
    }


def write_outputs(run: RunSpec, result: dict, meta: dict, out_dir: Path, args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    k = int(result["k"])
    components = result["components"][:k]
    explained = result["explained"][:k]
    cumulative = result["cumulative"][:k]
    eigvals = result["eigvals"][:k]
    score_std = result["pca_scores_std"][:k]

    np.savez_compressed(
        out_dir / "frame_p95_pca_directions.npz",
        run=run.name,
        latent_dim=run.latent_dim,
        total_stride=run.stride,
        strides=np.asarray(run.strides, dtype=np.int64),
        variance_target=float(result["variance_target"]),
        k_p95=k,
        mean=result["mean"].astype(np.float32),
        components=components.astype(np.float32),
        explained_variance_ratio=explained.astype(np.float64),
        cumulative_variance=cumulative.astype(np.float64),
        eigenvalues=eigvals.astype(np.float64),
        pca_score_std=score_std.astype(np.float64),
    )

    with (out_dir / "frame_p95_pca_component_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "component",
                "explained_variance_ratio",
                "cumulative_variance",
                "eigenvalue",
                "pca_score_std",
                "max_abs_channel",
                "max_abs_value",
                "l2_norm",
            ],
        )
        writer.writeheader()
        for i in range(k):
            vec = components[i]
            max_ch = int(np.argmax(np.abs(vec)))
            writer.writerow({
                "component": i,
                "explained_variance_ratio": float(explained[i]),
                "cumulative_variance": float(cumulative[i]),
                "eigenvalue": float(eigvals[i]),
                "pca_score_std": float(score_std[i]),
                "max_abs_channel": max_ch,
                "max_abs_value": float(vec[max_ch]),
                "l2_norm": float(np.linalg.norm(vec)),
            })

    with (out_dir / "frame_p95_pca_directions_long.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["component", "channel", "value"])
        writer.writeheader()
        for i, vec in enumerate(components):
            for ch, value in enumerate(vec):
                writer.writerow({"component": i, "channel": ch, "value": float(value)})

    with (out_dir / "latent_mean.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["channel", "mean"])
        writer.writeheader()
        for ch, value in enumerate(result["mean"]):
            writer.writerow({"channel": ch, "mean": float(value)})

    with (out_dir / "metadata.json").open("w") as f:
        json.dump({
            "run": run.name,
            "checkpoint": str(run.ckpt),
            "checkpoint_epoch": meta["checkpoint_epoch"],
            "checkpoint_best_snr": meta["checkpoint_best_snr"],
            "latent_dim": run.latent_dim,
            "total_stride": run.stride,
            "strides": run.strides,
            "n_samples": meta["n_samples"],
            "n_frames": int(args.n_frames_collected),
            "segment_length": args.segment_length,
            "frame_subsample": args.frame_subsample,
            "variance_target": args.variance_target,
            "k_p95": k,
            "sign_convention": "largest absolute loading is positive",
        }, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(np.arange(1, len(result["cumulative"]) + 1), result["cumulative"], label="cumulative")
    ax.axhline(args.variance_target, color="red", linestyle="--", linewidth=1, label=f"{args.variance_target:.0%}")
    ax.axvline(k, color="black", linestyle=":", linewidth=1, label=f"k={k}")
    ax.set_xlabel("PCA component")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title(f"{run.name} latent frame PCA")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "frame_pca_cumulative_variance.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=parse_run, required=True)
    parser.add_argument("--data", default="data/ljspeech_train_splits.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-eval", type=int, default=128)
    parser.add_argument("--segment-length", type=int, default=65536)
    parser.add_argument("--frame-subsample", type=int, default=1)
    parser.add_argument("--variance-target", type=float, default=0.95)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.data) as f:
        items = json.load(f)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    frames, meta = collect_latent_frames(args.run, items, args, device)
    args.n_frames_collected = len(frames)
    result = pca_directions(frames, args.variance_target)
    write_outputs(args.run, result, meta, args.out, args)
    print(f"run={args.run.name} frames={len(frames)} latent_dim={args.run.latent_dim} k_p95={result['k']} out={args.out}")


if __name__ == "__main__":
    main()
