"""Generalization stress test for WavVAE checkpoints.

The test reconstructs full-length samples and synthetic variants without
writing audio:

- full: original full utterance
- silence: same utterance with a short inserted silent span
- concat: two full utterances joined with a short silent seam

It reports global reconstruction metrics, time-local error curves, seam-aligned
error curves, and correlations between sample properties and reconstruction
error.

Usage:
  python -m autoencoder.scripts.generalization_stress_test \
    --out docs/v16.3/figures/generalization \
    --run V16.1=outputs/models/wav_vae_v16.1_hf_time_remote_main/best.pt:256:128:2,4,4,4 \
    --run V16.2=outputs/models/wav_vae_v16.2_256x_time_remote_main/best.pt:256:256:2,4,4,8 \
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
from scipy.signal import get_window, stft

from autoencoder.models.autoencoder import WavVAE


@dataclass
class RunSpec:
    name: str
    ckpt: Path
    latent_dim: int
    stride: int
    strides: list[int]


@dataclass
class Case:
    case_id: str
    case_type: str
    audio: np.ndarray
    sr: int
    source_a: str
    source_b: str
    boundary_sample: int
    inserted_silence_sec: float


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


def load_audio(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    peak = float(np.max(np.abs(audio))) + 1e-8
    if peak > 1.0:
        audio = audio / peak
    return audio, sr


def item_path(item: dict) -> str:
    return str(item.get("audio_path") or item.get("path") or item.get("wav_path") or item.get("file"))


def build_cases(items: list[dict], sr: int, args: argparse.Namespace) -> list[Case]:
    rng = random.Random(args.seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    single_ids = indices[: args.n_single]
    concat_ids = indices[args.n_single: args.n_single + 2 * args.n_concat]

    cases: list[Case] = []
    for j, idx in enumerate(single_ids):
        path = item_path(items[idx])
        audio, file_sr = load_audio(path)
        if file_sr != sr:
            raise ValueError(f"Unexpected sample rate {file_sr}: {path}")
        cases.append(Case(
            case_id=f"full_{j:03d}",
            case_type="full",
            audio=audio,
            sr=sr,
            source_a=Path(path).stem,
            source_b="",
            boundary_sample=-1,
            inserted_silence_sec=0.0,
        ))

        silence_sec = rng.uniform(args.silence_min_sec, args.silence_max_sec)
        silence_len = max(1, int(round(silence_sec * sr)))
        lo = max(1, int(0.2 * len(audio)))
        hi = max(lo + 1, int(0.8 * len(audio)))
        pos = rng.randrange(lo, hi)
        silence = np.zeros(silence_len, dtype=np.float32)
        with_silence = np.concatenate([audio[:pos], silence, audio[pos:]])
        cases.append(Case(
            case_id=f"silence_{j:03d}",
            case_type="silence",
            audio=with_silence,
            sr=sr,
            source_a=Path(path).stem,
            source_b="",
            boundary_sample=pos,
            inserted_silence_sec=silence_sec,
        ))

    for j in range(args.n_concat):
        idx_a = concat_ids[2 * j]
        idx_b = concat_ids[2 * j + 1]
        path_a = item_path(items[idx_a])
        path_b = item_path(items[idx_b])
        a, sr_a = load_audio(path_a)
        b, sr_b = load_audio(path_b)
        if sr_a != sr or sr_b != sr:
            raise ValueError(f"Unexpected sample rate in concat pair: {path_a}, {path_b}")
        silence_sec = rng.uniform(args.concat_silence_min_sec, args.concat_silence_max_sec)
        silence = np.zeros(max(1, int(round(silence_sec * sr))), dtype=np.float32)
        boundary = len(a)
        joined = np.concatenate([a, silence, b])
        cases.append(Case(
            case_id=f"concat_{j:03d}",
            case_type="concat",
            audio=joined,
            sr=sr,
            source_a=Path(path_a).stem,
            source_b=Path(path_b).stem,
            boundary_sample=boundary,
            inserted_silence_sec=silence_sec,
        ))

    return cases


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


def align_audio(audio: np.ndarray, stride: int) -> np.ndarray:
    n = (len(audio) // stride) * stride
    return audio[:n].astype(np.float32)


def reconstruct(model: WavVAE, audio: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(audio).view(1, 1, -1).to(device)
    with torch.no_grad():
        y = model(x)["x_hat"].squeeze().detach().cpu().float().numpy()
    return y[: len(audio)]


def snr_db(ref: np.ndarray, pred: np.ndarray) -> float:
    err = pred - ref
    return 10.0 * math.log10((float(np.mean(ref * ref)) + 1e-12) / (float(np.mean(err * err)) + 1e-12))


def local_snr_curve(ref: np.ndarray, pred: np.ndarray, win: int, hop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals = []
    err_rms = []
    centers = []
    for start in range(0, max(1, len(ref) - win + 1), hop):
        r = ref[start:start + win]
        p = pred[start:start + win]
        if len(r) < win:
            break
        e = p - r
        vals.append(10.0 * math.log10((float(np.mean(r * r)) + 1e-12) / (float(np.mean(e * e)) + 1e-12)))
        err_rms.append(math.sqrt(float(np.mean(e * e)) + 1e-12))
        centers.append(start + win / 2)
    return np.asarray(centers), np.asarray(vals), np.asarray(err_rms)


def resample_curve(values: np.ndarray, n: int) -> np.ndarray:
    if len(values) == 0:
        return np.full(n, np.nan)
    if len(values) == 1:
        return np.full(n, values[0])
    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, values)


def sample_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    rms = math.sqrt(float(np.mean(audio * audio)) + 1e-12)
    peak = float(np.max(np.abs(audio))) + 1e-12
    silence_frac = float(np.mean(np.abs(audio) < 1e-4))

    n_fft = 2048
    hop = 512
    if len(audio) < n_fft:
        return {
            "duration_sec": len(audio) / sr,
            "rms": rms,
            "peak": peak,
            "crest": peak / (rms + 1e-12),
            "silence_frac": silence_frac,
            "spectral_centroid": 0.0,
            "hf_energy_ratio_3500_9000": 0.0,
        }
    freqs, _, spec = stft(audio, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    mag2 = np.abs(spec) ** 2
    total = float(np.sum(mag2)) + 1e-12
    centroid = float(np.sum(freqs[:, None] * mag2) / total)
    hf_mask = (freqs >= 3500) & (freqs < 9000)
    hf_ratio = float(np.sum(mag2[hf_mask]) / total)
    return {
        "duration_sec": len(audio) / sr,
        "rms": rms,
        "peak": peak,
        "crest": peak / (rms + 1e-12),
        "silence_frac": silence_frac,
        "spectral_centroid": centroid,
        "hf_energy_ratio_3500_9000": hf_ratio,
    }


def boundary_metrics(ref: np.ndarray, pred: np.ndarray, boundary: int, sr: int) -> dict[str, float]:
    if boundary < 0:
        return {
            "boundary_local_snr": float("nan"),
            "pre_boundary_snr": float("nan"),
            "post_boundary_snr": float("nan"),
            "boundary_error_ratio": float("nan"),
        }
    radius = int(round(0.25 * sr))
    outer = int(round(1.0 * sr))
    n = len(ref)
    seam_lo = max(0, boundary - radius)
    seam_hi = min(n, boundary + radius)
    pre_lo = max(0, boundary - outer)
    pre_hi = max(0, boundary - radius)
    post_lo = min(n, boundary + radius)
    post_hi = min(n, boundary + outer)

    err = pred - ref

    def seg_snr(lo: int, hi: int) -> float:
        if hi <= lo:
            return float("nan")
        r = ref[lo:hi]
        e = err[lo:hi]
        return 10.0 * math.log10((float(np.mean(r * r)) + 1e-12) / (float(np.mean(e * e)) + 1e-12))

    seam_power = float(np.mean(err[seam_lo:seam_hi] ** 2)) if seam_hi > seam_lo else float("nan")
    global_power = float(np.mean(err * err)) + 1e-12
    return {
        "boundary_local_snr": seg_snr(seam_lo, seam_hi),
        "pre_boundary_snr": seg_snr(pre_lo, pre_hi),
        "post_boundary_snr": seg_snr(post_lo, post_hi),
        "boundary_error_ratio": seam_power / global_power,
    }


def boundary_curve(ref: np.ndarray, pred: np.ndarray, boundary: int, sr: int, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if boundary < 0:
        return np.array([]), np.array([])
    err = pred - ref
    half = int(round(args.boundary_window_sec * sr))
    win = int(round(args.local_win_sec * sr))
    hop = int(round(args.local_hop_sec * sr))
    centers = []
    vals = []
    for center in range(boundary - half, boundary + half + 1, hop):
        lo = center - win // 2
        hi = lo + win
        if lo < 0 or hi > len(ref):
            continue
        r = ref[lo:hi]
        e = err[lo:hi]
        vals.append(10.0 * math.log10((float(np.mean(r * r)) + 1e-12) / (float(np.mean(e * e)) + 1e-12)))
        centers.append((center - boundary) / sr)
    return np.asarray(centers), np.asarray(vals)


def pearson(x: list[float], y: list[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 3:
        return float("nan")
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if float(np.std(x_arr)) < 1e-12 or float(np.std(y_arr)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_snr_box(rows: list[dict], out_dir: Path) -> None:
    runs = sorted({r["run"] for r in rows})
    types = ["full", "silence", "concat"]
    fig, axes = plt.subplots(1, len(types), figsize=(13, 4.5), dpi=150, sharey=True)
    for ax, case_type in zip(axes, types):
        data = [[r["snr"] for r in rows if r["run"] == run and r["case_type"] == case_type] for run in runs]
        ax.boxplot(data, labels=runs, showmeans=True)
        ax.set_title(case_type)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Full-length SNR (dB)")
    fig.suptitle("Generalization Stress Test SNR")
    fig.tight_layout()
    fig.savefig(out_dir / "snr_boxplot.png")
    plt.close(fig)


def plot_normalized_curves(curves: dict[tuple[str, str], list[np.ndarray]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150, sharey=True)
    types = ["full", "silence", "concat"]
    x = np.linspace(0, 1, 128)
    for ax, case_type in zip(axes, types):
        for (run, typ), vals in sorted(curves.items()):
            if typ != case_type or not vals:
                continue
            arr = np.vstack(vals)
            mean = np.nanmean(arr, axis=0)
            p25 = np.nanpercentile(arr, 25, axis=0)
            p75 = np.nanpercentile(arr, 75, axis=0)
            ax.plot(x, mean, label=run)
            ax.fill_between(x, p25, p75, alpha=0.12)
        ax.set_title(case_type)
        ax.set_xlabel("Normalized time")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Local SNR (dB)")
    axes[-1].legend()
    fig.suptitle("Time-Local Reconstruction Error")
    fig.tight_layout()
    fig.savefig(out_dir / "time_local_snr_curves.png")
    plt.close(fig)


def plot_boundary_curves(curves: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150, sharey=True)
    for ax, case_type in zip(axes, ["silence", "concat"]):
        for (run, typ), vals in sorted(curves.items()):
            if typ != case_type or not vals:
                continue
            grid = np.linspace(-1.0, 1.0, 121)
            interp = []
            for x, y in vals:
                if len(x) > 1:
                    interp.append(np.interp(grid, x, y, left=np.nan, right=np.nan))
            if not interp:
                continue
            arr = np.vstack(interp)
            ax.plot(grid, np.nanmean(arr, axis=0), label=run)
        ax.axvline(0.0, color="black", linewidth=1, alpha=0.5)
        ax.set_title(f"{case_type} boundary")
        ax.set_xlabel("Seconds from boundary")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Local SNR (dB)")
    axes[-1].legend()
    fig.suptitle("Boundary-Aligned Error")
    fig.tight_layout()
    fig.savefig(out_dir / "boundary_local_snr_curves.png")
    plt.close(fig)


def plot_correlation(corr_rows: list[dict], out_dir: Path) -> None:
    features = sorted({r["feature"] for r in corr_rows})
    runs = sorted({r["run"] for r in corr_rows})
    mat = np.full((len(features), len(runs)), np.nan)
    for i, feat in enumerate(features):
        for j, run in enumerate(runs):
            vals = [r for r in corr_rows if r["feature"] == feat and r["run"] == run and r["target"] == "snr"]
            if vals:
                mat[i, j] = vals[0]["pearson_r"]
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(runs)))
    ax.set_xticklabels(runs)
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features)
    ax.set_title("Feature Correlation With SNR")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson r")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_snr_correlation.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=parse_run, action="append", required=True)
    parser.add_argument("--data", default="data/ljspeech_train_splits.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-single", type=int, default=24)
    parser.add_argument("--n-concat", type=int, default=12)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--silence-min-sec", type=float, default=0.15)
    parser.add_argument("--silence-max-sec", type=float, default=0.45)
    parser.add_argument("--concat-silence-min-sec", type=float, default=0.05)
    parser.add_argument("--concat-silence-max-sec", type=float, default=0.25)
    parser.add_argument("--local-win-sec", type=float, default=0.10)
    parser.add_argument("--local-hop-sec", type=float, default=0.05)
    parser.add_argument("--boundary-window-sec", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.data) as f:
        items = json.load(f)
    cases = build_cases(items, args.sr, args)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    per_case_rows: list[dict] = []
    normalized_curves: dict[tuple[str, str], list[np.ndarray]] = {}
    boundary_curves: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}

    win = max(16, int(round(args.local_win_sec * args.sr)))
    hop = max(8, int(round(args.local_hop_sec * args.sr)))

    for spec in args.run:
        model, ckpt = load_model(spec, device)
        for case in cases:
            audio = align_audio(case.audio, spec.stride)
            boundary = case.boundary_sample
            if boundary >= len(audio):
                boundary = -1
            if len(audio) < spec.stride * 2:
                continue
            pred = reconstruct(model, audio, device)
            centers, local_snr, _ = local_snr_curve(audio, pred, win, hop)
            features = sample_features(audio, case.sr)
            bmetrics = boundary_metrics(audio, pred, boundary, case.sr)
            row = {
                "run": spec.name,
                "checkpoint_epoch": ckpt.get("epoch"),
                "checkpoint_best_snr": ckpt.get("best_snr"),
                "latent_dim": spec.latent_dim,
                "total_stride": spec.stride,
                "case_id": case.case_id,
                "case_type": case.case_type,
                "source_a": case.source_a,
                "source_b": case.source_b,
                "boundary_sec": boundary / case.sr if boundary >= 0 else float("nan"),
                "inserted_silence_sec": case.inserted_silence_sec,
                "snr": snr_db(audio, pred),
                "local_snr_mean": float(np.nanmean(local_snr)),
                "local_snr_p10": float(np.nanpercentile(local_snr, 10)),
                "local_snr_min": float(np.nanmin(local_snr)),
            }
            row.update(features)
            row.update(bmetrics)
            per_case_rows.append(row)
            normalized_curves.setdefault((spec.name, case.case_type), []).append(resample_curve(local_snr, 128))
            bx, by = boundary_curve(audio, pred, boundary, case.sr, args)
            if len(bx) > 0:
                boundary_curves.setdefault((spec.name, case.case_type), []).append((bx, by))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    feature_keys = [
        "duration_sec",
        "rms",
        "crest",
        "silence_frac",
        "spectral_centroid",
        "hf_energy_ratio_3500_9000",
        "inserted_silence_sec",
        "boundary_error_ratio",
    ]
    targets = ["snr", "local_snr_p10", "boundary_local_snr"]
    corr_rows = []
    for run in sorted({r["run"] for r in per_case_rows}):
        run_rows = [r for r in per_case_rows if r["run"] == run]
        for feature in feature_keys:
            for target in targets:
                corr_rows.append({
                    "run": run,
                    "feature": feature,
                    "target": target,
                    "pearson_r": pearson([r.get(feature, float("nan")) for r in run_rows], [r.get(target, float("nan")) for r in run_rows]),
                    "n": len(run_rows),
                })

    write_csv(args.out / "per_case_metrics.csv", per_case_rows)
    write_csv(args.out / "feature_correlations.csv", corr_rows)
    plot_snr_box(per_case_rows, args.out)
    plot_normalized_curves(normalized_curves, args.out)
    plot_boundary_curves(boundary_curves, args.out)
    plot_correlation(corr_rows, args.out)
    print(f"Wrote {args.out}")
    print(f"cases={len(cases)}, rows={len(per_case_rows)}")


if __name__ == "__main__":
    main()
