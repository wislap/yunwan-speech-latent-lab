"""Diagnose text/speaker/residual structure in WavVAE latents.

This script is intentionally read-only: it loads a checkpoint and a crossed
manifest, extracts deterministic latent summaries, and reports simple factor
diagnostics that are easier to reason about than a training loss.

Example:
  python -m autoencoder.scripts.latent_factor_diagnostics \
    --checkpoint outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt \
    --manifest CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_long10x_v2/manifest_qc_8_13.jsonl \
    --path-root CosyVoice_main \
    --out outputs/diagnostics/v17_v16_3_latent_factor_qc_8_13
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

from autoencoder.models.autoencoder import WavVAE


def parse_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        items = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected manifest list, got {type(data).__name__}")
    return data


def item_audio_path(item: dict[str, Any]) -> str:
    for key in ("audio_path", "path", "wav_path", "file"):
        if item.get(key):
            return str(item[key])
    raise KeyError(f"No audio path key found in item {item.get('id', '<unknown>')}")


def resolve_audio_path(item: dict[str, Any], path_root: Path) -> Path:
    raw = Path(item_audio_path(item))
    if raw.is_absolute():
        return raw
    return path_root / raw


def effective_rank(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        return {
            "entropy_rank": 0.0,
            "participation_rank": 0.0,
            "rank_90": 0.0,
            "rank_95": 0.0,
            "rank_99": 0.0,
            "top1_var": 0.0,
            "top5_var": 0.0,
            "top10_var": 0.0,
        }
    values = values - values.mean(axis=0, keepdims=True)
    _, sigma, _ = np.linalg.svd(values, full_matrices=False)
    power = sigma**2
    total = power.sum() + 1e-30
    ev = power / total
    cum = np.cumsum(ev)
    return {
        "entropy_rank": float(np.exp(-np.sum(ev * np.log(ev + 1e-30)))),
        "participation_rank": float(total**2 / (np.sum(power**2) + 1e-30)),
        "rank_90": float(np.searchsorted(cum, 0.90) + 1),
        "rank_95": float(np.searchsorted(cum, 0.95) + 1),
        "rank_99": float(np.searchsorted(cum, 0.99) + 1),
        "top1_var": float(ev[0]),
        "top5_var": float(cum[min(4, len(cum) - 1)]),
        "top10_var": float(cum[min(9, len(cum) - 1)]),
    }


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[WavVAE, dict[str, Any]]:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = WavVAE(
        latent_dim=args.latent_dim,
        encoder_channels=parse_ints(args.encoder_channels),
        strides=parse_ints(args.strides),
        dilations=parse_ints(args.dilations),
        sample_latent=False,
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def load_audio(path: Path, target_sr: int, segment_samples: int, crop_mode: str, total_stride: int) -> torch.Tensor:
    audio_np, sr = sf.read(path, always_2d=False)
    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)
    audio = torch.from_numpy(audio_np.astype(np.float32))
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)

    if segment_samples > 0:
        if audio.numel() >= segment_samples:
            max_start = audio.numel() - segment_samples
            if crop_mode == "first":
                start = 0
            elif crop_mode == "last":
                start = max_start
            else:
                start = max_start // 2
            start = (start // total_stride) * total_stride
            audio = audio[start : start + segment_samples]
        else:
            audio = torch.nn.functional.pad(audio, (0, segment_samples - audio.numel()))

    aligned = (audio.numel() // total_stride) * total_stride
    if aligned < total_stride * 2:
        raise ValueError(f"Audio too short after alignment: {path}")
    return audio[:aligned].view(1, 1, -1)


def collect_latents(
    model: WavVAE,
    items: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    total_stride = math.prod(parse_ints(args.strides))
    segment_samples = 0
    if args.segment_seconds > 0:
        segment_samples = int(round(args.segment_seconds * args.sr))
        segment_samples = (segment_samples // total_stride) * total_stride

    rows: list[dict[str, Any]] = []
    mean_chunks: list[np.ndarray] = []
    frame_chunks: list[np.ndarray] = []

    for i, item in enumerate(items[: args.n_eval] if args.n_eval > 0 else items):
        path = resolve_audio_path(item, args.path_root)
        x = load_audio(path, args.sr, segment_samples, args.crop_mode, total_stride).to(device)
        with torch.no_grad():
            z, mean, std = model.encode(x)
        z_np = z.squeeze(0).detach().cpu().float().numpy()
        frames = z_np.T
        if args.frame_subsample > 1:
            frames = frames[:: args.frame_subsample]
        z_mean = z_np.mean(axis=1)

        text_id = str(item.get("same_text_group") or item.get("text_id") or item.get("id") or i)
        speaker_id = str(item.get("same_speaker_group") or item.get("speaker_id") or "unknown")
        row = {
            "row": i,
            "id": item.get("id", i),
            "text_id": text_id,
            "speaker_id": speaker_id,
            "split": item.get("split", ""),
            "domain": item.get("domain", ""),
            "audio_seconds": item.get("audio_seconds", ""),
            "latent_frames": int(z_np.shape[1]),
            "encoder_std_mean": float(std.mean().item()),
            "path": str(path),
        }
        rows.append(row)
        mean_chunks.append(z_mean.astype(np.float32))
        frame_chunks.append(frames.astype(np.float32))

        if args.progress_every > 0 and (i + 1) % args.progress_every == 0:
            print(f"encoded {i + 1}/{len(items)}")

    if not mean_chunks:
        raise RuntimeError("No latents collected")
    return rows, np.stack(mean_chunks, axis=0), np.concatenate(frame_chunks, axis=0)


def group_indices(rows: list[dict[str, Any]], key: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[str(row[key])].append(i)
    return dict(groups)


def add_prefixed(out: dict[str, Any], prefix: str, values: dict[str, float]) -> None:
    for key, value in values.items():
        out[f"{prefix}_{key}"] = value


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-30)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-30)
    return a @ b.T


def speaker_centroid_accuracy(z_mean: np.ndarray, rows: list[dict[str, Any]]) -> float:
    speakers = sorted({str(r["speaker_id"]) for r in rows})
    if len(speakers) < 2:
        return 0.0
    correct = 0
    for i, row in enumerate(rows):
        scores = []
        for spk in speakers:
            idx = [j for j, r in enumerate(rows) if str(r["speaker_id"]) == spk and j != i]
            if not idx:
                scores.append(-np.inf)
                continue
            centroid = z_mean[idx].mean(axis=0, keepdims=True)
            scores.append(float(cosine_matrix(z_mean[i : i + 1], centroid)[0, 0]))
        pred = speakers[int(np.argmax(scores))]
        correct += int(pred == str(row["speaker_id"]))
    return correct / len(rows)


def text_cross_speaker_retrieval(z_mean: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, float]:
    ranks = []
    top1 = 0
    for i, row in enumerate(rows):
        candidates = [j for j, r in enumerate(rows) if r["speaker_id"] != row["speaker_id"]]
        positives = [j for j in candidates if rows[j]["text_id"] == row["text_id"]]
        if not candidates or not positives:
            continue
        sims = cosine_matrix(z_mean[i : i + 1], z_mean[candidates])[0]
        order = np.argsort(-sims)
        ordered_candidates = [candidates[k] for k in order]
        rank = min(ordered_candidates.index(p) + 1 for p in positives)
        ranks.append(rank)
        top1 += int(rank == 1)
    if not ranks:
        return {"top1": 0.0, "mean_rank": 0.0, "n_queries": 0.0}
    return {"top1": float(top1 / len(ranks)), "mean_rank": float(np.mean(ranks)), "n_queries": float(len(ranks))}


def pca_basis(values: np.ndarray, max_k: int = 10) -> np.ndarray:
    if values.shape[0] < 2:
        return np.zeros((values.shape[1], 0), dtype=np.float64)
    centered = values.astype(np.float64) - values.mean(axis=0, keepdims=True)
    _, sigma, vh = np.linalg.svd(centered, full_matrices=False)
    keep = int(min(max_k, np.sum(sigma > 1e-8), vh.shape[0]))
    return vh[:keep].T


def subspace_overlap(a: np.ndarray, b: np.ndarray, max_k: int = 10) -> float:
    ua = pca_basis(a, max_k=max_k)
    ub = pca_basis(b, max_k=max_k)
    k = min(ua.shape[1], ub.shape[1])
    if k == 0:
        return 0.0
    return float(np.linalg.norm(ua[:, :k].T @ ub[:, :k], ord="fro") ** 2 / k)


def factor_diagnostics(z_mean: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    text_groups = group_indices(rows, "text_id")
    speaker_groups = group_indices(rows, "speaker_id")
    global_mean = z_mean.mean(axis=0)

    text_mean = {k: z_mean[idx].mean(axis=0) for k, idx in text_groups.items()}
    speaker_mean = {k: z_mean[idx].mean(axis=0) for k, idx in speaker_groups.items()}
    text_effect_by_row = np.stack([text_mean[r["text_id"]] - global_mean for r in rows], axis=0)
    speaker_effect_by_row = np.stack([speaker_mean[r["speaker_id"]] - global_mean for r in rows], axis=0)
    pred = global_mean + text_effect_by_row + speaker_effect_by_row
    residual = z_mean - pred
    centered = z_mean - global_mean

    total_var = float(np.mean(np.sum(centered**2, axis=1)) + 1e-30)
    text_var = float(np.mean(np.sum(text_effect_by_row**2, axis=1)))
    speaker_var = float(np.mean(np.sum(speaker_effect_by_row**2, axis=1)))
    residual_var = float(np.mean(np.sum(residual**2, axis=1)))

    text_effect_unique = np.stack([v - global_mean for v in text_mean.values()], axis=0)
    speaker_effect_unique = np.stack([v - global_mean for v in speaker_mean.values()], axis=0)

    out: dict[str, Any] = {
        "n_items": len(rows),
        "n_texts": len(text_groups),
        "n_speakers": len(speaker_groups),
        "total_utterance_mean_l2_var": total_var,
        "text_var_share": text_var / total_var,
        "speaker_var_share": speaker_var / total_var,
        "residual_var_share": residual_var / total_var,
        "text_speaker_subspace_overlap_top10": subspace_overlap(text_effect_unique, speaker_effect_unique),
        "speaker_centroid_acc": speaker_centroid_accuracy(z_mean, rows),
    }
    add_prefixed(out, "mean", effective_rank(z_mean))
    add_prefixed(out, "text_effect", effective_rank(text_effect_unique))
    add_prefixed(out, "speaker_effect", effective_rank(speaker_effect_unique))
    add_prefixed(out, "residual", effective_rank(residual))
    for key, value in text_cross_speaker_retrieval(z_mean, rows).items():
        out[f"text_cross_speaker_retrieval_{key}"] = value

    speakers = sorted(speaker_groups)
    paired_deltas = []
    if len(speakers) >= 2:
        base, other = speakers[0], speakers[1]
        for _, idx in text_groups.items():
            by_spk = {rows[i]["speaker_id"]: i for i in idx}
            if base in by_spk and other in by_spk:
                paired_deltas.append(z_mean[by_spk[other]] - z_mean[by_spk[base]])
    if paired_deltas:
        delta = np.stack(paired_deltas, axis=0)
        out["paired_speaker_delta_n"] = len(paired_deltas)
        out["paired_speaker_delta_l2_mean"] = float(np.linalg.norm(delta, axis=1).mean())
        out["paired_speaker_delta_l2_std"] = float(np.linalg.norm(delta, axis=1).std())
        add_prefixed(out, "paired_speaker_delta", effective_rank(delta))
    else:
        out["paired_speaker_delta_n"] = 0
    return out


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--latent-dim", type=int, default=384)
    parser.add_argument("--encoder-channels", default="128,256,512,1024,1024")
    parser.add_argument("--strides", default="2,4,4,8")
    parser.add_argument("--dilations", default="1,3,9")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--segment-seconds", type=float, default=10.0)
    parser.add_argument("--crop-mode", choices=["first", "center", "last"], default="center")
    parser.add_argument("--frame-subsample", type=int, default=4)
    parser.add_argument("--n-eval", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--save-arrays", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args, device)
    items = read_manifest(args.manifest)
    rows, z_mean, z_frames = collect_latents(model, items, args, device)

    summary = factor_diagnostics(z_mean, rows)
    add_prefixed(summary, "frame", effective_rank(z_frames))
    dim_std = z_frames.std(axis=0)
    summary.update({
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_best_snr": ckpt.get("best_snr"),
        "manifest": str(args.manifest),
        "path_root": str(args.path_root),
        "segment_seconds": args.segment_seconds,
        "crop_mode": args.crop_mode,
        "frame_subsample": args.frame_subsample,
        "n_frames": int(z_frames.shape[0]),
        "latent_dim": int(z_mean.shape[1]),
        "frame_std_mean": float(dim_std.mean()),
        "frame_std_min": float(dim_std.min()),
        "frame_std_max": float(dim_std.max()),
        "dead_dims_frame_std_lt_0_01": int((dim_std < 0.01).sum()),
        "low_dims_frame_std_lt_0_05": int((dim_std < 0.05).sum()),
        "active_dims_frame_std_gt_0_1": int((dim_std > 0.1).sum()),
        "encoder_std_mean": float(np.mean([r["encoder_std_mean"] for r in rows])),
    })

    with (args.out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_rows(rows, args.out / "items.csv")
    if args.save_arrays:
        np.savez_compressed(args.out / "latents.npz", z_mean=z_mean.astype(np.float32), z_frames=z_frames.astype(np.float32))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
