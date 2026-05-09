"""Diagnose V17.3 frozen-teacher distillation in AE latents."""

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
import torch.nn.functional as F
import torchaudio

from autoencoder.models.autoencoder import WavVAE
from autoencoder.v17_constraints import V17ConstraintModule


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def item_path(item: dict[str, Any]) -> str:
    for key in ("audio_path", "path", "wav_path", "file"):
        if item.get(key):
            return str(item[key])
    raise KeyError(f"No audio path in {item.get('id', '<unknown>')}")


def resolve_path(path_root: Path, item: dict[str, Any]) -> Path:
    path = Path(item_path(item))
    return path if path.is_absolute() else path_root / path


def text_key(item: dict[str, Any]) -> str:
    return str(item.get("same_text_group") or item.get("text_id") or item.get("text") or item.get("id"))


def speaker_key(item: dict[str, Any]) -> str:
    return str(item.get("same_speaker_group") or item.get("speaker_id") or item.get("speaker") or "")


def load_audio(path: Path, sr: int, segment_samples: int, total_stride: int, device: torch.device) -> torch.Tensor:
    data, file_sr = sf.read(path, always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    audio = torch.from_numpy(data.astype(np.float32))
    if file_sr != sr:
        audio = torchaudio.functional.resample(audio, file_sr, sr)
    if audio.numel() >= segment_samples:
        start = ((audio.numel() - segment_samples) // 2 // total_stride) * total_stride
        audio = audio[start : start + segment_samples]
    else:
        audio = F.pad(audio, (0, segment_samples - audio.numel()))
    aligned = (audio.numel() // total_stride) * total_stride
    return audio[:aligned].view(1, 1, -1).to(device)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-30)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-30)
    return a @ b.T


def retrieval_by_text_cross_speaker(features: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, float]:
    ranks = []
    top1 = 0
    for i, row in enumerate(rows):
        candidates = [j for j, r in enumerate(rows) if r["speaker_id"] != row["speaker_id"]]
        positives = [j for j in candidates if rows[j]["text_id"] == row["text_id"]]
        if not candidates or not positives:
            continue
        sims = cosine_matrix(features[i : i + 1], features[candidates])[0]
        ordered = [candidates[k] for k in np.argsort(-sims)]
        rank = min(ordered.index(p) + 1 for p in positives)
        ranks.append(rank)
        top1 += int(rank == 1)
    if not ranks:
        return {"top1": 0.0, "mean_rank": 0.0, "n": 0.0}
    return {"top1": float(top1 / len(ranks)), "mean_rank": float(np.mean(ranks)), "n": float(len(ranks))}


def speaker_centroid_accuracy(features: np.ndarray, rows: list[dict[str, Any]]) -> float:
    speakers = sorted({r["speaker_id"] for r in rows})
    correct = 0
    for i, row in enumerate(rows):
        scores = []
        for spk in speakers:
            idx = [j for j, r in enumerate(rows) if r["speaker_id"] == spk and j != i]
            if not idx:
                scores.append(-np.inf)
                continue
            centroid = features[idx].mean(axis=0, keepdims=True)
            scores.append(float(cosine_matrix(features[i : i + 1], centroid)[0, 0]))
        correct += int(speakers[int(np.argmax(scores))] == row["speaker_id"])
    return correct / max(len(rows), 1)


def paired_cosines(features: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, float]:
    by_text: dict[str, list[int]] = defaultdict(list)
    by_spk: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_text[row["text_id"]].append(i)
        by_spk[row["speaker_id"]].append(i)
    same_text_diff_speaker = []
    diff_text_same_speaker = []
    for idx in by_text.values():
        for pos, a in enumerate(idx):
            for b in idx[pos + 1 :]:
                if rows[a]["speaker_id"] != rows[b]["speaker_id"]:
                    same_text_diff_speaker.append(float(cosine_matrix(features[a : a + 1], features[b : b + 1])[0, 0]))
    for idx in by_spk.values():
        for a, b in zip(idx[::2], idx[1::2]):
            if rows[a]["text_id"] != rows[b]["text_id"]:
                diff_text_same_speaker.append(float(cosine_matrix(features[a : a + 1], features[b : b + 1])[0, 0]))
    return {
        "same_text_diff_speaker_cos_mean": float(np.mean(same_text_diff_speaker)) if same_text_diff_speaker else 0.0,
        "same_text_diff_speaker_cos_std": float(np.std(same_text_diff_speaker)) if same_text_diff_speaker else 0.0,
        "diff_text_same_speaker_cos_mean": float(np.mean(diff_text_same_speaker)) if diff_text_same_speaker else 0.0,
        "diff_text_same_speaker_cos_std": float(np.std(diff_text_same_speaker)) if diff_text_same_speaker else 0.0,
        "paired_n": float(len(same_text_diff_speaker)),
    }


def effective_rank(values: np.ndarray) -> dict[str, float]:
    if values.shape[0] < 2:
        return {"entropy": 0.0, "rank95": 0.0, "rank99": 0.0, "top1": 0.0}
    x = values.astype(np.float64) - values.mean(axis=0, keepdims=True)
    _, sigma, _ = np.linalg.svd(x, full_matrices=False)
    power = sigma**2
    ev = power / (power.sum() + 1e-30)
    cum = np.cumsum(ev)
    return {
        "entropy": float(np.exp(-np.sum(ev * np.log(ev + 1e-30)))),
        "rank95": float(np.searchsorted(cum, 0.95) + 1),
        "rank99": float(np.searchsorted(cum, 0.99) + 1),
        "top1": float(ev[0]),
    }


def feature_summary(name: str, features: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        f"{name}_text_cross_speaker_retrieval": retrieval_by_text_cross_speaker(features, rows),
        f"{name}_pair_cos": paired_cosines(features, rows),
        f"{name}_rank": effective_rank(features),
        f"{name}_speaker_centroid_acc": speaker_centroid_accuracy(features, rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-dim", type=int, default=384)
    parser.add_argument("--encoder-channels", type=int, nargs="+", default=[128, 256, 512, 1024, 1024])
    parser.add_argument("--strides", type=int, nargs="+", default=[2, 4, 4, 8])
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 3, 9])
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--segment-seconds", type=float, default=10.0)
    parser.add_argument("--n-eval", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = WavVAE(
        latent_dim=args.latent_dim,
        encoder_channels=args.encoder_channels,
        strides=args.strides,
        dilations=args.dilations,
        sample_latent=False,
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    v17 = V17ConstraintModule(
        latent_dim=args.latent_dim,
        teacher_distill_weight=1.0,
        teacher_checkpoint=str(args.teacher_checkpoint),
    )
    current = v17.state_dict()
    loaded = {
        k: v
        for k, v in ckpt.get("v17_constraints", {}).items()
        if k in current and tuple(current[k].shape) == tuple(v.shape)
    }
    v17.load_state_dict(loaded, strict=False)
    v17 = v17.to(device).eval()

    items = read_manifest(args.manifest)
    if args.n_eval > 0:
        items = items[: args.n_eval]
    total_stride = math.prod(args.strides)
    segment_samples = int(round(args.segment_seconds * args.sr))
    segment_samples = (segment_samples // total_stride) * total_stride

    rows = []
    adapter_features = []
    teacher_features = []
    latent_means = []
    cosines = []
    for i, item in enumerate(items):
        x = load_audio(resolve_path(args.path_root, item), args.sr, segment_samples, total_stride, device)
        with torch.no_grad():
            z, _, _ = model.encode(x)
            z_in = z.float()
            h = v17.teacher_distill.adapter(z_in)
            adapter = F.normalize(v17.teacher_distill.proj(v17.teacher_distill._pool(h)), dim=-1)
            teacher_audio = x[:, 0, :].float()
            with torch.amp.autocast(device_type=device.type, enabled=False):
                target = F.normalize(v17.teacher_distill.teacher(teacher_audio), dim=-1)
            latent_mean = F.normalize(z.squeeze(0).mean(dim=-1), dim=0)
        rows.append({"row": i, "id": item.get("id", i), "text_id": text_key(item), "speaker_id": speaker_key(item)})
        adapter_features.append(adapter.squeeze(0).cpu().numpy())
        teacher_features.append(target.squeeze(0).cpu().numpy())
        latent_means.append(latent_mean.cpu().numpy())
        cosines.append(float((adapter * target).sum(dim=-1).mean().item()))

    adapter_np = np.stack(adapter_features)
    teacher_np = np.stack(teacher_features)
    latent_np = np.stack(latent_means)
    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_best_snr": ckpt.get("best_snr"),
        "n_items": len(rows),
        "adapter_teacher_cos_mean": float(np.mean(cosines)),
        "adapter_teacher_cos_std": float(np.std(cosines)),
    }
    summary.update(feature_summary("adapter", adapter_np, rows))
    summary.update(feature_summary("teacher", teacher_np, rows))
    summary.update(feature_summary("latent_mean", latent_np, rows))

    with (args.out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (args.out / "items.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
