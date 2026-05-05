"""Diagnose V17 cross-factor projection heads.

This complements latent_factor_diagnostics.py: the cross-factor constraint acts
inside its phoneme/speaker projection heads, so raw utterance-mean latents can
look worse even when the constrained factor spaces improve.
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
import torch.nn.functional as F
import torchaudio

from autoencoder.models.autoencoder import WavVAE
from autoencoder.v17_constraints import V17ConstraintModule


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def item_path(item: dict[str, Any]) -> str:
    for key in ("audio_path", "path", "wav_path", "file"):
        if item.get(key):
            return str(item[key])
    raise KeyError(f"No audio path in {item.get('id', '<unknown>')}")


def resolve_path(path_root: Path, item: dict[str, Any]) -> Path:
    p = Path(item_path(item))
    return p if p.is_absolute() else path_root / p


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
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["text_id"]].append(i)
    same_text_diff_speaker = []
    diff_text_same_speaker = []
    for idx in groups.values():
        for a in idx:
            for b in idx:
                if a < b and rows[a]["speaker_id"] != rows[b]["speaker_id"]:
                    same_text_diff_speaker.append(float(cosine_matrix(features[a : a + 1], features[b : b + 1])[0, 0]))
    by_spk: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_spk[row["speaker_id"]].append(i)
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
    centered = values.astype(np.float64) - values.mean(axis=0, keepdims=True)
    _, sigma, _ = np.linalg.svd(centered, full_matrices=False)
    power = sigma**2
    ev = power / (power.sum() + 1e-30)
    cum = np.cumsum(ev)
    return {
        "entropy": float(np.exp(-np.sum(ev * np.log(ev + 1e-30)))),
        "rank95": float(np.searchsorted(cum, 0.95) + 1),
        "rank99": float(np.searchsorted(cum, 0.99) + 1),
        "top1": float(ev[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
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
        ext_weight=0.0,
        semantic_weight=0.0,
        cross_factor_weight=1.0,
        flow_weight=0.0,
        flow_hidden_dim=256,
        flow_depth=2,
    )
    current = v17.state_dict()
    loaded = {k: v for k, v in ckpt.get("v17_constraints", {}).items() if k in current and tuple(current[k].shape) == tuple(v.shape)}
    v17.load_state_dict(loaded, strict=False)
    v17 = v17.to(device).eval()

    items = read_manifest(args.manifest)
    if args.n_eval > 0:
        items = items[: args.n_eval]
    total_stride = math.prod(args.strides)
    segment_samples = int(round(args.segment_seconds * args.sr))
    segment_samples = (segment_samples // total_stride) * total_stride

    rows = []
    phoneme_features = []
    speaker_features = []
    latent_means = []
    for i, item in enumerate(items):
        x = load_audio(resolve_path(args.path_root, item), args.sr, segment_samples, total_stride, device)
        with torch.no_grad():
            z, _, _ = model.encode(x)
            phoneme, speaker = v17.cross_factor._encode(z)
        rows.append({
            "row": i,
            "id": item.get("id", i),
            "text_id": text_key(item),
            "speaker_id": speaker_key(item),
        })
        phoneme_features.append(phoneme.squeeze(0).detach().cpu().numpy())
        speaker_features.append(speaker.squeeze(0).detach().cpu().numpy())
        latent_means.append(z.squeeze(0).mean(dim=-1).detach().cpu().numpy())

    ph = np.stack(phoneme_features)
    sp = np.stack(speaker_features)
    lm = np.stack(latent_means)

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_best_snr": ckpt.get("best_snr"),
        "n_items": len(rows),
        "phoneme_text_cross_speaker_retrieval": retrieval_by_text_cross_speaker(ph, rows),
        "speaker_text_cross_speaker_retrieval": retrieval_by_text_cross_speaker(sp, rows),
        "latent_text_cross_speaker_retrieval": retrieval_by_text_cross_speaker(lm, rows),
        "phoneme_speaker_centroid_acc": speaker_centroid_accuracy(ph, rows),
        "speaker_speaker_centroid_acc": speaker_centroid_accuracy(sp, rows),
        "latent_speaker_centroid_acc": speaker_centroid_accuracy(lm, rows),
        "phoneme_pair_cos": paired_cosines(ph, rows),
        "speaker_pair_cos": paired_cosines(sp, rows),
        "latent_pair_cos": paired_cosines(lm, rows),
        "phoneme_rank": effective_rank(ph),
        "speaker_rank": effective_rank(sp),
        "latent_mean_rank": effective_rank(lm),
    }

    with (args.out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (args.out / "features.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
