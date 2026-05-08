"""Compare reconstruction and V17 constraint gradients on the AE backbone.

Queue-based V17 constraints first enqueue the paired same_text_group item and
then measure gradients on the other speaker variant. The script reports
gradient norms and cosine similarity per model component.
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

from autoencoder.losses import MultiScaleMelLoss, compute_reconstruction_loss
from autoencoder.losses_stft import MultiResolutionSTFTLoss
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
    raise KeyError(f"No audio path in item {item.get('id', '<unknown>')}")


def resolve_path(path_root: Path, item: dict[str, Any]) -> Path:
    p = Path(item_path(item))
    if p.is_absolute():
        return p
    return path_root / p


def load_audio(path: Path, sr: int, segment_length: int, total_stride: int, device: torch.device) -> torch.Tensor:
    data, file_sr = sf.read(path, always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    audio = torch.from_numpy(data.astype(np.float32))
    if file_sr != sr:
        audio = torchaudio.functional.resample(audio, file_sr, sr)
    if audio.numel() >= segment_length:
        start = ((audio.numel() - segment_length) // 2 // total_stride) * total_stride
        audio = audio[start : start + segment_length]
    else:
        audio = F.pad(audio, (0, segment_length - audio.numel()))
    aligned = (audio.numel() // total_stride) * total_stride
    return audio[:aligned].view(1, 1, -1).to(device)


def text_key(item: dict[str, Any]) -> str:
    return str(item.get("same_text_group") or item.get("text_id") or item.get("text") or item.get("id"))


def paired_items(items: list[dict[str, Any]], n_pairs: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[text_key(item)].append(item)
    pairs = []
    for rows in groups.values():
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda r: str(r.get("speaker_id", "")))
        pairs.append((rows[0], rows[1]))
        if len(pairs) >= n_pairs:
            break
    if not pairs:
        raise RuntimeError("No same_text_group pairs found")
    return pairs


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[WavVAE, V17ConstraintModule | None, dict[str, Any]]:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = WavVAE(
        latent_dim=args.latent_dim,
        encoder_channels=args.encoder_channels,
        strides=args.strides,
        dilations=args.dilations,
        sample_latent=False,
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device).train()

    v17 = None
    if "v17_constraints" in ckpt:
        v17 = V17ConstraintModule(
            latent_dim=args.latent_dim,
            ext_weight=0.0,
            flow_weight=0.0,
            semantic_weight=args.semantic_weight,
            semantic_hidden_dim=args.semantic_hidden_dim,
            semantic_proj_dim=args.semantic_proj_dim,
            semantic_depth=args.semantic_depth,
            semantic_infonce_weight=1.0,
            semantic_js_weight=args.semantic_js_weight,
            semantic_temperature=args.semantic_temperature,
            semantic_queue_size=args.semantic_queue_size,
            semantic_positive_key=args.semantic_positive_key,
            semantic_detach_adapter_latent=False,
            cross_factor_weight=args.cross_factor_weight,
            cross_factor_hidden_dim=args.cross_factor_hidden_dim,
            cross_factor_phoneme_dim=args.cross_factor_phoneme_dim,
            cross_factor_speaker_dim=args.cross_factor_speaker_dim,
            cross_factor_queue_size=args.cross_factor_queue_size,
            cross_factor_positive_key=args.cross_factor_positive_key,
            cross_factor_phoneme_weight=args.cross_factor_phoneme_weight,
            cross_factor_phoneme_nce_weight=args.cross_factor_phoneme_nce_weight,
            cross_factor_phoneme_temperature=args.cross_factor_phoneme_temperature,
            cross_factor_phoneme_uniform_weight=args.cross_factor_phoneme_uniform_weight,
            cross_factor_phoneme_soft_weight=args.cross_factor_phoneme_soft_weight,
            cross_factor_phoneme_soft_hard_weight=args.cross_factor_phoneme_soft_hard_weight,
            cross_factor_phoneme_soft_text_buckets=args.cross_factor_phoneme_soft_text_buckets,
            cross_factor_phoneme_same_speaker_neg_weight=args.cross_factor_phoneme_same_speaker_neg_weight,
            cross_factor_phoneme_same_speaker_neg_margin=args.cross_factor_phoneme_same_speaker_neg_margin,
            cross_factor_phoneme_same_speaker_neg_max_cos=args.cross_factor_phoneme_same_speaker_neg_max_cos,
            cross_factor_speaker_weight=args.cross_factor_speaker_weight,
            cross_factor_speaker_same_weight=args.cross_factor_speaker_same_weight,
            cross_factor_speaker_target_cos=args.cross_factor_speaker_target_cos,
            cross_factor_speaker_margin_mode=args.cross_factor_speaker_margin_mode,
            cross_factor_speaker_neg_margin=args.cross_factor_speaker_neg_margin,
            cross_factor_speaker_pos_margin=args.cross_factor_speaker_pos_margin,
            cross_factor_rank_guard_min_std=args.cross_factor_rank_guard_min_std,
            cross_factor_rank_guard_max_top1=args.cross_factor_rank_guard_max_top1,
            cross_factor_phoneme_rank_guard_weight=args.cross_factor_phoneme_rank_guard_weight,
            cross_factor_speaker_rank_guard_weight=args.cross_factor_speaker_rank_guard_weight,
            cross_factor_latent_rank_guard_weight=args.cross_factor_latent_rank_guard_weight,
            cross_factor_rank_guard_covariance_weight=args.cross_factor_rank_guard_covariance_weight,
            cross_factor_rank_guard_top1_weight=args.cross_factor_rank_guard_top1_weight,
            cross_factor_detach_latent=False,
        )
        current = v17.state_dict()
        loaded = {
            k: v
            for k, v in ckpt["v17_constraints"].items()
            if k in current and tuple(current[k].shape) == tuple(v.shape)
        }
        missing, unexpected = v17.load_state_dict(loaded, strict=False)
        skipped = len(ckpt["v17_constraints"]) - len(loaded)
        print(
            f"Loaded V17 constraints: loaded={len(loaded)} "
            f"skipped_shape_or_missing={skipped} missing={len(missing)} unexpected={len(unexpected)}"
        )
        v17 = v17.to(device).train()
    return model, v17, ckpt


def components(model: WavVAE) -> list[tuple[str, torch.nn.Module]]:
    out: list[tuple[str, torch.nn.Module]] = [
        ("encoder.stem", model.encoder.stem),
        ("encoder.proj", model.encoder.proj),
        ("decoder.proj", model.decoder.proj),
        ("decoder.tail", model.decoder.tail),
    ]
    for i, block in enumerate(model.encoder.blocks):
        out.append((f"encoder.block.{i}", block))
    for i, block in enumerate(model.decoder.blocks):
        out.append((f"decoder.block.{i}", block))
    return out


def grad_vector(module: torch.nn.Module) -> torch.Tensor | None:
    pieces = []
    for p in module.parameters():
        if p.grad is None:
            pieces.append(torch.zeros_like(p, dtype=torch.float32).flatten())
        else:
            pieces.append(p.grad.detach().float().flatten())
    if not pieces:
        return None
    return torch.cat(pieces)


def summarize_pair(name: str, recon: torch.Tensor, constraint: torch.Tensor, weight: float) -> dict[str, float | str]:
    recon_norm = float(recon.norm().item())
    constraint_norm = float(constraint.norm().item())
    weighted_constraint_norm = constraint_norm * weight
    if recon_norm <= 0.0 or constraint_norm <= 0.0:
        cosine = 0.0
    else:
        cosine = float(F.cosine_similarity(recon.unsqueeze(0), constraint.unsqueeze(0)).item())
    return {
        "component": name,
        "recon_norm": recon_norm,
        "constraint_norm_unweighted": constraint_norm,
        "constraint_norm_weighted": weighted_constraint_norm,
        "constraint_weighted_over_recon": weighted_constraint_norm / max(recon_norm, 1e-30),
        "constraint_unweighted_over_recon": constraint_norm / max(recon_norm, 1e-30),
        "cosine": cosine,
    }


def enqueue_seed(v17: V17ConstraintModule, z: torch.Tensor, metadata: list[dict[str, Any]], constraint: str) -> None:
    if constraint == "semantic":
        v17.semantic(z, metadata)
    elif constraint == "cross_factor":
        v17.cross_factor(z, metadata)
    else:
        raise ValueError(f"Unsupported constraint: {constraint}")


def constraint_loss(
    v17: V17ConstraintModule,
    z: torch.Tensor,
    metadata: list[dict[str, Any]],
    constraint: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], float]:
    if constraint == "semantic":
        loss, logs = v17.semantic(z, metadata)
        return loss, logs, float(v17.semantic_weight)
    if constraint == "cross_factor":
        loss, logs = v17.cross_factor(z, metadata)
        return loss, logs, float(v17.cross_factor_weight)
    raise ValueError(f"Unsupported constraint: {constraint}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--constraint", choices=["semantic", "cross_factor"], default="semantic")
    parser.add_argument("--n-pairs", type=int, default=12)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--segment-seconds", type=float, default=10.0)
    parser.add_argument("--latent-dim", type=int, default=384)
    parser.add_argument("--encoder-channels", type=int, nargs="+", default=[128, 256, 512, 1024, 1024])
    parser.add_argument("--strides", type=int, nargs="+", default=[2, 4, 4, 8])
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 3, 9])
    parser.add_argument("--semantic-weight", type=float, default=0.05)
    parser.add_argument("--semantic-hidden-dim", type=int, default=384)
    parser.add_argument("--semantic-proj-dim", type=int, default=192)
    parser.add_argument("--semantic-depth", type=int, default=4)
    parser.add_argument("--semantic-js-weight", type=float, default=0.05)
    parser.add_argument("--semantic-temperature", type=float, default=0.12)
    parser.add_argument("--semantic-queue-size", type=int, default=256)
    parser.add_argument("--semantic-positive-key", default="same_text_group")
    parser.add_argument("--cross-factor-weight", type=float, default=0.4)
    parser.add_argument("--cross-factor-hidden-dim", type=int, default=384)
    parser.add_argument("--cross-factor-phoneme-dim", type=int, default=192)
    parser.add_argument("--cross-factor-speaker-dim", type=int, default=64)
    parser.add_argument("--cross-factor-queue-size", type=int, default=256)
    parser.add_argument("--cross-factor-positive-key", default="same_text_group")
    parser.add_argument("--cross-factor-phoneme-weight", type=float, default=0.5)
    parser.add_argument("--cross-factor-phoneme-nce-weight", type=float, default=0.2)
    parser.add_argument("--cross-factor-phoneme-temperature", type=float, default=0.12)
    parser.add_argument("--cross-factor-phoneme-uniform-weight", type=float, default=0.05)
    parser.add_argument("--cross-factor-phoneme-soft-weight", type=float, default=1.0)
    parser.add_argument("--cross-factor-phoneme-soft-hard-weight", type=float, default=2.0)
    parser.add_argument("--cross-factor-phoneme-soft-text-buckets", type=int, default=512)
    parser.add_argument("--cross-factor-phoneme-same-speaker-neg-weight", type=float, default=0.0)
    parser.add_argument("--cross-factor-phoneme-same-speaker-neg-margin", type=float, default=0.15)
    parser.add_argument("--cross-factor-phoneme-same-speaker-neg-max-cos", type=float, default=0.65)
    parser.add_argument("--cross-factor-speaker-weight", type=float, default=0.75)
    parser.add_argument("--cross-factor-speaker-same-weight", type=float, default=0.0)
    parser.add_argument("--cross-factor-speaker-target-cos", type=float, default=-1.0)
    parser.add_argument("--cross-factor-speaker-margin-mode", action="store_true")
    parser.add_argument("--cross-factor-speaker-neg-margin", type=float, default=-0.2)
    parser.add_argument("--cross-factor-speaker-pos-margin", type=float, default=0.6)
    parser.add_argument("--cross-factor-rank-guard-min-std", type=float, default=0.03)
    parser.add_argument("--cross-factor-rank-guard-max-top1", type=float, default=0.75)
    parser.add_argument("--cross-factor-phoneme-rank-guard-weight", type=float, default=0.0)
    parser.add_argument("--cross-factor-speaker-rank-guard-weight", type=float, default=0.0)
    parser.add_argument("--cross-factor-latent-rank-guard-weight", type=float, default=0.0)
    parser.add_argument("--cross-factor-rank-guard-covariance-weight", type=float, default=0.0)
    parser.add_argument("--cross-factor-rank-guard-top1-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    total_stride = math.prod(args.strides)
    segment_length = int(round(args.segment_seconds * args.sr))
    segment_length = (segment_length // total_stride) * total_stride

    model, v17, ckpt = load_model(args, device)
    if v17 is None:
        raise RuntimeError("Checkpoint has no v17_constraints state")

    mrstft = MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_sizes=[512, 1024, 2048],
        sr=args.sr,
    ).to(device)
    mel = MultiScaleMelLoss(sr=args.sr).to(device)
    pairs = paired_items(read_manifest(args.manifest), args.n_pairs)
    comps = components(model)
    amp_enabled = device.type == "cuda" and not args.no_amp

    all_rows = []
    loss_rows = []
    for pair_idx, (seed_item, current_item) in enumerate(pairs):
        seed_x = load_audio(resolve_path(args.path_root, seed_item), args.sr, segment_length, total_stride, device)
        x = load_audio(resolve_path(args.path_root, current_item), args.sr, segment_length, total_stride, device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                seed_out = model(seed_x)
                enqueue_seed(v17, seed_out["z"], [seed_item], args.constraint)

        model.zero_grad(set_to_none=True)
        v17.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            out = model(x)
            recon_losses = compute_reconstruction_loss(
                x,
                out["x_hat"],
                mrstft,
                mel,
                out["reg_loss"],
                l1_weight=10.0,
                l2_weight=0.0,
                stft_weight=0.5,
                mel_weight=0.1,
            )
        recon_losses["total"].backward()
        recon_grads = {name: grad_vector(module).detach().cpu() for name, module in comps}

        model.zero_grad(set_to_none=True)
        v17.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            out = model(x)
            raw_constraint_loss, constraint_logs, active_weight = constraint_loss(
                v17, out["z"], [current_item], args.constraint
            )
        raw_constraint_loss.backward()
        constraint_grads = {name: grad_vector(module).detach().cpu() for name, module in comps}

        loss_row = {
            "pair": pair_idx,
            "seed_id": seed_item.get("id", ""),
            "current_id": current_item.get("id", ""),
            "text_key": text_key(current_item),
            "recon_loss": float(recon_losses["total"].detach().cpu()),
            "constraint": args.constraint,
            "constraint_loss_unweighted": float(raw_constraint_loss.detach().cpu()),
            "constraint_loss_weighted": float(raw_constraint_loss.detach().cpu()) * active_weight,
            "constraint_weight": active_weight,
        }
        for key, value in constraint_logs.items():
            if torch.is_tensor(value) and value.numel() == 1:
                loss_row[key] = float(value.detach().cpu())
        loss_rows.append(loss_row)
        for name, _ in comps:
            row = summarize_pair(name, recon_grads[name], constraint_grads[name], active_weight)
            row["pair"] = pair_idx
            row["constraint"] = args.constraint
            all_rows.append(row)

    keys = sorted({k for row in all_rows for k in row})
    with (args.out / "component_gradients.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_rows)

    with (args.out / "losses.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in loss_rows for k in row}))
        writer.writeheader()
        writer.writerows(loss_rows)

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_best_snr": ckpt.get("best_snr"),
        "manifest": str(args.manifest),
        "n_pairs": len(pairs),
        "constraint": args.constraint,
        "semantic_weight": args.semantic_weight,
        "cross_factor_weight": args.cross_factor_weight,
        "components": {},
    }
    for name, _ in comps:
        rows = [r for r in all_rows if r["component"] == name]
        summary["components"][name] = {
            "recon_norm_mean": float(np.mean([r["recon_norm"] for r in rows])),
            "constraint_norm_unweighted_mean": float(np.mean([r["constraint_norm_unweighted"] for r in rows])),
            "constraint_norm_weighted_mean": float(np.mean([r["constraint_norm_weighted"] for r in rows])),
            "constraint_weighted_over_recon_mean": float(np.mean([r["constraint_weighted_over_recon"] for r in rows])),
            "constraint_unweighted_over_recon_mean": float(np.mean([r["constraint_unweighted_over_recon"] for r in rows])),
            "cosine_mean": float(np.mean([r["cosine"] for r in rows])),
            "cosine_min": float(np.min([r["cosine"] for r in rows])),
            "cosine_max": float(np.max([r["cosine"] for r in rows])),
        }
    summary["losses"] = {
        "recon_loss_mean": float(np.mean([r["recon_loss"] for r in loss_rows])),
        "constraint_loss_unweighted_mean": float(np.mean([r["constraint_loss_unweighted"] for r in loss_rows])),
        "constraint_loss_weighted_mean": float(np.mean([r["constraint_loss_weighted"] for r in loss_rows])),
    }
    with (args.out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
