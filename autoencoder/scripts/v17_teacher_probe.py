"""Frozen-teacher probe for V17 content structure.

This script asks a narrow question:

    Can a lightweight sequence adapter extract pretrained speech-teacher
    content features from a frozen V16.3 WavVAE latent sequence?

It deliberately does not update the AE. If this fails, stronger V17 losses on
the AE backbone are unlikely to be the right next move. If it succeeds, future
experiments can warm up the adapter first and only then let small teacher
gradients reach high encoder/bottleneck layers.
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
import torch.nn as nn
import torch.nn.functional as F

from autoencoder.models.autoencoder import WavVAE


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected manifest list, got {type(data).__name__}")
    return data


def item_path(item: dict[str, Any]) -> str:
    for key in ("audio_path", "path", "wav_path", "file"):
        if item.get(key):
            return str(item[key])
    raise KeyError(f"No audio path in item {item.get('id', '<unknown>')}")


def resolve_path(path_root: Path, item: dict[str, Any]) -> Path:
    p = Path(item_path(item))
    return p if p.is_absolute() else path_root / p


def text_key(item: dict[str, Any]) -> str:
    return str(item.get("same_text_group") or item.get("text_id") or item.get("text") or item.get("id") or "")


def speaker_key(item: dict[str, Any]) -> str:
    return str(item.get("same_speaker_group") or item.get("speaker_id") or item.get("speaker") or "")


def parse_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def load_audio(path: Path, sr: int, segment_samples: int, crop_mode: str, total_stride: int) -> torch.Tensor:
    audio_np, file_sr = sf.read(path, always_2d=False)
    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)
    audio = torch.from_numpy(audio_np.astype(np.float32))
    if file_sr != sr:
        import torchaudio

        audio = torchaudio.functional.resample(audio, file_sr, sr)
    if segment_samples > 0:
        if audio.numel() >= segment_samples:
            max_start = audio.numel() - segment_samples
            if crop_mode == "first":
                start = 0
            elif crop_mode == "last":
                start = max_start
            elif crop_mode == "random":
                start = int(torch.randint(0, max_start + 1, (1,)).item())
            else:
                start = max_start // 2
            start = (start // total_stride) * total_stride
            audio = audio[start : start + segment_samples]
        else:
            audio = F.pad(audio, (0, segment_samples - audio.numel()))
    aligned = (audio.numel() // total_stride) * total_stride
    if aligned < total_stride * 2:
        audio = F.pad(audio, (0, total_stride * 2 - audio.numel()))
        aligned = (audio.numel() // total_stride) * total_stride
    return audio[:aligned]


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x) / math.sqrt(2.0)


class LatentTeacherAdapter(nn.Module):
    def __init__(self, latent_dim: int, teacher_dim: int, hidden_dim: int = 512, depth: int = 4):
        super().__init__()
        self.in_proj = nn.Conv1d(latent_dim, hidden_dim, 1)
        self.blocks = nn.ModuleList(
            [ResidualConvBlock(hidden_dim, dilation=2 ** (i % 3)) for i in range(depth)]
        )
        self.out_norm = nn.GroupNorm(8, hidden_dim)
        self.out_proj = nn.Conv1d(hidden_dim, teacher_dim, 1)

    def forward(self, z: torch.Tensor, target_frames: int) -> torch.Tensor:
        h = self.in_proj(z.float())
        for block in self.blocks:
            h = block(h)
        h = self.out_proj(F.silu(self.out_norm(h)))
        if h.shape[-1] != target_frames:
            h = F.interpolate(h, size=target_frames, mode="linear", align_corners=False)
        return h.transpose(1, 2)


def load_ae(args: argparse.Namespace, device: torch.device) -> WavVAE:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = WavVAE(
        latent_dim=args.latent_dim,
        encoder_channels=parse_ints(args.encoder_channels),
        strides=parse_ints(args.strides),
        dilations=parse_ints(args.dilations),
        sample_latent=False,
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_teacher(bundle_name: str, device: torch.device) -> tuple[nn.Module, int, int]:
    import torchaudio

    bundle = getattr(torchaudio.pipelines, bundle_name)
    teacher = bundle.get_model().to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    sample_rate = int(bundle.sample_rate)
    aux = getattr(bundle, "_params", {})
    teacher_dim = int(aux.get("encoder_embed_dim", 768)) if isinstance(aux, dict) else 768
    return teacher, sample_rate, teacher_dim


@torch.no_grad()
def teacher_features(teacher: nn.Module, audio: torch.Tensor, audio_sr: int, teacher_sr: int, device: torch.device) -> torch.Tensor:
    import torchaudio

    wav = audio.to(device)
    if audio_sr != teacher_sr:
        wav = torchaudio.functional.resample(wav, audio_sr, teacher_sr)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    feats, _ = teacher.extract_features(wav)
    return feats[-1].float()


def encode_batch(
    ae: WavVAE,
    teacher: nn.Module,
    adapter: LatentTeacherAdapter,
    audio: torch.Tensor,
    args: argparse.Namespace,
    teacher_sr: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    x = audio.unsqueeze(1).to(device)
    with torch.no_grad():
        z, _, _ = ae.encode(x)
        target = teacher_features(teacher, audio, args.sr, teacher_sr, device)
        target = F.layer_norm(target, (target.shape[-1],))
    pred = adapter(z, target.shape[1])
    pred = F.layer_norm(pred, (pred.shape[-1],))
    return pred, target


def alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth_l1_weight: float,
    pairwise_weight: float,
    variance_weight: float,
    min_std: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    frame_cos = (pred_n * target_n).sum(dim=-1)
    frame_loss = 1.0 - frame_cos.mean()
    pool_pred = F.normalize(pred.mean(dim=1), dim=-1)
    pool_target = F.normalize(target.mean(dim=1), dim=-1)
    pool_cos = (pool_pred * pool_target).sum(dim=-1)
    pool_loss = 1.0 - pool_cos.mean()
    l1 = F.smooth_l1_loss(pred, target)
    if pred.shape[0] > 1:
        student_sim = pool_pred @ pool_pred.T
        teacher_sim = pool_target @ pool_target.T
        eye = torch.eye(pred.shape[0], device=pred.device, dtype=torch.bool)
        pairwise = F.smooth_l1_loss(student_sim.masked_select(~eye), teacher_sim.masked_select(~eye))
        std = torch.sqrt(pred.mean(dim=1).float().var(dim=0, unbiased=False) + 1e-4)
        variance = F.relu(float(min_std) - std).mean()
    else:
        pairwise = pred.sum() * 0.0
        variance = pred.sum() * 0.0
    loss = (
        frame_loss
        + pool_loss
        + float(smooth_l1_weight) * l1
        + float(pairwise_weight) * pairwise
        + float(variance_weight) * variance
    )
    return loss, {
        "frame_cos": float(frame_cos.mean().detach().cpu()),
        "pool_cos": float(pool_cos.mean().detach().cpu()),
        "smooth_l1": float(l1.detach().cpu()),
        "pairwise": float(pairwise.detach().cpu()),
        "variance": float(variance.detach().cpu()),
    }


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


def paired_cosines(features: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, float]:
    by_text: dict[str, list[int]] = defaultdict(list)
    by_spk: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_text[row["text_id"]].append(i)
        by_spk[row["speaker_id"]].append(i)
    same_text_diff_speaker = []
    diff_text_same_speaker = []
    for idxs in by_text.values():
        for a in idxs:
            for b in idxs:
                if a < b and rows[a]["speaker_id"] != rows[b]["speaker_id"]:
                    same_text_diff_speaker.append(float(cosine_matrix(features[a : a + 1], features[b : b + 1])[0, 0]))
    for idxs in by_spk.values():
        for a, b in zip(idxs[::2], idxs[1::2]):
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


@torch.no_grad()
def evaluate(
    ae: WavVAE,
    teacher: nn.Module,
    adapter: LatentTeacherAdapter,
    items: list[dict[str, Any]],
    args: argparse.Namespace,
    teacher_sr: int,
    device: torch.device,
) -> dict[str, Any]:
    adapter.eval()
    total_stride = math.prod(parse_ints(args.strides))
    segment_samples = int(round(args.segment_seconds * args.sr))
    segment_samples = (segment_samples // total_stride) * total_stride
    rows = []
    student = []
    teach = []
    losses = []
    for i, item in enumerate(items[: args.eval_items] if args.eval_items > 0 else items):
        audio = load_audio(resolve_path(args.path_root, item), args.sr, segment_samples, args.eval_crop_mode, total_stride)
        pred, target = encode_batch(ae, teacher, adapter, audio, args, teacher_sr, device)
        loss, logs = alignment_loss(
            pred,
            target,
            args.smooth_l1_weight,
            args.pairwise_weight,
            args.variance_weight,
            args.min_std,
        )
        losses.append(float(loss.detach().cpu()))
        rows.append({
            "row": i,
            "id": item.get("id", i),
            "text_id": text_key(item),
            "speaker_id": speaker_key(item),
        })
        student.append(pred.mean(dim=1).squeeze(0).detach().cpu().numpy())
        teach.append(target.mean(dim=1).squeeze(0).detach().cpu().numpy())
    student_np = np.stack(student)
    teacher_np = np.stack(teach)
    return {
        "eval_loss": float(np.mean(losses)) if losses else 0.0,
        "student_text_cross_speaker_retrieval": retrieval_by_text_cross_speaker(student_np, rows),
        "teacher_text_cross_speaker_retrieval": retrieval_by_text_cross_speaker(teacher_np, rows),
        "student_pair_cos": paired_cosines(student_np, rows),
        "teacher_pair_cos": paired_cosines(teacher_np, rows),
        "student_rank": effective_rank(student_np),
        "teacher_rank": effective_rank(teacher_np),
        "n_items": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--teacher-bundle", default="WAV2VEC2_XLSR_300M")
    parser.add_argument("--latent-dim", type=int, default=384)
    parser.add_argument("--encoder-channels", default="128,256,512,1024,1024")
    parser.add_argument("--strides", default="2,4,4,8")
    parser.add_argument("--dilations", default="1,3,9")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--segment-seconds", type=float, default=10.0)
    parser.add_argument("--crop-mode", choices=["first", "center", "last", "random"], default="random")
    parser.add_argument("--eval-crop-mode", choices=["first", "center", "last", "random"], default="center")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--train-items", type=int, default=0)
    parser.add_argument("--eval-items", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--smooth-l1-weight", type=float, default=0.1)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--variance-weight", type=float, default=0.1)
    parser.add_argument("--min-std", type=float, default=0.05)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    total_stride = math.prod(parse_ints(args.strides))
    segment_samples = int(round(args.segment_seconds * args.sr))
    segment_samples = (segment_samples // total_stride) * total_stride

    train_items = read_manifest(args.train_manifest)
    if args.train_items > 0:
        train_items = train_items[: args.train_items]
    eval_items = read_manifest(args.eval_manifest)
    print(f"train_items={len(train_items)} eval_items={len(eval_items)} segment={segment_samples}")
    print(f"teacher={args.teacher_bundle}")

    ae = load_ae(args, device)
    teacher, teacher_sr, teacher_dim = load_teacher(args.teacher_bundle, device)
    adapter = LatentTeacherAdapter(args.latent_dim, teacher_dim, args.hidden_dim, args.depth).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float("inf")
    history = []
    global_step = 0
    for epoch in range(args.epochs):
        adapter.train()
        order = torch.randperm(len(train_items)).tolist()
        for local_idx in range(0, len(order), max(int(args.batch_size), 1)):
            item_indices = order[local_idx : local_idx + max(int(args.batch_size), 1)]
            audios = []
            for item_idx in item_indices:
                item = train_items[item_idx]
                audios.append(load_audio(resolve_path(args.path_root, item), args.sr, segment_samples, args.crop_mode, total_stride))
            audio = torch.stack(audios, dim=0)
            pred, target = encode_batch(ae, teacher, adapter, audio, args, teacher_sr, device)
            loss, logs = alignment_loss(
                pred,
                target,
                args.smooth_l1_weight,
                args.pairwise_weight,
                args.variance_weight,
                args.min_std,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
            global_step += 1

            row = {"step": global_step, "epoch": epoch, "loss": float(loss.detach().cpu()), **logs}
            history.append(row)
            if args.log_interval > 0 and global_step % args.log_interval == 0:
                print(
                    f"step {global_step} | loss={row['loss']:.4f} "
                    f"frame_cos={row['frame_cos']:.3f} pool_cos={row['pool_cos']:.3f} "
                    f"pair={row['pairwise']:.4f} var={row['variance']:.4f} l1={row['smooth_l1']:.4f}"
                )
            if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                summary = evaluate(ae, teacher, adapter, eval_items, args, teacher_sr, device)
                print("eval", json.dumps(summary, ensure_ascii=False))
                if summary["eval_loss"] < best_loss:
                    best_loss = summary["eval_loss"]
                    torch.save({"adapter": adapter.state_dict(), "args": vars(args), "summary": summary}, args.out / "best.pt")
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    summary = evaluate(ae, teacher, adapter, eval_items, args, teacher_sr, device)
    if summary["eval_loss"] < best_loss:
        best_loss = summary["eval_loss"]
        torch.save({"adapter": adapter.state_dict(), "args": vars(args), "summary": summary}, args.out / "best.pt")
    torch.save({"adapter": adapter.state_dict(), "args": vars(args), "summary": summary}, args.out / "latest.pt")

    with (args.out / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in history for k in row}))
        writer.writeheader()
        writer.writerows(history)
    with (args.out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
