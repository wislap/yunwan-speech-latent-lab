"""Train a frozen-V16.3-latent adapter toward the V17.3 content teacher.

This is a diagnostic, not a new AE training recipe. Both the WavVAE and the
Conformer teacher are frozen. Only a sequence adapter sees AE latents and tries
to reproduce the teacher's content geometry.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder.models.autoencoder import WavVAE
from autoencoder.scripts.v17_3_teacher_distill_diagnostics import (
    effective_rank,
    feature_summary,
)
from autoencoder.scripts.v17_3_train_conformer_teacher import LogMelConformerTeacher
from autoencoder.scripts.v17_3_train_content_teacher import (
    GroupBatcher,
    load_audio,
    read_manifest,
    resolve_path,
    same_speaker_hard_negative_loss,
    speaker_key,
    stable_id,
    supervised_contrastive_loss,
    text_key,
)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x) / math.sqrt(2.0)


class LatentTeacherAdapter(nn.Module):
    """Small sequence adapter with attention pooling.

    It is intentionally stronger than the direct-AE smoke adapter, but still
    modest enough to test whether existing latents expose teacher-readable
    content without letting the adapter become a full speech encoder.
    """

    def __init__(
        self,
        latent_dim: int = 384,
        hidden_dim: int = 384,
        emb_dim: int = 192,
        depth: int = 4,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualConvBlock(hidden_dim, dilation=2 ** (i % 4)) for i in range(depth)]
        )
        self.attn = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, 1, 1),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, z: torch.Tensor, return_raw: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = self.in_proj(z.float())
        for block in self.blocks:
            h = block(h)
        weight = torch.softmax(self.attn(h), dim=-1)
        pooled = (h * weight).sum(dim=-1)
        raw = self.proj(pooled)
        emb = F.normalize(raw, dim=-1)
        if return_raw:
            return emb, raw
        return emb


def relation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[0] < 2:
        return pred.sum() * 0.0
    eye = torch.eye(pred.shape[0], device=pred.device, dtype=torch.bool)
    pred_sim = pred @ pred.T
    target_sim = target @ target.T
    return F.smooth_l1_loss(pred_sim.masked_select(~eye), target_sim.masked_select(~eye))


def rank_guard_loss(raw: torch.Tensor, min_std: float, max_top1: float, cov_weight: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if raw.shape[0] < 2:
        zero = raw.sum() * 0.0
        return zero, {"std": zero.detach(), "top1": zero.detach(), "cov": zero.detach()}
    x = raw.float() - raw.float().mean(dim=0, keepdim=True)
    std = torch.sqrt(x.var(dim=0, unbiased=False) + 1e-4)
    var_loss = F.relu(float(min_std) - std).mean()
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    off_diag = cov - torch.diag_embed(torch.diag(cov))
    cov_loss = off_diag.pow(2).sum() / max(x.shape[1], 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    top1 = eigvals[-1] / eigvals.sum().clamp_min(1e-6)
    top1_loss = F.relu(top1 - float(max_top1)).pow(2)
    return var_loss + top1_loss + float(cov_weight) * cov_loss, {
        "std": std.mean().detach(),
        "top1": top1.detach(),
        "cov": cov_loss.detach(),
    }


def load_ae(args: argparse.Namespace, device: torch.device) -> WavVAE:
    ckpt = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
    model = WavVAE(
        latent_dim=args.latent_dim,
        encoder_channels=args.encoder_channels,
        strides=args.strides,
        dilations=args.dilations,
        sample_latent=False,
    )
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_teacher(args: argparse.Namespace, device: torch.device) -> LogMelConformerTeacher:
    model = LogMelConformerTeacher(
        sr=args.sr,
        n_fft=args.teacher_n_fft,
        hop_length=args.teacher_hop_length,
        n_mels=args.teacher_n_mels,
        hidden_dim=args.teacher_hidden_dim,
        emb_dim=args.emb_dim,
        num_layers=args.teacher_layers,
        num_heads=args.teacher_heads,
        ffn_dim=args.teacher_ffn_dim,
        conv_kernel=args.teacher_conv_kernel,
        dropout=0.0,
        text_buckets=args.text_buckets,
    )
    ckpt = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def encode_batch(
    ae: WavVAE,
    teacher: LogMelConformerTeacher,
    items: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    crop_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    segment_samples = int(round(args.segment_seconds * args.sr))
    segment_samples = (segment_samples // math.prod(args.strides)) * math.prod(args.strides)
    audio = torch.stack(
        [load_audio(resolve_path(args.path_root, item), args.sr, segment_samples, crop_mode) for item in items],
        dim=0,
    ).unsqueeze(1).to(device)
    with torch.amp.autocast(device_type=device.type, enabled=False):
        z, _, _ = ae.encode(audio.float())
        target = teacher(audio[:, 0, :].float())
    return z.detach(), target.detach()


@torch.no_grad()
def evaluate(
    ae: WavVAE,
    teacher: LogMelConformerTeacher,
    adapter: LatentTeacherAdapter,
    items: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    adapter.eval()
    eval_items = items[: args.eval_items] if args.eval_items > 0 else items
    rows = []
    pred_features = []
    teacher_features = []
    latent_features = []
    cosines = []
    for start in range(0, len(eval_items), args.eval_batch_size):
        batch_items = eval_items[start : start + args.eval_batch_size]
        z, target = encode_batch(ae, teacher, batch_items, args, device, args.eval_crop_mode)
        pred, _raw = adapter(z, return_raw=True)
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        latent = F.normalize(z.mean(dim=-1), dim=-1)
        cosines.extend((pred * target).sum(dim=-1).detach().cpu().tolist())
        pred_features.append(pred.detach().cpu().numpy())
        teacher_features.append(target.detach().cpu().numpy())
        latent_features.append(latent.detach().cpu().numpy())
        for item in batch_items:
            rows.append(
                {
                    "row": len(rows),
                    "id": item.get("id", len(rows)),
                    "text_id": text_key(item),
                    "speaker_id": speaker_key(item),
                }
            )
    pred_np = np.concatenate(pred_features, axis=0)
    teacher_np = np.concatenate(teacher_features, axis=0)
    latent_np = np.concatenate(latent_features, axis=0)
    summary: dict[str, Any] = {
        "n_items": len(rows),
        "adapter_teacher_cos_mean": float(np.mean(cosines)),
        "adapter_teacher_cos_std": float(np.std(cosines)),
    }
    summary.update(feature_summary("adapter", pred_np, rows))
    summary.update(feature_summary("teacher", teacher_np, rows))
    summary.update(feature_summary("latent_mean", latent_np, rows))
    summary["adapter_rank_direct"] = effective_rank(pred_np)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ae-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-dim", type=int, default=384)
    parser.add_argument("--encoder-channels", type=int, nargs="+", default=[128, 256, 512, 1024, 1024])
    parser.add_argument("--strides", type=int, nargs="+", default=[2, 4, 4, 8])
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 3, 9])
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--segment-seconds", type=float, default=10.0)
    parser.add_argument("--crop-mode", choices=["first", "center", "last", "random"], default="random")
    parser.add_argument("--eval-crop-mode", choices=["first", "center", "last", "random"], default="center")
    parser.add_argument("--adapter-hidden-dim", type=int, default=384)
    parser.add_argument("--adapter-depth", type=int, default=4)
    parser.add_argument("--emb-dim", type=int, default=192)
    parser.add_argument("--text-buckets", type=int, default=512)
    parser.add_argument("--teacher-hidden-dim", type=int, default=256)
    parser.add_argument("--teacher-layers", type=int, default=6)
    parser.add_argument("--teacher-heads", type=int, default=4)
    parser.add_argument("--teacher-ffn-dim", type=int, default=1024)
    parser.add_argument("--teacher-n-fft", type=int, default=1024)
    parser.add_argument("--teacher-hop-length", type=int, default=256)
    parser.add_argument("--teacher-n-mels", type=int, default=80)
    parser.add_argument("--teacher-conv-kernel", type=int, default=31)
    parser.add_argument("--groups-per-batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-items", type=int, default=112)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.12)
    parser.add_argument("--point-weight", type=float, default=1.0)
    parser.add_argument("--relation-weight", type=float, default=0.5)
    parser.add_argument("--nce-weight", type=float, default=0.5)
    parser.add_argument("--hard-neg-weight", type=float, default=0.25)
    parser.add_argument("--hard-neg-margin", type=float, default=0.35)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--rank-min-std", type=float, default=0.04)
    parser.add_argument("--rank-max-top1", type=float, default=0.45)
    parser.add_argument("--rank-cov-weight", type=float, default=0.01)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_items = read_manifest(args.train_manifest)
    eval_items = read_manifest(args.eval_manifest)
    batcher = GroupBatcher(train_items, args.groups_per_batch, args.path_root)
    ae = load_ae(args, device)
    teacher = load_teacher(args, device)
    adapter = LatentTeacherAdapter(
        latent_dim=args.latent_dim,
        hidden_dim=args.adapter_hidden_dim,
        emb_dim=args.emb_dim,
        depth=args.adapter_depth,
    ).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    text_table: dict[str, int] = {}
    speaker_table: dict[str, int] = {}
    history = []
    best_top1 = -1.0

    print(
        f"train_items={len(train_items)} groups={len(batcher.groups)} eval_items={len(eval_items)} "
        f"adapter_params={sum(p.numel() for p in adapter.parameters()):,}"
    )
    for step in range(1, args.steps + 1):
        indices = batcher.sample_indices()
        batch_items = [train_items[i] for i in indices]
        z, target = encode_batch(ae, teacher, batch_items, args, device, args.crop_mode)
        text_ids = torch.tensor([stable_id(text_key(item), text_table) for item in batch_items], device=device)
        speaker_ids = torch.tensor([stable_id(speaker_key(item), speaker_table) for item in batch_items], device=device)
        pred, raw = adapter(z, return_raw=True)
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)

        point = (1.0 - (pred * target).sum(dim=-1)).mean()
        rel = relation_loss(pred, target)
        nce, nce_top1 = supervised_contrastive_loss(pred, text_ids, args.temperature)
        hard, hard_cos = same_speaker_hard_negative_loss(pred, text_ids, speaker_ids, args.hard_neg_margin)
        rank, rank_logs = rank_guard_loss(raw, args.rank_min_std, args.rank_max_top1, args.rank_cov_weight)
        loss = (
            float(args.point_weight) * point
            + float(args.relation_weight) * rel
            + float(args.nce_weight) * nce
            + float(args.hard_neg_weight) * hard
            + float(args.rank_weight) * rank
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt.step()

        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "point": float(point.detach().cpu()),
            "rel": float(rel.detach().cpu()),
            "nce": float(nce.detach().cpu()),
            "nce_top1": float(nce_top1.detach().cpu()),
            "hard": float(hard.detach().cpu()),
            "hard_cos": float(hard_cos.detach().cpu()),
            "rank": float(rank.detach().cpu()),
            "rank_std": float(rank_logs["std"].detach().cpu()),
            "rank_top1": float(rank_logs["top1"].detach().cpu()),
            "rank_cov": float(rank_logs["cov"].detach().cpu()),
            "batch": len(batch_items),
        }
        history.append(row)
        if args.log_interval > 0 and step % args.log_interval == 0:
            print(
                f"step {step} | loss={row['loss']:.4f} point={row['point']:.3f} "
                f"rel={row['rel']:.3f} nce={row['nce']:.3f}/{row['nce_top1']:.2f} "
                f"hard={row['hard']:.3f}/{row['hard_cos']:.2f} "
                f"rank={row['rank_top1']:.2f} std={row['rank_std']:.3f}"
            )
        if args.eval_interval > 0 and step % args.eval_interval == 0:
            summary = evaluate(ae, teacher, adapter, eval_items, args, device)
            top1_eval = summary["adapter_text_cross_speaker_retrieval"]["top1"]
            print("eval", json.dumps(summary, ensure_ascii=False))
            if top1_eval > best_top1:
                best_top1 = top1_eval
                torch.save({"adapter": adapter.state_dict(), "args": vars(args), "summary": summary}, args.out / "best.pt")

    summary = evaluate(ae, teacher, adapter, eval_items, args, device)
    if summary["adapter_text_cross_speaker_retrieval"]["top1"] > best_top1:
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
