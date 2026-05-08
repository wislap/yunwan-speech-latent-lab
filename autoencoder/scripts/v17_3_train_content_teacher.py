"""V17.3 from-scratch content teacher.

This trains a standalone content representation teacher instead of modifying
the AE. The teacher only sees audio and crossed-data metadata:

- same_text_group across speakers should be close;
- different text should be separated, especially within the same speaker;
- embeddings should keep enough effective rank.

The resulting teacher can later be frozen and distilled into a V16.3 latent
adapter. It is intentionally independent from WavVAE reconstruction.
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

from autoencoder.v17_constraints import PinyinToneEncoder, _align_ids_to_frames, _duration_values, _heuristic_durations


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


def load_audio(path: Path, sr: int, segment_samples: int, crop_mode: str) -> torch.Tensor:
    audio_np, file_sr = sf.read(path, always_2d=False)
    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)
    audio = torch.from_numpy(audio_np.astype(np.float32))
    if file_sr != sr:
        import torchaudio

        audio = torchaudio.functional.resample(audio, file_sr, sr)
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
        audio = audio[start : start + segment_samples]
    else:
        audio = F.pad(audio, (0, segment_samples - audio.numel()))
    return audio


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x) / math.sqrt(2.0)


class ContentTeacher(nn.Module):
    """Small mature-style speech encoder: strided conv + TCN + attentive pool."""

    def __init__(self, hidden_dim: int = 384, emb_dim: int = 192, depth: int = 6, text_buckets: int = 512):
        super().__init__()
        self.text_buckets = text_buckets
        self.frontend = nn.Sequential(
            nn.Conv1d(1, 96, 9, stride=3, padding=4),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv1d(96, 192, 7, stride=4, padding=3),
            nn.GroupNorm(8, 192),
            nn.SiLU(),
            nn.Conv1d(192, hidden_dim, 5, stride=4, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualTCNBlock(hidden_dim, dilation=2 ** (i % 4)) for i in range(depth)]
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
        self.text_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, text_buckets),
        )
        self.ctc_head = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, text_buckets, 1),
        )

    def forward(
        self,
        audio: torch.Tensor,
        return_raw: bool = False,
        return_text: bool = False,
        return_ctc: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        h = self.frontend(audio.float())
        for block in self.blocks:
            h = block(h)
        weight = torch.softmax(self.attn(h), dim=-1)
        pooled = (h * weight).sum(dim=-1)
        raw = self.proj(pooled)
        emb = F.normalize(raw, dim=-1)
        text_logits = self.text_head(raw)
        ctc_logits = self.ctc_head(h).transpose(1, 2)
        if return_raw and return_text and return_ctc:
            return emb, raw, text_logits, ctc_logits
        if return_raw and return_text:
            return emb, raw, text_logits
        if return_text and return_ctc:
            return emb, text_logits, ctc_logits
        if return_text:
            return emb, text_logits
        if return_raw:
            return emb, raw
        return emb


class GroupBatcher:
    def __init__(self, items: list[dict[str, Any]], groups_per_batch: int, path_root: Path):
        self.items = items
        self.groups_per_batch = groups_per_batch
        self.path_root = path_root
        by_text: dict[str, list[int]] = defaultdict(list)
        for i, item in enumerate(items):
            by_text[text_key(item)].append(i)
        self.groups = [idxs for idxs in by_text.values() if len(idxs) >= 2]
        if not self.groups:
            raise RuntimeError("No same_text groups with at least two samples")

    def sample_indices(self) -> list[int]:
        order = torch.randperm(len(self.groups))[: self.groups_per_batch].tolist()
        indices: list[int] = []
        for group_idx in order:
            group = self.groups[group_idx]
            by_spk: dict[str, list[int]] = defaultdict(list)
            for idx in group:
                by_spk[speaker_key(self.items[idx])].append(idx)
            picked = []
            for rows in by_spk.values():
                picked.append(rows[int(torch.randint(0, len(rows), (1,)).item())])
            if len(picked) < 2:
                picked = group[:2]
            indices.extend(picked)
        return indices


def supervised_contrastive_loss(
    emb: torch.Tensor,
    text_ids: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = emb @ emb.T / max(float(temperature), 1e-4)
    eye = torch.eye(emb.shape[0], device=emb.device, dtype=torch.bool)
    pos = text_ids[:, None].eq(text_ids[None, :]) & ~eye
    valid = pos.any(dim=1)
    if not bool(valid.any()):
        zero = emb.sum() * 0.0
        return zero, torch.tensor(0.0, device=emb.device)
    logits = logits.masked_fill(eye, torch.finfo(logits.dtype).min)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss = -(log_prob.masked_fill(~pos, 0.0).sum(dim=1) / pos.sum(dim=1).clamp_min(1))[valid].mean()
    with torch.no_grad():
        top1 = pos.gather(1, logits.argmax(dim=1, keepdim=True)).float()
        acc = top1[valid].mean()
    return loss, acc


def same_speaker_hard_negative_loss(
    emb: torch.Tensor,
    text_ids: torch.Tensor,
    speaker_ids: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    same_spk_diff_text = speaker_ids[:, None].eq(speaker_ids[None, :]) & ~text_ids[:, None].eq(text_ids[None, :])
    eye = torch.eye(emb.shape[0], device=emb.device, dtype=torch.bool)
    mask = same_spk_diff_text & ~eye
    if not bool(mask.any()):
        zero = emb.sum() * 0.0
        return zero, zero.detach()
    sim = emb @ emb.T
    selected = sim[mask]
    return F.relu(selected - float(margin)).pow(2).mean(), selected.mean().detach()


def rank_guard_loss(raw: torch.Tensor, min_std: float, max_top1: float, cov_weight: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if raw.shape[0] < 2:
        zero = raw.sum() * 0.0
        return zero, {"std": zero.detach(), "top1": zero.detach()}
    x = raw.float()
    x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True, unbiased=False).mean().clamp_min(1e-3)
    x = x - x.mean(dim=0, keepdim=True)
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


def stable_id(value: str, table: dict[str, int]) -> int:
    if value not in table:
        table[value] = len(table)
    return table[value]


def text_bag_targets(items: list[dict[str, Any]], pinyin: PinyinToneEncoder, buckets: int, device: torch.device) -> torch.Tensor:
    targets = []
    for item in items:
        vec = torch.zeros(buckets, dtype=torch.float32)
        ids = pinyin.ids(str(item.get("text", "")))
        if ids:
            idx = torch.tensor(ids, dtype=torch.long).clamp(0, buckets - 1)
            vec.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
            vec = (vec > 0).float()
        else:
            vec[0] = 1.0
        targets.append(vec)
    return torch.stack(targets, dim=0).to(device)


def ctc_targets(
    items: list[dict[str, Any]],
    pinyin: PinyinToneEncoder,
    buckets: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    pieces = []
    lengths = []
    for item in items:
        ids = pinyin.ids(str(item.get("text", "")))
        ids = [int(max(1, min(buckets - 1, i))) for i in ids]
        if not ids:
            ids = [1]
        pieces.extend(ids)
        lengths.append(len(ids))
    return (
        torch.tensor(pieces, dtype=torch.long, device=device),
        torch.tensor(lengths, dtype=torch.long, device=device),
    )


def frame_targets(
    items: list[dict[str, Any]],
    pinyin: PinyinToneEncoder,
    buckets: int,
    frames: int,
    device: torch.device,
) -> torch.Tensor:
    targets = []
    for item in items:
        tokens, chars = pinyin.tokenize_with_chars(str(item.get("text", "")))
        ids = [int(max(1, min(buckets - 1, i))) for i in pinyin.ids(str(item.get("text", "")))]
        if len(ids) != len(tokens):
            ids = ids[: len(tokens)]
        durations = _duration_values(item)
        if durations is None:
            durations = _heuristic_durations(chars, item)
        targets.append(_align_ids_to_frames(ids, frames, device, durations))
    return torch.stack(targets, dim=0)


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
            else:
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
def evaluate(model: ContentTeacher, items: list[dict[str, Any]], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    model.eval()
    rows = []
    feats = []
    segment_samples = int(round(args.segment_seconds * args.sr))
    for i, item in enumerate(items[: args.eval_items] if args.eval_items > 0 else items):
        audio = load_audio(resolve_path(args.path_root, item), args.sr, segment_samples, args.eval_crop_mode)
        emb = model(audio.unsqueeze(0).to(device)).squeeze(0).detach().cpu().numpy()
        rows.append({"row": i, "id": item.get("id", i), "text_id": text_key(item), "speaker_id": speaker_key(item)})
        feats.append(emb)
    features = np.stack(feats)
    return {
        "n_items": len(rows),
        "text_cross_speaker_retrieval": retrieval_by_text_cross_speaker(features, rows),
        "pair_cos": paired_cosines(features, rows),
        "rank": effective_rank(features),
        "speaker_centroid_acc": speaker_centroid_accuracy(features, rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--segment-seconds", type=float, default=8.0)
    parser.add_argument("--crop-mode", choices=["first", "center", "last", "random"], default="random")
    parser.add_argument("--eval-crop-mode", choices=["first", "center", "last", "random"], default="center")
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--emb-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--text-buckets", type=int, default=512)
    parser.add_argument("--groups-per-batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-items", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.12)
    parser.add_argument("--hard-neg-weight", type=float, default=0.5)
    parser.add_argument("--hard-neg-margin", type=float, default=0.2)
    parser.add_argument("--rank-weight", type=float, default=0.1)
    parser.add_argument("--rank-min-std", type=float, default=0.08)
    parser.add_argument("--rank-max-top1", type=float, default=0.55)
    parser.add_argument("--rank-cov-weight", type=float, default=0.01)
    parser.add_argument("--text-weight", type=float, default=0.5)
    parser.add_argument("--ctc-weight", type=float, default=0.5)
    parser.add_argument("--frame-weight", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_items = read_manifest(args.train_manifest)
    eval_items = read_manifest(args.eval_manifest)
    batcher = GroupBatcher(train_items, args.groups_per_batch, args.path_root)
    model = ContentTeacher(args.hidden_dim, args.emb_dim, args.depth, args.text_buckets).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pinyin = PinyinToneEncoder(buckets=args.text_buckets)
    text_table: dict[str, int] = {}
    speaker_table: dict[str, int] = {}
    segment_samples = int(round(args.segment_seconds * args.sr))
    history = []
    best_top1 = -1.0

    print(
        f"train_items={len(train_items)} groups={len(batcher.groups)} eval_items={len(eval_items)} "
        f"segment={segment_samples} params={sum(p.numel() for p in model.parameters()):,}"
    )

    for step in range(1, args.steps + 1):
        indices = batcher.sample_indices()
        batch_items = [train_items[i] for i in indices]
        audio = torch.stack(
            [load_audio(resolve_path(args.path_root, item), args.sr, segment_samples, args.crop_mode) for item in batch_items],
            dim=0,
        ).to(device)
        text_ids = torch.tensor([stable_id(text_key(item), text_table) for item in batch_items], device=device)
        speaker_ids = torch.tensor([stable_id(speaker_key(item), speaker_table) for item in batch_items], device=device)
        emb, raw, text_logits, ctc_logits = model(audio, return_raw=True, return_text=True, return_ctc=True)
        nce, top1 = supervised_contrastive_loss(emb, text_ids, args.temperature)
        hard, hard_cos = same_speaker_hard_negative_loss(emb, text_ids, speaker_ids, args.hard_neg_margin)
        rank, rank_logs = rank_guard_loss(raw, args.rank_min_std, args.rank_max_top1, args.rank_cov_weight)
        text_targets = text_bag_targets(batch_items, pinyin, args.text_buckets, device)
        text_loss = F.binary_cross_entropy_with_logits(text_logits, text_targets)
        ctc_target, ctc_target_lengths = ctc_targets(batch_items, pinyin, args.text_buckets, device)
        ctc_input_lengths = torch.full(
            (ctc_logits.shape[0],),
            ctc_logits.shape[1],
            dtype=torch.long,
            device=device,
        )
        ctc_loss = F.ctc_loss(
            F.log_softmax(ctc_logits, dim=-1).transpose(0, 1),
            ctc_target,
            ctc_input_lengths,
            ctc_target_lengths,
            blank=0,
            zero_infinity=True,
        )
        frame_target = frame_targets(batch_items, pinyin, args.text_buckets, ctc_logits.shape[1], device)
        frame_loss = F.cross_entropy(ctc_logits.transpose(1, 2), frame_target)
        with torch.no_grad():
            frame_acc = (ctc_logits.argmax(dim=-1) == frame_target).float().mean()
        with torch.no_grad():
            text_pred = (torch.sigmoid(text_logits) > 0.5).float()
            text_f1_num = 2.0 * (text_pred * text_targets).sum(dim=1)
            text_f1_den = (text_pred + text_targets).sum(dim=1).clamp_min(1.0)
            text_f1 = (text_f1_num / text_f1_den).mean()
        loss = (
            nce
            + float(args.hard_neg_weight) * hard
            + float(args.rank_weight) * rank
            + float(args.text_weight) * text_loss
            + float(args.ctc_weight) * ctc_loss
            + float(args.frame_weight) * frame_loss
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "nce": float(nce.detach().cpu()),
            "top1": float(top1.detach().cpu()),
            "hard": float(hard.detach().cpu()),
            "hard_cos": float(hard_cos.detach().cpu()),
            "rank": float(rank.detach().cpu()),
            "rank_std": float(rank_logs["std"].detach().cpu()),
            "rank_top1": float(rank_logs["top1"].detach().cpu()),
            "rank_cov": float(rank_logs["cov"].detach().cpu()),
            "text": float(text_loss.detach().cpu()),
            "text_f1": float(text_f1.detach().cpu()),
            "ctc": float(ctc_loss.detach().cpu()),
            "frame": float(frame_loss.detach().cpu()),
            "frame_acc": float(frame_acc.detach().cpu()),
            "batch": len(batch_items),
        }
        history.append(row)
        if args.log_interval > 0 and step % args.log_interval == 0:
            print(
                f"step {step} | loss={row['loss']:.4f} nce={row['nce']:.4f} "
                f"top1={row['top1']:.2f} hard={row['hard']:.4f}/{row['hard_cos']:.2f} "
                f"rank={row['rank_top1']:.2f} std={row['rank_std']:.3f} cov={row['rank_cov']:.3f} "
                f"text={row['text']:.4f}/{row['text_f1']:.2f} "
                f"ctc={row['ctc']:.3f} frame={row['frame']:.3f}/{row['frame_acc']:.2f}"
            )
        if args.eval_interval > 0 and step % args.eval_interval == 0:
            summary = evaluate(model, eval_items, args, device)
            print("eval", json.dumps(summary, ensure_ascii=False))
            top1_eval = summary["text_cross_speaker_retrieval"]["top1"]
            if top1_eval > best_top1:
                best_top1 = top1_eval
                torch.save({"model": model.state_dict(), "args": vars(args), "summary": summary}, args.out / "best.pt")

    summary = evaluate(model, eval_items, args, device)
    if summary["text_cross_speaker_retrieval"]["top1"] > best_top1:
        torch.save({"model": model.state_dict(), "args": vars(args), "summary": summary}, args.out / "best.pt")
    torch.save({"model": model.state_dict(), "args": vars(args), "summary": summary}, args.out / "latest.pt")
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
