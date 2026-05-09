"""V17.3 mature content teacher: log-mel + Conformer.

This is the stronger version of the standalone content teacher. It replaces the
raw-waveform TCN with a mature speech encoder shape:

    waveform -> log-mel -> conv subsampling -> Conformer -> frame/utterance heads

The teacher is still independent from WavVAE. It must first prove that it can
learn a clean content space before any AE distillation is attempted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder.scripts.v17_3_train_content_teacher import (
    GroupBatcher,
    ctc_targets,
    effective_rank,
    frame_targets,
    load_audio,
    paired_cosines,
    rank_guard_loss,
    read_manifest,
    resolve_path,
    retrieval_by_text_cross_speaker,
    same_speaker_hard_negative_loss,
    speaker_centroid_accuracy,
    speaker_key,
    stable_id,
    supervised_contrastive_loss,
    text_bag_targets,
    text_key,
)
from autoencoder.v17_constraints import PinyinToneEncoder


def _subsample_lengths(lengths: torch.Tensor, n_times: int = 2) -> torch.Tensor:
    out = lengths
    for _ in range(n_times):
        out = torch.div(out + 1, 2, rounding_mode="floor")
    return out.clamp_min(1)


class LogMelConformerTeacher(nn.Module):
    def __init__(
        self,
        sr: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        hidden_dim: int = 256,
        emb_dim: int = 192,
        num_layers: int = 6,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        text_buckets: int = 512,
    ):
        super().__init__()
        import torchaudio
        from torchaudio.models import Conformer

        self.hop_length = hop_length
        self.text_buckets = text_buckets
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            n_mels=n_mels,
            f_min=40.0,
            f_max=min(sr / 2, 11000.0),
            power=2.0,
            normalized=False,
        )
        self.subsample = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        self.conformer = Conformer(
            input_dim=hidden_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=conv_kernel,
            dropout=dropout,
            use_group_norm=False,
            convolution_first=False,
        )
        self.attn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
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
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, text_buckets),
        )

    def _features(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        mel = self.mel(audio.float()).clamp_min(1e-6).log()
        lengths = torch.full(
            (audio.shape[0],),
            mel.shape[-1],
            device=audio.device,
            dtype=torch.long,
        )
        h = self.subsample(mel).transpose(1, 2)
        lengths = _subsample_lengths(lengths, 2).clamp_max(h.shape[1])
        h, lengths = self.conformer(h, lengths)
        return h, lengths

    def forward(
        self,
        audio: torch.Tensor,
        return_raw: bool = False,
        return_text: bool = False,
        return_ctc: bool = False,
    ):
        h, lengths = self._features(audio)
        max_t = h.shape[1]
        mask = torch.arange(max_t, device=h.device)[None, :] < lengths[:, None]
        attn_logits = self.attn(h).squeeze(-1).masked_fill(~mask, torch.finfo(h.dtype).min)
        weight = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)
        pooled = (h * weight).sum(dim=1)
        raw = self.proj(pooled)
        emb = F.normalize(raw, dim=-1)
        text_logits = self.text_head(raw)
        ctc_logits = self.ctc_head(h)
        if return_raw and return_text and return_ctc:
            return emb, raw, text_logits, ctc_logits, lengths
        if return_raw and return_text:
            return emb, raw, text_logits
        if return_text and return_ctc:
            return emb, text_logits, ctc_logits, lengths
        if return_text:
            return emb, text_logits
        if return_raw:
            return emb, raw
        return emb


@torch.no_grad()
def evaluate(model: LogMelConformerTeacher, items: list[dict], args: argparse.Namespace, device: torch.device) -> dict:
    import numpy as np

    model.eval()
    rows = []
    feats = []
    segment_samples = int(round(args.segment_seconds * args.sr))
    eval_items = items[: args.eval_items] if args.eval_items > 0 else items
    for i, item in enumerate(eval_items):
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
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--conv-kernel", type=int, default=31)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--text-buckets", type=int, default=512)
    parser.add_argument("--groups-per-batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-items", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.12)
    parser.add_argument("--hard-neg-weight", type=float, default=1.0)
    parser.add_argument("--hard-neg-margin", type=float, default=0.2)
    parser.add_argument("--rank-weight", type=float, default=2.0)
    parser.add_argument("--rank-min-std", type=float, default=0.08)
    parser.add_argument("--rank-max-top1", type=float, default=0.55)
    parser.add_argument("--rank-cov-weight", type=float, default=0.01)
    parser.add_argument("--text-weight", type=float, default=0.5)
    parser.add_argument("--ctc-weight", type=float, default=0.0)
    parser.add_argument("--frame-weight", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_items = read_manifest(args.train_manifest)
    eval_items = read_manifest(args.eval_manifest)
    batcher = GroupBatcher(train_items, args.groups_per_batch, args.path_root)
    model = LogMelConformerTeacher(
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        conv_kernel=args.conv_kernel,
        dropout=args.dropout,
        text_buckets=args.text_buckets,
    ).to(device)
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
        emb, raw, text_logits, ctc_logits, lengths = model(audio, return_raw=True, return_text=True, return_ctc=True)

        nce, top1 = supervised_contrastive_loss(emb, text_ids, args.temperature)
        hard, hard_cos = same_speaker_hard_negative_loss(emb, text_ids, speaker_ids, args.hard_neg_margin)
        rank, rank_logs = rank_guard_loss(raw, args.rank_min_std, args.rank_max_top1, args.rank_cov_weight)
        text_targets = text_bag_targets(batch_items, pinyin, args.text_buckets, device)
        text_loss = F.binary_cross_entropy_with_logits(text_logits, text_targets)
        ctc_target, ctc_target_lengths = ctc_targets(batch_items, pinyin, args.text_buckets, device)
        ctc_loss = F.ctc_loss(
            F.log_softmax(ctc_logits, dim=-1).transpose(0, 1),
            ctc_target,
            lengths,
            ctc_target_lengths,
            blank=0,
            zero_infinity=True,
        )
        frame_target = frame_targets(batch_items, pinyin, args.text_buckets, ctc_logits.shape[1], device)
        frame_loss = F.cross_entropy(ctc_logits.transpose(1, 2), frame_target)
        with torch.no_grad():
            text_pred = (torch.sigmoid(text_logits) > 0.5).float()
            text_f1 = (2.0 * (text_pred * text_targets).sum(dim=1) / (text_pred + text_targets).sum(dim=1).clamp_min(1.0)).mean()
            frame_acc = (ctc_logits.argmax(dim=-1) == frame_target).float().mean()
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
                f"step {step} | loss={row['loss']:.4f} nce={row['nce']:.4f} top1={row['top1']:.2f} "
                f"hard={row['hard']:.4f}/{row['hard_cos']:.2f} rank={row['rank_top1']:.2f} "
                f"std={row['rank_std']:.3f} text={row['text']:.4f}/{row['text_f1']:.2f} "
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
