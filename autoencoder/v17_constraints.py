"""V17 external-modality adapter and flow constraints for AE latents."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _stable_bucket(value: str, buckets: int) -> int:
    if buckets <= 1:
        return 0
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % buckets


def _metadata_list(metadata: Any, batch_size: int) -> list[dict]:
    if isinstance(metadata, list):
        return [m if isinstance(m, dict) else {} for m in metadata]
    if isinstance(metadata, dict):
        rows = []
        for i in range(batch_size):
            row = {}
            for key, value in metadata.items():
                if isinstance(value, (list, tuple)) and i < len(value):
                    row[key] = value[i]
                else:
                    row[key] = value
            rows.append(row)
        return rows
    return [{} for _ in range(batch_size)]


class PinyinToneEncoder:
    """Convert Chinese text to pinyin+tone tokens.

    This is an external symbolic modality encoder. It intentionally lives
    outside the AE and produces fixed labels/features from text metadata.
    """

    def __init__(self, buckets: int = 512):
        self.buckets = buckets
        self._pypinyin = None

    def _load(self):
        if self._pypinyin is None:
            try:
                from pypinyin import Style, lazy_pinyin
            except ImportError as exc:
                raise ImportError(
                    "V17 pinyin constraints require pypinyin. Install it or "
                    "disable v17_ext_text_weight / v17_flow_weight."
                ) from exc
            self._pypinyin = (lazy_pinyin, Style)
        return self._pypinyin

    def tokenize_with_chars(self, text: str) -> tuple[list[str], list[str]]:
        text = str(text).strip()
        if not text:
            return ["<blank>"], [""]
        lazy_pinyin, Style = self._load()
        tokens = []
        chars = []
        for ch in text:
            py = lazy_pinyin(ch, style=Style.TONE3, neutral_tone_with_five=True, errors="ignore")
            if py:
                tokens.append(py[0])
                chars.append(ch)
        if not tokens:
            return ["<blank>"], [""]
        return tokens, chars

    def tokenize(self, text: str) -> list[str]:
        tokens, _ = self.tokenize_with_chars(text)
        return tokens

    def ids(self, text: str) -> list[int]:
        return [_stable_bucket(t, self.buckets - 1) + 1 for t in self.tokenize(text)]


def _duration_values(row: dict) -> list[float] | None:
    for key in ("pinyin_durations", "token_durations", "phoneme_durations", "durations"):
        value = row.get(key)
        if isinstance(value, list) and value:
            try:
                return [max(float(v), 0.0) for v in value]
            except (TypeError, ValueError):
                return None
    return None


def _heuristic_durations(chars: list[str], row: dict) -> list[float]:
    text = str(row.get("text", ""))
    speed = max(float(row.get("speed", 1.0) or 1.0), 0.25)
    weights = []
    char_index = 0
    punctuation_bonus = 0.0
    pause_weights = {
        "，": 0.8,
        ",": 0.8,
        "、": 0.5,
        "；": 1.0,
        ";": 1.0,
        "：": 0.8,
        ":": 0.8,
        "。": 1.4,
        "！": 1.4,
        "!": 1.4,
        "？": 1.4,
        "?": 1.4,
    }
    for ch in text:
        if char_index >= len(chars):
            break
        if ch == chars[char_index]:
            base = 1.0 + punctuation_bonus
            weights.append(base / speed)
            punctuation_bonus = 0.0
            char_index += 1
        elif ch in pause_weights:
            punctuation_bonus += pause_weights[ch]
    while len(weights) < len(chars):
        weights.append(1.0 / speed)
    return weights[: len(chars)]


def _align_ids_to_frames(
    ids: list[int],
    frames: int,
    device: torch.device,
    durations: list[float] | None = None,
) -> torch.Tensor:
    if frames <= 0:
        raise ValueError("frames must be positive")
    if not ids:
        ids = [0]
    if durations is None or len(durations) != len(ids) or sum(durations) <= 0:
        src = torch.tensor(ids, device=device, dtype=torch.float32).view(1, 1, -1)
        aligned = F.interpolate(src, size=frames, mode="nearest").view(frames).long()
        return aligned

    weights = torch.tensor(durations, device=device, dtype=torch.float32).clamp_min(0.0)
    weights = weights / weights.sum().clamp_min(1e-6)
    raw = weights * frames
    counts = torch.floor(raw).long()
    remainder = frames - int(counts.sum().item())
    if remainder > 0:
        frac_order = torch.argsort(raw - counts.float(), descending=True)
        counts[frac_order[:remainder]] += 1
    elif remainder < 0:
        removable = torch.argsort(raw - counts.float())
        for idx in removable:
            if remainder == 0:
                break
            take = min(int(counts[idx].item()), -remainder)
            counts[idx] -= take
            remainder += take
    pieces = []
    for token_id, count in zip(ids, counts.tolist()):
        if count > 0:
            pieces.append(torch.full((count,), int(token_id), device=device, dtype=torch.long))
    if not pieces:
        return torch.full((frames,), int(ids[0]), device=device, dtype=torch.long)
    aligned = torch.cat(pieces, dim=0)
    if aligned.numel() < frames:
        aligned = F.pad(aligned, (0, frames - aligned.numel()), value=int(ids[-1]))
    elif aligned.numel() > frames:
        aligned = aligned[:frames]
    return aligned


class ExternalAdapterConstraint(nn.Module):
    """Shallow sequence-aware adapter from latent frames to external targets."""

    def __init__(
        self,
        latent_dim: int,
        text_buckets: int = 512,
        hidden_dim: int = 256,
        speaker_buckets: int = 64,
        style_buckets: int = 32,
        text_weight: float = 1.0,
        speaker_weight: float = 0.0,
        style_weight: float = 0.0,
        speed_weight: float = 0.1,
        detach_latent: bool = False,
    ):
        super().__init__()
        self.pinyin = PinyinToneEncoder(buckets=text_buckets)
        self.text_embedding = nn.Embedding(text_buckets, hidden_dim)
        self.frame_adapter = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.text_head = nn.Conv1d(hidden_dim, text_buckets, kernel_size=1)
        self.utt_norm = nn.LayerNorm(latent_dim)
        self.utt_adapter = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.speaker_head = nn.Linear(hidden_dim, speaker_buckets)
        self.style_head = nn.Linear(hidden_dim, style_buckets)
        self.speed_head = nn.Linear(hidden_dim, 1)
        self.text_buckets = text_buckets
        self.speaker_buckets = speaker_buckets
        self.style_buckets = style_buckets
        self.text_weight = text_weight
        self.speaker_weight = speaker_weight
        self.style_weight = style_weight
        self.speed_weight = speed_weight
        self.detach_latent = detach_latent

    @property
    def condition_dim(self) -> int:
        return self.text_embedding.embedding_dim + self.speaker_buckets + self.style_buckets + 1

    def _text_targets(self, rows: list[dict], frames: int, device: torch.device) -> torch.Tensor:
        targets = []
        for row in rows:
            tokens, chars = self.pinyin.tokenize_with_chars(str(row.get("text", "")))
            ids = [_stable_bucket(t, self.text_buckets - 1) + 1 for t in tokens]
            durations = _duration_values(row)
            if durations is None:
                durations = _heuristic_durations(chars, row)
            targets.append(_align_ids_to_frames(ids, frames, device, durations))
        return torch.stack(targets, dim=0)

    def condition_vector(self, metadata: Any, batch_size: int, device: torch.device) -> torch.Tensor:
        rows = _metadata_list(metadata, batch_size)
        text_vecs = []
        for row in rows:
            ids = torch.tensor(self.pinyin.ids(str(row.get("text", ""))), device=device, dtype=torch.long)
            text_vecs.append(self.text_embedding(ids).mean(dim=0))
        text = torch.stack(text_vecs, dim=0)
        speaker_ids = torch.tensor(
            [_stable_bucket(str(r.get("speaker_id", "")), self.speaker_buckets) for r in rows],
            device=device,
            dtype=torch.long,
        )
        style_ids = torch.tensor(
            [_stable_bucket(str(r.get("style", "")), self.style_buckets) for r in rows],
            device=device,
            dtype=torch.long,
        )
        speaker = F.one_hot(speaker_ids, self.speaker_buckets).float()
        style = F.one_hot(style_ids, self.style_buckets).float()
        speed = torch.tensor(
            [float(r.get("speed", 1.0) or 1.0) for r in rows],
            device=device,
            dtype=torch.float32,
        ).unsqueeze(1)
        speed = (speed - 1.0) / 0.25
        return torch.cat([text, speaker, style, speed], dim=-1)

    def forward(self, z: torch.Tensor, metadata: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, _, frames = z.shape
        device = z.device
        rows = _metadata_list(metadata, b)
        z_in = z.detach() if self.detach_latent else z

        losses: dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=device)

        if self.text_weight > 0 and any(str(r.get("text", "")) for r in rows):
            frame_h = self.frame_adapter(z_in.float())
            text_logits = self.text_head(frame_h)
            text_targets = self._text_targets(rows, frames, device)
            text_loss = F.cross_entropy(text_logits, text_targets)
            total = total + self.text_weight * text_loss
            losses["v17_ext_text"] = text_loss

        pooled = z_in.mean(dim=-1)
        utt_h = self.utt_adapter(self.utt_norm(pooled.float()))

        if self.speaker_weight > 0 and any("speaker_id" in r for r in rows):
            speaker_ids = torch.tensor(
                [_stable_bucket(str(r.get("speaker_id", "")), self.speaker_buckets) for r in rows],
                device=device,
                dtype=torch.long,
            )
            speaker_loss = F.cross_entropy(self.speaker_head(utt_h), speaker_ids)
            total = total + self.speaker_weight * speaker_loss
            losses["v17_ext_speaker"] = speaker_loss

        if self.style_weight > 0 and any("style" in r for r in rows):
            style_ids = torch.tensor(
                [_stable_bucket(str(r.get("style", "")), self.style_buckets) for r in rows],
                device=device,
                dtype=torch.long,
            )
            style_loss = F.cross_entropy(self.style_head(utt_h), style_ids)
            total = total + self.style_weight * style_loss
            losses["v17_ext_style"] = style_loss

        if self.speed_weight > 0 and any("speed" in r for r in rows):
            speed = torch.tensor(
                [float(r.get("speed", 1.0) or 1.0) for r in rows],
                device=device,
                dtype=torch.float32,
            ).unsqueeze(1)
            speed = (speed - 1.0) / 0.25
            speed_loss = F.smooth_l1_loss(self.speed_head(utt_h), speed)
            total = total + self.speed_weight * speed_loss
            losses["v17_ext_speed"] = speed_loss

        losses["v17_ext_total"] = total
        return total, losses


class LatentFlowConstraint(nn.Module):
    """Small conditional flow-matching model over AE latent frames."""

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        hidden_dim: int = 512,
        depth: int = 4,
        detach_target: bool = True,
        min_t: float = 0.0,
        max_t: float = 1.0,
    ):
        super().__init__()
        self.detach_target = detach_target
        self.min_t = min_t
        self.max_t = max_t
        self.cond_proj = nn.Linear(condition_dim + 1, hidden_dim)
        self.in_proj = nn.Conv1d(latent_dim + hidden_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1) for _ in range(depth)]
        )
        self.norms = nn.ModuleList([nn.GroupNorm(8, hidden_dim) for _ in range(depth)])
        self.out_proj = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, _, frames = z.shape
        target = z.float()
        if self.detach_target:
            target = target.detach()
        noise = torch.randn_like(target)
        t = self.min_t + (self.max_t - self.min_t) * torch.rand(b, 1, 1, device=z.device)
        x_t = (1.0 - t) * noise + t * target
        v_target = target - noise

        cond_t = torch.cat([condition.float(), t.squeeze(-1)], dim=-1)
        cond = self.cond_proj(cond_t).unsqueeze(-1).expand(-1, -1, frames)
        h = self.in_proj(torch.cat([x_t, cond], dim=1))
        for norm, block in zip(self.norms, self.blocks):
            h = h + block(F.silu(norm(h))) / math.sqrt(max(len(self.blocks), 1))
        v_pred = self.out_proj(F.silu(h))
        loss = F.mse_loss(v_pred, v_target)
        return loss, {"v17_flow": loss}


class V17ConstraintModule(nn.Module):
    """Combined V17 external adapter and flow constraints."""

    def __init__(
        self,
        latent_dim: int,
        ext_weight: float = 0.0,
        flow_weight: float = 0.0,
        adapter_hidden_dim: int = 256,
        text_buckets: int = 512,
        speaker_buckets: int = 64,
        style_buckets: int = 32,
        text_weight: float = 1.0,
        speaker_weight: float = 0.0,
        style_weight: float = 0.0,
        speed_weight: float = 0.1,
        flow_hidden_dim: int = 512,
        flow_depth: int = 4,
        external_detach_latent: bool = False,
        flow_detach_target: bool = True,
        flow_min_t: float = 0.0,
        flow_max_t: float = 1.0,
    ):
        super().__init__()
        self.ext_weight = ext_weight
        self.flow_weight = flow_weight
        self.external = ExternalAdapterConstraint(
            latent_dim=latent_dim,
            text_buckets=text_buckets,
            hidden_dim=adapter_hidden_dim,
            speaker_buckets=speaker_buckets,
            style_buckets=style_buckets,
            text_weight=text_weight,
            speaker_weight=speaker_weight,
            style_weight=style_weight,
            speed_weight=speed_weight,
            detach_latent=external_detach_latent,
        )
        self.flow = LatentFlowConstraint(
            latent_dim=latent_dim,
            condition_dim=self.external.condition_dim,
            hidden_dim=flow_hidden_dim,
            depth=flow_depth,
            detach_target=flow_detach_target,
            min_t=flow_min_t,
            max_t=flow_max_t,
        )

    @property
    def enabled(self) -> bool:
        return self.ext_weight > 0 or self.flow_weight > 0

    def forward(self, z: torch.Tensor, metadata: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total = torch.tensor(0.0, device=z.device)
        logs: dict[str, torch.Tensor] = {}
        b = z.shape[0]
        if self.ext_weight > 0:
            ext_loss, ext_logs = self.external(z, metadata)
            total = total + self.ext_weight * ext_loss
            logs.update(ext_logs)
        if self.flow_weight > 0:
            cond = self.external.condition_vector(metadata, b, z.device)
            flow_loss, flow_logs = self.flow(z, cond)
            total = total + self.flow_weight * flow_loss
            logs.update(flow_logs)
        logs["v17_total"] = total
        return total, logs
