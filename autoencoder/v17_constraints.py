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


def _stable_int(value: str) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) & ((1 << 63) - 1)


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


def _masked_logsumexp(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    return torch.logsumexp(logits.masked_fill(~mask, neg_inf), dim=dim)


def _masked_infonce(
    query: torch.Tensor,
    key: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = query @ key.T / max(float(temperature), 1e-4)
    all_mask = torch.ones_like(positive_mask, dtype=torch.bool)
    valid = positive_mask.any(dim=1)
    if not bool(valid.any()):
        zero = query.sum() * 0.0
        return zero, torch.tensor(0.0, device=query.device)
    log_pos = _masked_logsumexp(logits, positive_mask, dim=1)
    log_all = _masked_logsumexp(logits, all_mask, dim=1)
    loss = -(log_pos[valid] - log_all[valid]).mean()
    with torch.no_grad():
        top1 = positive_mask.gather(1, logits.argmax(dim=1, keepdim=True)).float()
        acc = top1[valid].mean() if bool(valid.any()) else torch.tensor(0.0, device=query.device)
    return loss, acc


def _pairwise_js_loss(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    if a.shape[0] < 2:
        return a.sum() * 0.0
    temp = max(float(temperature), 1e-4)
    sim_a = (a @ a.T) / temp
    sim_b = (b @ b.T) / temp
    eye = torch.eye(a.shape[0], device=a.device, dtype=torch.bool)
    sim_a = sim_a.masked_fill(eye, torch.finfo(sim_a.dtype).min)
    sim_b = sim_b.masked_fill(eye, torch.finfo(sim_b.dtype).min)
    log_pa = F.log_softmax(sim_a, dim=-1)
    log_pb = F.log_softmax(sim_b, dim=-1)
    pa = log_pa.exp()
    pb = log_pb.exp()
    m = 0.5 * (pa + pb)
    js = 0.5 * F.kl_div(log_pa, m, reduction="batchmean") + 0.5 * F.kl_div(log_pb, m, reduction="batchmean")
    return js


def _row_text_key(row: dict, positive_key: str = "same_text_group") -> str:
    return str(
        row.get(positive_key)
        or row.get("same_text_group")
        or row.get("text_id")
        or row.get("text")
        or row.get("id")
        or ""
    )


def _row_speaker_key(row: dict) -> str:
    return str(row.get("same_speaker_group") or row.get("speaker_id") or row.get("speaker") or "")


def _rank_guard_loss(
    values: torch.Tensor,
    min_std: float,
    max_top1: float,
    variance_weight: float,
    covariance_weight: float,
    top1_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """VICReg-style anti-collapse guard for factor or full-latent features.

    This is deliberately not a residual supervision target. It only preserves
    enough active dimensions and discourages a single principal direction from
    swallowing the shared/free capacity.
    """
    if values.shape[0] < 2:
        zero = values.sum() * 0.0
        return zero, {
            "var": zero.detach(),
            "cov": zero.detach(),
            "top1": zero.detach(),
            "std": zero.detach(),
        }

    with torch.amp.autocast(device_type=values.device.type, enabled=False):
        x = values.float()
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0, unbiased=False) + 1e-4)
        var_loss = F.relu(float(min_std) - std).mean()

        denom = max(x.shape[0] - 1, 1)
        cov = (x.T @ x) / denom
        diag = torch.diag(cov)
        off_diag = cov - torch.diag_embed(diag)
        cov_loss = off_diag.pow(2).sum() / max(x.shape[1], 1)

        eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
        total_power = eigvals.sum().clamp_min(1e-6)
        top1 = eigvals[-1] / total_power
        top1_loss = F.relu(top1 - float(max_top1)).pow(2)

    loss = (
        float(variance_weight) * var_loss
        + float(covariance_weight) * cov_loss
        + float(top1_weight) * top1_loss
    )
    return loss, {
        "var": var_loss.detach(),
        "cov": cov_loss.detach(),
        "top1": top1.detach(),
        "std": std.mean().detach(),
    }


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


class ResidualTemporalBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.GroupNorm(8, hidden_dim)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation)
        self.ff = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm(x))
        h = self.conv(h)
        h = F.silu(h)
        h = self.ff(h)
        return x + h / math.sqrt(2.0)


class SameModalityEncoderAdapterConstraint(nn.Module):
    """Learned same-modality encoder target with adapter-side alignment.

    The target encoder receives detached AE latents and forms an external
    same-modality anchor. The adapter path can send gradients back into the AE
    latent when detach_adapter_latent=False. A small memory queue keeps
    contrastive positives useful even when audio batches are size 1.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 384,
        proj_dim: int = 192,
        depth: int = 4,
        weight: float = 0.0,
        infonce_weight: float = 1.0,
        js_weight: float = 0.1,
        temperature: float = 0.12,
        queue_size: int = 256,
        positive_key: str = "same_text_group",
        detach_adapter_latent: bool = False,
    ):
        super().__init__()
        self.weight = weight
        self.infonce_weight = infonce_weight
        self.js_weight = js_weight
        self.temperature = temperature
        self.queue_size = queue_size
        self.positive_key = positive_key
        self.detach_adapter_latent = detach_adapter_latent

        self.target_in = nn.Conv1d(latent_dim, hidden_dim, kernel_size=1)
        self.target_blocks = nn.ModuleList(
            [ResidualTemporalBlock(hidden_dim, dilation=2 ** (i % 3)) for i in range(depth)]
        )
        self.target_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

        self.adapter_in = nn.Conv1d(latent_dim, hidden_dim, kernel_size=1)
        self.adapter_blocks = nn.ModuleList(
            [ResidualTemporalBlock(hidden_dim, dilation=2 ** (i % 3)) for i in range(max(depth // 2, 1))]
        )
        self.adapter_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

        self.register_buffer("queue_features", torch.zeros(queue_size, proj_dim), persistent=True)
        self.register_buffer("queue_keys", torch.full((queue_size,), -1, dtype=torch.long), persistent=True)
        self.register_buffer("queue_ptr", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("queue_filled", torch.zeros((), dtype=torch.long), persistent=True)

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        return h.mean(dim=-1)

    def _encode_target(self, z: torch.Tensor) -> torch.Tensor:
        h = self.target_in(z.detach().float())
        for block in self.target_blocks:
            h = block(h)
        return F.normalize(self.target_proj(self._pool(h)), dim=-1)

    def _encode_adapter(self, z: torch.Tensor) -> torch.Tensor:
        z_in = z.detach() if self.detach_adapter_latent else z
        h = self.adapter_in(z_in.float())
        for block in self.adapter_blocks:
            h = block(h)
        return F.normalize(self.adapter_proj(self._pool(h)), dim=-1)

    def _positive_ids(self, metadata: Any, batch_size: int, device: torch.device) -> torch.Tensor:
        rows = _metadata_list(metadata, batch_size)
        values = []
        for row in rows:
            values.append(_stable_int(_row_text_key(row, self.positive_key)))
        return torch.tensor(values, device=device, dtype=torch.long)

    @torch.no_grad()
    def _enqueue(self, features: torch.Tensor, keys: torch.Tensor) -> None:
        if self.queue_size <= 0:
            return
        features = features.detach()
        keys = keys.detach()
        n = features.shape[0]
        if n == 0:
            return
        if n >= self.queue_size:
            self.queue_features.copy_(features[-self.queue_size :])
            self.queue_keys.copy_(keys[-self.queue_size :])
            self.queue_ptr.zero_()
            self.queue_filled.fill_(self.queue_size)
            return
        ptr = int(self.queue_ptr.item())
        end = ptr + n
        if end <= self.queue_size:
            self.queue_features[ptr:end].copy_(features)
            self.queue_keys[ptr:end].copy_(keys)
        else:
            first = self.queue_size - ptr
            self.queue_features[ptr:].copy_(features[:first])
            self.queue_keys[ptr:].copy_(keys[:first])
            self.queue_features[: end - self.queue_size].copy_(features[first:])
            self.queue_keys[: end - self.queue_size].copy_(keys[first:])
        self.queue_ptr.fill_(end % self.queue_size)
        self.queue_filled.fill_(min(self.queue_size, int(self.queue_filled.item()) + n))

    def forward(self, z: torch.Tensor, metadata: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b = z.shape[0]
        device = z.device
        query = self._encode_adapter(z)
        key = self._encode_target(z)
        ids = self._positive_ids(metadata, b, device)

        bank_n = int(self.queue_filled.item())
        if bank_n > 0:
            bank_features = self.queue_features[:bank_n].to(device=device, dtype=key.dtype)
            bank_keys = self.queue_keys[:bank_n].to(device=device)
            all_keys = torch.cat([key, bank_features], dim=0)
            all_ids = torch.cat([ids, bank_keys], dim=0)
        else:
            all_keys = key
            all_ids = ids

        positive_mask = ids[:, None].eq(all_ids[None, :])
        own = torch.arange(b, device=device)
        own_mask = torch.zeros_like(positive_mask)
        own_mask[own, own] = True
        cross_positive_mask = positive_mask & ~own_mask
        has_cross_positive = cross_positive_mask.any(dim=1, keepdim=True)
        positive_mask = torch.where(has_cross_positive, cross_positive_mask, own_mask)

        nce, acc = _masked_infonce(query, all_keys, positive_mask, self.temperature)
        js = _pairwise_js_loss(query, key, self.temperature)
        total = self.infonce_weight * nce + self.js_weight * js

        with torch.no_grad():
            positives = positive_mask.sum(dim=1).float().mean()
            cross_positives = cross_positive_mask.sum(dim=1).float().mean()
            self._enqueue(key, ids)

        logs = {
            "v17_semantic_total": total,
            "v17_semantic_infonce": nce,
            "v17_semantic_js": js,
            "v17_semantic_top1": acc.detach(),
            "v17_semantic_pos": positives.detach(),
            "v17_semantic_cross_pos": cross_positives.detach(),
            "v17_semantic_queue": torch.tensor(float(bank_n), device=device),
        }
        return total, logs


class CrossSampleFactorAlignmentConstraint(nn.Module):
    """Cross-sample phoneme/speaker factor constraint for crossed data.

    For queued samples with the same phoneme/text key but a different speaker:
    - phoneme projection is pulled together.
    - speaker projection is pushed toward the opposite direction.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 384,
        phoneme_dim: int = 192,
        speaker_dim: int = 64,
        queue_size: int = 256,
        positive_key: str = "same_text_group",
        phoneme_weight: float = 1.0,
        phoneme_nce_weight: float = 1.0,
        phoneme_temperature: float = 0.12,
        phoneme_uniform_weight: float = 0.05,
        phoneme_soft_weight: float = 1.0,
        phoneme_soft_hard_weight: float = 2.0,
        phoneme_soft_text_buckets: int = 512,
        phoneme_same_speaker_neg_weight: float = 0.0,
        phoneme_same_speaker_neg_margin: float = 0.15,
        phoneme_same_speaker_neg_max_cos: float = 0.65,
        speaker_opposite_weight: float = 1.0,
        speaker_same_weight: float = 0.0,
        speaker_target_cos: float = -1.0,
        speaker_margin_mode: bool = False,
        speaker_neg_margin: float = -0.2,
        speaker_pos_margin: float = 0.6,
        rank_guard_min_std: float = 0.03,
        rank_guard_max_top1: float = 0.75,
        phoneme_rank_guard_weight: float = 0.0,
        speaker_rank_guard_weight: float = 0.0,
        latent_rank_guard_weight: float = 0.0,
        rank_guard_covariance_weight: float = 0.0,
        rank_guard_top1_weight: float = 1.0,
        detach_latent: bool = False,
    ):
        super().__init__()
        self.queue_size = queue_size
        self.positive_key = positive_key
        self.phoneme_weight = phoneme_weight
        self.phoneme_nce_weight = phoneme_nce_weight
        self.phoneme_temperature = phoneme_temperature
        self.phoneme_uniform_weight = phoneme_uniform_weight
        self.phoneme_soft_weight = phoneme_soft_weight
        self.phoneme_soft_hard_weight = phoneme_soft_hard_weight
        self.phoneme_text_buckets = phoneme_soft_text_buckets
        self.phoneme_same_speaker_neg_weight = phoneme_same_speaker_neg_weight
        self.phoneme_same_speaker_neg_margin = phoneme_same_speaker_neg_margin
        self.phoneme_same_speaker_neg_max_cos = phoneme_same_speaker_neg_max_cos
        self.speaker_opposite_weight = speaker_opposite_weight
        self.speaker_same_weight = speaker_same_weight
        self.speaker_target_cos = speaker_target_cos
        self.speaker_margin_mode = speaker_margin_mode
        self.speaker_neg_margin = speaker_neg_margin
        self.speaker_pos_margin = speaker_pos_margin
        self.rank_guard_min_std = rank_guard_min_std
        self.rank_guard_max_top1 = rank_guard_max_top1
        self.phoneme_rank_guard_weight = phoneme_rank_guard_weight
        self.speaker_rank_guard_weight = speaker_rank_guard_weight
        self.latent_rank_guard_weight = latent_rank_guard_weight
        self.rank_guard_covariance_weight = rank_guard_covariance_weight
        self.rank_guard_top1_weight = rank_guard_top1_weight
        self.detach_latent = detach_latent
        self.pinyin = PinyinToneEncoder(buckets=phoneme_soft_text_buckets)

        self.norm = nn.LayerNorm(latent_dim)
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.phoneme_head = nn.Linear(hidden_dim, phoneme_dim)
        self.speaker_head = nn.Linear(hidden_dim, speaker_dim)

        self.register_buffer("queue_phoneme", torch.zeros(queue_size, phoneme_dim), persistent=True)
        self.register_buffer("queue_speaker", torch.zeros(queue_size, speaker_dim), persistent=True)
        self.register_buffer("queue_latent", torch.zeros(queue_size, latent_dim), persistent=True)
        self.register_buffer("queue_text_vec", torch.zeros(queue_size, phoneme_soft_text_buckets), persistent=True)
        self.register_buffer("queue_text_keys", torch.full((queue_size,), -1, dtype=torch.long), persistent=True)
        self.register_buffer("queue_speaker_keys", torch.full((queue_size,), -1, dtype=torch.long), persistent=True)
        self.register_buffer("queue_ptr", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("queue_filled", torch.zeros((), dtype=torch.long), persistent=True)

    def _project_pooled(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(self.norm(pooled))
        phoneme = F.normalize(self.phoneme_head(h), dim=-1)
        speaker = F.normalize(self.speaker_head(h), dim=-1)
        return phoneme, speaker

    def _encode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_in = z.detach() if self.detach_latent else z
        pooled = z_in.mean(dim=-1).float()
        return self._project_pooled(pooled)

    def _keys(self, metadata: Any, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        rows = _metadata_list(metadata, batch_size)
        text = [_stable_int(_row_text_key(row, self.positive_key)) for row in rows]
        speaker = [_stable_int(_row_speaker_key(row)) for row in rows]
        return (
            torch.tensor(text, device=device, dtype=torch.long),
            torch.tensor(speaker, device=device, dtype=torch.long),
        )

    def _text_vectors(self, metadata: Any, batch_size: int, device: torch.device) -> torch.Tensor:
        rows = _metadata_list(metadata, batch_size)
        vectors = []
        for row in rows:
            text = str(row.get("text", ""))
            ids = self.pinyin.ids(text)
            vec = torch.zeros(self.phoneme_text_buckets, device=device, dtype=torch.float32)
            if ids:
                idx = torch.tensor(ids, device=device, dtype=torch.long).clamp(0, self.phoneme_text_buckets - 1)
                vec.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
            else:
                vec[0] = 1.0
            vec = vec / vec.norm().clamp_min(1e-6)
            vectors.append(vec)
        return torch.stack(vectors, dim=0)

    @torch.no_grad()
    def _enqueue(
        self,
        phoneme: torch.Tensor,
        speaker: torch.Tensor,
        latent_pooled: torch.Tensor,
        text_vec: torch.Tensor,
        text_keys: torch.Tensor,
        speaker_keys: torch.Tensor,
    ) -> None:
        if self.queue_size <= 0:
            return
        phoneme = phoneme.detach()
        speaker = speaker.detach()
        latent_pooled = latent_pooled.detach()
        text_vec = text_vec.detach()
        text_keys = text_keys.detach()
        speaker_keys = speaker_keys.detach()
        n = phoneme.shape[0]
        if n == 0:
            return
        if n >= self.queue_size:
            self.queue_phoneme.copy_(phoneme[-self.queue_size :])
            self.queue_speaker.copy_(speaker[-self.queue_size :])
            self.queue_latent.copy_(latent_pooled[-self.queue_size :])
            self.queue_text_vec.copy_(text_vec[-self.queue_size :])
            self.queue_text_keys.copy_(text_keys[-self.queue_size :])
            self.queue_speaker_keys.copy_(speaker_keys[-self.queue_size :])
            self.queue_ptr.zero_()
            self.queue_filled.fill_(self.queue_size)
            return

        ptr = int(self.queue_ptr.item())
        end = ptr + n
        if end <= self.queue_size:
            self.queue_phoneme[ptr:end].copy_(phoneme)
            self.queue_speaker[ptr:end].copy_(speaker)
            self.queue_latent[ptr:end].copy_(latent_pooled)
            self.queue_text_vec[ptr:end].copy_(text_vec)
            self.queue_text_keys[ptr:end].copy_(text_keys)
            self.queue_speaker_keys[ptr:end].copy_(speaker_keys)
        else:
            first = self.queue_size - ptr
            rest = end - self.queue_size
            self.queue_phoneme[ptr:].copy_(phoneme[:first])
            self.queue_speaker[ptr:].copy_(speaker[:first])
            self.queue_latent[ptr:].copy_(latent_pooled[:first])
            self.queue_text_vec[ptr:].copy_(text_vec[:first])
            self.queue_text_keys[ptr:].copy_(text_keys[:first])
            self.queue_speaker_keys[ptr:].copy_(speaker_keys[:first])
            self.queue_phoneme[:rest].copy_(phoneme[first:])
            self.queue_speaker[:rest].copy_(speaker[first:])
            self.queue_latent[:rest].copy_(latent_pooled[first:])
            self.queue_text_vec[:rest].copy_(text_vec[first:])
            self.queue_text_keys[:rest].copy_(text_keys[first:])
            self.queue_speaker_keys[:rest].copy_(speaker_keys[first:])
        self.queue_ptr.fill_(end % self.queue_size)
        self.queue_filled.fill_(min(self.queue_size, int(self.queue_filled.item()) + n))

    def forward(self, z: torch.Tensor, metadata: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b = z.shape[0]
        device = z.device
        z_in = z.detach() if self.detach_latent else z
        pooled = z_in.mean(dim=-1).float()
        phoneme, speaker = self._project_pooled(pooled)
        latent_pooled = F.layer_norm(pooled, (pooled.shape[-1],))
        text_keys, speaker_keys = self._keys(metadata, b, device)
        text_vec = self._text_vectors(metadata, b, device)

        bank_n = int(self.queue_filled.item())
        if bank_n > 0:
            bank_pooled = self.queue_latent[:bank_n].to(device=device, dtype=pooled.dtype).clone()
            bank_phoneme, bank_speaker = self._project_pooled(bank_pooled)
            bank_latent = F.layer_norm(bank_pooled, (bank_pooled.shape[-1],))
            bank_text_vec = self.queue_text_vec[:bank_n].to(device=device, dtype=text_vec.dtype).clone()
            bank_text = self.queue_text_keys[:bank_n].to(device=device)
            bank_spk = self.queue_speaker_keys[:bank_n].to(device=device)
            pair_mask = text_keys[:, None].eq(bank_text[None, :]) & ~speaker_keys[:, None].eq(bank_spk[None, :])
            nce_positive_mask = text_keys[:, None].eq(bank_text[None, :])
            same_speaker_mask = speaker_keys[:, None].eq(bank_spk[None, :]) & ~text_keys[:, None].eq(bank_text[None, :])
        else:
            bank_phoneme = phoneme.new_zeros((0, phoneme.shape[-1]))
            bank_speaker = speaker.new_zeros((0, speaker.shape[-1]))
            bank_latent = latent_pooled.new_zeros((0, latent_pooled.shape[-1]))
            bank_text_vec = text_vec.new_zeros((0, text_vec.shape[-1]))
            pair_mask = torch.zeros((b, 0), device=device, dtype=torch.bool)
            nce_positive_mask = torch.zeros((b, 0), device=device, dtype=torch.bool)
            same_speaker_mask = torch.zeros((b, 0), device=device, dtype=torch.bool)

        valid = pair_mask.any(dim=1)
        if bool(valid.any()):
            phoneme_cos = phoneme @ bank_phoneme.T
            speaker_cos = speaker @ bank_speaker.T
            phoneme_loss = (1.0 - phoneme_cos)[pair_mask].mean()
            if self.speaker_margin_mode:
                speaker_loss = F.relu(speaker_cos[pair_mask] - self.speaker_neg_margin).pow(2).mean()
            else:
                speaker_loss = (speaker_cos[pair_mask] - self.speaker_target_cos).pow(2).mean()
            phoneme_cos_mean = phoneme_cos[pair_mask].mean()
            speaker_cos_mean = speaker_cos[pair_mask].mean()
            pairs = pair_mask.sum().float()
        else:
            zero = phoneme.sum() * 0.0
            phoneme_loss = zero
            speaker_loss = zero
            phoneme_cos_mean = zero.detach()
            speaker_cos_mean = zero.detach()
            pairs = torch.tensor(0.0, device=device)

        same_speaker_valid = same_speaker_mask.any(dim=1)
        if bool(same_speaker_valid.any()):
            speaker_same_cos = speaker @ bank_speaker.T
            if self.speaker_margin_mode:
                speaker_same_loss = F.relu(self.speaker_pos_margin - speaker_same_cos[same_speaker_mask]).pow(2).mean()
            else:
                speaker_same_loss = (1.0 - speaker_same_cos)[same_speaker_mask].mean()
            speaker_same_cos_mean = speaker_same_cos[same_speaker_mask].mean()
            same_speaker_pairs = same_speaker_mask.sum().float()
        else:
            speaker_same_loss = speaker.sum() * 0.0
            speaker_same_cos_mean = speaker_same_loss.detach()
            same_speaker_pairs = torch.tensor(0.0, device=device)

        nce_valid = nce_positive_mask.any(dim=1)
        if bool(nce_valid.any()):
            phoneme_nce, phoneme_top1 = _masked_infonce(
                phoneme,
                bank_phoneme,
                nce_positive_mask,
                self.phoneme_temperature,
            )
        else:
            phoneme_nce = phoneme.sum() * 0.0
            phoneme_top1 = torch.tensor(0.0, device=device)

        if bank_n > 1:
            all_phoneme = torch.cat([phoneme, bank_phoneme], dim=0)
            sim = all_phoneme @ all_phoneme.T
            eye = torch.eye(sim.shape[0], device=device, dtype=torch.bool)
            # Penalize excessive global similarity. This is zero below the margin
            # and rises when features collapse into a narrow cone.
            phoneme_uniform = F.relu(sim.masked_select(~eye) - 0.20).pow(2).mean()
        else:
            phoneme_uniform = phoneme.sum() * 0.0
            all_phoneme = phoneme

        all_speaker = torch.cat([speaker, bank_speaker], dim=0) if bank_n > 0 else speaker
        all_latent = torch.cat([latent_pooled, bank_latent], dim=0) if bank_n > 0 else latent_pooled
        phoneme_rank_guard, phoneme_rank_logs = _rank_guard_loss(
            all_phoneme,
            self.rank_guard_min_std,
            self.rank_guard_max_top1,
            self.phoneme_rank_guard_weight,
            self.rank_guard_covariance_weight,
            self.rank_guard_top1_weight,
        )
        speaker_rank_guard, speaker_rank_logs = _rank_guard_loss(
            all_speaker,
            self.rank_guard_min_std,
            self.rank_guard_max_top1,
            self.speaker_rank_guard_weight,
            self.rank_guard_covariance_weight,
            self.rank_guard_top1_weight,
        )
        latent_rank_guard, latent_rank_logs = _rank_guard_loss(
            all_latent,
            self.rank_guard_min_std,
            self.rank_guard_max_top1,
            self.latent_rank_guard_weight,
            self.rank_guard_covariance_weight,
            self.rank_guard_top1_weight,
        )

        if bank_n > 0:
            all_phoneme_cos = phoneme @ bank_phoneme.T
            target_sim = (text_vec @ bank_text_vec.T).clamp(0.0, 1.0)
            target_sim = torch.where(nce_positive_mask, torch.ones_like(target_sim), target_sim)
            hard_weight = 1.0 + self.phoneme_soft_hard_weight * target_sim.detach()
            phoneme_soft = (F.smooth_l1_loss(all_phoneme_cos, target_sim, reduction="none") * hard_weight).mean()
            diff_mask = ~nce_positive_mask
            soft_target_mean = target_sim.mean()
            soft_cos_mean = all_phoneme_cos.mean()
            if bool(diff_mask.any()):
                soft_neg_target_mean = target_sim[diff_mask].mean()
                soft_neg_cos_mean = all_phoneme_cos[diff_mask].mean()
            else:
                soft_neg_target_mean = phoneme.sum() * 0.0
                soft_neg_cos_mean = phoneme.sum() * 0.0
            if self.phoneme_same_speaker_neg_weight > 0 and bool(same_speaker_mask.any()):
                # Same speaker but different text is the hardest anti-collapse
                # case for the phoneme head: speaker cues are held constant, so
                # remaining similarity should be bounded by symbolic text overlap.
                same_speaker_target = (
                    target_sim[same_speaker_mask].detach() + float(self.phoneme_same_speaker_neg_margin)
                ).clamp(max=float(self.phoneme_same_speaker_neg_max_cos))
                same_speaker_phoneme_cos = all_phoneme_cos[same_speaker_mask]
                phoneme_same_speaker_neg = F.relu(same_speaker_phoneme_cos - same_speaker_target).pow(2).mean()
                phoneme_same_speaker_neg_cos = same_speaker_phoneme_cos.mean()
                phoneme_same_speaker_neg_target = same_speaker_target.mean()
            else:
                phoneme_same_speaker_neg = phoneme.sum() * 0.0
                phoneme_same_speaker_neg_cos = phoneme.sum() * 0.0
                phoneme_same_speaker_neg_target = phoneme.sum() * 0.0
        else:
            phoneme_soft = phoneme.sum() * 0.0
            soft_target_mean = phoneme.sum() * 0.0
            soft_cos_mean = phoneme.sum() * 0.0
            soft_neg_target_mean = phoneme.sum() * 0.0
            soft_neg_cos_mean = phoneme.sum() * 0.0
            phoneme_same_speaker_neg = phoneme.sum() * 0.0
            phoneme_same_speaker_neg_cos = phoneme.sum() * 0.0
            phoneme_same_speaker_neg_target = phoneme.sum() * 0.0

        total = (
            self.phoneme_weight * phoneme_loss
            + self.phoneme_nce_weight * phoneme_nce
            + self.phoneme_uniform_weight * phoneme_uniform
            + self.phoneme_soft_weight * phoneme_soft
            + self.phoneme_same_speaker_neg_weight * phoneme_same_speaker_neg
            + self.speaker_opposite_weight * speaker_loss
            + self.speaker_same_weight * speaker_same_loss
            + phoneme_rank_guard
            + speaker_rank_guard
            + latent_rank_guard
        )
        with torch.no_grad():
            self._enqueue(phoneme, speaker, pooled, text_vec, text_keys, speaker_keys)

        logs = {
            "v17_cross_factor_total": total,
            "v17_cross_phoneme": phoneme_loss,
            "v17_cross_phoneme_nce": phoneme_nce,
            "v17_cross_phoneme_top1": phoneme_top1.detach(),
            "v17_cross_phoneme_uniform": phoneme_uniform,
            "v17_cross_phoneme_soft": phoneme_soft,
            "v17_cross_phoneme_same_speaker_neg": phoneme_same_speaker_neg,
            "v17_cross_phoneme_same_speaker_neg_cos": phoneme_same_speaker_neg_cos.detach(),
            "v17_cross_phoneme_same_speaker_neg_target": phoneme_same_speaker_neg_target.detach(),
            "v17_cross_soft_target": soft_target_mean.detach(),
            "v17_cross_soft_cos": soft_cos_mean.detach(),
            "v17_cross_soft_neg_target": soft_neg_target_mean.detach(),
            "v17_cross_soft_neg_cos": soft_neg_cos_mean.detach(),
            "v17_cross_speaker_opp": speaker_loss,
            "v17_cross_speaker_same": speaker_same_loss,
            "v17_cross_pairs": pairs.detach(),
            "v17_cross_same_speaker_pairs": same_speaker_pairs.detach(),
            "v17_cross_nce_pos": nce_positive_mask.sum().float().detach(),
            "v17_cross_phoneme_cos": phoneme_cos_mean.detach(),
            "v17_cross_speaker_cos": speaker_cos_mean.detach(),
            "v17_cross_speaker_same_cos": speaker_same_cos_mean.detach(),
            "v17_cross_phoneme_rank_guard": phoneme_rank_guard.detach(),
            "v17_cross_speaker_rank_guard": speaker_rank_guard.detach(),
            "v17_cross_latent_rank_guard": latent_rank_guard.detach(),
            "v17_cross_phoneme_rank_top1": phoneme_rank_logs["top1"],
            "v17_cross_speaker_rank_top1": speaker_rank_logs["top1"],
            "v17_cross_latent_rank_top1": latent_rank_logs["top1"],
            "v17_cross_phoneme_std": phoneme_rank_logs["std"],
            "v17_cross_speaker_std": speaker_rank_logs["std"],
            "v17_cross_latent_std": latent_rank_logs["std"],
            "v17_cross_queue": torch.tensor(float(bank_n), device=device),
        }
        return total, logs


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
        semantic_weight: float = 0.0,
        semantic_hidden_dim: int = 384,
        semantic_proj_dim: int = 192,
        semantic_depth: int = 4,
        semantic_infonce_weight: float = 1.0,
        semantic_js_weight: float = 0.1,
        semantic_temperature: float = 0.12,
        semantic_queue_size: int = 256,
        semantic_positive_key: str = "same_text_group",
        semantic_detach_adapter_latent: bool = False,
        cross_factor_weight: float = 0.0,
        cross_factor_hidden_dim: int = 384,
        cross_factor_phoneme_dim: int = 192,
        cross_factor_speaker_dim: int = 64,
        cross_factor_queue_size: int = 256,
        cross_factor_positive_key: str = "same_text_group",
        cross_factor_phoneme_weight: float = 1.0,
        cross_factor_phoneme_nce_weight: float = 1.0,
        cross_factor_phoneme_temperature: float = 0.12,
        cross_factor_phoneme_uniform_weight: float = 0.05,
        cross_factor_phoneme_soft_weight: float = 1.0,
        cross_factor_phoneme_soft_hard_weight: float = 2.0,
        cross_factor_phoneme_soft_text_buckets: int = 512,
        cross_factor_phoneme_same_speaker_neg_weight: float = 0.0,
        cross_factor_phoneme_same_speaker_neg_margin: float = 0.15,
        cross_factor_phoneme_same_speaker_neg_max_cos: float = 0.65,
        cross_factor_speaker_weight: float = 1.0,
        cross_factor_speaker_same_weight: float = 0.0,
        cross_factor_speaker_target_cos: float = -1.0,
        cross_factor_speaker_margin_mode: bool = False,
        cross_factor_speaker_neg_margin: float = -0.2,
        cross_factor_speaker_pos_margin: float = 0.6,
        cross_factor_rank_guard_min_std: float = 0.03,
        cross_factor_rank_guard_max_top1: float = 0.75,
        cross_factor_phoneme_rank_guard_weight: float = 0.0,
        cross_factor_speaker_rank_guard_weight: float = 0.0,
        cross_factor_latent_rank_guard_weight: float = 0.0,
        cross_factor_rank_guard_covariance_weight: float = 0.0,
        cross_factor_rank_guard_top1_weight: float = 1.0,
        cross_factor_detach_latent: bool = False,
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
        self.semantic_weight = semantic_weight
        self.cross_factor_weight = cross_factor_weight
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
        self.semantic = SameModalityEncoderAdapterConstraint(
            latent_dim=latent_dim,
            hidden_dim=semantic_hidden_dim,
            proj_dim=semantic_proj_dim,
            depth=semantic_depth,
            weight=semantic_weight,
            infonce_weight=semantic_infonce_weight,
            js_weight=semantic_js_weight,
            temperature=semantic_temperature,
            queue_size=semantic_queue_size,
            positive_key=semantic_positive_key,
            detach_adapter_latent=semantic_detach_adapter_latent,
        )
        self.cross_factor = CrossSampleFactorAlignmentConstraint(
            latent_dim=latent_dim,
            hidden_dim=cross_factor_hidden_dim,
            phoneme_dim=cross_factor_phoneme_dim,
            speaker_dim=cross_factor_speaker_dim,
            queue_size=cross_factor_queue_size,
            positive_key=cross_factor_positive_key,
            phoneme_weight=cross_factor_phoneme_weight,
            phoneme_nce_weight=cross_factor_phoneme_nce_weight,
            phoneme_temperature=cross_factor_phoneme_temperature,
            phoneme_uniform_weight=cross_factor_phoneme_uniform_weight,
            phoneme_soft_weight=cross_factor_phoneme_soft_weight,
            phoneme_soft_hard_weight=cross_factor_phoneme_soft_hard_weight,
            phoneme_soft_text_buckets=cross_factor_phoneme_soft_text_buckets,
            phoneme_same_speaker_neg_weight=cross_factor_phoneme_same_speaker_neg_weight,
            phoneme_same_speaker_neg_margin=cross_factor_phoneme_same_speaker_neg_margin,
            phoneme_same_speaker_neg_max_cos=cross_factor_phoneme_same_speaker_neg_max_cos,
            speaker_opposite_weight=cross_factor_speaker_weight,
            speaker_same_weight=cross_factor_speaker_same_weight,
            speaker_target_cos=cross_factor_speaker_target_cos,
            speaker_margin_mode=cross_factor_speaker_margin_mode,
            speaker_neg_margin=cross_factor_speaker_neg_margin,
            speaker_pos_margin=cross_factor_speaker_pos_margin,
            rank_guard_min_std=cross_factor_rank_guard_min_std,
            rank_guard_max_top1=cross_factor_rank_guard_max_top1,
            phoneme_rank_guard_weight=cross_factor_phoneme_rank_guard_weight,
            speaker_rank_guard_weight=cross_factor_speaker_rank_guard_weight,
            latent_rank_guard_weight=cross_factor_latent_rank_guard_weight,
            rank_guard_covariance_weight=cross_factor_rank_guard_covariance_weight,
            rank_guard_top1_weight=cross_factor_rank_guard_top1_weight,
            detach_latent=cross_factor_detach_latent,
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
        return self.ext_weight > 0 or self.semantic_weight > 0 or self.cross_factor_weight > 0 or self.flow_weight > 0

    def forward(self, z: torch.Tensor, metadata: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total = torch.tensor(0.0, device=z.device)
        logs: dict[str, torch.Tensor] = {}
        b = z.shape[0]
        if self.ext_weight > 0:
            ext_loss, ext_logs = self.external(z, metadata)
            total = total + self.ext_weight * ext_loss
            logs.update(ext_logs)
        if self.semantic_weight > 0:
            sem_loss, sem_logs = self.semantic(z, metadata)
            total = total + self.semantic_weight * sem_loss
            logs.update(sem_logs)
        if self.cross_factor_weight > 0:
            cross_loss, cross_logs = self.cross_factor(z, metadata)
            total = total + self.cross_factor_weight * cross_loss
            logs.update(cross_logs)
        if self.flow_weight > 0:
            cond = self.external.condition_vector(metadata, b, z.device)
            flow_loss, flow_logs = self.flow(z, cond)
            total = total + self.flow_weight * flow_loss
            logs.update(flow_logs)
        logs["v17_total"] = total
        return total, logs
