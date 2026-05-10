"""V17.4 Flow + Analogy structural constraint for AE latents.

Joint training module that adds two constraints to the AE:
1. Flow matching on a learned semantic subspace (generation-friendly)
2. Analogy loss for direction consistency (semantic linearity)

Both operate on z_proj = SemanticProjection(z), a low-dimensional
L2-normalized projection of the full latent sequence.

Usage in train.py:
    v17_4 = V174FlowStructureModule(latent_dim=384, ...).to(device)
    ...
    v17_4_loss, v17_4_logs = v17_4(z, metadata)
    total_loss = recon_loss + v17_4_loss
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Semantic Projection ────────────────────────────────────


class SemanticProjection(nn.Module):
    """Conv + attention pooling + linear projection to semantic subspace.

    Maps z [B, latent_dim, T] -> z_proj [B, proj_dim], L2-normalized.
    """

    def __init__(self, latent_dim: int = 384, proj_dim: int = 64, hidden_dim: int = 256, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(latent_dim, hidden_dim, 1), nn.SiLU()]
        for i in range(depth):
            dilation = 2 ** i
            pad = (3 // 2) * dilation
            layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, 3, padding=pad, dilation=dilation),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(),
            ])
        self.conv = nn.Sequential(*layers)
        self.attn = nn.Conv1d(hidden_dim, 1, 1)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_dim, bias=False),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """[B, latent_dim, T] -> [B, proj_dim] normalized."""
        h = self.conv(z)
        w = torch.softmax(self.attn(h), dim=-1)
        pooled = (h * w).sum(dim=-1)
        raw = self.proj(pooled)
        return F.normalize(raw, dim=-1)


# ─── Flow Matching Network ───────────────────────────────────


class FlowMatchingNet(nn.Module):
    """Predicts velocity v(x_t, t, cond) for conditional flow matching."""

    def __init__(self, proj_dim: int, cond_dim: int, hidden_dim: int = 256, depth: int = 4):
        super().__init__()
        input_dim = proj_dim + 1 + cond_dim
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, proj_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_t, t, cond], dim=-1))


# ─── Condition Encoder ───────────────────────────────────────


class ConditionEncoder(nn.Module):
    """Learned embeddings for text_group + speaker_id."""

    def __init__(self, n_text_groups: int = 1024, n_speakers: int = 16, emb_dim: int = 128):
        super().__init__()
        self.text_emb = nn.Embedding(n_text_groups, emb_dim)
        self.speaker_emb = nn.Embedding(n_speakers, emb_dim)
        self.out_dim = emb_dim * 2

    def forward(self, text_ids: torch.Tensor, speaker_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.text_emb(text_ids), self.speaker_emb(speaker_ids)], dim=-1)


# ─── Main Module ─────────────────────────────────────────────


def _text_key(row: dict) -> str:
    return str(
        row.get("same_text_group")
        or row.get("text_id")
        or row.get("text")
        or row.get("id")
        or ""
    )


def _speaker_key(row: dict) -> str:
    return str(
        row.get("same_speaker_group")
        or row.get("speaker_id")
        or row.get("speaker")
        or ""
    )


class V174FlowStructureModule(nn.Module):
    """V17.4 joint flow matching + analogy constraint.

    Integrates into the AE training loop. Receives z and metadata,
    returns a scalar loss and diagnostic logs.
    """

    def __init__(
        self,
        latent_dim: int = 384,
        proj_dim: int = 64,
        proj_hidden_dim: int = 256,
        proj_depth: int = 3,
        flow_hidden_dim: int = 256,
        flow_depth: int = 4,
        cond_emb_dim: int = 128,
        n_text_groups: int = 1024,
        n_speakers: int = 16,
        flow_weight: float = 1.0,
        analogy_weight: float = 1.0,
        diversity_weight: float = 0.5,
        diversity_max_cos: float = 0.5,
        analogy_min_cos: float = 0.0,
        speaker_centroid_momentum: float = 0.99,
    ):
        super().__init__()
        self.flow_weight = flow_weight
        self.analogy_weight = analogy_weight
        self.diversity_weight = diversity_weight
        self.diversity_max_cos = diversity_max_cos
        self.analogy_min_cos = analogy_min_cos
        self.momentum = speaker_centroid_momentum

        self.semantic_proj = SemanticProjection(latent_dim, proj_dim, proj_hidden_dim, proj_depth)

        cond_dim = cond_emb_dim * 2
        self.flow_net = FlowMatchingNet(proj_dim, cond_dim, flow_hidden_dim, flow_depth)
        self.cond_enc = ConditionEncoder(n_text_groups, n_speakers, cond_emb_dim)

        # EMA speaker centroids for centering
        self.register_buffer("speaker_centroids", torch.zeros(n_speakers, proj_dim))
        self.register_buffer("speaker_counts", torch.zeros(n_speakers))
        self.register_buffer("centroids_initialized", torch.tensor(False))

        # z_proj memory bank for batch-like flow matching with small batches
        self.bank_size = 64
        self.register_buffer("bank_z_proj", torch.zeros(self.bank_size, proj_dim))
        self.register_buffer("bank_text_ids", torch.zeros(self.bank_size, dtype=torch.long))
        self.register_buffer("bank_speaker_ids", torch.zeros(self.bank_size, dtype=torch.long))
        self.register_buffer("bank_valid", torch.zeros(self.bank_size, dtype=torch.bool))
        self.register_buffer("bank_ptr", torch.tensor(0, dtype=torch.long))

        # Queue of speaker-change directions for analogy loss
        # Each entry: a unit-normalized speaker-change direction from a text group
        self.analogy_queue_size = 64
        self.register_buffer(
            "analogy_queue", torch.zeros(self.analogy_queue_size, proj_dim)
        )
        self.register_buffer(
            "analogy_queue_valid", torch.zeros(self.analogy_queue_size, dtype=torch.bool)
        )
        self.register_buffer("analogy_queue_ptr", torch.tensor(0, dtype=torch.long))

        # ID mappings (set externally before training)
        self._text_to_id: dict[str, int] = {}
        self._speaker_to_id: dict[str, int] = {}

    def set_id_mappings(self, text_to_id: dict[str, int], speaker_to_id: dict[str, int]) -> None:
        self._text_to_id = text_to_id
        self._speaker_to_id = speaker_to_id

    def _get_ids(self, metadata: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract text and speaker IDs from metadata list."""
        text_ids = []
        speaker_ids = []
        for row in metadata:
            tk = _text_key(row)
            sk = _speaker_key(row)
            text_ids.append(self._text_to_id.get(tk, 0))
            speaker_ids.append(self._speaker_to_id.get(sk, 0))
        device = self.speaker_centroids.device
        return (
            torch.tensor(text_ids, device=device),
            torch.tensor(speaker_ids, device=device),
        )

    @torch.no_grad()
    def _update_centroids(self, z_proj: torch.Tensor, speaker_ids: torch.Tensor) -> None:
        """EMA update of speaker centroids."""
        for i in range(z_proj.shape[0]):
            sid = int(speaker_ids[i].item())
            if sid >= self.speaker_centroids.shape[0]:
                continue
            if not self.centroids_initialized:
                self.speaker_centroids[sid] = z_proj[i]
                self.speaker_counts[sid] += 1
            else:
                self.speaker_centroids[sid] = (
                    self.momentum * self.speaker_centroids[sid]
                    + (1 - self.momentum) * z_proj[i]
                )
        # Normalize centroids
        norms = self.speaker_centroids.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.speaker_centroids.div_(norms)
        # Mark initialized after first full pass
        if self.speaker_counts.min() > 0:
            self.centroids_initialized.fill_(True)

    def _flow_loss(
        self, z_proj: torch.Tensor, text_ids: torch.Tensor, speaker_ids: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Flow matching on speaker-centered z_proj.

        Combines current batch with memory bank so small batches still get
        enough condition diversity for flow matching to learn meaningful
        mappings.
        """
        B = z_proj.shape[0]
        if B < 1:
            zero = z_proj.sum() * 0.0
            return zero, {}

        # Speaker centering for current batch (with grad)
        spk_cents = self.speaker_centroids[speaker_ids]
        z_residual = z_proj - spk_cents
        z_residual = F.normalize(z_residual, dim=-1)

        # Collect bank samples (no grad, for additional flow matching samples)
        bank_valid = self.bank_valid
        if bank_valid.any():
            bank_z = self.bank_z_proj[bank_valid]  # [N_bank, proj_dim]
            bank_tids = self.bank_text_ids[bank_valid]
            bank_sids = self.bank_speaker_ids[bank_valid]
            bank_cents = self.speaker_centroids[bank_sids]
            bank_residual = bank_z - bank_cents
            bank_residual = F.normalize(bank_residual, dim=-1)
            bank_cond = self.cond_enc(bank_tids, bank_sids)

            # Concatenate current (with grad) + bank (no grad)
            all_residual = torch.cat([z_residual, bank_residual], dim=0)
            all_cond = torch.cat([cond, bank_cond], dim=0)
        else:
            all_residual = z_residual
            all_cond = cond

        # Flow matching on combined batch
        N = all_residual.shape[0]
        t = torch.rand(N, 1, device=z_proj.device)
        noise = torch.randn_like(all_residual)
        x_t = (1 - t) * noise + t * all_residual
        target_v = all_residual - noise
        pred_v = self.flow_net(x_t, t, all_cond)

        loss = F.mse_loss(pred_v, target_v)

        # Update bank with current batch (no grad)
        with torch.no_grad():
            for i in range(B):
                ptr = int(self.bank_ptr.item())
                self.bank_z_proj[ptr] = z_proj[i].detach()
                self.bank_text_ids[ptr] = text_ids[i]
                self.bank_speaker_ids[ptr] = speaker_ids[i]
                self.bank_valid[ptr] = True
                self.bank_ptr.fill_((ptr + 1) % self.bank_size)

        logs = {
            "flow_mse": loss.detach(),
            "z_residual_var": z_residual.var(dim=0).mean().detach() if B > 1 else torch.tensor(0.0, device=z_proj.device),
            "flow_batch_n": torch.tensor(float(N), device=z_proj.device),
        }
        return loss, logs

    def _analogy_loss(
        self, z_proj: torch.Tensor, text_ids: torch.Tensor, speaker_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Direction consistency across text groups.

        Collect speaker-change directions from the current batch and compare
        against a queue of past directions. Encourage cosine similarity
        (same speaker-change direction regardless of which text).
        """
        B = z_proj.shape[0]
        if B < 2:
            zero = z_proj.sum() * 0.0
            return zero, {"analogy_cos": zero.detach(), "analogy_pairs": torch.tensor(0.0)}

        # Group by text to find speaker-change pairs in current batch
        text_groups: dict[int, dict[int, int]] = {}  # text_id -> {speaker_id: batch_idx}
        for i in range(B):
            tid = int(text_ids[i].item())
            sid = int(speaker_ids[i].item())
            text_groups.setdefault(tid, {})[sid] = i

        # Collect current-batch directions (with grad)
        current_dirs: list[torch.Tensor] = []
        for tid, spk_map in text_groups.items():
            if len(spk_map) < 2:
                continue
            spk_list = list(spk_map.keys())
            i0 = spk_map[spk_list[0]]
            i1 = spk_map[spk_list[1]]
            d = z_proj[i1] - z_proj[i0]
            d_norm = F.normalize(d.unsqueeze(0), dim=-1).squeeze(0)
            current_dirs.append(d_norm)

        if not current_dirs:
            zero = z_proj.sum() * 0.0
            return zero, {"analogy_cos": zero.detach(), "analogy_pairs": torch.tensor(0.0)}

        # Compare current directions against queue (past directions)
        loss = z_proj.sum() * 0.0
        n_pairs = 0
        queue_valid = self.analogy_queue_valid
        if queue_valid.any():
            queue = self.analogy_queue[queue_valid]  # [N_queue, proj_dim]
            for d in current_dirs:
                # Cosine with all queue entries
                cos_values = (d.unsqueeze(0) @ queue.T).squeeze(0)  # [N_queue]
                # Encourage high cosine
                loss = loss + (1.0 - cos_values.mean())
                # Soft margin penalty if too negative
                loss = loss + F.relu(self.analogy_min_cos - cos_values).mean()
                n_pairs += cos_values.shape[0]

            if n_pairs > 0:
                loss = loss / max(len(current_dirs), 1)

        # Update queue with current directions (no grad)
        with torch.no_grad():
            for d in current_dirs:
                ptr = int(self.analogy_queue_ptr.item())
                self.analogy_queue[ptr] = d.detach()
                self.analogy_queue_valid[ptr] = True
                self.analogy_queue_ptr.fill_((ptr + 1) % self.analogy_queue_size)

        mean_cos = torch.tensor(0.0, device=z_proj.device)
        if n_pairs > 0 and queue_valid.any():
            queue = self.analogy_queue[self.analogy_queue_valid]
            all_cos = []
            for d in current_dirs:
                all_cos.append((d.unsqueeze(0) @ queue.T).squeeze(0).detach())
            mean_cos = torch.cat(all_cos).mean()

        logs = {
            "analogy_cos": mean_cos,
            "analogy_pairs": torch.tensor(float(n_pairs), device=z_proj.device),
        }
        return loss, logs

    def _diversity_loss(self, z_proj: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Prevent all z_proj from collapsing to same direction."""
        B = z_proj.shape[0]
        if B < 2:
            zero = z_proj.sum() * 0.0
            return zero, {"mean_cos": zero.detach()}

        cos_matrix = z_proj @ z_proj.T
        mask = ~torch.eye(B, device=z_proj.device, dtype=torch.bool)
        mean_cos = cos_matrix[mask].mean()
        loss = F.relu(mean_cos - self.diversity_max_cos)
        return loss, {"mean_cos": mean_cos.detach()}

    def forward(
        self, z: torch.Tensor, metadata: list[dict] | Any, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            z: [B, latent_dim, T] encoder output
            metadata: list of dicts with text/speaker info
        Returns:
            loss: scalar
            logs: dict of diagnostic tensors
        """
        if not isinstance(metadata, list):
            metadata = [metadata] if isinstance(metadata, dict) else [{}]

        B = z.shape[0]
        text_ids, speaker_ids = self._get_ids(metadata[:B])

        # Project to semantic subspace
        z_proj = self.semantic_proj(z)  # [B, proj_dim], normalized

        # Update speaker centroids (no grad)
        self._update_centroids(z_proj.detach(), speaker_ids)

        # Flow matching loss (uses bank for larger effective batch)
        cond = self.cond_enc(text_ids, speaker_ids)
        flow_loss, flow_logs = self._flow_loss(z_proj, text_ids, speaker_ids, cond)

        # Analogy loss
        analogy_loss, analogy_logs = self._analogy_loss(z_proj, text_ids, speaker_ids)

        # Diversity loss
        div_loss, div_logs = self._diversity_loss(z_proj)

        # Total
        total = (
            self.flow_weight * flow_loss
            + self.analogy_weight * analogy_loss
            + self.diversity_weight * div_loss
        )

        logs: dict[str, torch.Tensor] = {
            "v174_total": total.detach(),
            "v174_flow": flow_loss.detach(),
            "v174_analogy": analogy_loss.detach(),
            "v174_diversity": div_loss.detach(),
            **{f"v174_{k}": v for k, v in flow_logs.items()},
            **{f"v174_{k}": v for k, v in analogy_logs.items()},
            **{f"v174_{k}": v for k, v in div_logs.items()},
        }
        return total, logs
