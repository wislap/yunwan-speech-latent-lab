"""V17.4 Phase 1: Joint AE + Flow Subspace + Analogy Training.

Unfreezes the AE encoder and jointly trains:
  1. Reconstruction (existing losses)
  2. Flow matching on learned semantic subspace (condition-predictability)
  3. Analogy loss on crossed pairs (direction consistency)

The flow matching and analogy losses provide structural pressure on the encoder
to organize latent information into condition-predictable, direction-consistent
subspaces — without requiring strict orthogonal factorization.

Usage:
  python -m autoencoder.scripts.v17_4_joint_train \
    --ae-checkpoint outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt \
    --probe-checkpoint outputs/models/v17_4_flow_probe_v5_spk_centered/best.pt \
    --data-path CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_train_phalign_neighbors.jsonl \
    --eval-data-path CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_val_phalign_neighbors.jsonl \
    --output-dir outputs/models/v17_4_joint_phase1
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from autoencoder.models.autoencoder import WavVAE
from autoencoder.losses_stft import MultiResolutionSTFTLoss
from autoencoder.losses import MultiScaleMelLoss, compute_reconstruction_loss


# ─── Reuse modules from Phase 0 probe ───────────────────────


class FlowMatchingNet(nn.Module):
    def __init__(self, proj_dim: int, cond_dim: int, hidden_dim: int = 256, depth: int = 4):
        super().__init__()
        input_dim = proj_dim + 1 + cond_dim
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, proj_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_t, t, cond], dim=-1))


class ConditionEncoder(nn.Module):
    def __init__(self, n_text_groups: int = 1024, n_speakers: int = 16, emb_dim: int = 128):
        super().__init__()
        self.text_emb = nn.Embedding(n_text_groups, emb_dim)
        self.speaker_emb = nn.Embedding(n_speakers, emb_dim)
        self.out_dim = emb_dim * 2

    def forward(self, text_ids: torch.Tensor, speaker_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.text_emb(text_ids), self.speaker_emb(speaker_ids)], dim=-1)


class SemanticProjection(nn.Module):
    def __init__(self, latent_dim: int = 384, proj_dim: int = 64, hidden_dim: int = 256, depth: int = 3):
        super().__init__()
        layers = [nn.Conv1d(latent_dim, hidden_dim, 1), nn.SiLU()]
        for i in range(depth):
            dilation = 2 ** i
            pad = (3 // 2) * dilation
            layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, 3, padding=pad, dilation=dilation),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(),
            ])
        self.conv = nn.Sequential(*layers)
        self.attn = nn.Sequential(nn.Conv1d(hidden_dim, 1, 1))
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_dim, bias=False),
        )

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        h = self.conv(z_seq)
        attn_weights = torch.softmax(self.attn(h), dim=-1)
        pooled = (h * attn_weights).sum(dim=-1)
        raw = self.proj(pooled)
        return F.normalize(raw, dim=-1)


# ─── Analogy Loss ────────────────────────────────────────────


def compute_analogy_loss(
    z_proj: torch.Tensor,
    text_ids: list[int],
    speaker_ids: list[int],
    min_cos: float = 0.3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Direction consistency loss using crossed pairs.

    For each pair of texts (A, B) that both appear under speakers (1, 2):
      d_spk_A = z_proj[A, spk1] - z_proj[A, spk2]  (speaker direction for text A)
      d_spk_B = z_proj[B, spk1] - z_proj[B, spk2]  (speaker direction for text B)
      loss += relu(min_cos - cos(d_spk_A, d_spk_B))

    This encourages speaker directions to be consistent across texts.
    Similarly for text directions across speakers.
    """
    device = z_proj.device
    B = z_proj.shape[0]

    # Build index: (text_id, speaker_id) -> batch index
    pair_to_idx: dict[tuple[int, int], int] = {}
    for i in range(B):
        pair_to_idx[(text_ids[i], speaker_ids[i])] = i

    # Find crossed quads: (text_A, spk_1), (text_A, spk_2), (text_B, spk_1), (text_B, spk_2)
    texts_by_speaker: dict[int, set[int]] = defaultdict(set)
    speakers_by_text: dict[int, set[int]] = defaultdict(set)
    for i in range(B):
        texts_by_speaker[speaker_ids[i]].add(text_ids[i])
        speakers_by_text[text_ids[i]].add(speaker_ids[i])

    # Speaker direction consistency
    spk_cos_losses = []
    spk_cos_values = []

    speaker_list = list(texts_by_speaker.keys())
    if len(speaker_list) >= 2:
        spk_1, spk_2 = speaker_list[0], speaker_list[1]
        shared_texts = texts_by_speaker[spk_1] & texts_by_speaker[spk_2]
        shared_texts = sorted(shared_texts)

        if len(shared_texts) >= 2:
            # Compute speaker directions for all shared texts
            directions = []
            for tid in shared_texts:
                idx1 = pair_to_idx.get((tid, spk_1))
                idx2 = pair_to_idx.get((tid, spk_2))
                if idx1 is not None and idx2 is not None:
                    d = z_proj[idx1] - z_proj[idx2]
                    directions.append(d)

            if len(directions) >= 2:
                dirs = torch.stack(directions)  # [N_texts, proj_dim]
                dirs_norm = F.normalize(dirs, dim=-1)
                # Pairwise cosine
                cos_matrix = dirs_norm @ dirs_norm.T
                mask = ~torch.eye(len(directions), device=device, dtype=torch.bool)
                pairwise_cos = cos_matrix[mask]
                # Loss: penalize if cos < min_cos
                spk_loss = F.relu(min_cos - pairwise_cos).mean()
                spk_cos_losses.append(spk_loss)
                spk_cos_values.append(pairwise_cos.mean().item())

    # Text direction consistency (across speakers)
    txt_cos_losses = []
    txt_cos_values = []

    text_list = [t for t, spks in speakers_by_text.items() if len(spks) >= 2]
    if len(text_list) >= 2:
        # For each speaker pair, compute text directions
        for spk_pair_idx in range(min(len(speaker_list), 3)):
            for spk_pair_idx2 in range(spk_pair_idx + 1, min(len(speaker_list), 4)):
                spk_a = speaker_list[spk_pair_idx]
                spk_b = speaker_list[spk_pair_idx2]
                # Text directions under spk_a vs spk_b
                dirs_a = []
                dirs_b = []
                common_text_pairs = []
                for i, t1 in enumerate(text_list[:8]):
                    for t2 in text_list[i+1:i+4]:
                        idx_a1 = pair_to_idx.get((t1, spk_a))
                        idx_a2 = pair_to_idx.get((t2, spk_a))
                        idx_b1 = pair_to_idx.get((t1, spk_b))
                        idx_b2 = pair_to_idx.get((t2, spk_b))
                        if all(x is not None for x in [idx_a1, idx_a2, idx_b1, idx_b2]):
                            d_a = z_proj[idx_a1] - z_proj[idx_a2]
                            d_b = z_proj[idx_b1] - z_proj[idx_b2]
                            dirs_a.append(d_a)
                            dirs_b.append(d_b)

                if dirs_a:
                    dirs_a_t = torch.stack(dirs_a)
                    dirs_b_t = torch.stack(dirs_b)
                    cos_vals = F.cosine_similarity(
                        F.normalize(dirs_a_t, dim=-1),
                        F.normalize(dirs_b_t, dim=-1),
                        dim=-1,
                    )
                    txt_loss = F.relu(min_cos - cos_vals).mean()
                    txt_cos_losses.append(txt_loss)
                    txt_cos_values.append(cos_vals.mean().item())

    # Combine
    total_loss = z_proj.sum() * 0.0  # zero on correct device
    if spk_cos_losses:
        total_loss = total_loss + torch.stack(spk_cos_losses).mean()
    if txt_cos_losses:
        total_loss = total_loss + torch.stack(txt_cos_losses).mean()

    logs = {
        "spk_dir_cos": np.mean(spk_cos_values) if spk_cos_values else 0.0,
        "txt_dir_cos": np.mean(txt_cos_values) if txt_cos_values else 0.0,
        "n_spk_pairs": len(spk_cos_losses),
        "n_txt_pairs": len(txt_cos_losses),
    }
    return total_loss, logs


# ─── Data ────────────────────────────────────────────────────


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def resolve_audio_path(item: dict, roots: list[Path]) -> Path:
    for key in ("audio_path", "path", "wav_path", "file"):
        if key in item:
            p = Path(item[key])
            if p.exists():
                return p
            for root in roots:
                candidate = root / p
                if candidate.exists():
                    return candidate
    raise FileNotFoundError(f"Cannot find audio for {item.get('id', item)}")


def load_audio(path: Path, sr: int, max_samples: int, total_stride: int) -> torch.Tensor:
    data, file_sr = sf.read(path, always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    audio = torch.from_numpy(data.astype(np.float32))
    if file_sr != sr:
        audio = torchaudio.functional.resample(audio, file_sr, sr)
    if max_samples > 0 and audio.numel() > max_samples:
        start = (audio.numel() - max_samples) // 2
        start = (start // total_stride) * total_stride
        audio = audio[start: start + max_samples]
    aligned_len = (audio.numel() // total_stride) * total_stride
    if aligned_len < total_stride:
        audio = F.pad(audio, (0, total_stride - audio.numel()))
    else:
        audio = audio[:aligned_len]
    return audio


def text_key(item: dict) -> str:
    return str(item.get("same_text_group") or item.get("text_id") or item.get("text") or item.get("id") or "")


def speaker_key(item: dict) -> str:
    return str(item.get("same_speaker_group") or item.get("speaker_id") or item.get("speaker") or "")


def compute_snr(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    noise = x_hat - x
    signal_power = (x ** 2).mean()
    noise_power = (noise ** 2).mean()
    if noise_power < 1e-10:
        return 100.0
    return 10.0 * math.log10(signal_power.item() / noise_power.item())


# ─── Grouped Batch Sampler ───────────────────────────────────


class CrossedGroupBatcher:
    """Yields batches where each batch contains multiple (text, speaker) pairs.

    Ensures crossed structure: same text appears under multiple speakers in
    each batch, enabling analogy loss computation.
    """

    def __init__(self, items: list[dict], batch_groups: int = 8):
        self.batch_groups = batch_groups
        # Group items by text
        self.text_groups: dict[str, list[int]] = defaultdict(list)
        for i, item in enumerate(items):
            self.text_groups[text_key(item)].append(i)
        # Keep only groups with multiple speakers
        self.valid_groups = [
            (tk, indices) for tk, indices in self.text_groups.items()
            if len(indices) >= 2
        ]

    def sample_batch(self) -> list[int]:
        """Sample batch_groups text groups, return all their items."""
        if len(self.valid_groups) < self.batch_groups:
            groups = self.valid_groups
        else:
            perm = torch.randperm(len(self.valid_groups))[:self.batch_groups]
            groups = [self.valid_groups[i] for i in perm.tolist()]
        indices = []
        for _, group_indices in groups:
            indices.extend(group_indices)
        return indices


# ─── Training ────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sr = args.sr
    total_stride = 1
    for s in args.strides:
        total_stride *= s
    max_samples = args.segment_samples

    # ── Load AE (unfrozen) ───────────────────────────────────
    print("Loading AE...")
    ae_ckpt = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
    ae = WavVAE(
        latent_dim=args.latent_dim,
        encoder_channels=args.encoder_channels,
        strides=args.strides,
        dilations=args.dilations,
        sample_latent=False,
    )
    ae.load_state_dict(ae_ckpt.get("model", ae_ckpt), strict=True)
    ae = ae.to(device).train()
    print(f"  AE params: {sum(p.numel() for p in ae.parameters()):,}")

    # ── Load structural modules ──────────────────────────────
    proj_dim = args.proj_dim
    cond_dim = args.cond_emb_dim * 2

    semantic_proj = SemanticProjection(
        args.latent_dim, proj_dim,
        hidden_dim=args.proj_hidden_dim,
        depth=args.proj_depth,
    ).to(device)

    flow_net = FlowMatchingNet(
        proj_dim=proj_dim,
        cond_dim=cond_dim,
        hidden_dim=args.flow_hidden_dim,
        depth=args.flow_depth,
    ).to(device)

    # Load data first to determine vocab sizes
    data_path = Path(args.data_path)
    roots = [Path.cwd(), data_path.parent]
    if "outputs" in data_path.parts:
        idx = data_path.parts.index("outputs")
        if idx > 0:
            roots.append(Path(*data_path.parts[:idx]))

    train_items = read_manifest(data_path)
    eval_items = read_manifest(Path(args.eval_data_path)) if args.eval_data_path else train_items[:50]

    all_items = train_items + eval_items
    text_groups = sorted(set(text_key(item) for item in all_items))
    speakers = sorted(set(speaker_key(item) for item in all_items))
    text_to_id = {t: i for i, t in enumerate(text_groups)}
    speaker_to_id = {s: i for i, s in enumerate(speakers)}

    cond_enc = ConditionEncoder(
        n_text_groups=len(text_groups) + 1,
        n_speakers=len(speakers) + 1,
        emb_dim=args.cond_emb_dim,
    ).to(device)

    print(f"  Train items: {len(train_items)}, Eval items: {len(eval_items)}")
    print(f"  Text groups: {len(text_groups)}, Speakers: {len(speakers)}")

    # Optionally load probe checkpoint to initialize projection + flow
    if args.probe_checkpoint and Path(args.probe_checkpoint).exists():
        print(f"  Loading probe checkpoint: {args.probe_checkpoint}")
        probe_ckpt = torch.load(args.probe_checkpoint, map_location="cpu", weights_only=False)
        semantic_proj.load_state_dict(probe_ckpt["semantic_proj"], strict=False)
        flow_net.load_state_dict(probe_ckpt["flow_net"], strict=False)
        # Don't load cond_enc — vocab size may differ
        print("  Loaded semantic_proj and flow_net from probe")

    # ── Loss functions ───────────────────────────────────────
    mrstft_loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_sizes=[512, 1024, 2048],
        sr=sr,
    ).to(device)

    mel_loss_fn = MultiScaleMelLoss(
        sr=sr,
        fft_sizes=(512, 1024, 2048),
        n_mels_list=(64, 128, 256),
    ).to(device)

    # ── Optimizers ───────────────────────────────────────────
    # Separate lr for AE encoder vs structural modules
    ae_encoder_params = list(ae.encoder.parameters()) + list(ae.bottleneck.parameters())
    ae_decoder_params = list(ae.decoder.parameters())
    struct_params = (
        list(semantic_proj.parameters())
        + list(flow_net.parameters())
        + list(cond_enc.parameters())
    )

    param_groups = [
        {"params": ae_encoder_params, "lr": args.encoder_lr},
        {"params": ae_decoder_params, "lr": args.decoder_lr},
        {"params": struct_params, "lr": args.struct_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    total_params = sum(p.numel() for pg in param_groups for p in pg["params"])
    print(f"  Total trainable: {total_params:,}")
    print(f"    Encoder+Bottleneck: {sum(p.numel() for p in ae_encoder_params):,} (lr={args.encoder_lr})")
    print(f"    Decoder: {sum(p.numel() for p in ae_decoder_params):,} (lr={args.decoder_lr})")
    print(f"    Structural: {sum(p.numel() for p in struct_params):,} (lr={args.struct_lr})")

    # Cosine schedule
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.total_steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Batch sampler ────────────────────────────────────────
    batcher = CrossedGroupBatcher(train_items, batch_groups=args.batch_groups)

    # ── Speaker centroids (updated periodically) ─────────────
    n_speakers = len(speakers)
    speaker_centroids = torch.zeros(n_speakers + 1, proj_dim, device=device)

    # ── Training loop ────────────────────────────────────────
    print(f"\nStarting Phase 1 joint training:")
    print(f"  total_steps={args.total_steps}, warmup={args.warmup_steps}")
    print(f"  lambda_flow={args.lambda_flow}, lambda_analogy={args.lambda_analogy}")
    print(f"  encoder_freeze_steps={args.encoder_freeze_steps}")
    print()

    step = 0
    running_recon = 0.0
    running_flow = 0.0
    running_analogy = 0.0
    t_start = time.time()

    # Freeze encoder for initial warmup
    encoder_frozen = True
    for p in ae_encoder_params:
        p.requires_grad_(False)

    while step < args.total_steps:
        batch_indices = batcher.sample_batch()
        if not batch_indices:
            continue

        # Unfreeze encoder after warmup
        if encoder_frozen and step >= args.encoder_freeze_steps:
            encoder_frozen = False
            for p in ae_encoder_params:
                p.requires_grad_(True)
            print(f"  [step {step}] Encoder unfrozen")

        # Load batch audio
        audios = []
        batch_text_ids = []
        batch_speaker_ids = []
        for idx in batch_indices:
            item = train_items[idx]
            try:
                path = resolve_audio_path(item, roots)
                audio = load_audio(path, sr, max_samples, total_stride)
                audios.append(audio)
                batch_text_ids.append(text_to_id.get(text_key(item), 0))
                batch_speaker_ids.append(speaker_to_id.get(speaker_key(item), 0))
            except (FileNotFoundError, RuntimeError):
                continue

        if len(audios) < 4:
            continue

        # Pad to same length
        max_len = max(a.numel() for a in audios)
        max_len = (max_len // total_stride) * total_stride
        audio_batch = torch.zeros(len(audios), 1, max_len, device=device)
        for i, a in enumerate(audios):
            audio_batch[i, 0, :a.numel()] = a.to(device)

        B = audio_batch.shape[0]

        # ── Forward: AE reconstruction ───────────────────────
        ae.set_step(step)
        out = ae(audio_batch)
        x_hat = out["x_hat"]
        z = out["z"]  # [B, latent_dim, T]
        reg_loss = out["reg_loss"]

        recon_losses = compute_reconstruction_loss(
            audio_batch, x_hat, mrstft_loss_fn, mel_loss_fn, reg_loss,
            l1_weight=args.l1_weight, l2_weight=0.0,
            stft_weight=args.stft_weight, mel_weight=args.mel_weight,
        )
        recon_loss = recon_losses["total"]

        # ── Forward: Semantic projection ─────────────────────
        z_proj = semantic_proj(z)  # [B, proj_dim], normalized

        # ── Flow matching loss (speaker-centered) ────────────
        text_ids_t = torch.tensor(batch_text_ids, device=device)
        speaker_ids_t = torch.tensor(batch_speaker_ids, device=device)
        cond = cond_enc(text_ids_t, speaker_ids_t)

        # Speaker centering
        spk_cents = speaker_centroids[speaker_ids_t]
        z_residual = F.normalize(z_proj - spk_cents, dim=-1)

        t_flow = torch.rand(B, 1, device=device)
        noise = torch.randn_like(z_residual)
        x_t = (1 - t_flow) * noise + t_flow * z_residual
        target_v = z_residual - noise
        pred_v = flow_net(x_t, t_flow, cond)
        flow_loss = F.mse_loss(pred_v, target_v)

        # ── Analogy loss ─────────────────────────────────────
        analogy_loss, analogy_logs = compute_analogy_loss(
            z_proj, batch_text_ids, batch_speaker_ids,
            min_cos=args.analogy_min_cos,
        )

        # ── Total loss ───────────────────────────────────────
        # Scale structural losses based on training phase
        flow_weight = args.lambda_flow
        analogy_weight = args.lambda_analogy
        if step < args.encoder_freeze_steps:
            # During warmup, full structural weight on projection only
            flow_weight = args.lambda_flow
            analogy_weight = args.lambda_analogy

        total_loss = recon_loss + flow_weight * flow_loss + analogy_weight * analogy_loss

        # ── Backward ─────────────────────────────────────────
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for pg in param_groups for p in pg["params"] if p.requires_grad],
            args.grad_clip,
        )
        optimizer.step()
        scheduler.step()

        running_recon += recon_loss.item()
        running_flow += flow_loss.item()
        running_analogy += analogy_loss.item()
        step += 1

        # Update speaker centroids periodically
        if step % 100 == 0:
            with torch.no_grad():
                speaker_centroids.zero_()
                counts = torch.zeros(n_speakers + 1, device=device)
                for i in range(B):
                    sid = batch_speaker_ids[i]
                    speaker_centroids[sid] += z_proj[i]
                    counts[sid] += 1
                # EMA update with global centroids
                for sid in range(n_speakers + 1):
                    if counts[sid] > 0:
                        new_cent = speaker_centroids[sid] / counts[sid]
                        speaker_centroids[sid] = F.normalize(new_cent, dim=-1)

        # ── Logging ──────────────────────────────────────────
        if step % args.log_interval == 0:
            avg_recon = running_recon / args.log_interval
            avg_flow = running_flow / args.log_interval
            avg_analogy = running_analogy / args.log_interval
            elapsed = time.time() - t_start
            lr_enc = optimizer.param_groups[0]["lr"]
            lr_struct = optimizer.param_groups[2]["lr"]
            frozen_tag = " [enc frozen]" if encoder_frozen else ""
            print(
                f"  step {step}/{args.total_steps} | "
                f"recon={avg_recon:.4f} flow={avg_flow:.4f} analogy={avg_analogy:.4f} | "
                f"spk_cos={analogy_logs['spk_dir_cos']:.3f} txt_cos={analogy_logs['txt_dir_cos']:.3f} | "
                f"lr_enc={lr_enc:.2e} lr_s={lr_struct:.2e} | "
                f"{elapsed:.1f}s{frozen_tag}",
                flush=True,
            )
            running_recon = 0.0
            running_flow = 0.0
            running_analogy = 0.0

        # ── Evaluation ───────────────────────────────────────
        if step % args.eval_interval == 0 or step == args.total_steps:
            ae.eval()
            semantic_proj.eval()
            snrs = []
            eval_flow_losses = []
            eval_z_projs = []
            eval_text_ids = []
            eval_speaker_ids = []

            with torch.no_grad():
                for item in eval_items[:50]:
                    try:
                        path = resolve_audio_path(item, roots)
                        audio = load_audio(path, sr, max_samples, total_stride)
                    except (FileNotFoundError, RuntimeError):
                        continue
                    audio_t = audio.view(1, 1, -1).to(device)
                    out = ae(audio_t)
                    snrs.append(compute_snr(out["x_hat"], audio_t))

                    zp = semantic_proj(out["z"])
                    eval_z_projs.append(zp.cpu())
                    tid = text_to_id.get(text_key(item), 0)
                    sid = speaker_to_id.get(speaker_key(item), 0)
                    eval_text_ids.append(tid)
                    eval_speaker_ids.append(sid)

            # Cross-speaker retrieval on z_proj residual
            if eval_z_projs:
                zp_all = torch.cat(eval_z_projs, dim=0).numpy()
                zp_norm = zp_all / (np.linalg.norm(zp_all, axis=1, keepdims=True) + 1e-30)
                sim = zp_norm @ zp_norm.T
                top1 = 0
                count = 0
                for i in range(len(eval_text_ids)):
                    cands = [j for j in range(len(eval_text_ids))
                             if j != i and eval_speaker_ids[j] != eval_speaker_ids[i]]
                    pos = [j for j in cands if eval_text_ids[j] == eval_text_ids[i]]
                    if not cands or not pos:
                        continue
                    best = max(cands, key=lambda j: sim[i, j])
                    if best in pos:
                        top1 += 1
                    count += 1
                retrieval = top1 / max(count, 1)
            else:
                retrieval = 0.0

            mean_snr = np.mean(snrs) if snrs else 0.0
            print(
                f"\n  [Eval step {step}] SNR={mean_snr:.2f}dB | "
                f"retrieval_top1={retrieval:.4f} | n={len(snrs)}\n",
                flush=True,
            )

            # Save checkpoint
            save_dict = {
                "step": step,
                "model": ae.state_dict(),
                "semantic_proj": semantic_proj.state_dict(),
                "flow_net": flow_net.state_dict(),
                "cond_enc": cond_enc.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": {"snr": mean_snr, "retrieval_top1": retrieval},
                "args": vars(args),
                "text_to_id": text_to_id,
                "speaker_to_id": speaker_to_id,
            }
            torch.save(save_dict, output_dir / "latest.pt")

            # Save metrics
            with (output_dir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, "snr": mean_snr, "retrieval": retrieval}) + "\n")

            ae.train()
            semantic_proj.train()

    print(f"\nTraining complete. Final SNR={mean_snr:.2f}dB, retrieval={retrieval:.4f}")


# ─── CLI ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="V17.4 Phase 1: Joint AE + Flow + Analogy")
    parser.add_argument("--ae-checkpoint", type=str, required=True)
    parser.add_argument("--probe-checkpoint", type=str, default="")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--eval-data-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="outputs/models/v17_4_joint_phase1")

    # AE architecture
    parser.add_argument("--latent-dim", type=int, default=384)
    parser.add_argument("--strides", type=int, nargs="+", default=[2, 4, 4, 8])
    parser.add_argument("--encoder-channels", type=int, nargs="+", default=[128, 256, 512, 1024, 1024])
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 3, 9])

    # Structural modules
    parser.add_argument("--proj-dim", type=int, default=64)
    parser.add_argument("--proj-hidden-dim", type=int, default=256)
    parser.add_argument("--proj-depth", type=int, default=3)
    parser.add_argument("--flow-hidden-dim", type=int, default=256)
    parser.add_argument("--flow-depth", type=int, default=4)
    parser.add_argument("--cond-emb-dim", type=int, default=128)

    # Loss weights
    parser.add_argument("--lambda-flow", type=float, default=0.05)
    parser.add_argument("--lambda-analogy", type=float, default=0.1)
    parser.add_argument("--analogy-min-cos", type=float, default=0.3)
    parser.add_argument("--l1-weight", type=float, default=10.0)
    parser.add_argument("--stft-weight", type=float, default=0.5)
    parser.add_argument("--mel-weight", type=float, default=0.1)

    # Training
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--encoder-freeze-steps", type=int, default=1000,
                        help="Steps to keep encoder frozen (warmup structural modules)")
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--decoder-lr", type=float, default=3e-5)
    parser.add_argument("--struct-lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--batch-groups", type=int, default=8,
                        help="Number of text groups per batch (each with multiple speakers)")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--segment-samples", type=int, default=220500)
    parser.add_argument("--sr", type=int, default=22050)

    # Logging
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=500)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
