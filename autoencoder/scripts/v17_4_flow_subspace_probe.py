"""V17.4 Phase 0: Frozen-AE flow subspace probe.

Freezes the V16.3 WavVAE encoder. Trains only:
  1. A learned linear projection W_proj: z_pooled (384) -> z_proj (proj_dim)
  2. A small conditional flow matching network: condition -> z_proj

This tests whether V16.3 latent contains a low-dimensional linear subspace
that is condition-predictable (text_group + speaker -> z_proj via flow matching).

If flow loss decreases, it proves the latent already has condition-predictable
structure in some linear subspace. Phase 1 can then unfreeze the encoder to
strengthen that subspace.

Usage:
  python -m autoencoder.scripts.v17_4_flow_subspace_probe \
    --ae-checkpoint outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt \
    --data-path CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_train_phalign_neighbors.jsonl \
    --eval-data-path CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_val_phalign_neighbors.jsonl \
    --output-dir outputs/models/v17_4_flow_subspace_probe
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from autoencoder.models.autoencoder import WavVAE


# ─── Flow Matching Network ───────────────────────────────────


class FlowMatchingNet(nn.Module):
    """Small MLP that predicts velocity v(x_t, t, cond) for flow matching.

    Input: x_t (proj_dim) + t (1) + condition (cond_dim)
    Output: velocity (proj_dim)
    """

    def __init__(self, proj_dim: int, cond_dim: int, hidden_dim: int = 256, depth: int = 4):
        super().__init__()
        input_dim = proj_dim + 1 + cond_dim
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, proj_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: [B, proj_dim] noisy projected latent
            t: [B, 1] time step
            cond: [B, cond_dim] condition embedding
        Returns:
            velocity: [B, proj_dim]
        """
        inp = torch.cat([x_t, t, cond], dim=-1)
        return self.net(inp)


# ─── Condition Encoder ───────────────────────────────────────


class ConditionEncoder(nn.Module):
    """Encodes text_group + speaker_id into a condition vector.

    Uses learned embeddings for both. Simple but sufficient for Phase 0.
    """

    def __init__(self, n_text_groups: int = 1024, n_speakers: int = 16, emb_dim: int = 64):
        super().__init__()
        self.text_emb = nn.Embedding(n_text_groups, emb_dim)
        self.speaker_emb = nn.Embedding(n_speakers, emb_dim)
        self.out_dim = emb_dim * 2

    def forward(self, text_ids: torch.Tensor, speaker_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_ids: [B] integer text group indices
            speaker_ids: [B] integer speaker indices
        Returns:
            cond: [B, out_dim]
        """
        return torch.cat([self.text_emb(text_ids), self.speaker_emb(speaker_ids)], dim=-1)


# ─── Semantic Projection ────────────────────────────────────


class SemanticProjection(nn.Module):
    """Learned projection from full latent sequence to semantic subspace.

    Uses a small Conv + attention pooling to selectively attend to
    content-relevant frames before projecting to the semantic subspace.
    This avoids the mean-pooling problem where speaker statistics dominate.
    """

    def __init__(self, latent_dim: int = 384, proj_dim: int = 64, hidden_dim: int = 256, depth: int = 2):
        super().__init__()
        # Sequence processing: lightweight conv blocks
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

        # Attention pooling: learn which frames matter for content
        self.attn = nn.Sequential(
            nn.Conv1d(hidden_dim, 1, 1),
        )

        # Final projection to semantic subspace
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_dim, bias=False),
        )

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_seq: [B, latent_dim, T] full latent sequence
        Returns:
            z_proj: [B, proj_dim] L2-normalized
        """
        h = self.conv(z_seq)  # [B, hidden_dim, T]
        attn_weights = torch.softmax(self.attn(h), dim=-1)  # [B, 1, T]
        pooled = (h * attn_weights).sum(dim=-1)  # [B, hidden_dim]
        raw = self.proj(pooled)  # [B, proj_dim]
        return F.normalize(raw, dim=-1)  # unit sphere, prevents collapse


# ─── Data Loading ────────────────────────────────────────────


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
    """Load and prepare audio: resample, crop/pad, stride-align."""
    data, file_sr = sf.read(path, always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    audio = torch.from_numpy(data.astype(np.float32))
    if file_sr != sr:
        audio = torchaudio.functional.resample(audio, file_sr, sr)
    # Crop to max_samples if needed
    if max_samples > 0 and audio.numel() > max_samples:
        start = (audio.numel() - max_samples) // 2
        start = (start // total_stride) * total_stride
        audio = audio[start: start + max_samples]
    # Stride-align
    aligned_len = (audio.numel() // total_stride) * total_stride
    if aligned_len < total_stride:
        audio = F.pad(audio, (0, total_stride - audio.numel()))
    else:
        audio = audio[:aligned_len]
    return audio


def text_key(item: dict) -> str:
    return str(
        item.get("same_text_group")
        or item.get("text_id")
        or item.get("text")
        or item.get("id")
        or ""
    )


def speaker_key(item: dict) -> str:
    return str(
        item.get("same_speaker_group")
        or item.get("speaker_id")
        or item.get("speaker")
        or ""
    )


# ─── Evaluation Metrics ─────────────────────────────────────


@torch.no_grad()
def evaluate_probe(
    ae: WavVAE,
    semantic_proj: SemanticProjection,
    flow_net: FlowMatchingNet,
    cond_enc: ConditionEncoder,
    items: list[dict],
    text_to_id: dict[str, int],
    speaker_to_id: dict[str, int],
    roots: list[Path],
    device: torch.device,
    sr: int,
    max_samples: int,
    total_stride: int,
    speaker_centroids: torch.Tensor | None = None,
) -> dict[str, float]:
    """Evaluate flow matching quality and z_proj structure."""
    ae.eval()
    semantic_proj.eval()
    flow_net.eval()
    cond_enc.eval()

    flow_losses = []
    z_projs = []
    z_residuals = []
    text_ids_list = []
    speaker_ids_list = []

    for item in items:
        try:
            path = resolve_audio_path(item, roots)
            audio = load_audio(path, sr, max_samples, total_stride)
        except (FileNotFoundError, RuntimeError):
            continue

        audio_t = audio.view(1, 1, -1).to(device)
        with torch.no_grad():
            out = ae(audio_t)
            z = out["z"]  # [1, latent_dim, T_frames]

        z_proj = semantic_proj(z)  # [1, proj_dim]

        tk = text_key(item)
        sk = speaker_key(item)
        tid = text_to_id.get(tk, 0)
        sid = speaker_to_id.get(sk, 0)

        text_ids_t = torch.tensor([tid], device=device)
        speaker_ids_t = torch.tensor([sid], device=device)
        cond = cond_enc(text_ids_t, speaker_ids_t)

        # Speaker centering for flow loss
        if speaker_centroids is not None:
            spk_cent = speaker_centroids[sid:sid+1]
            z_res = F.normalize(z_proj - spk_cent, dim=-1)
        else:
            z_res = z_proj

        # Flow matching loss at random t
        t = torch.rand(1, 1, device=device)
        noise = torch.randn_like(z_res)
        x_t = (1 - t) * noise + t * z_res
        target_v = z_res - noise
        pred_v = flow_net(x_t, t, cond)
        loss = F.mse_loss(pred_v, target_v)
        flow_losses.append(loss.item())

        z_projs.append(z_proj.cpu())
        z_residuals.append(z_res.cpu())
        text_ids_list.append(tid)
        speaker_ids_list.append(sid)

    if not flow_losses:
        return {"flow_loss": 999.0}

    # Compute cross-speaker text retrieval on RESIDUAL (not raw z_proj)
    z_res_np = torch.cat(z_residuals, dim=0).numpy()
    z_res_norm = z_res_np / (np.linalg.norm(z_res_np, axis=1, keepdims=True) + 1e-30)
    sim_matrix = z_res_norm @ z_res_norm.T

    # Retrieval: for each item, find same-text different-speaker
    top1_hits = 0
    retrieval_count = 0
    for i in range(len(text_ids_list)):
        candidates = [
            j for j in range(len(text_ids_list))
            if j != i and speaker_ids_list[j] != speaker_ids_list[i]
        ]
        positives = [j for j in candidates if text_ids_list[j] == text_ids_list[i]]
        if not candidates or not positives:
            continue
        sims = [(sim_matrix[i, j], j) for j in candidates]
        sims.sort(key=lambda x: -x[0])
        if sims[0][1] in positives:
            top1_hits += 1
        retrieval_count += 1

    # Variance
    z_projs_t = torch.cat(z_projs, dim=0).numpy()
    proj_var = np.var(z_projs_t, axis=0).sum()
    res_var = np.var(z_res_np, axis=0).sum()

    metrics = {
        "flow_loss": float(np.mean(flow_losses)),
        "retrieval_top1": top1_hits / max(retrieval_count, 1),
        "retrieval_count": retrieval_count,
        "z_proj_var": float(proj_var),
        "z_residual_var": float(res_var),
        "n_eval": len(flow_losses),
    }
    return metrics


# ─── Training ────────────────────────────────────────────────


def train_probe(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sr = args.sr
    total_stride = 1
    for s in args.strides:
        total_stride *= s
    max_samples = args.segment_samples

    # ── Load frozen AE ───────────────────────────────────────
    print("Loading frozen AE...")
    ckpt = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
    ae = WavVAE(
        latent_dim=args.latent_dim,
        encoder_channels=args.encoder_channels,
        strides=args.strides,
        dilations=args.dilations,
        sample_latent=False,
    )
    state = ckpt.get("model", ckpt)
    ae.load_state_dict(state, strict=True)
    ae = ae.to(device).eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    print(f"  AE loaded, total_stride={total_stride}")

    # ── Load data ────────────────────────────────────────────
    data_path = Path(args.data_path)
    roots = [Path.cwd(), data_path.parent]
    if "outputs" in data_path.parts:
        idx = data_path.parts.index("outputs")
        if idx > 0:
            roots.append(Path(*data_path.parts[:idx]))

    train_items = read_manifest(data_path)
    eval_items = read_manifest(Path(args.eval_data_path)) if args.eval_data_path else train_items[:50]
    print(f"  Train items: {len(train_items)}, Eval items: {len(eval_items)}")

    # Build ID mappings
    all_items = train_items + eval_items
    text_groups = sorted(set(text_key(item) for item in all_items))
    speakers = sorted(set(speaker_key(item) for item in all_items))
    text_to_id = {t: i for i, t in enumerate(text_groups)}
    speaker_to_id = {s: i for i, s in enumerate(speakers)}
    print(f"  Text groups: {len(text_groups)}, Speakers: {len(speakers)}")

    # ── Build trainable modules ──────────────────────────────
    proj_dim = args.proj_dim
    cond_dim = args.cond_emb_dim * 2  # text_emb + speaker_emb

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
    cond_enc = ConditionEncoder(
        n_text_groups=len(text_groups) + 1,
        n_speakers=len(speakers) + 1,
        emb_dim=args.cond_emb_dim,
    ).to(device)

    trainable_params = (
        list(semantic_proj.parameters())
        + list(flow_net.parameters())
        + list(cond_enc.parameters())
    )
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"  Trainable params: {total_trainable:,}")
    print(f"    SemanticProjection: {sum(p.numel() for p in semantic_proj.parameters()):,}")
    print(f"    FlowMatchingNet: {sum(p.numel() for p in flow_net.parameters()):,}")
    print(f"    ConditionEncoder: {sum(p.numel() for p in cond_enc.parameters()):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    # Cosine schedule
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.total_steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ────────────────────────────────────────
    print(f"\nStarting training: {args.total_steps} steps, proj_dim={proj_dim}")
    print(f"  flow_hidden={args.flow_hidden_dim}, flow_depth={args.flow_depth}")
    print(f"  lr={args.lr}, warmup={args.warmup_steps}")
    print()

    semantic_proj.train()
    flow_net.train()
    cond_enc.train()

    step = 0
    epoch = 0
    running_loss = 0.0
    t_start = time.time()

    # Pre-compute all latents for batch training (frozen AE, so this is safe)
    print("  Pre-computing latent representations...")
    all_z_seqs = []  # list of [1, latent_dim, T] tensors
    all_text_ids = []
    all_speaker_ids = []
    valid_indices = []

    for idx, item in enumerate(train_items):
        try:
            path = resolve_audio_path(item, roots)
            audio = load_audio(path, sr, max_samples, total_stride)
        except (FileNotFoundError, RuntimeError):
            continue

        audio_t = audio.view(1, 1, -1).to(device)
        with torch.no_grad():
            out = ae(audio_t)
            z = out["z"]  # [1, latent_dim, T_frames]

        all_z_seqs.append(z.cpu())
        tk = text_key(item)
        sk = speaker_key(item)
        all_text_ids.append(text_to_id.get(tk, 0))
        all_speaker_ids.append(speaker_to_id.get(sk, 0))
        valid_indices.append(idx)

    n_valid = len(all_z_seqs)
    print(f"  Pre-computed {n_valid} latents")

    # Pre-compute speaker means for centering
    # First pass: compute z_proj for all items to get speaker centroids
    print("  Computing speaker centroids...")
    semantic_proj.eval()
    all_z_proj_init = []
    with torch.no_grad():
        for i in range(n_valid):
            z_i = all_z_seqs[i].to(device)
            zp = semantic_proj(z_i)  # [1, proj_dim]
            all_z_proj_init.append(zp.cpu())
    all_z_proj_init = torch.cat(all_z_proj_init, dim=0)  # [N, proj_dim]

    # Compute per-speaker centroids
    n_speakers = len(speakers)
    speaker_centroids = torch.zeros(n_speakers + 1, proj_dim, device=device)
    speaker_counts = torch.zeros(n_speakers + 1, device=device)
    for i in range(n_valid):
        sid = all_speaker_ids[i]
        speaker_centroids[sid] += all_z_proj_init[i].to(device)
        speaker_counts[sid] += 1
    for sid in range(n_speakers + 1):
        if speaker_counts[sid] > 0:
            speaker_centroids[sid] /= speaker_counts[sid]
    # Normalize centroids (since z_proj is on unit sphere)
    speaker_centroids = F.normalize(speaker_centroids, dim=-1)
    print(f"  Speaker centroids computed ({n_speakers} speakers)")

    semantic_proj.train()

    # Batch training
    batch_size = min(args.batch_size, n_valid)
    print(f"  Batch size: {batch_size}")
    print()

    while step < args.total_steps:
        perm = torch.randperm(n_valid).tolist()
        epoch += 1

        for batch_start in range(0, n_valid - batch_size + 1, batch_size):
            if step >= args.total_steps:
                break

            batch_indices = perm[batch_start: batch_start + batch_size]

            # Gather batch: pad z sequences to same length
            z_batch = []
            max_t = max(all_z_seqs[i].shape[-1] for i in batch_indices)
            for i in batch_indices:
                z_i = all_z_seqs[i].to(device)
                if z_i.shape[-1] < max_t:
                    z_i = F.pad(z_i, (0, max_t - z_i.shape[-1]))
                z_batch.append(z_i)
            z_batch = torch.cat(z_batch, dim=0)  # [B, latent_dim, T]

            text_ids_t = torch.tensor([all_text_ids[i] for i in batch_indices], device=device)
            speaker_ids_t = torch.tensor([all_speaker_ids[i] for i in batch_indices], device=device)

            # Forward
            z_proj = semantic_proj(z_batch)  # [B, proj_dim], normalized
            cond = cond_enc(text_ids_t, speaker_ids_t)  # [B, cond_dim]

            # Speaker centering: subtract speaker centroid, re-normalize
            # This forces flow net to predict text-dependent residual, not speaker mean
            spk_cents = speaker_centroids[speaker_ids_t]  # [B, proj_dim]
            z_residual = z_proj - spk_cents  # [B, proj_dim]
            # Re-normalize residual to prevent scale collapse
            z_residual = F.normalize(z_residual, dim=-1)

            # Flow matching on speaker-residual
            t = torch.rand(batch_size, 1, device=device)
            noise = torch.randn_like(z_residual)
            x_t = (1 - t) * noise + t * z_residual
            target_v = z_residual - noise
            pred_v = flow_net(x_t, t, cond)

            loss = F.mse_loss(pred_v, target_v)

            # Diversity encouragement on residual (not speaker-dominated z_proj)
            if batch_size > 1:
                cos_matrix = z_residual @ z_residual.T  # [B, B]
                mask = ~torch.eye(batch_size, device=device, dtype=torch.bool)
                mean_off_diag_cos = cos_matrix[mask].mean()
                diversity_loss = F.relu(mean_off_diag_cos - 0.5)
                total_loss = loss + 0.5 * diversity_loss
            else:
                diversity_loss = torch.tensor(0.0, device=device)
                mean_off_diag_cos = torch.tensor(0.0, device=device)
                total_loss = loss

            proj_var = z_residual.var(dim=0).mean()

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            step += 1

            # Update speaker centroids periodically (projection is changing)
            if step % 200 == 0:
                semantic_proj.eval()
                speaker_centroids.zero_()
                speaker_counts.zero_()
                with torch.no_grad():
                    for i in range(n_valid):
                        z_i = all_z_seqs[i].to(device)
                        zp = semantic_proj(z_i)  # [1, proj_dim]
                        sid = all_speaker_ids[i]
                        speaker_centroids[sid] += zp.squeeze(0)
                        speaker_counts[sid] += 1
                for sid in range(n_speakers + 1):
                    if speaker_counts[sid] > 0:
                        speaker_centroids[sid] /= speaker_counts[sid]
                speaker_centroids.copy_(F.normalize(speaker_centroids, dim=-1))
                semantic_proj.train()

            # Logging
            if step % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                elapsed = time.time() - t_start
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  step {step}/{args.total_steps} | "
                    f"flow_loss={avg_loss:.6f} | "
                    f"proj_var={proj_var.item():.4f} | "
                    f"div_loss={diversity_loss.item():.4f} | "
                    f"mean_cos={mean_off_diag_cos.item():.4f} | "
                    f"lr={lr:.2e} | "
                    f"{elapsed:.1f}s",
                    flush=True,
                )
                running_loss = 0.0

            # Evaluation
            if step % args.eval_interval == 0 or step == args.total_steps:
                print(f"\n  [Eval at step {step}]")
                metrics = evaluate_probe(
                    ae, semantic_proj, flow_net, cond_enc,
                    eval_items, text_to_id, speaker_to_id,
                    roots, device, sr, max_samples, total_stride,
                    speaker_centroids=speaker_centroids,
                )
                print(
                    f"    flow_loss={metrics['flow_loss']:.6f} | "
                    f"retrieval_top1={metrics['retrieval_top1']:.4f} | "
                    f"z_proj_var={metrics['z_proj_var']:.4f} | "
                    f"res_var={metrics['z_residual_var']:.4f} | "
                    f"n_eval={metrics['n_eval']}",
                    flush=True,
                )
                print()

                # Save checkpoint
                save_dict = {
                    "step": step,
                    "semantic_proj": semantic_proj.state_dict(),
                    "flow_net": flow_net.state_dict(),
                    "cond_enc": cond_enc.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": metrics,
                    "args": vars(args),
                    "text_to_id": text_to_id,
                    "speaker_to_id": speaker_to_id,
                }
                torch.save(save_dict, output_dir / "latest.pt")

                # Save metrics log
                metrics_log = output_dir / "metrics.jsonl"
                with metrics_log.open("a") as f:
                    entry = {"step": step, **metrics}
                    f.write(json.dumps(entry) + "\n")

    # ── Final evaluation ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Final evaluation")
    print("=" * 60)
    final_metrics = evaluate_probe(
        ae, semantic_proj, flow_net, cond_enc,
        eval_items, text_to_id, speaker_to_id,
        roots, device, sr, max_samples, total_stride,
        speaker_centroids=speaker_centroids,
    )
    print(f"  flow_loss={final_metrics['flow_loss']:.6f}")
    print(f"  retrieval_top1={final_metrics['retrieval_top1']:.4f}")
    print(f"  z_proj_var={final_metrics['z_proj_var']:.4f}")
    print(f"  z_residual_var={final_metrics['z_residual_var']:.4f}")
    print(f"  n_eval={final_metrics['n_eval']}")

    # Also evaluate on train set for comparison
    print("\n  [Train set eval]")
    train_metrics = evaluate_probe(
        ae, semantic_proj, flow_net, cond_enc,
        train_items[:200], text_to_id, speaker_to_id,
        roots, device, sr, max_samples, total_stride,
        speaker_centroids=speaker_centroids,
    )
    print(f"  flow_loss={train_metrics['flow_loss']:.6f}")
    print(f"  retrieval_top1={train_metrics['retrieval_top1']:.4f}")
    print(f"  z_proj_var={train_metrics['z_proj_var']:.4f}")

    # Save final
    save_dict = {
        "step": step,
        "semantic_proj": semantic_proj.state_dict(),
        "flow_net": flow_net.state_dict(),
        "cond_enc": cond_enc.state_dict(),
        "final_metrics": final_metrics,
        "train_metrics": train_metrics,
        "args": vars(args),
        "text_to_id": text_to_id,
        "speaker_to_id": speaker_to_id,
    }
    torch.save(save_dict, output_dir / "best.pt")
    print(f"\nSaved to {output_dir / 'best.pt'}")


# ─── CLI ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="V17.4 Phase 0: Flow subspace probe")
    parser.add_argument("--ae-checkpoint", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--eval-data-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="outputs/models/v17_4_flow_subspace_probe")

    # AE architecture (must match checkpoint)
    parser.add_argument("--latent-dim", type=int, default=384)
    parser.add_argument("--strides", type=int, nargs="+", default=[2, 4, 4, 8])
    parser.add_argument("--encoder-channels", type=int, nargs="+", default=[128, 256, 512, 1024, 1024])
    parser.add_argument("--dilations", type=int, nargs="+", default=[1, 3, 9])

    # Probe architecture
    parser.add_argument("--proj-dim", type=int, default=64,
                        help="Dimension of the learned semantic subspace")
    parser.add_argument("--proj-hidden-dim", type=int, default=256,
                        help="Hidden dim of the sequence projection conv")
    parser.add_argument("--proj-depth", type=int, default=2,
                        help="Number of conv layers in sequence projection")
    parser.add_argument("--flow-hidden-dim", type=int, default=256)
    parser.add_argument("--flow-depth", type=int, default=4)
    parser.add_argument("--cond-emb-dim", type=int, default=64)

    # Training
    parser.add_argument("--total-steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for flow matching (uses pre-computed latents)")
    parser.add_argument("--segment-samples", type=int, default=220500,
                        help="Max audio samples per item (10s at 22050Hz)")
    parser.add_argument("--sr", type=int, default=22050)

    # Logging
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=500)

    args = parser.parse_args()
    train_probe(args)


if __name__ == "__main__":
    main()
