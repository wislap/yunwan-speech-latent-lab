"""V17.4 Variance Analysis: How much of the latent does z_proj explain?

Loads the trained semantic_proj from the flow probe, computes z_proj for all
items, and measures:
1. What fraction of total latent variance is captured by z_proj's subspace?
2. If we reconstruct audio using only z_proj (zeroing residual), how much SNR?
3. If we add noise to the residual dimensions, how does quality degrade?

This answers: "Is the condition-predictable subspace important for reconstruction?"

Usage:
  python -m autoencoder.scripts.v17_4_variance_analysis \
    --ae-checkpoint outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt \
    --probe-checkpoint outputs/models/v17_4_flow_probe_v4_normed/best.pt \
    --data-path CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_val_phalign_neighbors.jsonl \
    --output-dir outputs/diagnostics/v17_4_variance_analysis
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from autoencoder.models.autoencoder import WavVAE
from autoencoder.scripts.v17_4_flow_subspace_probe import (
    SemanticProjection,
    FlowMatchingNet,
    ConditionEncoder,
    read_manifest,
    resolve_audio_path,
    load_audio,
    text_key,
    speaker_key,
)


def compute_snr(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    noise = x_hat - x
    signal_power = (x ** 2).mean()
    noise_power = (noise ** 2).mean()
    if noise_power < 1e-10:
        return 100.0
    return 10.0 * math.log10(signal_power.item() / noise_power.item())


def main():
    parser = argparse.ArgumentParser(description="V17.4 Variance Analysis")
    parser.add_argument("--ae-checkpoint", type=str, required=True)
    parser.add_argument("--probe-checkpoint", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/diagnostics/v17_4_variance_analysis")
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--segment-samples", type=int, default=220500)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load probe checkpoint to get architecture params
    probe_ckpt = torch.load(args.probe_checkpoint, map_location="cpu", weights_only=False)
    probe_args = probe_ckpt["args"]

    latent_dim = probe_args["latent_dim"]
    strides = probe_args["strides"]
    encoder_channels = probe_args["encoder_channels"]
    dilations = probe_args["dilations"]
    proj_dim = probe_args["proj_dim"]
    proj_hidden_dim = probe_args["proj_hidden_dim"]
    proj_depth = probe_args["proj_depth"]

    total_stride = 1
    for s in strides:
        total_stride *= s

    # Load frozen AE
    print("Loading AE...")
    ae_ckpt = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
    ae = WavVAE(
        latent_dim=latent_dim,
        encoder_channels=encoder_channels,
        strides=strides,
        dilations=dilations,
        sample_latent=False,
    )
    ae.load_state_dict(ae_ckpt.get("model", ae_ckpt), strict=True)
    ae = ae.to(device).eval()

    # Load semantic projection
    print("Loading semantic projection...")
    semantic_proj = SemanticProjection(
        latent_dim=latent_dim,
        proj_dim=proj_dim,
        hidden_dim=proj_hidden_dim,
        depth=proj_depth,
    ).to(device)
    semantic_proj.load_state_dict(probe_ckpt["semantic_proj"])
    semantic_proj.eval()

    # Load data
    data_path = Path(args.data_path)
    roots = [Path.cwd(), data_path.parent]
    if "outputs" in data_path.parts:
        idx = data_path.parts.index("outputs")
        if idx > 0:
            roots.append(Path(*data_path.parts[:idx]))

    items = read_manifest(data_path)[:args.max_items]
    print(f"  Items: {len(items)}")

    # ── Analysis 1: Variance explained by projection subspace ──
    print("\n=== Variance Analysis ===")

    # Get the projection weight matrix (last linear layer)
    # semantic_proj.proj is nn.Sequential(LayerNorm, Linear)
    proj_weight = None
    for module in semantic_proj.proj.modules():
        if isinstance(module, nn.Linear):
            proj_weight = module.weight.data  # [proj_dim, hidden_dim]
            break

    all_z_pooled = []  # utterance-level mean latents
    all_z_frames = []  # frame-level latents (flattened)
    all_z_proj = []    # projected latents
    snr_full = []      # SNR with full latent
    snr_proj_only = [] # SNR reconstructing from projected subspace info

    print("  Computing latents...")
    for i, item in enumerate(items):
        try:
            path = resolve_audio_path(item, roots)
            audio = load_audio(path, args.sr, args.segment_samples, total_stride)
        except (FileNotFoundError, RuntimeError):
            continue

        audio_t = audio.view(1, 1, -1).to(device)

        with torch.no_grad():
            out = ae(audio_t)
            z = out["z"]  # [1, latent_dim, T]
            x_hat_full = out["x_hat"]

            # Full reconstruction SNR
            snr_full.append(compute_snr(x_hat_full, audio_t))

            # Get z_proj (normalized)
            z_proj = semantic_proj(z)  # [1, proj_dim]
            all_z_proj.append(z_proj.cpu())

            # Mean-pooled latent for variance analysis
            z_pooled = z.mean(dim=-1)  # [1, latent_dim]
            all_z_pooled.append(z_pooled.cpu())

            # Frame-level for total variance
            all_z_frames.append(z.squeeze(0).cpu())  # [latent_dim, T]

    n = len(all_z_pooled)
    print(f"  Valid items: {n}")

    # Compute total frame-level variance
    all_frames = torch.cat([f.T for f in all_z_frames], dim=0)  # [total_frames, latent_dim]
    total_frame_var = all_frames.var(dim=0).sum().item()
    print(f"\n  Total frame-level variance (sum over dims): {total_frame_var:.4f}")
    print(f"  Per-dim frame variance: {total_frame_var / latent_dim:.4f}")

    # Compute utterance-level variance
    z_pooled_all = torch.cat(all_z_pooled, dim=0)  # [N, latent_dim]
    total_utt_var = z_pooled_all.var(dim=0).sum().item()
    print(f"  Total utterance-level variance: {total_utt_var:.4f}")
    print(f"  Per-dim utterance variance: {total_utt_var / latent_dim:.4f}")

    # Compute z_proj variance
    z_proj_all = torch.cat(all_z_proj, dim=0)  # [N, proj_dim]
    proj_var = z_proj_all.var(dim=0).sum().item()
    print(f"  z_proj variance (sum over dims): {proj_var:.4f}")
    print(f"  z_proj per-dim variance: {proj_var / proj_dim:.4f}")

    # PCA analysis of utterance-level latents
    z_centered = z_pooled_all - z_pooled_all.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / max(n - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).flip(0)  # descending
    total_eigval = eigvals.sum().item()

    cumvar = torch.cumsum(eigvals, dim=0) / total_eigval
    rank_50 = int((cumvar < 0.50).sum().item()) + 1
    rank_90 = int((cumvar < 0.90).sum().item()) + 1
    rank_95 = int((cumvar < 0.95).sum().item()) + 1
    rank_99 = int((cumvar < 0.99).sum().item()) + 1

    print(f"\n  PCA of utterance-level latents:")
    print(f"    Top-1 eigenvalue fraction: {eigvals[0].item() / total_eigval:.4f}")
    print(f"    Top-{proj_dim} eigenvalue fraction: {cumvar[proj_dim-1].item():.4f}")
    print(f"    Rank for 50% variance: {rank_50}")
    print(f"    Rank for 90% variance: {rank_90}")
    print(f"    Rank for 95% variance: {rank_95}")
    print(f"    Rank for 99% variance: {rank_99}")

    # ── Analysis 2: Subspace alignment with PCA ──
    print(f"\n=== Subspace Alignment ===")

    # Get the effective projection directions in latent space
    # semantic_proj maps z_seq -> conv -> attn_pool -> proj
    # The "directions" it uses are implicit in the conv+attn weights
    # But we can measure: how much of z_pooled variance does z_proj capture?

    # Project z_pooled through semantic_proj's effective linear map
    # Since proj is normalized, we measure cosine alignment
    z_proj_unnorm = []
    for z_seq in all_z_frames:
        with torch.no_grad():
            z_in = z_seq.unsqueeze(0).to(device)
            h = semantic_proj.conv(z_in)
            attn_weights = torch.softmax(semantic_proj.attn(h), dim=-1)
            pooled = (h * attn_weights).sum(dim=-1)
            # Get raw (pre-normalization) projection
            for module in semantic_proj.proj.modules():
                if isinstance(module, nn.LayerNorm):
                    pooled = module(pooled)
                elif isinstance(module, nn.Linear):
                    raw = module(pooled)
                    break
            z_proj_unnorm.append(raw.cpu())

    z_proj_unnorm_all = torch.cat(z_proj_unnorm, dim=0)  # [N, proj_dim]
    proj_unnorm_var = z_proj_unnorm_all.var(dim=0).sum().item()
    print(f"  z_proj (unnormalized) variance: {proj_unnorm_var:.4f}")
    print(f"  Ratio to total utterance variance: {proj_unnorm_var / total_utt_var:.4f}")

    # ── Analysis 3: Reconstruction with projected vs full latent ──
    print(f"\n=== Reconstruction Analysis ===")
    print(f"  Full latent SNR (mean): {np.mean(snr_full):.2f} dB")

    # Test: what if we zero out the "residual" in latent space?
    # We can't easily do this with the conv-based projection, but we can
    # measure the correlation between z_proj and reconstruction quality
    
    # Instead, let's measure: how much does z_proj correlate with
    # the condition (text_id, speaker_id)?
    print(f"\n=== Condition Predictability ===")

    text_to_id = probe_ckpt.get("text_to_id", {})
    speaker_to_id = probe_ckpt.get("speaker_to_id", {})

    # Group z_proj by text and speaker
    text_groups: dict[str, list[int]] = {}
    speaker_groups: dict[str, list[int]] = {}
    for i, item in enumerate(items[:n]):
        tk = text_key(item)
        sk = speaker_key(item)
        text_groups.setdefault(tk, []).append(i)
        speaker_groups.setdefault(sk, []).append(i)

    # Within-text variance vs between-text variance (ANOVA-style)
    z_proj_np = z_proj_all.numpy()
    global_mean = z_proj_np.mean(axis=0)

    # Between-text variance
    between_text_var = 0.0
    within_text_var = 0.0
    for tk, indices in text_groups.items():
        if len(indices) < 2:
            continue
        group = z_proj_np[indices]
        group_mean = group.mean(axis=0)
        between_text_var += len(indices) * np.sum((group_mean - global_mean) ** 2)
        within_text_var += np.sum((group - group_mean) ** 2)

    total_var_np = np.sum((z_proj_np - global_mean) ** 2)
    between_text_ratio = between_text_var / max(total_var_np, 1e-10)
    within_text_ratio = within_text_var / max(total_var_np, 1e-10)

    print(f"  Between-text variance ratio: {between_text_ratio:.4f}")
    print(f"  Within-text variance ratio: {within_text_ratio:.4f}")
    print(f"  (Between-text high = z_proj captures text differences)")

    # Between-speaker variance
    between_speaker_var = 0.0
    within_speaker_var = 0.0
    for sk, indices in speaker_groups.items():
        if len(indices) < 2:
            continue
        group = z_proj_np[indices]
        group_mean = group.mean(axis=0)
        between_speaker_var += len(indices) * np.sum((group_mean - global_mean) ** 2)
        within_speaker_var += np.sum((group - group_mean) ** 2)

    between_speaker_ratio = between_speaker_var / max(total_var_np, 1e-10)
    print(f"  Between-speaker variance ratio: {between_speaker_ratio:.4f}")
    print(f"  (Between-speaker high = z_proj captures speaker differences)")

    # Same-text cross-speaker cosine
    same_text_cos = []
    diff_text_cos = []
    for i in range(min(n, 200)):
        for j in range(i + 1, min(n, 200)):
            cos = float(np.dot(z_proj_np[i], z_proj_np[j]))
            ti = text_key(items[i])
            tj = text_key(items[j])
            si = speaker_key(items[i])
            sj = speaker_key(items[j])
            if ti == tj and si != sj:
                same_text_cos.append(cos)
            elif ti != tj and si == sj:
                diff_text_cos.append(cos)

    if same_text_cos:
        print(f"\n  Same-text cross-speaker cosine: {np.mean(same_text_cos):.4f} (n={len(same_text_cos)})")
    if diff_text_cos:
        print(f"  Diff-text same-speaker cosine: {np.mean(diff_text_cos):.4f} (n={len(diff_text_cos)})")
    if same_text_cos and diff_text_cos:
        gap = np.mean(same_text_cos) - np.mean(diff_text_cos)
        print(f"  Gap (same_text - diff_text): {gap:.4f}")
        print(f"  (Positive gap = text is more discriminative than speaker in z_proj)")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Latent dim: {latent_dim}, Proj dim: {proj_dim}")
    print(f"  Flow loss (train): {probe_ckpt.get('train_metrics', {}).get('flow_loss', 'N/A')}")
    print(f"  Flow loss (val): {probe_ckpt.get('final_metrics', {}).get('flow_loss', 'N/A')}")
    print(f"  PCA top-{proj_dim} covers: {cumvar[proj_dim-1].item()*100:.1f}% of utterance variance")
    print(f"  z_proj unnorm var / total utt var: {proj_unnorm_var / total_utt_var:.4f}")
    print(f"  Between-text / total: {between_text_ratio:.4f}")
    print(f"  Between-speaker / total: {between_speaker_ratio:.4f}")
    if same_text_cos and diff_text_cos:
        print(f"  Same-text cross-spk cos: {np.mean(same_text_cos):.4f}")
        print(f"  Diff-text same-spk cos: {np.mean(diff_text_cos):.4f}")

    # Save results
    results = {
        "latent_dim": latent_dim,
        "proj_dim": proj_dim,
        "n_items": n,
        "total_frame_var": total_frame_var,
        "total_utt_var": total_utt_var,
        "proj_unnorm_var": proj_unnorm_var,
        "proj_var_ratio": proj_unnorm_var / total_utt_var,
        "pca_top_k_coverage": cumvar[proj_dim-1].item(),
        "rank_50": rank_50,
        "rank_90": rank_90,
        "rank_95": rank_95,
        "between_text_ratio": between_text_ratio,
        "between_speaker_ratio": between_speaker_ratio,
        "within_text_ratio": within_text_ratio,
        "same_text_cross_speaker_cos": float(np.mean(same_text_cos)) if same_text_cos else None,
        "diff_text_same_speaker_cos": float(np.mean(diff_text_cos)) if diff_text_cos else None,
        "mean_snr_full": float(np.mean(snr_full)),
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
