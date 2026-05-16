"""Block B3 — Supervised subspaces and principal angles.

For V16.3 latent:
1. Construct phoneme subspace U_ph from class-mean SVD (frame-level
   grouping by pinyin id).
2. Construct speaker subspace U_sp from utterance-mean grouping by
   speaker id. (CosyVoice 4 speakers, so U_sp has dim <= 3.)
3. Compute principal angles between U_ph and U_sp.
4. Build a shuffle null by permuting phoneme labels 1000 times and
   recomputing U_ph_null's principal angles with U_sp.
5. Compute variance fractions: how much of the total frame variance
   lives inside each subspace.
6. Within / between variance decomposition (LDA-style) for phoneme
   and speaker.

Outputs:
- <out>/b3_subspaces.npz
- <out>/b3_summary.json
- <out>/figures/b3_principal_angles.png
- <out>/figures/b3_shuffle_null.png
- <out>/figures/b3_variance_fractions.png

Usage:
  python -m autoencoder.scripts.anatomy.subspace \
    --checkpoint outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt \
    --dataset-name cosyvoice_4spk_val \
    --manifest CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_val_phalign_neighbors.jsonl \
    --path-root CosyVoice_main --path-root . \
    --n-samples 112 --segment-samples 131072 \
    --phoneme-min-count 100 \
    --variance-target 0.90 \
    --n-shuffles 1000 \
    --out docs_ae/v16_3_latent_anatomy/data/b3_v16_3_cosyvoice
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .common import ANATOMY_SEED, load_checkpoint_auto, load_manifest, set_global_seed
from .dataset import encode_and_label, prepare_samples, stack_frames
from .pca_semantics import broadcast_speaker


# ---------------------------------------------------------------------------
# Class-mean SVD
# ---------------------------------------------------------------------------


def class_mean_svd(
    features: np.ndarray,  # [N, D]
    labels: np.ndarray,  # [N] integer
    min_count: int = 50,
    variance_target: float = 0.90,
) -> dict:
    """SVD of centered class-mean matrix.

    Returns dict with:
    - U: [d, D] orthonormal basis of the principal subspace
    - dim: truncation dimension (smallest so cumulative variance >= target)
    - class_means: [K, D] (centered around the grand mean of the classes)
    - grand_mean: [D]
    - class_ids: list[int] (labels kept)
    - class_counts: list[int]
    - singular_values
    - explained_variance_ratio
    - cumulative_variance
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    keep = counts >= min_count
    keep_ids = unique[keep]
    keep_counts = counts[keep]
    if len(keep_ids) < 2:
        raise ValueError(f"Need ≥2 classes with count≥{min_count}, got {len(keep_ids)}")
    means = np.stack(
        [features[labels == c].mean(axis=0) for c in keep_ids], axis=0
    )  # [K, D]
    grand = means.mean(axis=0)
    centered = means - grand
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    power = s**2
    ev = power / (power.sum() + 1e-30)
    cum = np.cumsum(ev)
    dim = int(np.searchsorted(cum, variance_target) + 1)
    # Right singular vectors of centered = basis of subspace, stored as rows
    U = vt[:dim].copy()  # [dim, D]
    return {
        "U": U,
        "dim": dim,
        "class_means": means,
        "grand_mean": grand,
        "class_ids": keep_ids.tolist(),
        "class_counts": keep_counts.tolist(),
        "singular_values": s,
        "explained_variance_ratio": ev,
        "cumulative_variance": cum,
    }


def class_mean_subspace_only(
    features: np.ndarray,
    labels: np.ndarray,
    min_count: int,
    dim: int,
) -> np.ndarray:
    """Lean version used in the shuffle null inner loop. Returns U [dim, D]."""
    unique, counts = np.unique(labels, return_counts=True)
    keep = counts >= min_count
    keep_ids = unique[keep]
    if len(keep_ids) < 2:
        return np.zeros((dim, features.shape[1]), dtype=np.float64)
    # Remap kept labels to dense indices for scatter-mean
    label_to_idx = {int(c): i for i, c in enumerate(keep_ids)}
    # For scatter, we need per-frame dense index; -1 for filtered
    dense = np.full(len(labels), -1, dtype=np.int64)
    for lab, idx in label_to_idx.items():
        dense[labels == lab] = idx
    mask = dense >= 0
    X = features[mask]
    d_idx = dense[mask]
    K = len(keep_ids)
    D = features.shape[1]
    sum_buf = np.zeros((K, D), dtype=np.float64)
    cnt_buf = np.zeros(K, dtype=np.int64)
    np.add.at(sum_buf, d_idx, X)
    np.add.at(cnt_buf, d_idx, 1)
    means = sum_buf / cnt_buf[:, None]
    grand = means.mean(axis=0)
    centered = means - grand
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:dim].copy()


# ---------------------------------------------------------------------------
# Principal angles
# ---------------------------------------------------------------------------


def principal_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Principal angles (radians) between two subspaces given by row-basis
    matrices U [m, D], V [n, D]. Returns sorted angles of length min(m, n).
    """
    # QR-orthonormalize just in case (should already be orthonormal)
    Q_u, _ = np.linalg.qr(U.T)
    Q_v, _ = np.linalg.qr(V.T)
    s = np.linalg.svd(Q_u.T @ Q_v, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(s)
    return np.sort(angles)  # smallest angle first


# ---------------------------------------------------------------------------
# Variance decomposition
# ---------------------------------------------------------------------------


def subspace_variance_fraction(
    features: np.ndarray, U: np.ndarray
) -> tuple[float, float]:
    """Projection variance fraction.

    Returns (inside_fraction, total_variance).
    `inside_fraction` = Σ ||U x_t||² / Σ ||x_t||², after centering features.
    """
    X = np.asarray(features, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    total = float(np.sum(X * X))
    if total < 1e-10:
        return (0.0, 0.0)
    projected = X @ U.T  # [N, dim]
    inside = float(np.sum(projected * projected))
    return (inside / total, total)


def within_between_variance(
    features: np.ndarray, labels: np.ndarray, min_count: int
) -> dict:
    """Compute per-label within-class and between-class variance ratios.

    SS_within = Σ_k Σ_{t in k} ||x_t − μ_k||²
    SS_between = Σ_k n_k ||μ_k − μ_grand||²
    SS_total = Σ_t ||x_t − μ_grand||²
    Returns {"within_frac", "between_frac", "n_classes", "total"}.
    """
    X = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    keep = counts >= min_count
    kept_labels = unique[keep]
    mask = np.isin(labels, kept_labels)
    X = X[mask]
    labels = labels[mask]
    grand = X.mean(axis=0)
    ss_total = float(np.sum((X - grand) ** 2))
    ss_between = 0.0
    ss_within = 0.0
    for c in kept_labels:
        m = labels == c
        if m.sum() < 1:
            continue
        mu_c = X[m].mean(axis=0)
        ss_between += float(m.sum()) * float(np.sum((mu_c - grand) ** 2))
        ss_within += float(np.sum((X[m] - mu_c) ** 2))
    return {
        "within_frac": ss_within / ss_total if ss_total > 0 else float("nan"),
        "between_frac": ss_between / ss_total if ss_total > 0 else float("nan"),
        "n_classes_kept": int(len(kept_labels)),
        "n_frames_kept": int(mask.sum()),
        "ss_total": ss_total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-name", default="cosyvoice_4spk_val")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, action="append", default=[])
    parser.add_argument("--n-samples", type=int, default=112)
    parser.add_argument("--segment-samples", type=int, default=131072)
    parser.add_argument("--crop-mode", default="center", choices=["first", "center", "last"])
    parser.add_argument("--phoneme-min-count", type=int, default=100)
    parser.add_argument("--speaker-min-count", type=int, default=1)
    parser.add_argument("--variance-target", type=float, default=0.90)
    parser.add_argument("--n-shuffles", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=ANATOMY_SEED)
    parser.add_argument("--tag", default="", help="Optional suffix for output files (for re-use on V18)")
    parser.add_argument("--model-kind", default="auto", choices=["auto", "v16_3", "v18"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    set_global_seed(args.seed)
    out_dir = args.out
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[B3] loading checkpoint {args.checkpoint}")
    bundle = load_checkpoint_auto(args.checkpoint, device, kind=args.model_kind)
    items = load_manifest(args.manifest)
    items = prepare_samples(items, args.n_samples, seed=args.seed)
    print(f"[B3] selected {len(items)} items from {args.manifest}")

    path_roots = [Path(".").resolve()] + [Path(r) for r in args.path_root]
    pairs, vocab = encode_and_label(
        items,
        bundle=bundle,
        path_roots=path_roots,
        segment_samples=args.segment_samples,
        crop_mode=args.crop_mode,
        build_phoneme_vocab=True,
        device=device,
        verbose=args.verbose,
    )
    stacked = stack_frames(pairs, vocab)
    print(f"[B3] stacked frames: {stacked.latent.shape}")

    X = stacked.latent.astype(np.float64)
    phoneme = stacked.labels_frame["phoneme"]
    speaker = broadcast_speaker(stacked.per_utt_speaker, stacked.utterance_index)

    # --- Phoneme subspace: frame-level grouping ---
    phoneme_mask = phoneme > 0
    X_ph = X[phoneme_mask]
    phoneme_valid = phoneme[phoneme_mask]
    print(
        f"[B3] phoneme frames for subspace = {len(X_ph)}, "
        f"unique classes = {len(np.unique(phoneme_valid))}"
    )
    phoneme_result = class_mean_svd(
        X_ph, phoneme_valid, min_count=args.phoneme_min_count,
        variance_target=args.variance_target,
    )
    print(
        f"[B3] phoneme subspace: dim={phoneme_result['dim']}, "
        f"kept_classes={len(phoneme_result['class_ids'])}, "
        f"top SV = {phoneme_result['singular_values'][0]:.3f}, "
        f"cum@dim = {phoneme_result['cumulative_variance'][phoneme_result['dim']-1]:.3f}"
    )

    # --- Speaker subspace: utterance-mean grouping ---
    # Use per-utterance latent means, so subspace reflects between-speaker
    # variation at utterance level.
    utt_means = stacked.per_utt_latent_mean.astype(np.float64)
    utt_speaker = np.array(
        [broadcast_speaker(stacked.per_utt_speaker, np.array([i]))[0] for i in range(len(utt_means))],
        dtype=np.int64,
    )
    valid_spk = utt_speaker >= 0
    utt_means_k = utt_means[valid_spk]
    utt_speaker_k = utt_speaker[valid_spk]
    print(f"[B3] speaker utterances = {len(utt_means_k)}, unique speakers = {len(np.unique(utt_speaker_k))}")
    speaker_result = class_mean_svd(
        utt_means_k, utt_speaker_k, min_count=args.speaker_min_count,
        variance_target=args.variance_target,
    )
    print(
        f"[B3] speaker subspace: dim={speaker_result['dim']}, "
        f"kept_speakers={len(speaker_result['class_ids'])}, "
        f"top SV = {speaker_result['singular_values'][0]:.3f}"
    )

    # --- Principal angles: phoneme vs speaker ---
    U_ph = phoneme_result["U"]  # [d_ph, D]
    U_sp = speaker_result["U"]  # [d_sp, D]
    angles_rad = principal_angles(U_ph, U_sp)
    angles_deg = np.rad2deg(angles_rad)
    print(f"[B3] principal angles (deg): {np.round(angles_deg, 2).tolist()}")

    # --- Shuffle null ---
    # Permute phoneme labels at frame level while keeping marginal distribution.
    print(f"[B3] computing phoneme-label shuffle null ({args.n_shuffles} permutations) ...")
    rng = np.random.default_rng(args.seed)
    null_angle_means = np.zeros(args.n_shuffles, dtype=np.float64)
    null_min_angles = np.zeros(args.n_shuffles, dtype=np.float64)
    for s in range(args.n_shuffles):
        perm = rng.permutation(len(phoneme_valid))
        shuffled = phoneme_valid[perm]
        try:
            U_ph_null = class_mean_subspace_only(
                X_ph, shuffled, min_count=args.phoneme_min_count, dim=phoneme_result["dim"]
            )
        except Exception:
            continue
        null_angles = principal_angles(U_ph_null, U_sp)
        null_angle_means[s] = np.rad2deg(null_angles).mean()
        null_min_angles[s] = np.rad2deg(null_angles).min()
        if args.verbose and (s + 1) % 200 == 0:
            print(f"[B3]   shuffle {s+1}/{args.n_shuffles}")

    observed_mean = float(angles_deg.mean())
    observed_min = float(angles_deg.min())
    p_mean = float((null_angle_means <= observed_mean).mean())
    p_min = float((null_min_angles <= observed_min).mean())

    print(
        f"[B3] shuffle null: mean angle observed={observed_mean:.2f}°, "
        f"null mean={null_angle_means.mean():.2f}° (CI [{np.quantile(null_angle_means, 0.025):.2f}, {np.quantile(null_angle_means, 0.975):.2f}]), "
        f"p(mean <= observed) = {p_mean:.4f}"
    )
    print(
        f"[B3] shuffle null: min angle observed={observed_min:.2f}°, "
        f"p(min <= observed) = {p_min:.4f}"
    )

    # --- Variance fractions ---
    v_ph_frac, _ = subspace_variance_fraction(X, U_ph)
    v_sp_frac_frame, _ = subspace_variance_fraction(X, U_sp)
    v_sp_frac_utt, _ = subspace_variance_fraction(utt_means_k, U_sp)
    print(
        f"[B3] variance fractions: "
        f"U_ph ∩ frame-X = {v_ph_frac:.4f}, "
        f"U_sp ∩ frame-X = {v_sp_frac_frame:.4f}, "
        f"U_sp ∩ utt-mean-X = {v_sp_frac_utt:.4f}"
    )

    # --- Within/between variance ---
    wb_ph = within_between_variance(X_ph, phoneme_valid, min_count=args.phoneme_min_count)
    wb_sp_utt = within_between_variance(utt_means_k, utt_speaker_k, min_count=args.speaker_min_count)
    print(
        f"[B3] within-between: phoneme within={wb_ph['within_frac']:.3f} / "
        f"between={wb_ph['between_frac']:.3f} ({wb_ph['n_classes_kept']} classes)"
    )
    print(
        f"[B3] within-between: speaker (utt-mean) within={wb_sp_utt['within_frac']:.3f} / "
        f"between={wb_sp_utt['between_frac']:.3f} ({wb_sp_utt['n_classes_kept']} speakers)"
    )

    # --- Save ---
    summary = {
        "block": "B3",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": bundle.checkpoint.get("epoch"),
        "checkpoint_best_snr": bundle.checkpoint.get("best_snr"),
        "dataset_name": args.dataset_name,
        "n_samples": len(pairs),
        "n_frames": int(stacked.latent.shape[0]),
        "latent_dim": int(stacked.latent.shape[1]),
        "phoneme_subspace": {
            "dim": int(phoneme_result["dim"]),
            "n_classes_kept": len(phoneme_result["class_ids"]),
            "min_count": args.phoneme_min_count,
            "variance_target": args.variance_target,
            "variance_fraction_on_all_frames": v_ph_frac,
            "within_frac": wb_ph["within_frac"],
            "between_frac": wb_ph["between_frac"],
            "singular_values": phoneme_result["singular_values"].tolist(),
            "cumulative_variance_among_classes": phoneme_result["cumulative_variance"].tolist(),
        },
        "speaker_subspace": {
            "dim": int(speaker_result["dim"]),
            "n_classes_kept": len(speaker_result["class_ids"]),
            "variance_target": args.variance_target,
            "variance_fraction_on_all_frames": v_sp_frac_frame,
            "variance_fraction_on_utt_means": v_sp_frac_utt,
            "within_frac_utt": wb_sp_utt["within_frac"],
            "between_frac_utt": wb_sp_utt["between_frac"],
            "singular_values": speaker_result["singular_values"].tolist(),
            "cumulative_variance_among_classes": speaker_result["cumulative_variance"].tolist(),
        },
        "principal_angles_deg": angles_deg.tolist(),
        "principal_angles_mean_deg": observed_mean,
        "principal_angles_min_deg": observed_min,
        "shuffle_null": {
            "n": args.n_shuffles,
            "mean_angle_mean": float(null_angle_means.mean()),
            "mean_angle_ci": [
                float(np.quantile(null_angle_means, 0.025)),
                float(np.quantile(null_angle_means, 0.975)),
            ],
            "min_angle_mean": float(null_min_angles.mean()),
            "p_mean": p_mean,
            "p_min": p_min,
        },
    }
    with (out_dir / f"b3_summary{suffix}.json").open("w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        out_dir / f"b3_subspaces{suffix}.npz",
        U_phoneme=U_ph,
        U_speaker=U_sp,
        principal_angles_deg=angles_deg,
        null_angle_means=null_angle_means,
        null_min_angles=null_min_angles,
        phoneme_sv=phoneme_result["singular_values"],
        speaker_sv=speaker_result["singular_values"],
    )

    # --- Plots ---
    # (1) Principal angles + shuffle null distribution
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    axes[0].bar(np.arange(len(angles_deg)) + 1, angles_deg, color="tab:blue", alpha=0.85)
    axes[0].axhline(90, color="gray", linestyle="--", lw=1, label="orthogonal")
    axes[0].set_xlabel("principal angle index")
    axes[0].set_ylabel("angle (deg)")
    axes[0].set_title(f"Observed principal angles U_ph ({U_ph.shape[0]}d) vs U_sp ({U_sp.shape[0]}d)")
    axes[0].set_ylim(0, 95)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].hist(null_angle_means, bins=40, color="lightgray", edgecolor="black", alpha=0.85, label="null")
    axes[1].axvline(observed_mean, color="red", lw=2, label=f"observed = {observed_mean:.2f}°")
    axes[1].set_xlabel("mean principal angle (deg)")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"Shuffle null ({args.n_shuffles} perms), p = {p_mean:.3f}")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures_dir / f"b3_principal_angles{suffix}.png")
    plt.close(fig)

    # (2) Variance fractions bar
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=150)
    labels_plot = [
        f"U_ph ({U_ph.shape[0]}d) ∩ frame-X",
        f"U_sp ({U_sp.shape[0]}d) ∩ frame-X",
        f"U_sp ({U_sp.shape[0]}d) ∩ utt-mean-X",
        "phoneme between-class",
        "speaker (utt) between-class",
    ]
    vals = [
        v_ph_frac,
        v_sp_frac_frame,
        v_sp_frac_utt,
        wb_ph["between_frac"],
        wb_sp_utt["between_frac"],
    ]
    ax.barh(labels_plot, vals, color=["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:purple"])
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("variance fraction")
    ax.set_title(f"V16.3 latent subspace variance ({args.dataset_name})")
    ax.set_xlim(0, max(vals) * 1.2 + 0.05)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(figures_dir / f"b3_variance_fractions{suffix}.png")
    plt.close(fig)

    print(f"\n[B3] === SUMMARY ===")
    print(f"  U_ph dim={U_ph.shape[0]}, variance_fraction(frame)={v_ph_frac:.4f}")
    print(f"  U_sp dim={U_sp.shape[0]}, variance_fraction(frame)={v_sp_frac_frame:.4f}, (utt-mean)={v_sp_frac_utt:.4f}")
    print(f"  principal angles (deg): {[f'{a:.2f}' for a in angles_deg]}")
    print(f"  observed mean angle={observed_mean:.2f}, null mean={null_angle_means.mean():.2f}, p={p_mean:.4f}")
    print(f"  phoneme within-frac={wb_ph['within_frac']:.3f}, between-frac={wb_ph['between_frac']:.3f}")
    print(f"  speaker(utt) within-frac={wb_sp_utt['within_frac']:.3f}, between-frac={wb_sp_utt['between_frac']:.3f}")
    print(f"  output: {out_dir}")


if __name__ == "__main__":
    main()
