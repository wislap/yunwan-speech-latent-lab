"""Block B2 — PCA directions vs semantic labels.

For each of the top-K PCA directions of V16.3 latent, compute a scalar
association measure between the per-frame projection coefficient `α_i(t)`
and each of six semantic labels:
  phoneme   — eta² correlation ratio
  speaker   — eta² correlation ratio (CosyVoice only, single value per utt,
              broadcast to all frames in that utterance)
  f0        — R² via ridge CV, restricted to voiced frames
  energy    — R² via ridge CV
  centroid  — R² via ridge CV
  voiced    — R² via ridge CV on the binary flag

Each direction is assigned a "primary label" by argmax score, if
score > threshold; otherwise "unassigned".

Outputs:
- <out>/b2_heatmap.npz
- <out>/b2_summary.json
- <out>/figures/b2_heatmap.png
- <out>/figures/b2_bar_primary.png

Usage:
  python -m autoencoder.scripts.anatomy.pca_semantics \
    --checkpoint outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt \
    --dataset-name cosyvoice_4spk_val \
    --manifest CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_val_phalign_neighbors.jsonl \
    --path-root CosyVoice_main --path-root . \
    --n-samples 112 --segment-samples 131072 \
    --k-directions 32 \
    --out docs_ae/v16_3_latent_anatomy/data/b2_v16_3_cosyvoice
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
from .probes import eta_squared_cv, r2_cv


LABEL_LIST = ["phoneme", "speaker", "f0", "energy", "centroid", "voiced"]
ASSIGN_THRESHOLDS = {
    "phoneme": 0.30,     # eta²
    "speaker": 0.30,     # eta²
    "f0": 0.30,          # R²
    "energy": 0.30,      # R²
    "centroid": 0.30,    # R²
    "voiced": 0.30,      # R²
}


def compute_pca_basis(frame_latent: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, top-k components [K, D], explained variance ratio)."""
    mean = frame_latent.mean(axis=0).astype(np.float64)
    centered = frame_latent.astype(np.float64) - mean
    # SVD on centered X = U S Vt, components are rows of Vt
    _, sigma, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(k, vt.shape[0])
    # Deterministic sign: largest absolute loading positive
    components = vt[:k].copy()
    for i in range(k):
        j = int(np.argmax(np.abs(components[i])))
        if components[i, j] < 0:
            components[i] *= -1.0
    ev = (sigma**2) / float((sigma**2).sum() + 1e-30)
    return mean.astype(np.float32), components.astype(np.float64), ev[:k]


def score_direction(
    alpha: np.ndarray,
    stacked_labels: dict[str, np.ndarray],
    utterance_index: np.ndarray,
    speaker_labels: np.ndarray,
    n_folds: int,
    seed: int,
) -> dict[str, float]:
    """Score a single PCA direction against all six labels."""
    out: dict[str, float] = {}
    voiced_mask = stacked_labels["voiced"].astype(bool)

    # phoneme (categorical)
    phoneme = stacked_labels["phoneme"]
    phoneme_mask = phoneme > 0  # exclude silence/unknown (id 0 and -1)
    eta_ph = eta_squared_cv(alpha, phoneme, utterance_index, n_folds=n_folds, mask=phoneme_mask, seed=seed)
    out["phoneme_eta2"] = eta_ph["full"]
    out["phoneme_n"] = eta_ph["n_samples"]

    # speaker (categorical, from per-utt broadcast)
    eta_sp = eta_squared_cv(alpha, speaker_labels, utterance_index, n_folds=n_folds, seed=seed)
    out["speaker_eta2"] = eta_sp["full"]
    out["speaker_n"] = eta_sp["n_samples"]

    # f0 (continuous, voiced frames only)
    r2_f0 = r2_cv(alpha, stacked_labels["f0"], utterance_index, n_folds=n_folds, mask=voiced_mask, seed=seed)
    out["f0_r2"] = r2_f0["mean"]
    out["f0_n"] = r2_f0["n_samples"]

    # energy
    r2_en = r2_cv(alpha, stacked_labels["energy"], utterance_index, n_folds=n_folds, seed=seed)
    out["energy_r2"] = r2_en["mean"]

    # centroid
    r2_ce = r2_cv(alpha, stacked_labels["centroid"], utterance_index, n_folds=n_folds, seed=seed)
    out["centroid_r2"] = r2_ce["mean"]

    # voiced (binary 0/1)
    r2_vo = r2_cv(
        alpha,
        stacked_labels["voiced"].astype(np.float64),
        utterance_index,
        n_folds=n_folds,
        seed=seed,
    )
    out["voiced_r2"] = r2_vo["mean"]

    return out


def broadcast_speaker(per_utt_speaker: list[str | None], utterance_index: np.ndarray) -> np.ndarray:
    """Map per-utterance speaker strings to per-frame integer IDs.

    Returns int array of shape [N_frames_total]. Missing speakers become -1.
    """
    labeled = {}
    for spk in per_utt_speaker:
        if spk is None or spk == "":
            continue
        if spk not in labeled:
            labeled[spk] = len(labeled)
    ids_per_utt = np.array(
        [labeled[s] if (s in labeled) else -1 for s in per_utt_speaker],
        dtype=np.int64,
    )
    return ids_per_utt[utterance_index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, action="append", default=[])
    parser.add_argument("--n-samples", type=int, default=112)
    parser.add_argument("--segment-samples", type=int, default=131072)
    parser.add_argument("--crop-mode", default="center", choices=["first", "center", "last"])
    parser.add_argument("--k-directions", type=int, default=32)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=ANATOMY_SEED)
    parser.add_argument("--shuffle-labels", action="store_true",
                        help="Frame-level permute all labels for shuffle-null baseline.")
    parser.add_argument("--model-kind", default="auto", choices=["auto", "v16_3", "v18"])
    parser.add_argument("--tag", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    set_global_seed(args.seed)
    out_dir = args.out
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[B2] loading checkpoint {args.checkpoint}")
    bundle = load_checkpoint_auto(args.checkpoint, device, kind=args.model_kind)
    items = load_manifest(args.manifest)
    items = prepare_samples(items, args.n_samples, seed=args.seed)
    print(f"[B2] prepared {len(items)} items from {args.manifest}")

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
    print(f"[B2] encoded {len(pairs)} / {len(items)}; phoneme vocab size {len(vocab)}")
    if len(pairs) < args.n_folds * 2:
        raise RuntimeError(f"Not enough samples ({len(pairs)}) for {args.n_folds}-fold CV")

    stacked = stack_frames(pairs, vocab)
    print(
        f"[B2] stacked frames: total={stacked.latent.shape[0]}, dim={stacked.latent.shape[1]}, "
        f"n_utts={stacked.n_utts}"
    )

    phoneme_ids = stacked.labels_frame["phoneme"]
    n_phoneme_frames = int((phoneme_ids > 0).sum())
    uniq_phoneme = int(len(np.unique(phoneme_ids[phoneme_ids > 0])))
    print(f"[B2] phoneme-labeled frames = {n_phoneme_frames}, unique phoneme classes = {uniq_phoneme}")

    speaker_frame_id = broadcast_speaker(stacked.per_utt_speaker, stacked.utterance_index)
    n_speakers = int((np.unique(speaker_frame_id) != -1).sum())
    print(f"[B2] speakers labelled = {n_speakers}")

    if args.shuffle_labels:
        print("[B2] SHUFFLING labels (frame-level) for null baseline")
        rng_shuf = np.random.default_rng(args.seed + 999)
        perm = rng_shuf.permutation(len(stacked.utterance_index))
        stacked.labels_frame["phoneme"] = stacked.labels_frame["phoneme"][perm]
        stacked.labels_frame["f0"] = stacked.labels_frame["f0"][perm]
        stacked.labels_frame["voiced"] = stacked.labels_frame["voiced"][perm]
        stacked.labels_frame["energy"] = stacked.labels_frame["energy"][perm]
        stacked.labels_frame["centroid"] = stacked.labels_frame["centroid"][perm]
        phoneme_ids = stacked.labels_frame["phoneme"]
        speaker_frame_id = speaker_frame_id[perm]

    # --- PCA basis ---
    print(f"[B2] computing top-{args.k_directions} PCA directions ...")
    mean_vec, components, ev = compute_pca_basis(stacked.latent, args.k_directions)
    centered = stacked.latent.astype(np.float64) - mean_vec[None, :]
    # alpha[t, i] = components[i] . centered[t]
    alpha_mat = centered @ components.T  # [N_frames, K]
    print(f"[B2] alpha matrix shape = {alpha_mat.shape}")

    # --- Score each direction ---
    print(f"[B2] scoring {args.k_directions} directions against {len(LABEL_LIST)} labels ...")
    K = args.k_directions
    score_matrix = np.full((K, len(LABEL_LIST)), np.nan, dtype=np.float64)
    per_direction = []
    score_key = {
        "phoneme": "phoneme_eta2",
        "speaker": "speaker_eta2",
        "f0": "f0_r2",
        "energy": "energy_r2",
        "centroid": "centroid_r2",
        "voiced": "voiced_r2",
    }
    for i in range(K):
        alpha = alpha_mat[:, i]
        scores = score_direction(
            alpha=alpha,
            stacked_labels=stacked.labels_frame,
            utterance_index=stacked.utterance_index,
            speaker_labels=speaker_frame_id,
            n_folds=args.n_folds,
            seed=args.seed,
        )
        row = [scores.get(score_key[label], float("nan")) for label in LABEL_LIST]
        score_matrix[i] = row
        # primary label
        clean_row = np.array([x if np.isfinite(x) else -np.inf for x in row])
        best = int(np.argmax(clean_row))
        best_label = LABEL_LIST[best]
        best_val = float(clean_row[best])
        if not np.isfinite(best_val) or best_val < ASSIGN_THRESHOLDS[best_label]:
            primary = "unassigned"
        else:
            primary = best_label
        per_direction.append(
            {
                "component": i,
                "explained_variance_ratio": float(ev[i]),
                "scores": {label: (float(x) if np.isfinite(x) else None) for label, x in zip(LABEL_LIST, row)},
                "primary": primary,
                "primary_score": best_val if np.isfinite(best_val) else None,
            }
        )
        if args.verbose and (i + 1) % 8 == 0:
            print(f"[B2]   scored {i + 1}/{K} directions")

    # --- Aggregate summary ---
    primary_counts = {label: 0 for label in LABEL_LIST}
    primary_counts["unassigned"] = 0
    for d in per_direction:
        primary_counts[d["primary"]] += 1

    max_per_label = {}
    for j, label in enumerate(LABEL_LIST):
        col = score_matrix[:, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            max_per_label[label] = None
        else:
            max_per_label[label] = float(finite.max())

    # --- Save arrays ---
    suffix_raw = "_shuffled" if args.shuffle_labels else ""
    if args.tag:
        suffix_raw = f"_{args.tag}" + suffix_raw
    suffix = suffix_raw
    np.savez_compressed(
        out_dir / f"b2_heatmap{suffix}.npz",
        score_matrix=score_matrix.astype(np.float64),
        labels=np.asarray(LABEL_LIST, dtype=object),
        ev=ev.astype(np.float64),
        components=components.astype(np.float32),
        mean_vec=mean_vec.astype(np.float32),
    )

    summary = {
        "block": "B2",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": bundle.checkpoint.get("epoch"),
        "checkpoint_best_snr": bundle.checkpoint.get("best_snr"),
        "dataset_name": args.dataset_name,
        "manifest": str(args.manifest),
        "n_samples_requested": args.n_samples,
        "n_samples_encoded": len(pairs),
        "n_frames_total": int(stacked.latent.shape[0]),
        "latent_dim": int(stacked.latent.shape[1]),
        "k_directions": K,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "phoneme_vocab_size": len(vocab),
        "phoneme_labeled_frames": n_phoneme_frames,
        "phoneme_unique_classes": uniq_phoneme,
        "n_speakers": n_speakers,
        "assignment_thresholds": ASSIGN_THRESHOLDS,
        "primary_counts": primary_counts,
        "max_score_per_label": max_per_label,
        "ev_top32": ev.tolist(),
        "ev_top32_sum": float(ev.sum()),
        "per_direction": per_direction,
    }
    with (out_dir / f"b2_summary{suffix}.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- Heatmap ---
    fig, ax = plt.subplots(figsize=(9.5, 7.0), dpi=150)
    display = np.where(np.isfinite(score_matrix), score_matrix, 0.0)
    im = ax.imshow(display, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(LABEL_LIST)))
    ax.set_xticklabels(LABEL_LIST, rotation=30, ha="right")
    ax.set_yticks(np.arange(K))
    ax.set_yticklabels([f"PC{i + 1}" for i in range(K)], fontsize=7)
    ax.set_title(f"V16.3 latent — PCA direction × semantic label ({args.dataset_name})")
    for i in range(K):
        for j in range(len(LABEL_LIST)):
            v = score_matrix[i, j]
            if np.isfinite(v) and v > 0.15:
                color = "white" if v < 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("R² / η²")
    fig.tight_layout()
    fig.savefig(figures_dir / f"b2_heatmap{suffix}.png")
    plt.close(fig)

    # --- Primary bar chart ---
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=150)
    labels_ordered = LABEL_LIST + ["unassigned"]
    counts = [primary_counts[l] for l in labels_ordered]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "lightgray"]
    ax.bar(labels_ordered, counts, color=colors, edgecolor="black")
    for i, c in enumerate(counts):
        ax.text(i, c + 0.15, str(c), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(f"# of top-{K} PCA directions (primary label)")
    ax.set_title(f"V16.3 latent — PCA direction primary assignment ({args.dataset_name})"
                 + (" [shuffled]" if args.shuffle_labels else ""))
    ax.grid(alpha=0.25, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(figures_dir / f"b2_bar_primary{suffix}.png")
    plt.close(fig)

    # --- Console summary ---
    print("\n[B2] === SUMMARY ===")
    print(f"  dataset={args.dataset_name}  K={K}  folds={args.n_folds}")
    print(f"  EV of top-{K} PCA = {ev.sum():.3f}")
    print("  primary label counts:")
    for label in labels_ordered:
        print(f"    {label:12s} {primary_counts[label]}")
    print("  best score per label:")
    for label, v in max_per_label.items():
        print(f"    {label:12s} {v if v is None else f'{v:.3f}'}")
    print(f"  output: {out_dir}")


if __name__ == "__main__":
    main()
