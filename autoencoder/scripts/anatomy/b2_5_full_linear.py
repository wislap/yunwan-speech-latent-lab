"""B2.5 S2 — Full-384-dim linear probe as ceiling.

B2 used top-32 PCs (covering 55.8% variance). Maybe energy and other
labels are encoded in the tail. This runs the linear probe on the
full 384-dim latent with 5-fold utterance-split CV for each label.

For categorical targets (phoneme, speaker) we use multi-class logistic
regression with L2 regularization. For continuous (f0, energy,
centroid) we use ridge. For binary (voiced) we use ridge on {0, 1}.

Outputs:
- <out>/b2_5_full_linear.json
- <out>/figures/b2_5_full384_vs_top32.png (comparison bar chart)
- <out>/figures/b2_5_label_scores.png

Usage:
  python -m autoencoder.scripts.anatomy.b2_5_full_linear \
    --checkpoint outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt \
    --manifest CosyVoice_main/outputs/tts/cosyvoice3/crossed_zh_4spk_4h_scale_v0_4spk_qc_8p0_13p5_val_phalign_neighbors.jsonl \
    --path-root CosyVoice_main --path-root . \
    --n-samples 112 --segment-samples 131072 \
    --out docs_ae/v16_3_latent_anatomy/data/b2_5
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
from .probes import ridge_fit_predict, r2_score_safe, utterance_kfold_indices


# ---------------------------------------------------------------------------
# Multi-class logistic regression (L2, simple gradient descent)
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / (e.sum(axis=1, keepdims=True) + 1e-30)


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    alpha: float = 1.0,
    lr: float = 0.5,
    n_iter: int = 300,
    batch_size: int | None = None,
    seed: int = ANATOMY_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit multi-class logistic regression with L2 regularization.

    Returns (W [D, C], b [C]). Features are centered/standardized internally.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    N, D = X.shape
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - mu) / sd
    W = np.zeros((D, n_classes), dtype=np.float32)
    b = np.zeros(n_classes, dtype=np.float32)
    Y_onehot = np.zeros((N, n_classes), dtype=np.float32)
    Y_onehot[np.arange(N), y] = 1.0
    batch = batch_size or min(N, 4096)
    for it in range(n_iter):
        if batch >= N:
            idx = np.arange(N)
        else:
            idx = rng.choice(N, size=batch, replace=False)
        xb = Xn[idx]
        yb = Y_onehot[idx]
        logits = xb @ W + b
        p = _softmax(logits)
        dW = xb.T @ (p - yb) / len(idx) + alpha * W / N
        db = (p - yb).mean(axis=0)
        W -= lr * dW
        b -= lr * db
        if it == n_iter - 1 or it == 10 or it == 50:
            if batch < N:
                # decay lr later
                lr *= 0.98
    return (W / sd.reshape(-1, 1), b - (mu @ (W / sd.reshape(-1, 1))).reshape(-1))


def predict_logistic(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    logits = X @ W + b
    return np.argmax(logits, axis=1)


# ---------------------------------------------------------------------------
# Unified CV driver
# ---------------------------------------------------------------------------


def ridge_cv(
    X: np.ndarray,
    y: np.ndarray,
    utt_idx: np.ndarray,
    mask: np.ndarray | None,
    n_folds: int,
    alpha: float,
    seed: int,
) -> dict:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    keep = mask & np.isfinite(y)
    Xk, yk, utt_k = X[keep], y[keep], utt_idx[keep]
    folds = utterance_kfold_indices(utt_k, n_folds=n_folds, seed=seed)
    fold_r2 = []
    for tr, te in folds:
        if len(tr) < 2 or len(te) < 2:
            fold_r2.append(float("nan"))
            continue
        y_hat = ridge_fit_predict(Xk[tr], yk[tr], Xk[te], alpha=alpha)
        fold_r2.append(r2_score_safe(yk[te], y_hat))
    arr = np.array(fold_r2, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    return {
        "mean": float(valid.mean()) if valid.size else float("nan"),
        "std": float(valid.std(ddof=1)) if valid.size > 1 else 0.0,
        "per_fold": [float(v) for v in arr],
        "n_samples": int(len(yk)),
    }


def logistic_cv(
    X: np.ndarray,
    y: np.ndarray,
    utt_idx: np.ndarray,
    mask: np.ndarray | None,
    n_folds: int,
    alpha: float,
    n_iter: int,
    seed: int,
) -> dict:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    keep = mask & (y >= 0)
    Xk, yk, utt_k = X[keep], y[keep], utt_idx[keep]
    # Remap labels to dense 0..C-1
    classes = np.unique(yk)
    remap = {int(c): i for i, c in enumerate(classes)}
    y_dense = np.array([remap[int(v)] for v in yk], dtype=np.int64)
    n_classes = len(classes)
    folds = utterance_kfold_indices(utt_k, n_folds=n_folds, seed=seed)
    fold_acc = []
    fold_top5 = []
    for tr, te in folds:
        if len(tr) < 2 or len(te) < 2:
            fold_acc.append(float("nan"))
            fold_top5.append(float("nan"))
            continue
        W, b = fit_logistic(Xk[tr], y_dense[tr], n_classes, alpha=alpha, n_iter=n_iter, seed=seed)
        logits = Xk[te] @ W + b
        pred = np.argmax(logits, axis=1)
        acc = float((pred == y_dense[te]).mean())
        fold_acc.append(acc)
        # Top-5 accuracy if C > 5
        if n_classes > 5:
            top5 = np.argsort(-logits, axis=1)[:, :5]
            fold_top5.append(float(np.any(top5 == y_dense[te][:, None], axis=1).mean()))
        else:
            fold_top5.append(float("nan"))
    acc_arr = np.array(fold_acc, dtype=np.float64)
    t5_arr = np.array(fold_top5, dtype=np.float64)
    valid_acc = acc_arr[np.isfinite(acc_arr)]
    valid_t5 = t5_arr[np.isfinite(t5_arr)]
    # Chance baseline: fraction of majority class
    counts = np.bincount(y_dense)
    chance = float(counts.max() / counts.sum())
    return {
        "acc_mean": float(valid_acc.mean()) if valid_acc.size else float("nan"),
        "acc_std": float(valid_acc.std(ddof=1)) if valid_acc.size > 1 else 0.0,
        "top5_mean": float(valid_t5.mean()) if valid_t5.size else float("nan"),
        "acc_per_fold": [float(v) for v in acc_arr],
        "n_classes": int(n_classes),
        "majority_chance": chance,
        "n_samples": int(len(yk)),
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
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--logistic-alpha", type=float, default=1.0)
    parser.add_argument("--logistic-iter", type=int, default=300)
    parser.add_argument("--b2-json", type=Path, default=None,
                        help="Optional path to b2_summary.json for top-32 comparison")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=ANATOMY_SEED)
    parser.add_argument("--shuffle-labels", action="store_true",
                        help="If set, permute all labels within their dtype before running. For S3 shuffle null.")
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
    bundle = load_checkpoint_auto(args.checkpoint, device, kind=args.model_kind)
    items = load_manifest(args.manifest)
    items = prepare_samples(items, args.n_samples, seed=args.seed)
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
    print(f"[B2.5-S2] stacked frames: {stacked.latent.shape}")

    X = stacked.latent.astype(np.float32)
    utt_idx = stacked.utterance_index
    phoneme = stacked.labels_frame["phoneme"].astype(np.int64)
    voiced = stacked.labels_frame["voiced"].astype(bool)
    f0 = stacked.labels_frame["f0"]
    energy = stacked.labels_frame["energy"]
    centroid = stacked.labels_frame["centroid"]
    speaker = broadcast_speaker(stacked.per_utt_speaker, utt_idx)

    if args.shuffle_labels:
        print("[B2.5-S2] SHUFFLING labels (frame-level permutation) for null baseline")
        rng_shuf = np.random.default_rng(args.seed + 999)
        perm = rng_shuf.permutation(len(utt_idx))
        phoneme = phoneme[perm]
        voiced = voiced[perm]
        f0 = f0[perm]
        energy = energy[perm]
        centroid = centroid[perm]
        speaker = speaker[perm]

    # Scores
    print("[B2.5-S2] ridge CV: f0 (voiced only), energy, centroid, voiced")
    scores = {}
    scores["f0"] = ridge_cv(X, f0, utt_idx, voiced, args.n_folds, args.ridge_alpha, args.seed)
    scores["energy"] = ridge_cv(X, energy, utt_idx, None, args.n_folds, args.ridge_alpha, args.seed)
    scores["centroid"] = ridge_cv(X, centroid, utt_idx, None, args.n_folds, args.ridge_alpha, args.seed)
    scores["voiced"] = ridge_cv(X, voiced.astype(np.float64), utt_idx, None, args.n_folds, args.ridge_alpha, args.seed)

    print(f"[B2.5-S2] logistic CV: phoneme (frames with id>0), speaker")
    phoneme_mask = phoneme > 0
    scores["phoneme"] = logistic_cv(
        X, phoneme, utt_idx, phoneme_mask, args.n_folds, args.logistic_alpha, args.logistic_iter, args.seed
    )
    scores["speaker"] = logistic_cv(
        X, speaker, utt_idx, None, args.n_folds, args.logistic_alpha, args.logistic_iter, args.seed
    )

    # Console readout
    print("\n[B2.5-S2] === LINEAR PROBE (full 384-dim) ===")
    for k in ["phoneme", "speaker"]:
        s = scores[k]
        print(
            f"  {k:9s}  acc={s['acc_mean']:.3f}  top5={s['top5_mean']:.3f}  "
            f"chance={s['majority_chance']:.3f}  n_cls={s['n_classes']}  n={s['n_samples']}"
        )
    for k in ["f0", "energy", "centroid", "voiced"]:
        s = scores[k]
        print(f"  {k:9s}  R²={s['mean']:.3f} ± {s['std']:.3f}  n={s['n_samples']}")

    # Load B2 top-32 multi-var for comparison if available
    top32_ref = {}
    if args.b2_json and args.b2_json.exists():
        with args.b2_json.open() as f:
            b2data = json.load(f)
        # B2 primary summary provides CV-per-direction, not multivariate. We grab
        # the max_score_per_label as comparator (best single PC).
        top32_ref = {k: v for k, v in b2data.get("max_score_per_label", {}).items() if v is not None}

    summary = {
        "block": "B2.5-S2",
        "shuffle_labels": args.shuffle_labels,
        "checkpoint": str(args.checkpoint),
        "dataset_name": args.dataset_name,
        "n_samples": len(pairs),
        "n_frames": int(stacked.latent.shape[0]),
        "latent_dim": int(stacked.latent.shape[1]),
        "n_folds": args.n_folds,
        "ridge_alpha": args.ridge_alpha,
        "logistic_alpha": args.logistic_alpha,
        "logistic_iter": args.logistic_iter,
        "scores": scores,
        "b2_top32_best_per_label": top32_ref,
    }
    suffix_raw = "_shuffled" if args.shuffle_labels else ""
    if args.tag:
        suffix_raw = f"_{args.tag}" + suffix_raw
    suffix = suffix_raw
    with (out_dir / f"b2_5_full_linear{suffix}.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- Bar comparison: full-384 vs top-32 (best single PC) ---
    labels_plot = ["phoneme", "speaker", "f0", "energy", "centroid", "voiced"]

    def _score_to_plot(label: str) -> tuple[float, str]:
        s = scores[label]
        if label in ("phoneme", "speaker"):
            return float(s["acc_mean"]), "accuracy"
        return float(s["mean"]), "R²"

    full384_vals = [_score_to_plot(l)[0] for l in labels_plot]
    top32_vals = []
    for label in labels_plot:
        # top-32 best single-PC CV score
        if label == "phoneme":
            v = top32_ref.get("phoneme", 0.0)
        elif label == "speaker":
            v = top32_ref.get("speaker", 0.0)
        else:
            v = top32_ref.get(label, 0.0)
        top32_vals.append(v if v is not None else 0.0)

    fig, ax = plt.subplots(figsize=(8.5, 4.0), dpi=150)
    x = np.arange(len(labels_plot))
    ax.bar(x - 0.2, top32_vals, width=0.4, color="tab:gray",
           label="best single PC (B2)")
    ax.bar(x + 0.2, full384_vals, width=0.4, color="tab:blue",
           label="full 384-dim linear (B2.5)")
    for i, (t, f) in enumerate(zip(top32_vals, full384_vals)):
        ax.text(i - 0.2, t + 0.01, f"{t:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, f + 0.01, f"{f:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot)
    ax.set_ylabel("score (accuracy or R²)")
    ax.set_title(f"V16.3 latent linear probe: full-384 vs best-single-PC ({args.dataset_name})"
                 + (" [shuffled]" if args.shuffle_labels else ""))
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / f"b2_5_full384_vs_top32{suffix}.png")
    plt.close(fig)

    print(f"\n  output: {out_dir}")


if __name__ == "__main__":
    main()
