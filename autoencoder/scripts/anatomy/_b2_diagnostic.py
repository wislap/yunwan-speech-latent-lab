"""Diagnostic for B2 — check raw (no CV) per-direction correlations
and multi-variate predictive strength, to distinguish 'no info' from
'distributed info'.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from autoencoder.scripts.anatomy.common import load_manifest, load_v16_3, set_global_seed
from autoencoder.scripts.anatomy.dataset import encode_and_label, prepare_samples, stack_frames
from autoencoder.scripts.anatomy.pca_semantics import broadcast_speaker, compute_pca_basis
from autoencoder.scripts.anatomy.probes import eta_squared, r2_score_safe


def multivariate_r2(X: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None, alpha: float = 1.0) -> float:
    """In-sample ridge R² on mean-centered X against y. No CV. Diagnostic only."""
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    keep = mask & np.isfinite(y)
    X = X[keep]
    y = y[keep]
    if len(y) < 20:
        return float("nan")
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    gram = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
    w = np.linalg.solve(gram, Xc.T @ yc)
    y_hat = Xc @ w + y.mean()
    return r2_score_safe(y, y_hat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--path-root", action="append", default=[], type=Path)
    parser.add_argument("--n-samples", type=int, default=112)
    parser.add_argument("--segment-samples", type=int, default=131072)
    parser.add_argument("--k-directions", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_v16_3(args.checkpoint, device)
    items = load_manifest(args.manifest)
    items = prepare_samples(items, args.n_samples, seed=args.seed)
    path_roots = [Path(".").resolve()] + list(args.path_root)
    pairs, vocab = encode_and_label(
        items,
        bundle=bundle,
        path_roots=path_roots,
        segment_samples=args.segment_samples,
        build_phoneme_vocab=True,
        device=device,
    )
    stacked = stack_frames(pairs, vocab)

    mean_vec, components, ev = compute_pca_basis(stacked.latent, args.k_directions)
    centered = stacked.latent.astype(np.float64) - mean_vec[None, :]
    alpha = centered @ components.T  # [N, K]

    phoneme = stacked.labels_frame["phoneme"]
    voiced = stacked.labels_frame["voiced"].astype(bool)
    f0 = stacked.labels_frame["f0"]
    energy = stacked.labels_frame["energy"]
    centroid = stacked.labels_frame["centroid"]
    voiced_f = voiced.astype(np.float64)
    speaker_id = broadcast_speaker(stacked.per_utt_speaker, stacked.utterance_index)

    per_dir = []
    for i in range(args.k_directions):
        a = alpha[:, i]
        # raw in-sample R² (correlation squared)
        def sse_r2(y, m):
            m = m & np.isfinite(y)
            if m.sum() < 20:
                return float("nan")
            x = a[m]
            yv = y[m]
            r = np.corrcoef(x, yv)[0, 1]
            return float(r * r)
        per_dir.append({
            "pc": i + 1,
            "ev": float(ev[i]),
            "f0_r2_raw_voiced": sse_r2(f0, voiced),
            "energy_r2_raw": sse_r2(energy, np.ones_like(voiced)),
            "centroid_r2_raw": sse_r2(centroid, np.ones_like(voiced)),
            "voiced_r2_raw": sse_r2(voiced_f, np.ones_like(voiced)),
            "phoneme_eta2_raw": float(eta_squared(a, phoneme, min_class_size=20) if (phoneme > 0).sum() > 100 else float("nan")),
            "speaker_eta2_raw": float(eta_squared(a, speaker_id, min_class_size=20) if (speaker_id >= 0).sum() > 100 else float("nan")),
        })

    # Multivariate R² using all 32 alpha as features (in-sample, no CV)
    mv = {
        "f0_multi_r2": multivariate_r2(alpha, f0, mask=voiced),
        "energy_multi_r2": multivariate_r2(alpha, energy),
        "centroid_multi_r2": multivariate_r2(alpha, centroid),
        "voiced_multi_r2": multivariate_r2(alpha, voiced_f),
    }

    summary = {
        "n_frames": int(len(alpha)),
        "n_voiced": int(voiced.sum()),
        "n_phoneme_labeled": int((phoneme > 0).sum()),
        "n_unique_phoneme": int(len(np.unique(phoneme[phoneme > 0]))),
        "n_speakers": int((np.unique(speaker_id) != -1).sum()),
        "ev_total": float(ev.sum()),
        "per_direction": per_dir,
        "multivariate_in_sample": mv,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "b2_diagnostic.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("[B2-diag] === KEY FIGURES ===")
    print(f"  n_frames = {summary['n_frames']}  n_voiced = {summary['n_voiced']}")
    print(f"  EV top-{args.k_directions} = {summary['ev_total']:.3f}")
    print("  max per-direction RAW scores (no CV):")
    arr = {k: [d[k] for d in per_dir if d[k] is not None and not np.isnan(d[k])] for k in ("f0_r2_raw_voiced", "energy_r2_raw", "centroid_r2_raw", "voiced_r2_raw", "phoneme_eta2_raw", "speaker_eta2_raw")}
    for k, vals in arr.items():
        if vals:
            top3 = sorted(vals, reverse=True)[:3]
            print(f"    {k:22s} max={max(vals):.4f}  top3={top3}")
    print("  multivariate in-sample R² (all 32 PC as features):")
    for k, v in mv.items():
        print(f"    {k:22s} = {v:.4f}")


if __name__ == "__main__":
    main()
