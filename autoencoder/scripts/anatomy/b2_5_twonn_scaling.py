"""B2.5 S1 — TwoNN N-scaling sanity check.

Runs TwoNN on N ∈ {500, 1000, 2000, 4000, 8000, 16000, 32000} to check
whether the B1 local_ID estimate was truncated by sample size. Also
runs Levina-Bickel MLE as an independent ID estimator, and a synthetic
calibration on `normal(0, I_d)` data with d ∈ {10, 20, 40, 80, 160}.

Outputs:
- <out>/b2_5_twonn_scaling.npz
- <out>/b2_5_twonn_scaling.json
- <out>/figures/twonn_vs_N.png
- <out>/figures/id_estimator_calibration.png

Usage:
  python -m autoencoder.scripts.anatomy.b2_5_twonn_scaling \
    --checkpoint outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt \
    --manifest data/ljspeech_train_splits.json \
    --n-samples 128 --segment-samples 65536 \
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

from .common import ANATOMY_SEED, load_manifest, load_v16_3, set_global_seed
from .dataset import encode_and_label, prepare_samples, stack_frames
from .spectrum import twonn_estimate


# ---------------------------------------------------------------------------
# Levina-Bickel MLE
# ---------------------------------------------------------------------------


def levina_bickel_mle(
    points: np.ndarray,
    k_values: list[int] = (5, 10, 20),
    sample_size: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Levina-Bickel MLE intrinsic dimension estimator.

    For each query point i and its k nearest neighbours at distances
    T_1(i) < T_2(i) < ... < T_k(i):
        m_k(i) = (k - 1)^{-1} Σ_{j=1}^{k-1} log(T_k(i) / T_j(i))
    The MLE is 1 / mean_i m_k(i).

    Returns a dict {"k={k}": estimate, ...} and "mean": cross-k mean.
    """
    rng = rng or np.random.default_rng(ANATOMY_SEED)
    x = np.asarray(points, dtype=np.float64)
    n = x.shape[0]
    if sample_size is not None and sample_size < n:
        idx = rng.choice(n, size=sample_size, replace=False)
        x = x[idx]
        n = sample_size

    k_max = max(k_values)
    chunk = 512
    # Compute per-point sorted distances up to k_max
    per_point_sorted = np.zeros((n, k_max + 1), dtype=np.float64)
    sq_norms = np.sum(x * x, axis=1)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = x[start:end]
        block_sq = sq_norms[start:end]
        d2 = block_sq[:, None] + sq_norms[None, :] - 2.0 * (block @ x.T)
        np.maximum(d2, 0.0, out=d2)
        d = np.sqrt(d2)
        for i in range(end - start):
            d[i, start + i] = np.inf
        part = np.partition(d, kth=k_max, axis=1)[:, : k_max + 1]
        part = np.sort(part, axis=1)
        per_point_sorted[start:end] = part

    out = {}
    for k in k_values:
        # m_k(i) = mean over j=1..k-1 of log(T_k / T_j)
        # T_j is per_point_sorted[:, j-1] ... but the self distance is inf so offset
        # We want nearest neighbor distances T_1 .. T_k. Since we set self to inf,
        # sorted part[:, 0] is T_1, part[:, k-1] is T_k.
        T_k = per_point_sorted[:, k - 1]
        valid = T_k > 1e-12
        m_vals = np.zeros(n, dtype=np.float64)
        for j in range(1, k):
            T_j = per_point_sorted[:, j - 1]
            denom = T_j
            mask = valid & (denom > 1e-12)
            ratio = np.where(mask, T_k / denom, 1.0)
            m_vals += np.log(ratio)
        m_vals = m_vals / (k - 1)
        m_vals = m_vals[valid]
        if m_vals.size == 0 or not np.isfinite(m_vals.mean()):
            out[f"k={k}"] = float("nan")
        else:
            out[f"k={k}"] = float(1.0 / m_vals.mean())
    valid_vals = [v for v in out.values() if np.isfinite(v)]
    out["mean"] = float(np.mean(valid_vals)) if valid_vals else float("nan")
    return out


# ---------------------------------------------------------------------------
# Synthetic calibration
# ---------------------------------------------------------------------------


def synthetic_gaussian_baseline(
    dims: list[int],
    n: int,
    k_twonn: int = 2,
    seed: int = ANATOMY_SEED,
) -> list[dict[str, float]]:
    """For each true d, generate N(0, I_d) points and measure TwoNN + L-B."""
    out = []
    rng = np.random.default_rng(seed)
    for d in dims:
        x = rng.standard_normal(size=(n, d))
        twonn = twonn_estimate(x, k=k_twonn, rng=rng)
        lb = levina_bickel_mle(x, k_values=[10, 20], rng=rng)
        out.append({
            "true_d": d,
            "twonn": twonn,
            "lb_k10": lb["k=10"],
            "lb_k20": lb["k=20"],
            "lb_mean": lb["mean"],
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, action="append", default=[])
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--segment-samples", type=int, default=65536)
    parser.add_argument("--sweep-n", type=str, default="500,1000,2000,4000,8000,16000,32000")
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--calibration-dims", type=str, default="10,20,40,80,160")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=ANATOMY_SEED)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    set_global_seed(args.seed)
    out_dir = args.out
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[B2.5-S1] loading checkpoint {args.checkpoint}")
    bundle = load_v16_3(args.checkpoint, device)
    items = load_manifest(args.manifest)
    items = prepare_samples(items, args.n_samples, seed=args.seed)
    path_roots = [Path(".").resolve()] + [Path(r) for r in args.path_root]
    pairs, vocab = encode_and_label(
        items,
        bundle=bundle,
        path_roots=path_roots,
        segment_samples=args.segment_samples,
        build_phoneme_vocab=False,
        device=device,
        verbose=args.verbose,
    )
    stacked = stack_frames(pairs, vocab)
    all_frames = stacked.latent
    n_total = all_frames.shape[0]
    print(f"[B2.5-S1] collected {n_total} frames")

    sweep_n = [int(v) for v in args.sweep_n.split(",") if v.strip()]
    calibration_dims = [int(v) for v in args.calibration_dims.split(",") if v.strip()]
    rng = np.random.default_rng(args.seed)

    # --- TwoNN N-scaling on V16.3 latent ---
    print(f"[B2.5-S1] TwoNN N-scaling sweep on V16.3 frames: {sweep_n}")
    sweep_results = []
    for n in sweep_n:
        n_eff = min(n, n_total)
        twonn_vals = []
        for rep in range(args.n_repeats):
            idx = rng.choice(n_total, size=n_eff, replace=False)
            vals = twonn_estimate(all_frames[idx], k=2, rng=rng)
            twonn_vals.append(vals)
        # Levina-Bickel only for N >= 500 (cheap enough; skip N > 16000 to save time)
        lb = None
        if n_eff <= 16000:
            sample_for_lb = min(n_eff, 4000)
            idx_lb = rng.choice(n_total, size=sample_for_lb, replace=False)
            lb = levina_bickel_mle(all_frames[idx_lb], k_values=[10, 20], rng=rng)
        arr = np.array(twonn_vals)
        entry = {
            "N": int(n_eff),
            "twonn_median": float(np.median(arr)),
            "twonn_p05": float(np.quantile(arr, 0.05)),
            "twonn_p95": float(np.quantile(arr, 0.95)),
            "lb_k10": lb["k=10"] if lb else None,
            "lb_k20": lb["k=20"] if lb else None,
            "lb_mean": lb["mean"] if lb else None,
        }
        sweep_results.append(entry)
        print(
            f"[B2.5-S1]   N={n_eff}  TwoNN={entry['twonn_median']:.2f} "
            f"(p05-p95 {entry['twonn_p05']:.2f}-{entry['twonn_p95']:.2f})  "
            f"L-B mean={entry['lb_mean']}"
        )

    # --- Synthetic calibration ---
    print(f"[B2.5-S1] synthetic Gaussian calibration: dims={calibration_dims}, N=4000")
    calibration = synthetic_gaussian_baseline(calibration_dims, n=4000, k_twonn=2, seed=args.seed)
    for c in calibration:
        print(
            f"[B2.5-S1]   true d={c['true_d']}  TwoNN={c['twonn']:.2f}  "
            f"L-B k10={c['lb_k10']:.2f}  L-B k20={c['lb_k20']:.2f}"
        )

    # --- Save ---
    summary = {
        "block": "B2.5-S1",
        "checkpoint": str(args.checkpoint),
        "n_samples": len(pairs),
        "n_frames": int(n_total),
        "sweep_n": sweep_n,
        "n_repeats": args.n_repeats,
        "sweep_results": sweep_results,
        "calibration_dims": calibration_dims,
        "calibration": calibration,
    }
    with (out_dir / "b2_5_twonn_scaling.json").open("w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        out_dir / "b2_5_twonn_scaling.npz",
        sweep_n=np.asarray(sweep_n),
        twonn_medians=np.asarray([s["twonn_median"] for s in sweep_results]),
        twonn_p05=np.asarray([s["twonn_p05"] for s in sweep_results]),
        twonn_p95=np.asarray([s["twonn_p95"] for s in sweep_results]),
        lb_k20=np.asarray([s["lb_k20"] if s["lb_k20"] is not None else np.nan for s in sweep_results]),
        cal_dims=np.asarray(calibration_dims),
        cal_twonn=np.asarray([c["twonn"] for c in calibration]),
        cal_lb_k10=np.asarray([c["lb_k10"] for c in calibration]),
        cal_lb_k20=np.asarray([c["lb_k20"] for c in calibration]),
    )

    # --- Plots ---
    # (1) TwoNN vs N on real latent + L-B
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=150)
    ns = [s["N"] for s in sweep_results]
    medians = [s["twonn_median"] for s in sweep_results]
    p05 = [s["twonn_p05"] for s in sweep_results]
    p95 = [s["twonn_p95"] for s in sweep_results]
    ax.plot(ns, medians, marker="o", color="tab:blue", label="TwoNN (k=2)")
    ax.fill_between(ns, p05, p95, color="tab:blue", alpha=0.2)
    lb_vals = [s["lb_k20"] for s in sweep_results]
    # Plot Levina-Bickel only for non-None
    lb_xs = [n for n, v in zip(ns, lb_vals) if v is not None]
    lb_ys = [v for v in lb_vals if v is not None]
    if lb_ys:
        ax.plot(lb_xs, lb_ys, marker="s", color="tab:orange", label="Levina-Bickel (k=20)")
    ax.set_xscale("log")
    ax.set_xlabel("N (number of frames)")
    ax.set_ylabel("Estimated intrinsic dimension")
    ax.set_title("V16.3 local intrinsic dim vs sample size")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "twonn_vs_N.png")
    plt.close(fig)

    # (2) Synthetic calibration: true d vs estimate
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=150)
    dims = [c["true_d"] for c in calibration]
    twonn_estimates = [c["twonn"] for c in calibration]
    lb_estimates = [c["lb_k20"] for c in calibration]
    max_d = max(dims)
    ax.plot([0, max_d], [0, max_d], linestyle="--", color="gray", label="y=x (ideal)")
    ax.plot(dims, twonn_estimates, marker="o", color="tab:blue", label="TwoNN")
    ax.plot(dims, lb_estimates, marker="s", color="tab:orange", label="Levina-Bickel (k=20)")
    ax.set_xlabel("true d (Gaussian dimension)")
    ax.set_ylabel("estimated d")
    ax.set_title("ID estimator calibration on N(0, I_d), N=4000")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "id_estimator_calibration.png")
    plt.close(fig)

    print("\n[B2.5-S1] === SUMMARY ===")
    print(f"  sweep_N       TwoNN (median)  L-B (k=20)")
    for s in sweep_results:
        lb_str = f"{s['lb_k20']:.2f}" if s["lb_k20"] is not None else "    -"
        print(f"  {s['N']:>6d}        {s['twonn_median']:>6.2f}        {lb_str}")
    print(f"\n  calibration: true d -> TwoNN / L-B")
    for c in calibration:
        print(f"    d={c['true_d']:>3d} -> TwoNN={c['twonn']:>6.2f}  L-B={c['lb_k20']:>6.2f}")
    print(f"\n  output: {out_dir}")


if __name__ == "__main__":
    main()
