"""Block B1 — Global and local spectrum of V16.3 latent.

Produces:
- Full SVD spectrum of frame-level latents
- Entropy rank, participation ratio, rank_90/95/99
- Per-dimension std distribution, dead-dim count
- Local intrinsic dimension via TwoNN (Facco et al. 2017)
- Temporal autocorrelation and effective frame count

Outputs:
- `<out>/b1_spectrum.npz` — numeric arrays
- `<out>/b1_summary.json` — citable summary numbers with CI
- `<out>/figures/b1_cumulative_variance.png`
- `<out>/figures/b1_dim_std_histogram.png`
- `<out>/figures/b1_local_id_histogram.png`
- `<out>/figures/b1_temporal_autocorr.png`

Usage:
  python -m autoencoder.scripts.anatomy.spectrum \
    --checkpoint outputs/models/wav_vae_v16.3_256x_dim384_time_remote_main/best.pt \
    --dataset-name ljspeech \
    --manifest data/ljspeech_train_splits.json \
    --n-samples 128 \
    --segment-samples 65536 \
    --out docs_ae/v16_3_latent_anatomy/data/b1_v16_3_ljspeech
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .common import (
    ANATOMY_SEED,
    bootstrap_ci,
    load_checkpoint_auto,
    load_manifest,
    set_global_seed,
    svd_spectrum,
)
from .dataset import encode_and_label, prepare_samples, stack_frames


# ---------------------------------------------------------------------------
# Local intrinsic dimension (TwoNN)
# ---------------------------------------------------------------------------


def twonn_estimate(
    points: np.ndarray,
    k: int = 2,
    sample_size: int | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """TwoNN intrinsic dimension estimator (Facco et al. 2017).

    The original TwoNN uses the ratio of distances to the 1st and 2nd nearest
    neighbors. We extend with `k` so that multi-scale variants are easy; k=2
    recovers the original.

    This uses the max-likelihood estimator:
        d_hat = N / sum(log(r_{k} / r_{1}))
    where r_1, r_k are distances to the 1st and kth nearest neighbors, and
    the sum is over N points (duplicates/zero distances filtered out).
    """
    rng = rng or np.random.default_rng(ANATOMY_SEED)
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"twonn_estimate expects [N, D], got {x.shape}")
    n = x.shape[0]
    if sample_size is not None and sample_size < n:
        idx = rng.choice(n, size=sample_size, replace=False)
        x = x[idx]
        n = sample_size
    # kth NN requires computing k+1 nearest distances (including self)
    k_search = k + 1
    # Squared L2 via dot products to avoid O(N²D) memory for diff tensors.
    sq_norms = np.sum(x * x, axis=1)  # [N]
    chunk = 512
    ratios = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = x[start:end]
        block_sq = sq_norms[start:end]
        # (a-b)^2 = a^2 + b^2 - 2 a.b
        d2 = block_sq[:, None] + sq_norms[None, :] - 2.0 * (block @ x.T)
        np.maximum(d2, 0.0, out=d2)
        d = np.sqrt(d2)
        # set self-distance to inf so we skip it
        for i in range(end - start):
            d[i, start + i] = np.inf
        part = np.partition(d, kth=k_search - 1, axis=1)[:, :k_search]
        part = np.sort(part, axis=1)
        r1 = part[:, 0]
        rk = part[:, k_search - 1]
        valid = (r1 > 1e-12) & (rk > r1 * (1 + 1e-8))
        ratios.append(rk[valid] / r1[valid])
    ratios_arr = np.concatenate(ratios, axis=0)
    if ratios_arr.size == 0:
        return float("nan")
    # ML estimator for TwoNN generalization: d_hat = n / sum(log(rk / r1))
    # For k=2 this is the standard TwoNN.
    return float(ratios_arr.size / np.sum(np.log(ratios_arr)))


def twonn_bootstrap(
    points: np.ndarray,
    k: int = 2,
    sample_size: int = 2000,
    n_resamples: int = 20,
    seed: int = ANATOMY_SEED,
) -> tuple[float, float, float]:
    """Bootstrap TwoNN by resampling point subsets. Returns (median, q05, q95)."""
    rng = np.random.default_rng(seed)
    estimates = []
    n = points.shape[0]
    actual_n = min(sample_size, n)
    for _ in range(n_resamples):
        idx = rng.choice(n, size=actual_n, replace=False)
        estimates.append(twonn_estimate(points[idx], k=k, rng=rng))
    estimates = np.array([e for e in estimates if np.isfinite(e)])
    if estimates.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    return (
        float(np.median(estimates)),
        float(np.quantile(estimates, 0.05)),
        float(np.quantile(estimates, 0.95)),
    )


# ---------------------------------------------------------------------------
# Temporal autocorrelation
# ---------------------------------------------------------------------------


def temporal_autocorrelation(
    per_utt_latents: list[np.ndarray],
    max_lag: int = 16,
) -> dict[str, np.ndarray]:
    """Compute per-channel autocorrelation averaged across utterances.

    For each utterance, compute zero-mean, unit-std autocorrelation per channel
    for lags 0..max_lag, then average over utterances.

    Returns:
    - "per_channel": [D, max_lag+1]
    - "mean_over_channels": [max_lag+1]
    - "effective_frames": float — harmonic average effective frame count
    """
    # Aggregate lag-wise
    all_acs = []
    all_effective = []
    for z in per_utt_latents:
        T, D = z.shape
        if T < max_lag + 2:
            continue
        zc = z - z.mean(axis=0, keepdims=True)
        denom = np.sum(zc * zc, axis=0) + 1e-30
        utt_ac = np.zeros((D, max_lag + 1), dtype=np.float64)
        utt_ac[:, 0] = 1.0
        for lag in range(1, max_lag + 1):
            num = np.sum(zc[:-lag] * zc[lag:], axis=0)
            utt_ac[:, lag] = num / denom
        all_acs.append(utt_ac)
        # Effective frames using per-utterance utt_ac mean-over-channel
        mean_ac = utt_ac.mean(axis=0)
        weights = 1.0 - np.arange(0, max_lag + 1) / max(T, 1)
        denom_eff = 1.0 + 2.0 * float(np.sum(weights[1:] * mean_ac[1:]))
        all_effective.append(T / max(denom_eff, 1.0))
    if not all_acs:
        return {
            "per_channel": np.zeros((0, max_lag + 1)),
            "mean_over_channels": np.zeros(max_lag + 1),
            "effective_frames": float("nan"),
            "effective_frames_p05": float("nan"),
            "effective_frames_p95": float("nan"),
        }
    stacked = np.stack(all_acs, axis=0)
    per_channel = stacked.mean(axis=0)
    return {
        "per_channel": per_channel,
        "mean_over_channels": per_channel.mean(axis=0),
        "effective_frames": float(np.mean(all_effective)),
        "effective_frames_p05": float(np.quantile(all_effective, 0.05)),
        "effective_frames_p95": float(np.quantile(all_effective, 0.95)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-name", default="ljspeech")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, action="append", default=[])
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--segment-samples", type=int, default=65536)
    parser.add_argument("--crop-mode", default="center", choices=["first", "center", "last"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--twonn-k", type=int, default=2)
    parser.add_argument("--twonn-subsample", type=int, default=4000)
    parser.add_argument("--twonn-bootstrap", type=int, default=20)
    parser.add_argument("--max-lag", type=int, default=16)
    parser.add_argument("--seed", type=int, default=ANATOMY_SEED)
    parser.add_argument("--n-sigma-plot", type=int, default=384)
    parser.add_argument("--model-kind", default="auto", choices=["auto", "v16_3", "v18"])
    parser.add_argument("--tag", default="", help="Optional output suffix for v18 reruns")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    set_global_seed(args.seed)
    out_dir = args.out
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[B1] loading checkpoint {args.checkpoint}")
    bundle = load_checkpoint_auto(args.checkpoint, device, kind=args.model_kind)
    print(
        f"[B1] model loaded: kind={bundle.kind}, latent_dim={bundle.latent_dim}, stride={bundle.total_stride}, "
        f"ckpt_epoch={bundle.checkpoint.get('epoch')}, ckpt_best_snr={bundle.checkpoint.get('best_snr')}"
    )

    print(f"[B1] loading manifest {args.manifest}")
    items = load_manifest(args.manifest)
    print(f"[B1] manifest has {len(items)} items")
    items = prepare_samples(items, args.n_samples, seed=args.seed)
    print(f"[B1] selected {len(items)} samples")

    path_roots = [Path(".").resolve()] + [Path(r) for r in args.path_root]
    pairs, vocab = encode_and_label(
        items,
        bundle=bundle,
        path_roots=path_roots,
        segment_samples=args.segment_samples,
        crop_mode=args.crop_mode,
        build_phoneme_vocab=False,  # B1 does not need phoneme labels
        device=device,
        verbose=args.verbose,
    )
    print(f"[B1] encoded {len(pairs)} / {len(items)} samples")
    if not pairs:
        raise RuntimeError("No samples successfully encoded")

    stacked = stack_frames(pairs, vocab)
    print(
        f"[B1] stacked frames: total={stacked.latent.shape[0]}, "
        f"dim={stacked.latent.shape[1]}, n_utts={stacked.n_utts}"
    )

    # --- SVD spectrum ---
    print(f"[B1] computing frame-level SVD ...")
    frame_spec = svd_spectrum(stacked.latent)
    print(f"[B1] computing utterance-mean SVD ...")
    mean_spec = svd_spectrum(stacked.per_utt_latent_mean)

    # --- Per-dim std ---
    dim_std_frame = stacked.latent.std(axis=0)
    dim_std_mean = stacked.per_utt_latent_mean.std(axis=0)

    # --- Local ID via TwoNN ---
    print(
        f"[B1] estimating local intrinsic dimension "
        f"(TwoNN k={args.twonn_k}, subsample={args.twonn_subsample}, "
        f"bootstrap={args.twonn_bootstrap}) ..."
    )
    local_id_med, local_id_p05, local_id_p95 = twonn_bootstrap(
        stacked.latent,
        k=args.twonn_k,
        sample_size=args.twonn_subsample,
        n_resamples=args.twonn_bootstrap,
        seed=args.seed,
    )

    # --- Temporal autocorrelation ---
    print(f"[B1] computing temporal autocorrelation (max_lag={args.max_lag}) ...")
    per_utt_latents = [p.latent for p in pairs]
    ac_result = temporal_autocorrelation(per_utt_latents, max_lag=args.max_lag)

    # --- Bootstrap CI on the global rank metrics by utterance resampling ---
    print("[B1] bootstrapping rank metrics over utterances ...")
    utt_indices = np.arange(stacked.n_utts)
    rng = np.random.default_rng(args.seed)

    def _rank_stat(name: str, kind: str = "frame"):
        def _fn(idx: np.ndarray) -> float:
            if kind == "frame":
                mask = np.isin(stacked.utterance_index, idx)
                subset = stacked.latent[mask]
            else:
                subset = stacked.per_utt_latent_mean[idx]
            if subset.shape[0] < 2:
                return float("nan")
            s = svd_spectrum(subset)
            return float(s[name])
        return _fn

    n_boot = 20
    frame_rank95_est, frame_rank95_lo, frame_rank95_hi = bootstrap_ci(
        utt_indices, _rank_stat("rank_95", "frame"), n_resamples=n_boot, rng=rng
    )
    frame_entropy_est, frame_entropy_lo, frame_entropy_hi = bootstrap_ci(
        utt_indices, _rank_stat("entropy_rank", "frame"), n_resamples=n_boot, rng=rng
    )
    mean_rank95_est, mean_rank95_lo, mean_rank95_hi = bootstrap_ci(
        utt_indices, _rank_stat("rank_95", "mean"), n_resamples=n_boot, rng=rng
    )

    # --- Save arrays ---
    np.savez_compressed(
        out_dir / f"b1_spectrum{suffix}.npz",
        frame_sigma=frame_spec["sigma"],
        frame_ev=frame_spec["explained_variance_ratio"],
        frame_cum=frame_spec["cumulative_variance"],
        mean_sigma=mean_spec["sigma"],
        mean_ev=mean_spec["explained_variance_ratio"],
        mean_cum=mean_spec["cumulative_variance"],
        dim_std_frame=dim_std_frame,
        dim_std_mean=dim_std_mean,
        temporal_ac_per_channel=ac_result["per_channel"],
        temporal_ac_mean=ac_result["mean_over_channels"],
    )

    # --- Summary JSON ---
    summary = {
        "block": "B1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": bundle.checkpoint.get("epoch"),
        "checkpoint_best_snr": bundle.checkpoint.get("best_snr"),
        "dataset_name": args.dataset_name,
        "manifest": str(args.manifest),
        "n_samples_requested": args.n_samples,
        "n_samples_encoded": len(pairs),
        "n_frames_total": int(stacked.latent.shape[0]),
        "latent_dim": int(stacked.latent.shape[1]),
        "segment_samples": args.segment_samples,
        "crop_mode": args.crop_mode,
        "seed": args.seed,
        "frame": {
            "entropy_rank": frame_spec["entropy_rank"],
            "participation_rank": frame_spec["participation_rank"],
            "rank_90": frame_spec["rank_90"],
            "rank_95": frame_spec["rank_95"],
            "rank_99": frame_spec["rank_99"],
            "rank_95_bootstrap": {
                "point": frame_rank95_est,
                "ci_lo": frame_rank95_lo,
                "ci_hi": frame_rank95_hi,
                "n_boot": n_boot,
            },
            "entropy_rank_bootstrap": {
                "point": frame_entropy_est,
                "ci_lo": frame_entropy_lo,
                "ci_hi": frame_entropy_hi,
                "n_boot": n_boot,
            },
        },
        "mean": {
            "entropy_rank": mean_spec["entropy_rank"],
            "participation_rank": mean_spec["participation_rank"],
            "rank_90": mean_spec["rank_90"],
            "rank_95": mean_spec["rank_95"],
            "rank_99": mean_spec["rank_99"],
            "rank_95_bootstrap": {
                "point": mean_rank95_est,
                "ci_lo": mean_rank95_lo,
                "ci_hi": mean_rank95_hi,
                "n_boot": n_boot,
            },
        },
        "dim_activity": {
            "frame_std_mean": float(dim_std_frame.mean()),
            "frame_std_min": float(dim_std_frame.min()),
            "frame_std_max": float(dim_std_frame.max()),
            "frame_dead_dims_lt_0_01": int((dim_std_frame < 0.01).sum()),
            "frame_low_dims_lt_0_05": int((dim_std_frame < 0.05).sum()),
            "frame_active_dims_gt_0_1": int((dim_std_frame > 0.1).sum()),
            "mean_std_mean": float(dim_std_mean.mean()),
        },
        "local_id": {
            "estimator": "TwoNN",
            "k": args.twonn_k,
            "sample_size": args.twonn_subsample,
            "n_bootstrap": args.twonn_bootstrap,
            "median": local_id_med,
            "p05": local_id_p05,
            "p95": local_id_p95,
            "global_entropy_rank": frame_spec["entropy_rank"],
            "local_global_ratio": local_id_med / frame_spec["entropy_rank"] if frame_spec["entropy_rank"] else float("nan"),
        },
        "temporal": {
            "max_lag": args.max_lag,
            "effective_frames_per_sample": ac_result["effective_frames"],
            "effective_frames_p05": ac_result["effective_frames_p05"],
            "effective_frames_p95": ac_result["effective_frames_p95"],
            "mean_autocorr": ac_result["mean_over_channels"].tolist(),
        },
    }
    with (out_dir / f"b1_summary{suffix}.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- Plots ---
    # (1) Cumulative variance
    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=150)
    n = min(args.n_sigma_plot, len(frame_spec["cumulative_variance"]))
    ax.plot(np.arange(1, n + 1), frame_spec["cumulative_variance"][:n], label="frame", color="steelblue")
    ax.plot(
        np.arange(1, min(n, len(mean_spec["cumulative_variance"])) + 1),
        mean_spec["cumulative_variance"][: min(n, len(mean_spec["cumulative_variance"]))],
        label="utterance mean",
        color="darkorange",
    )
    ax.axhline(0.95, color="red", linestyle="--", lw=1, alpha=0.6)
    ax.set_xlabel("Component index")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title(f"V16.3 latent spectrum ({args.dataset_name}, {len(pairs)} samples)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / f"b1_cumulative_variance{suffix}.png")
    plt.close(fig)

    # (2) Per-dim std histogram
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=150)
    axes[0].hist(dim_std_frame, bins=40, color="steelblue", edgecolor="black", alpha=0.85)
    axes[0].set_title("frame-level per-dim std")
    axes[0].set_xlabel("std")
    axes[0].set_ylabel("count")
    axes[0].grid(alpha=0.3)
    axes[0].axvline(0.01, color="red", linestyle="--", lw=1)
    axes[1].hist(dim_std_mean, bins=40, color="darkorange", edgecolor="black", alpha=0.85)
    axes[1].set_title("utterance-mean per-dim std")
    axes[1].set_xlabel("std")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / f"b1_dim_std_histogram{suffix}.png")
    plt.close(fig)

    # (3) Local ID histogram-like: we only have aggregated stat, so plot median + CI band
    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=150)
    ax.bar(["local ID"], [local_id_med], yerr=[[local_id_med - local_id_p05], [local_id_p95 - local_id_med]], capsize=8, color="teal", alpha=0.85)
    ax.axhline(frame_spec["entropy_rank"], color="red", linestyle="--", lw=1, label=f"global entropy rank {frame_spec['entropy_rank']:.1f}")
    ax.set_ylabel("dimension")
    ax.set_title(f"TwoNN local ID vs global entropy rank")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures_dir / f"b1_local_id{suffix}.png")
    plt.close(fig)

    # (4) Temporal autocorrelation
    fig, ax = plt.subplots(figsize=(7.5, 3.5), dpi=150)
    lags = np.arange(0, ac_result["mean_over_channels"].shape[0])
    ax.plot(lags, ac_result["mean_over_channels"], marker="o", color="purple")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("lag (frames)")
    ax.set_ylabel("mean autocorrelation")
    ax.set_title(
        f"Temporal autocorrelation — effective frames per sample ≈ {ac_result['effective_frames']:.1f}"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / f"b1_temporal_autocorr{suffix}.png")
    plt.close(fig)

    # --- Console summary ---
    print("\n[B1] === SUMMARY ===")
    print(f"  frame entropy_rank  = {frame_spec['entropy_rank']:.2f}")
    print(
        f"  frame rank_95       = {frame_spec['rank_95']}  "
        f"(boot CI {frame_rank95_lo:.1f}-{frame_rank95_hi:.1f})"
    )
    print(f"  frame rank_99       = {frame_spec['rank_99']}")
    print(f"  mean  rank_95       = {mean_spec['rank_95']}  "
          f"(boot CI {mean_rank95_lo:.1f}-{mean_rank95_hi:.1f})")
    print(f"  local ID (TwoNN)    = {local_id_med:.2f}  (5-95% {local_id_p05:.2f}-{local_id_p95:.2f})")
    print(
        f"  local/global ratio  = {summary['local_id']['local_global_ratio']:.3f}"
    )
    print(
        f"  effective frames    = {ac_result['effective_frames']:.1f}  "
        f"(5-95% {ac_result['effective_frames_p05']:.1f}-{ac_result['effective_frames_p95']:.1f})"
    )
    print(f"  output: {out_dir}")


if __name__ == "__main__":
    main()
