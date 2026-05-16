"""Probe primitives for B2 and B4.

Provides:
- Utterance-grouped k-fold splitter so that frames from one utterance never
  appear in both train and test.
- Ridge univariate / multivariate R² with CV.
- eta² (correlation ratio) for categorical labels: measures how much a
  scalar feature varies between classes, bounded in [0, 1].
- Bootstrap CI helpers for both.

All probes here are designed for per-PCA-direction analysis in B2, where
each feature is a single scalar. B4 uses these extended to full 384-dim
features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ANATOMY_SEED = 20260510


# ---------------------------------------------------------------------------
# Utterance-grouped fold splitter
# ---------------------------------------------------------------------------


def utterance_kfold_indices(
    utterance_index: np.ndarray,
    n_folds: int = 5,
    seed: int = ANATOMY_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_frame_idx, test_frame_idx) pairs.

    Utterances (unique IDs in `utterance_index`) are randomly partitioned
    into `n_folds` groups. Frames in test fold utterances are never in the
    training fold.
    """
    rng = np.random.default_rng(seed)
    unique_utts = np.unique(utterance_index)
    rng.shuffle(unique_utts)
    fold_utts = np.array_split(unique_utts, n_folds)
    folds = []
    for k in range(n_folds):
        test_utts = set(fold_utts[k].tolist())
        test_mask = np.array([u in test_utts for u in utterance_index])
        train_idx = np.nonzero(~test_mask)[0]
        test_idx = np.nonzero(test_mask)[0]
        folds.append((train_idx, test_idx))
    return folds


# ---------------------------------------------------------------------------
# R² with ridge regression (continuous target)
# ---------------------------------------------------------------------------


def ridge_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Closed-form ridge fit, predict on test.

    X: [N, D], y: [N] continuous. Returns predictions on X_test.
    """
    X_train = np.atleast_2d(X_train)
    X_test = np.atleast_2d(X_test)
    if X_train.ndim == 1:
        X_train = X_train[:, None]
    if X_test.ndim == 1:
        X_test = X_test[:, None]
    # Standardize target (helps with regularization behaviour on scalar features)
    y_mean = float(y_train.mean())
    Xc = X_train - X_train.mean(axis=0, keepdims=True)
    # Append intercept via separate mean handling
    d = Xc.shape[1]
    gram = Xc.T @ Xc + alpha * np.eye(d)
    w = np.linalg.solve(gram, Xc.T @ (y_train - y_mean))
    y_test_hat = (X_test - X_train.mean(axis=0, keepdims=True)) @ w + y_mean
    return y_test_hat


def r2_score_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² = 1 - SS_res / SS_tot. Clip at -1.0 to avoid useless extreme values
    from tiny-variance test folds.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-10:
        return float("nan")
    r2 = 1.0 - ss_res / ss_tot
    return max(r2, -1.0)


def r2_cv(
    X: np.ndarray,
    y: np.ndarray,
    utterance_index: np.ndarray,
    n_folds: int = 5,
    alpha: float = 1.0,
    mask: np.ndarray | None = None,
    seed: int = ANATOMY_SEED,
) -> dict:
    """K-fold CV R² with utterance-level splits.

    mask: optional boolean mask to drop invalid samples (e.g. unvoiced frames
    when y = F0). Applied before folds are constructed so that fold balance is
    on valid samples.
    """
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    keep = mask & np.isfinite(y)
    X_k = X[keep]
    y_k = y[keep]
    utt_k = utterance_index[keep]
    if X_k.ndim == 1:
        X_k = X_k[:, None]
    if len(y_k) < n_folds * 2:
        return {"mean": float("nan"), "per_fold": [], "n_samples": int(len(y_k))}

    folds = utterance_kfold_indices(utt_k, n_folds=n_folds, seed=seed)
    fold_r2 = []
    for train_idx, test_idx in folds:
        if len(train_idx) < 2 or len(test_idx) < 2:
            fold_r2.append(float("nan"))
            continue
        y_hat = ridge_fit_predict(X_k[train_idx], y_k[train_idx], X_k[test_idx], alpha=alpha)
        fold_r2.append(r2_score_safe(y_k[test_idx], y_hat))
    arr = np.array(fold_r2, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    return {
        "mean": float(valid.mean()) if valid.size else float("nan"),
        "std": float(valid.std(ddof=1)) if valid.size > 1 else 0.0,
        "per_fold": [float(v) for v in arr],
        "n_samples": int(len(y_k)),
    }


# ---------------------------------------------------------------------------
# eta² (correlation ratio) for categorical targets
# ---------------------------------------------------------------------------


def eta_squared(alpha: np.ndarray, labels: np.ndarray, min_class_size: int = 2) -> float:
    """Single-feature correlation ratio.

    eta² = SS_between / SS_total where SS_between = Σ_k n_k (μ_k − μ)².
    Returns NaN if no class has enough members or total variance is zero.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    labels = np.asarray(labels)
    mu = float(alpha.mean())
    ss_total = float(np.sum((alpha - mu) ** 2))
    if ss_total < 1e-10:
        return float("nan")
    ss_between = 0.0
    valid = False
    for c in np.unique(labels):
        mask = labels == c
        n_c = int(mask.sum())
        if n_c < min_class_size:
            continue
        mu_c = float(alpha[mask].mean())
        ss_between += n_c * (mu_c - mu) ** 2
        valid = True
    if not valid:
        return float("nan")
    return min(max(ss_between / ss_total, 0.0), 1.0)


def eta_squared_cv(
    alpha: np.ndarray,
    labels: np.ndarray,
    utterance_index: np.ndarray,
    n_folds: int = 5,
    mask: np.ndarray | None = None,
    min_class_size: int = 2,
    seed: int = ANATOMY_SEED,
) -> dict:
    """Fold-wise eta². Reports per-fold test-set eta² and the full-data eta².

    Note: unlike regression where the model is trained on train and evaluated
    on test, eta² is an association statistic, not a predictive one. We report
    both the full-data eta² (our primary summary) and the across-fold mean to
    check stability.
    """
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    keep = mask & (labels != -1) & np.isfinite(alpha)
    a = alpha[keep]
    lab = labels[keep]
    utt = utterance_index[keep]
    if len(a) < n_folds * 2:
        return {
            "full": float("nan"),
            "fold_mean": float("nan"),
            "per_fold": [],
            "n_samples": int(len(a)),
            "n_classes": int(len(np.unique(lab))),
        }
    full = eta_squared(a, lab, min_class_size=min_class_size)
    folds = utterance_kfold_indices(utt, n_folds=n_folds, seed=seed)
    fold_vals = []
    for _, test_idx in folds:
        if len(test_idx) < min_class_size * 2:
            fold_vals.append(float("nan"))
            continue
        fold_vals.append(eta_squared(a[test_idx], lab[test_idx], min_class_size=min_class_size))
    arr = np.array(fold_vals, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    return {
        "full": full,
        "fold_mean": float(valid.mean()) if valid.size else float("nan"),
        "fold_std": float(valid.std(ddof=1)) if valid.size > 1 else 0.0,
        "per_fold": [float(v) for v in arr],
        "n_samples": int(len(a)),
        "n_classes": int(len(np.unique(lab))),
    }


__all__ = [
    "utterance_kfold_indices",
    "ridge_fit_predict",
    "r2_score_safe",
    "r2_cv",
    "eta_squared",
    "eta_squared_cv",
]
