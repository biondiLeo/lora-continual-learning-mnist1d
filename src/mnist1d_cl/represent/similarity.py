"""Representation-similarity metrics between two feature sets of the SAME samples.

Used to compare the backbone (Task A) features with the FFT/LoRA features on the
same Task A samples (rows aligned). All metrics assume identical row ordering.

  * linear_cka                : Centered Kernel Alignment (1 = identical up to
                                orthogonal transf. + isotropic scaling; 0 = unrelated).
  * pairwise_distance_correlation : Pearson correlation of the pairwise Euclidean
                                distance matrices (relational geometry preservation).
  * procrustes_residual       : orthogonal-Procrustes disparity after standardizing
                                (0 = same shape; higher = more distortion).
  * mean_cosine_same_sample   : mean per-sample cosine (same feature dim required).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.spatial import procrustes


def _center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two (N x d) feature matrices (columns centered)."""
    X = _center(np.asarray(X, dtype=np.float64))
    Y = _center(np.asarray(Y, dtype=np.float64))
    hsic = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    denom = (np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro"))
    return float(hsic / denom) if denom > 0 else float("nan")


def _pairwise_dists(X: np.ndarray) -> np.ndarray:
    # condensed upper-triangle pairwise Euclidean distances
    diff = X[:, None, :] - X[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    iu = np.triu_indices(X.shape[0], k=1)
    return d[iu]


def pairwise_distance_correlation(X: np.ndarray, Y: np.ndarray,
                                  max_n: int = 800, seed: int = 0) -> float:
    """Pearson correlation between the pairwise-distance vectors of X and Y.

    Subsamples to ``max_n`` rows for tractability (deterministic given seed).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]
    if n > max_n:
        idx = np.random.default_rng(seed).choice(n, size=max_n, replace=False)
        idx.sort()
        X, Y = X[idx], Y[idx]
    dx, dy = _pairwise_dists(X), _pairwise_dists(Y)
    if dx.std() == 0 or dy.std() == 0:
        return float("nan")
    return float(np.corrcoef(dx, dy)[0, 1])


def procrustes_residual(X: np.ndarray, Y: np.ndarray) -> float:
    """Orthogonal Procrustes disparity (0 = identical shape). Requires same dim."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape != Y.shape:
        return float("nan")
    try:
        _m1, _m2, disparity = procrustes(X, Y)
        return float(disparity)
    except Exception:
        return float("nan")


def _mean_cosine_same_sample(X: np.ndarray, Y: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    nx = np.linalg.norm(X, axis=1)
    ny = np.linalg.norm(Y, axis=1)
    valid = (nx >= eps) & (ny >= eps)
    cos = np.full(X.shape[0], np.nan)
    cos[valid] = (X[valid] * Y[valid]).sum(axis=1) / (nx[valid] * ny[valid])
    return {"mean_cosine_same_sample": float(np.nanmean(cos)) if valid.any() else float("nan"),
            "pct_valid_cosine": float(100.0 * valid.mean())}


def representation_similarity(X_ref: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
    """Bundle of representation-similarity metrics between reference X_ref and Y."""
    out = {
        "linear_cka": linear_cka(X_ref, Y),
        "pairwise_distance_correlation": pairwise_distance_correlation(X_ref, Y),
        "procrustes_residual": procrustes_residual(X_ref, Y),
        "n": int(X_ref.shape[0]),
    }
    out.update(_mean_cosine_same_sample(np.asarray(X_ref, float), np.asarray(Y, float)))
    return out
