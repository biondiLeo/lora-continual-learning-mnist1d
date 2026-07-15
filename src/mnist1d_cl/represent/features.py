"""Feature-space diagnostics.

Separation metrics are computed in the ORIGINAL feature space (h2). PCA is used
ONLY for visualization (and any PCA-space metric is reported separately, never
as a replacement for the original-space value).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def effective_rank(features: np.ndarray, center: bool = True,
                   eps: float = 1e-12) -> float:
    """Roy-Vetterli effective rank: exp(entropy of normalized singular values).

    Computed on the (optionally mean-centered) feature matrix. Ranges in
    [1, dim]; a value << dim indicates the representation collapses onto fewer
    effective directions than the nominal feature dimensionality.
    """
    X = np.asarray(features, dtype=np.float64)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    s = s[s > eps]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def feature_stats(features: np.ndarray, near_zero_eps: float = 1e-6) -> Dict[str, float]:
    norms = np.linalg.norm(features, axis=1)
    return {
        "mean_feature_norm": float(norms.mean()),
        "std_feature_norm": float(norms.std()),
        "mean_feature_variance": float(features.var(axis=0).mean()),
        "pct_zero_features": float(np.mean(features == 0.0) * 100.0),
        "pct_near_zero_features": float(np.mean(np.abs(features) < near_zero_eps) * 100.0),
        "pct_zero_norm_samples": float(np.mean(norms < near_zero_eps) * 100.0),
        "n": int(features.shape[0]),
        "dim": int(features.shape[1]),
    }


def _centroids(features: np.ndarray, labels: np.ndarray):
    classes = sorted(np.unique(labels).tolist())
    cents = {c: features[labels == c].mean(axis=0) for c in classes}
    return classes, cents


def separation_metrics(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Class-separation metrics in the original feature space."""
    classes, cents = _centroids(features, labels)
    result: Dict[str, float] = {}

    # intra: mean distance of each sample to its own class centroid.
    intra = []
    for c in classes:
        f = features[labels == c]
        if f.shape[0] > 0:
            intra.append(np.linalg.norm(f - cents[c], axis=1).mean())
    mean_intra = float(np.mean(intra)) if intra else float("nan")

    # inter: mean pairwise distance between class centroids.
    inter_pairs = []
    cl = list(classes)
    for i in range(len(cl)):
        for j in range(i + 1, len(cl)):
            inter_pairs.append(np.linalg.norm(cents[cl[i]] - cents[cl[j]]))
    mean_inter = float(np.mean(inter_pairs)) if inter_pairs else float("nan")

    # nearest-centroid accuracy.
    C = np.stack([cents[c] for c in classes], axis=0)
    d = np.linalg.norm(features[:, None, :] - C[None, :, :], axis=2)
    nn_class = np.array([classes[i] for i in d.argmin(axis=1)])
    nc_acc = float((nn_class == labels).mean())

    # silhouette (needs >=2 classes and >=2 samples per configuration).
    sil = float("nan")
    if len(classes) >= 2 and features.shape[0] > len(classes):
        try:
            sil = float(silhouette_score(features, labels))
        except Exception:
            sil = float("nan")

    result.update({
        "silhouette": sil,
        "mean_intra_class_distance": mean_intra,
        "mean_inter_class_distance": mean_inter,
        "inter_intra_ratio": float(mean_inter / mean_intra) if mean_intra > 0 else float("nan"),
        "nearest_centroid_accuracy": nc_acc,
        "effective_rank": effective_rank(features),
        "n_classes": len(classes),
    })
    result.update(feature_stats(features))
    return result


def drift_metrics(feat_before: np.ndarray, feat_after: np.ndarray,
                  ids_before: Optional[np.ndarray] = None,
                  ids_after: Optional[np.ndarray] = None,
                  eps: float = 1e-8) -> Dict[str, float]:
    """Per-sample feature drift. Assumes rows are aligned (same samples, order).

    Cosine is computed only on samples whose both vectors are non-degenerate;
    the fraction of valid samples is reported.
    """
    if ids_before is not None and ids_after is not None:
        if not np.array_equal(np.asarray(ids_before), np.asarray(ids_after)):
            raise ValueError("drift_metrics requires aligned sample IDs")
    if feat_before.shape != feat_after.shape:
        raise ValueError("feature matrices must have the same shape")

    nb = np.linalg.norm(feat_before, axis=1)
    na = np.linalg.norm(feat_after, axis=1)
    valid = (nb >= eps) & (na >= eps)
    n_total = feat_before.shape[0]

    cos = np.full(n_total, np.nan)
    cos[valid] = (np.sum(feat_before[valid] * feat_after[valid], axis=1)
                  / (nb[valid] * na[valid]))
    euclid = np.linalg.norm(feat_after - feat_before, axis=1)

    return {
        "mean_cosine_same_sample": float(np.nanmean(cos)) if valid.any() else float("nan"),
        "std_cosine_same_sample": float(np.nanstd(cos)) if valid.any() else float("nan"),
        "mean_euclidean_same_sample": float(euclid.mean()),
        "n_total": int(n_total),
        "n_valid_cosine": int(valid.sum()),
        "n_excluded_cosine": int((~valid).sum()),
        "pct_valid_cosine": float(100.0 * valid.mean()),
    }


def fit_common_pca(feature_list: Sequence[np.ndarray], n_components: int = 2):
    """Fit a single PCA on the union of the provided feature matrices."""
    stacked = np.concatenate(list(feature_list), axis=0)
    n_comp = int(min(n_components, stacked.shape[1], stacked.shape[0]))
    pca = PCA(n_components=n_comp)
    pca.fit(stacked)
    return pca, pca.explained_variance_ratio_.tolist()


def pca_transform(pca, features: np.ndarray) -> np.ndarray:
    return pca.transform(features)
