"""Representation diagnostics: separation (original space), drift, PCA."""

import numpy as np

from mnist1d_cl.represent import (
    separation_metrics, drift_metrics, fit_common_pca, pca_transform, feature_stats,
)


def _separable(n=40, dim=6, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(3):
        center = np.zeros(dim); center[c] = 5.0
        xs.append(center + 0.2 * rng.normal(size=(n, dim)))
        ys.append(np.full(n, c))
    return np.concatenate(xs), np.concatenate(ys)


def test_separation_metrics_separable():
    x, y = _separable()
    m = separation_metrics(x, y)
    assert m["inter_intra_ratio"] > 1.0
    assert m["nearest_centroid_accuracy"] > 0.95
    assert m["silhouette"] > 0.5


def test_drift_identical():
    x, _ = _separable()
    d = drift_metrics(x, x)
    assert abs(d["mean_cosine_same_sample"] - 1.0) < 1e-6
    assert d["pct_valid_cosine"] == 100.0
    assert d["mean_euclidean_same_sample"] == 0.0


def test_drift_reports_excluded_zero_vectors():
    x, _ = _separable(n=10, dim=4)
    x2 = x.copy()
    x2[0] = 0.0  # degenerate vector
    x_zero = x.copy(); x_zero[0] = 0.0
    d = drift_metrics(x_zero, x2)
    assert d["n_excluded_cosine"] >= 1
    assert d["pct_valid_cosine"] < 100.0


def test_common_pca():
    x, _ = _separable(dim=6)
    pca, evr = fit_common_pca([x], n_components=2)
    coords = pca_transform(pca, x)
    assert coords.shape == (x.shape[0], 2)
    assert len(evr) == 2 and sum(evr) <= 1.0 + 1e-6


def test_feature_stats_zero_fraction():
    x = np.zeros((5, 4))
    s = feature_stats(x)
    assert s["pct_zero_features"] == 100.0
    assert s["pct_zero_norm_samples"] == 100.0
