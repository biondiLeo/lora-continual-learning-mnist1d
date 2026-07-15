"""Internal representation analysis: separation, drift, PCA."""

from .features import (
    feature_stats,
    effective_rank,
    separation_metrics,
    drift_metrics,
    fit_common_pca,
    pca_transform,
)
from .similarity import (
    linear_cka,
    pairwise_distance_correlation,
    procrustes_residual,
    representation_similarity,
)

__all__ = [
    "feature_stats",
    "effective_rank",
    "separation_metrics",
    "drift_metrics",
    "fit_common_pca",
    "pca_transform",
    "linear_cka",
    "pairwise_distance_correlation",
    "procrustes_residual",
    "representation_similarity",
]
