"""Out-of-distribution analysis for classes 8 and 9 (never trained).

Convention (fixed): every OOD score is oriented so that a HIGHER value means
MORE out-of-distribution.
"""

from .scores import (
    msp_ood,
    predictive_entropy,
    energy_score,
    neg_logit_norm,
    max_logit_ood,
    seen_ood_scores,
    centroids_from_features,
    centroid_distance_ood,
    nearest_seen_class,
    all_ood_scores,
)
from .detection import (auroc, aupr, aupr_in, aupr_out, fpr_at_tpr, evaluate_ood,
                        classify_direction)

__all__ = [
    "msp_ood",
    "predictive_entropy",
    "energy_score",
    "neg_logit_norm",
    "max_logit_ood",
    "seen_ood_scores",
    "centroids_from_features",
    "centroid_distance_ood",
    "nearest_seen_class",
    "all_ood_scores",
    "auroc",
    "aupr",
    "aupr_in",
    "aupr_out",
    "fpr_at_tpr",
    "evaluate_ood",
    "classify_direction",
]
