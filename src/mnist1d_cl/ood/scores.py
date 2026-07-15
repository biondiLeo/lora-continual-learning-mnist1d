"""OOD scores. ALL scores are oriented so higher == more OOD.

  * msp_ood            = 1 - max softmax prob        (ID confident -> low)
  * predictive_entropy = entropy of softmax          (ID confident -> low)
  * energy_score       = -T * logsumexp(logits/T)    (ID -> low/negative)
  * neg_logit_norm     = -||logits||_2               (ID large norm -> low)
  * centroid_distance  = min L2 dist to seen centroids (ID close -> low)
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp

from ..metrics.classification import softmax_np


def msp_ood(logits: np.ndarray) -> np.ndarray:
    return 1.0 - softmax_np(logits).max(axis=1)


def predictive_entropy(logits: np.ndarray) -> np.ndarray:
    p = softmax_np(logits)
    return -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1)


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    T = float(temperature)
    energy = -T * logsumexp(logits / T, axis=1)   # E = -T logsumexp(z/T)
    return energy  # ID -> low (very negative); OOD -> higher


def neg_logit_norm(logits: np.ndarray) -> np.ndarray:
    return -np.linalg.norm(logits, axis=1)


def max_logit_ood(logits: np.ndarray) -> np.ndarray:
    """OOD score from the maximum logit; ID has high max-logit -> negate."""
    return -logits.max(axis=1)


def seen_ood_scores(logits_full: np.ndarray, seen_classes) -> Dict[str, np.ndarray]:
    """OOD scores computed EXCLUSIVELY from the seen-class logits.

    The 8/9 head rows are unsupervised and must NOT enter the scores. All scores
    are oriented so higher == more OOD.
    """
    z = np.asarray(logits_full)[:, list(seen_classes)]
    return {
        "MSP": msp_ood(z),
        "MaxLogit": max_logit_ood(z),
        "Energy_T1": energy_score(z, temperature=1.0),
        "Entropy": predictive_entropy(z),
    }


def centroids_from_features(features: np.ndarray, labels: np.ndarray,
                            classes: Sequence[int]) -> Dict[int, np.ndarray]:
    cents: Dict[int, np.ndarray] = {}
    for c in classes:
        mask = labels == c
        if mask.any():
            cents[int(c)] = features[mask].mean(axis=0)
    return cents


def _dist_to_centroids(features: np.ndarray,
                       centroids: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    classes = sorted(centroids.keys())
    C = np.stack([centroids[c] for c in classes], axis=0)          # (K, d)
    # pairwise euclidean distances (N, K)
    d = np.linalg.norm(features[:, None, :] - C[None, :, :], axis=2)
    nearest_idx = d.argmin(axis=1)
    nearest_class = np.array([classes[i] for i in nearest_idx])
    min_dist = d.min(axis=1)
    return min_dist, nearest_class


def centroid_distance_ood(features: np.ndarray,
                          centroids: Dict[int, np.ndarray]) -> np.ndarray:
    return _dist_to_centroids(features, centroids)[0]


def nearest_seen_class(features: np.ndarray,
                       centroids: Dict[int, np.ndarray]) -> np.ndarray:
    return _dist_to_centroids(features, centroids)[1]


def all_ood_scores(logits: np.ndarray, features: np.ndarray | None = None,
                   centroids: Dict[int, np.ndarray] | None = None,
                   energy_temperature: float = 1.0) -> Dict[str, np.ndarray]:
    scores: Dict[str, np.ndarray] = {
        "msp": msp_ood(logits),
        "entropy": predictive_entropy(logits),
        "energy": energy_score(logits, energy_temperature),
        "neg_logit_norm": neg_logit_norm(logits),
    }
    if features is not None and centroids:
        scores["centroid_distance"] = centroid_distance_ood(features, centroids)
    return scores
