"""OOD detection metrics. Positive class = OOD; higher score => more OOD.

AUROC / AUPR use sklearn. FPR@TPR is computed manually so its orientation is
explicit and testable: we find the score threshold at which the OOD true-positive
rate reaches ``tpr_target`` and report the ID false-positive rate there.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def auroc(scores: np.ndarray, is_ood: np.ndarray) -> float:
    is_ood = np.asarray(is_ood).astype(int)
    if len(np.unique(is_ood)) < 2:
        return float("nan")
    return float(roc_auc_score(is_ood, scores))


def aupr(scores: np.ndarray, is_ood: np.ndarray) -> float:
    """AUPR with OOD as the positive class (alias of aupr_out)."""
    is_ood = np.asarray(is_ood).astype(int)
    if len(np.unique(is_ood)) < 2:
        return float("nan")
    return float(average_precision_score(is_ood, scores))


def aupr_out(scores: np.ndarray, is_ood: np.ndarray) -> float:
    """AUPR treating OOD as positive; score is OOD-ness (higher == more OOD)."""
    return aupr(scores, is_ood)


def aupr_in(scores: np.ndarray, is_ood: np.ndarray) -> float:
    """AUPR treating ID as positive; ID-ness score = -OOD-ness."""
    is_ood = np.asarray(is_ood).astype(int)
    if len(np.unique(is_ood)) < 2:
        return float("nan")
    is_id = 1 - is_ood
    return float(average_precision_score(is_id, -np.asarray(scores)))


def fpr_at_tpr(scores: np.ndarray, is_ood: np.ndarray,
               tpr_target: float = 0.95) -> float:
    """FPR (on ID) at the threshold where OOD TPR >= ``tpr_target``.

    Higher score => more OOD, so we detect OOD as ``score >= threshold``.
    """
    scores = np.asarray(scores, dtype=float)
    is_ood = np.asarray(is_ood).astype(bool)
    pos = scores[is_ood]      # OOD
    neg = scores[~is_ood]     # ID
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # threshold = the (1 - tpr_target) quantile of positive scores from the top:
    # we want TPR >= target, i.e. threshold such that fraction(pos >= thr) >= target.
    thr = np.quantile(pos, 1.0 - tpr_target, method="lower")
    fpr = float((neg >= thr).mean())
    return fpr


def classify_direction(auroc_val: float, cap_threshold: float = 0.55) -> str:
    """Classify an OOD score (orientation FIXED a priori as higher==more OOD).

    Distinguishes three cases (sign-convention inversion is ruled out by
    construction, since orientations follow the standard definitions):
      * consistent_with_definition : AUROC >= 0.5 and separable;
      * empirical_ID_OOD_inversion : AUROC < 0.5 but separable (the score orders
        OOD as more ID-like) — NOT a sign error, NOT absence of capacity;
      * no_discriminative_capacity : separability (max(AUROC,1-AUROC)) < threshold.
    """
    if not np.isfinite(auroc_val):
        return "undefined"
    sep = max(auroc_val, 1.0 - auroc_val)
    if sep < cap_threshold:
        return "no_discriminative_capacity"
    return "consistent_with_definition" if auroc_val >= 0.5 else "empirical_ID_OOD_inversion"


def evaluate_ood(id_scores: np.ndarray, ood_scores: np.ndarray,
                 tpr_target: float = 0.95) -> Dict[str, float]:
    """All scores must be oriented A PRIORI so higher == more OOD.

    Positive class: OOD for AUROC / AUPR-OUT / FPR@TPR; ID for AUPR-IN.
    ``separability_auroc`` = max(AUROC, 1-AUROC) is a DIAGNOSTIC only; the
    reported AUROC keeps the fixed orientation and is never flipped to inflate it.
    """
    scores = np.concatenate([id_scores, ood_scores])
    is_ood = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    mean_id = float(np.mean(id_scores)) if len(id_scores) else float("nan")
    mean_ood = float(np.mean(ood_scores)) if len(ood_scores) else float("nan")
    au = auroc(scores, is_ood)
    return {
        "auroc": au,                       # positive=OOD, fixed-orientation score
        "aupr_in": aupr_in(scores, is_ood),   # positive=ID,  score=-OOD-score
        "aupr_out": aupr_out(scores, is_ood),  # positive=OOD, score=OOD-score
        f"fpr@{int(tpr_target * 100)}tpr": fpr_at_tpr(scores, is_ood, tpr_target),
        "separability_auroc": float(max(au, 1.0 - au)) if np.isfinite(au) else float("nan"),
        "direction_class": classify_direction(au),
        "mean_id_score": mean_id,
        "mean_ood_score": mean_ood,
        "direction_ok": bool(mean_ood >= mean_id),
        "n_id": int(len(id_scores)),
        "n_ood": int(len(ood_scores)),
    }
