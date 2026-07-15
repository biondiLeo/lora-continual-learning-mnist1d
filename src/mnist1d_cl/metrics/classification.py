"""Classification metrics computed on numpy logits/labels.

Masked vs non-masked accuracy is the key diagnostic:
  * non-masked: argmax over all 10 outputs (realistic class-incremental setting);
  * masked:     argmax restricted to a task's ``allowed_classes``.
Their gap distinguishes representation loss (both drop) from a decision bias /
head miscalibration (only non-masked drops).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ..constants import NUM_CLASSES


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def predictions(logits: np.ndarray) -> np.ndarray:
    return logits.argmax(axis=1)


def masked_predictions(logits: np.ndarray,
                       allowed_classes: Optional[Sequence[int]]) -> np.ndarray:
    if allowed_classes is None:
        return predictions(logits)
    masked = np.full_like(logits, -np.inf)
    idx = np.asarray(list(allowed_classes), dtype=int)
    masked[:, idx] = logits[:, idx]
    return masked.argmax(axis=1)


def accuracy(logits: np.ndarray, labels: np.ndarray,
             allowed_classes: Optional[Sequence[int]] = None) -> float:
    pred = masked_predictions(logits, allowed_classes)
    return float((pred == labels).mean()) if labels.size else float("nan")


def per_class_accuracy(logits: np.ndarray, labels: np.ndarray,
                       classes: Optional[Sequence[int]] = None,
                       allowed_classes: Optional[Sequence[int]] = None
                       ) -> Dict[int, float]:
    pred = masked_predictions(logits, allowed_classes)
    if classes is None:
        classes = sorted(np.unique(labels).tolist())
    out: Dict[int, float] = {}
    for c in classes:
        mask = labels == c
        out[int(c)] = float((pred[mask] == c).mean()) if mask.any() else float("nan")
    return out


def confusion_matrix(logits: np.ndarray, labels: np.ndarray,
                     num_classes: int = NUM_CLASSES,
                     allowed_classes: Optional[Sequence[int]] = None,
                     normalize: bool = False) -> np.ndarray:
    pred = masked_predictions(logits, allowed_classes)
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(labels, pred):
        cm[int(t), int(p)] += 1.0
    if normalize:
        row = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    return cm


def prediction_distribution(logits: np.ndarray, num_classes: int = NUM_CLASSES,
                            allowed_classes: Optional[Sequence[int]] = None
                            ) -> Dict[str, object]:
    pred = masked_predictions(logits, allowed_classes)
    counts = np.bincount(pred, minlength=num_classes).astype(int)
    total = int(counts.sum())
    frac = counts / total if total else counts.astype(float)
    return {"counts": counts.tolist(), "fractions": frac.tolist(), "total": total}


def forgetting(acc_before: float, acc_after: float) -> float:
    """Forgetting = accuracy drop on an old task (positive => forgetting)."""
    return float(acc_before - acc_after)


def mean_seen_accuracy(logits: np.ndarray, labels: np.ndarray,
                       seen_classes: Sequence[int],
                       allowed_classes: Optional[Sequence[int]] = None) -> float:
    """Mean over per-class accuracies of the seen classes (balanced)."""
    pca = per_class_accuracy(logits, labels, classes=seen_classes,
                             allowed_classes=allowed_classes)
    vals = [v for v in pca.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def confidence_stats(logits: np.ndarray) -> Dict[str, float]:
    p = softmax_np(logits)
    conf = p.max(axis=1)
    logit_norm = np.linalg.norm(logits, axis=1)
    return {
        "mean_confidence": float(conf.mean()),
        "std_confidence": float(conf.std()),
        "mean_logit_norm": float(logit_norm.mean()),
        "mean_max_logit": float(logits.max(axis=1).mean()),
    }


def full_report(logits: np.ndarray, labels: np.ndarray,
                task_classes: Sequence[int],
                num_classes: int = NUM_CLASSES) -> Dict[str, object]:
    """Bundle of metrics for a set of samples (a task's evaluation samples)."""
    return {
        "accuracy_non_masked": accuracy(logits, labels, None),
        "accuracy_masked": accuracy(logits, labels, task_classes),
        "per_class_accuracy_non_masked": per_class_accuracy(
            logits, labels, classes=task_classes, allowed_classes=None),
        "per_class_accuracy_masked": per_class_accuracy(
            logits, labels, classes=task_classes, allowed_classes=task_classes),
        "prediction_distribution_non_masked": prediction_distribution(
            logits, num_classes, None),
        "confusion_non_masked": confusion_matrix(
            logits, labels, num_classes, None).tolist(),
        "confidence": confidence_stats(logits),
        "n": int(labels.size),
    }
