"""Test #11: masked vs non-masked metrics."""

import numpy as np

from mnist1d_cl.metrics import (
    accuracy, per_class_accuracy, confusion_matrix, prediction_distribution,
    forgetting, full_report,
)


def _logits_prefer_ood_then_class2():
    # 3 samples of true class 2; non-masked argmax = 8, masked(0-4) argmax = 2
    logits = np.zeros((3, 10), dtype=np.float64)
    logits[:, 8] = 5.0
    logits[:, 2] = 3.0
    labels = np.array([2, 2, 2])
    return logits, labels


def test_masked_vs_non_masked_accuracy():
    logits, labels = _logits_prefer_ood_then_class2()
    assert accuracy(logits, labels, None) == 0.0
    assert accuracy(logits, labels, [0, 1, 2, 3, 4]) == 1.0


def test_per_class_and_confusion():
    logits, labels = _logits_prefer_ood_then_class2()
    pca = per_class_accuracy(logits, labels, classes=[2], allowed_classes=[0, 1, 2, 3, 4])
    assert pca[2] == 1.0
    cm = confusion_matrix(logits, labels, num_classes=10, allowed_classes=None)
    assert cm[2, 8] == 3  # all predicted as 8 when non-masked
    dist = prediction_distribution(logits, allowed_classes=None)
    assert dist["counts"][8] == 3


def test_forgetting_and_report():
    assert forgetting(0.9, 0.4) == 0.5
    logits, labels = _logits_prefer_ood_then_class2()
    rep = full_report(logits, labels, task_classes=[0, 1, 2, 3, 4])
    assert rep["accuracy_masked"] == 1.0
    assert rep["accuracy_non_masked"] == 0.0
