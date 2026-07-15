"""Tests #16, #17: OOD metrics and score orientation (higher == more OOD)."""

import numpy as np

from mnist1d_cl.ood import (
    msp_ood, energy_score, predictive_entropy, evaluate_ood, auroc,
)


def _id_ood_logits(seed=0):
    rng = np.random.default_rng(seed)
    n_id, n_ood, K = 200, 100, 10
    id_logits = 0.1 * rng.normal(size=(n_id, K))
    id_labels = rng.integers(0, 8, size=n_id)
    id_logits[np.arange(n_id), id_labels] += 6.0  # confident ID
    ood_logits = 0.1 * rng.normal(size=(n_ood, K))  # near-uniform OOD
    return id_logits, ood_logits


def test_msp_orientation_separable():
    id_logits, ood_logits = _id_ood_logits()
    res = evaluate_ood(msp_ood(id_logits), msp_ood(ood_logits))
    # higher msp_ood == more OOD; ID confident -> low, OOD -> high -> AUROC ~1
    assert res["auroc"] > 0.9
    assert res["mean_ood_score"] > res["mean_id_score"]
    assert res["fpr@95tpr"] < 0.2


def test_energy_orientation_separable():
    id_logits, ood_logits = _id_ood_logits()
    res = evaluate_ood(energy_score(id_logits), energy_score(ood_logits))
    assert res["auroc"] > 0.9  # ID low energy, OOD high energy


def test_swapped_orientation_low_auroc():
    id_logits, ood_logits = _id_ood_logits()
    # If orientation were flipped (ID treated as OOD), AUROC collapses.
    res = evaluate_ood(msp_ood(ood_logits), msp_ood(id_logits))
    assert res["auroc"] < 0.1


def test_identical_distributions_half():
    id_logits, _ = _id_ood_logits()
    s = msp_ood(id_logits)
    assert abs(evaluate_ood(s, s)["auroc"] - 0.5) < 1e-6


def test_entropy_nonnegative():
    id_logits, ood_logits = _id_ood_logits()
    assert (predictive_entropy(ood_logits) >= -1e-9).all()
