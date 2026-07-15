"""Tests #19, #20: SVD spectra and subspace comparison / intruder dimensions."""

import numpy as np

from mnist1d_cl.svd import (
    singular_values, compare_layer, principal_angles, subspace_similarity,
    intruder_dimensions, top_left_subspace,
)


def test_singular_values_of_diagonal():
    W = np.diag([3.0, 2.0, 1.0, 0.5])
    sv = singular_values(W)
    assert np.allclose(np.sort(sv)[::-1], [3.0, 2.0, 1.0, 0.5])


def test_identical_matrix_no_change():
    W = np.diag([3.0, 2.0, 1.0, 0.5])
    cmp = compare_layer(W, W, k=2)
    assert max(cmp["left_principal_angles_rad"]) < 1e-6
    assert abs(cmp["left_subspace_similarity"] - 1.0) < 1e-6
    assert all(not d["is_intruder"] for d in cmp["left_intruders"])


def test_intruder_detection():
    e = np.eye(4)
    W_before = 3 * np.outer(e[0], e[0]) + 2 * np.outer(e[1], e[1])
    W_after = 3 * np.outer(e[2], e[2]) + 2 * np.outer(e[3], e[3])
    dims = intruder_dimensions(W_before, W_after, k_after=2, k_before=2, tau=0.5)
    assert all(d["is_intruder"] for d in dims)  # orthogonal new directions
    # subspaces are orthogonal -> similarity ~0, angles ~pi/2
    sim = subspace_similarity(top_left_subspace(W_before, 2),
                              top_left_subspace(W_after, 2))
    assert sim < 1e-6
    ang = principal_angles(top_left_subspace(W_before, 2),
                           top_left_subspace(W_after, 2))
    assert np.all(ang > (np.pi / 2 - 1e-6))


def test_intruder_sensitivity_monotone():
    e = np.eye(4)
    W_before = 3 * np.outer(e[0], e[0]) + 2 * np.outer(e[1], e[1])
    # slightly rotated after: still mostly aligned
    W_after = W_before + 0.01 * np.outer(e[2], e[0])
    cmp = compare_layer(W_before, W_after, k=2)
    sens = cmp["left_intruder_sensitivity"]
    counts = [sens[f"tau={t}"] for t in (0.3, 0.5, 0.7)]
    assert counts == sorted(counts)  # non-decreasing in tau
