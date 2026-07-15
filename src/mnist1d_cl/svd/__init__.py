"""Spectral (SVD) analysis of weight matrices before/after Task B."""

from .spectral import (
    svd,
    singular_values,
    top_left_subspace,
    top_right_subspace,
    principal_angles,
    subspace_similarity,
    singular_vector_alignment,
    intruder_dimensions,
    intruder_sensitivity,
    compare_layer,
    effective_rank_sv,
    stable_rank_sv,
    numerical_rank_sv,
    rank_for_energy,
    update_metrics,
    compare_updates,
)

__all__ = [
    "svd",
    "singular_values",
    "top_left_subspace",
    "top_right_subspace",
    "principal_angles",
    "subspace_similarity",
    "singular_vector_alignment",
    "intruder_dimensions",
    "intruder_sensitivity",
    "compare_layer",
    "effective_rank_sv",
    "stable_rank_sv",
    "numerical_rank_sv",
    "rank_for_energy",
    "update_metrics",
    "compare_updates",
]
