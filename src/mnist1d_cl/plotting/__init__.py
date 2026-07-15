"""Plotting helpers (matplotlib, Agg backend). Fixed class colors, shared axes."""

from .plots import (
    CLASS_COLORS,
    common_limits,
    panel_scatter,
    plot_confusion,
    plot_prediction_histogram,
    plot_svd_spectrum,
    plot_score_distributions,
    plot_similarity_distributions,
    plot_roc,
    plot_lines,
)

__all__ = [
    "CLASS_COLORS",
    "common_limits",
    "panel_scatter",
    "plot_confusion",
    "plot_prediction_histogram",
    "plot_svd_spectrum",
    "plot_score_distributions",
    "plot_similarity_distributions",
    "plot_roc",
    "plot_lines",
]
