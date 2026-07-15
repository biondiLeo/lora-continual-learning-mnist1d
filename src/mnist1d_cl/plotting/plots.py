"""Figure generation with fixed per-class colors and shared axis limits.

No visual manipulation: the same axis limits and the same class->color map are
used across compared models so that separation is comparable, never inflated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..utils.io import ensure_dir  # noqa: E402

# Fixed color per class 0..9 (tab10).
CLASS_COLORS = {i: c for i, c in enumerate(plt.get_cmap("tab10").colors)}


def common_limits(coords_list: Sequence[np.ndarray], pad: float = 0.05
                  ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    allc = np.concatenate([c for c in coords_list if len(c)], axis=0)
    xmin, ymin = allc.min(axis=0)
    xmax, ymax = allc.max(axis=0)
    dx, dy = (xmax - xmin) or 1.0, (ymax - ymin) or 1.0
    return ((xmin - pad * dx, xmax + pad * dx), (ymin - pad * dy, ymax + pad * dy))


def _scatter(ax, coords: np.ndarray, labels: np.ndarray, title: str,
             xlim=None, ylim=None, classes: Optional[Sequence[int]] = None) -> None:
    if classes is None:
        classes = sorted(np.unique(labels).tolist())
    for c in classes:
        m = labels == c
        if m.any():
            ax.scatter(coords[m, 0], coords[m, 1], s=10, alpha=0.7,
                       color=CLASS_COLORS[int(c) % 10], label=str(int(c)))
    ax.set_title(title, fontsize=9)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xticks([]); ax.set_yticks([])


def panel_scatter(panels: List[Tuple[np.ndarray, np.ndarray, str]],
                  save_path: str | Path, suptitle: str = "",
                  share_limits: bool = True) -> Path:
    """Side-by-side scatter panels with shared limits and one legend."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6), squeeze=False)
    xlim = ylim = None
    if share_limits:
        xlim, ylim = common_limits([p[0] for p in panels])
    classes = sorted(set(int(c) for _, lab, _ in panels for c in np.unique(lab)))
    for ax, (coords, labels, title) in zip(axes[0], panels):
        _scatter(ax, coords, labels, title, xlim, ylim, classes)
    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=min(len(lbls), 10),
               fontsize=8, title="classe", bbox_to_anchor=(0.5, -0.02))
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_confusion(cm: np.ndarray, save_path: str | Path, title: str = "",
                   class_labels: Optional[Sequence[int]] = None) -> Path:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    n = cm.shape[0]
    ticks = list(range(n)) if class_labels is None else list(class_labels)
    ax.set_xticks(range(n)); ax.set_xticklabels(ticks, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(ticks, fontsize=7)
    ax.set_xlabel("predetto"); ax.set_ylabel("vero")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path


def plot_prediction_histogram(counts: Sequence[float], save_path: str | Path,
                              title: str = "") -> Path:
    counts = np.asarray(counts)
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.bar(range(len(counts)), counts,
           color=[CLASS_COLORS[i % 10] for i in range(len(counts))])
    ax.set_xlabel("classe predetta"); ax.set_ylabel("conteggio")
    ax.set_xticks(range(len(counts)))
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path


def plot_svd_spectrum(sv_before: Sequence[float], sv_after: Sequence[float],
                      save_path: str | Path, title: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(range(1, len(sv_before) + 1), sv_before, "o-", label="Task A")
    ax.plot(range(1, len(sv_after) + 1), sv_after, "s--", label="dopo Task B")
    ax.set_xlabel("indice valore singolare"); ax.set_ylabel("valore singolare")
    ax.set_title(title, fontsize=9); ax.legend(fontsize=8)
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path


def plot_lines(series: dict, save_path: str | Path, title: str = "",
               xlabel: str = "", ylabel: str = "", logy: bool = False,
               markers: bool = True) -> Path:
    """Plot several labelled y-series (x = 1..len). ``series`` maps label -> y."""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    style = "o-" if markers else "-"
    for label, y in series.items():
        y = np.asarray(y)
        ax.plot(np.arange(1, len(y) + 1), y, style, ms=3, lw=1.2, label=label)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9); ax.legend(fontsize=7)
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path


def plot_roc(curves: dict, save_path: str | Path, title: str = "") -> Path:
    """Plot one or more ROC curves. ``curves`` maps label -> (fpr, tpr, auroc)."""
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    for label, (fpr, tpr, au) in curves.items():
        ax.plot(fpr, tpr, label=f"{label} (AUROC={au:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("FPR (ID)"); ax.set_ylabel("TPR (OOD)")
    ax.set_title(title, fontsize=9); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path


def plot_similarity_distributions(same_sample, same_class_nonself, diff_class,
                                  save_path: str | Path, title: str = "") -> Path:
    """Overlay the three retrieval similarity distributions."""
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    bins = np.linspace(-1, 1, 60)
    if len(same_sample):
        ax.hist(same_sample, bins=bins, alpha=0.55, density=True, label="same-sample")
    ax.hist(same_class_nonself, bins=bins, alpha=0.55, density=True, label="same-class non-self")
    ax.hist(diff_class, bins=bins, alpha=0.55, density=True, label="different-class")
    ax.set_xlabel("cosine similarity"); ax.set_ylabel("densita")
    ax.set_title(title, fontsize=9); ax.legend(fontsize=8)
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path


def plot_score_distributions(id_scores: np.ndarray, ood_scores: np.ndarray,
                             save_path: str | Path, title: str = "",
                             score_name: str = "score") -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.hist(id_scores, bins=30, alpha=0.6, label="ID (0-7)", density=True)
    ax.hist(ood_scores, bins=30, alpha=0.6, label="OOD (8-9)", density=True)
    ax.set_xlabel(f"{score_name} (alto = piu OOD)"); ax.set_ylabel("densita")
    ax.set_title(title, fontsize=9); ax.legend(fontsize=8)
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path
