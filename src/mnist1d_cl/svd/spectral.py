"""SVD utilities for comparing weight geometry before vs after Task B.

Sign ambiguity of singular vectors is handled by always using ABS cosine.
When singular values are (nearly) degenerate individual vectors are unstable, so
subspace comparisons (principal angles, projection similarity) are preferred.

Intruder dimension (documented definition): a top-``k_after`` singular direction
of ``W_after`` whose maximum |cosine| against ALL top-``k_before`` singular
directions of ``W_before`` is below a threshold ``tau``. A sensitivity analysis
over several ``tau`` values is provided.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.linalg import subspace_angles


def svd(W: np.ndarray):
    """Return (U, S, Vt) with U (out, r), S (r,), Vt (r, in)."""
    W = np.asarray(W, dtype=np.float64)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    return U, S, Vt


def singular_values(W: np.ndarray) -> np.ndarray:
    return svd(W)[1]


def top_left_subspace(W: np.ndarray, k: int) -> np.ndarray:
    U = svd(W)[0]
    return U[:, :k]


def top_right_subspace(W: np.ndarray, k: int) -> np.ndarray:
    Vt = svd(W)[2]
    return Vt[:k, :].T


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between column spaces of A and B."""
    return np.asarray(subspace_angles(A, B))[::-1]  # subspace_angles returns descending


def subspace_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Projection metric ||A^T B||_F^2 / min(kA, kB) in [0, 1] (1 = identical)."""
    # Ensure orthonormal columns.
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    m = min(Qa.shape[1], Qb.shape[1])
    return float((np.linalg.norm(Qa.T @ Qb, "fro") ** 2) / max(m, 1))


def singular_vector_alignment(W_before: np.ndarray, W_after: np.ndarray,
                              k: int, side: str = "left") -> Dict[str, object]:
    """|cosine| between matched top-k singular vectors (diagonal) + full matrix."""
    if side == "left":
        Vb, Va = top_left_subspace(W_before, k), top_left_subspace(W_after, k)
    elif side == "right":
        Vb, Va = top_right_subspace(W_before, k), top_right_subspace(W_after, k)
    else:
        raise ValueError("side must be 'left' or 'right'")
    M = np.abs(Vb.T @ Va)                      # (k, k) |cos|
    return {"diagonal_abs_cos": np.diag(M).tolist(), "abs_cos_matrix": M.tolist()}


def intruder_dimensions(W_before: np.ndarray, W_after: np.ndarray,
                        k_after: int, k_before: int, tau: float = 0.5,
                        side: str = "left") -> List[Dict[str, object]]:
    """Identify intruder singular directions of ``W_after`` (see module doc)."""
    if side == "left":
        Vb = top_left_subspace(W_before, k_before)
        Ua, Sa, _ = svd(W_after)
        Va = Ua[:, :k_after]
    elif side == "right":
        Vb = top_right_subspace(W_before, k_before)
        _, Sa, Vta = svd(W_after)
        Va = Vta[:k_after, :].T
    else:
        raise ValueError("side must be 'left' or 'right'")

    align = np.abs(Va.T @ Vb)                  # (k_after, k_before)
    max_align = align.max(axis=1)
    out: List[Dict[str, object]] = []
    for j in range(k_after):
        out.append({
            "index": int(j),
            "singular_value": float(Sa[j]),
            "max_abs_cos_with_before": float(max_align[j]),
            "is_intruder": bool(max_align[j] < tau),
        })
    return out


def intruder_sensitivity(W_before: np.ndarray, W_after: np.ndarray,
                         k_after: int, k_before: int,
                         taus: Sequence[float] = (0.3, 0.5, 0.7),
                         side: str = "left") -> Dict[str, int]:
    """Count intruders as a function of the threshold ``tau``."""
    dims = intruder_dimensions(W_before, W_after, k_after, k_before,
                               tau=max(taus), side=side)
    max_align = np.array([d["max_abs_cos_with_before"] for d in dims])
    return {f"tau={t}": int((max_align < t).sum()) for t in taus}


def effective_rank_sv(s: np.ndarray, eps: float = 1e-12) -> float:
    s = np.asarray(s, dtype=np.float64)
    s = s[s > eps]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def stable_rank_sv(s: np.ndarray, eps: float = 1e-12) -> float:
    s = np.asarray(s, dtype=np.float64)
    if s.size == 0 or s[0] <= eps:
        return 0.0
    return float((s ** 2).sum() / (s[0] ** 2))


def numerical_rank_sv(s: np.ndarray, rtol: float = 1e-6) -> int:
    """Numerical rank = #{singular values > rtol * largest} (rtol documented)."""
    s = np.asarray(s, dtype=np.float64)
    if s.size == 0 or s[0] == 0:
        return 0
    return int((s > rtol * s[0]).sum())


def rank_for_energy(s: np.ndarray, frac: float) -> int:
    s = np.asarray(s, dtype=np.float64)
    total = (s ** 2).sum()
    if total <= 0:
        return 0
    cum = np.cumsum(s ** 2) / total
    return int(np.searchsorted(cum, frac) + 1)


def update_metrics(DeltaW: np.ndarray, W_old: np.ndarray,
                   ranks=(2, 4, 8, 16), rtol: float = 1e-6) -> Dict[str, object]:
    """Structural metrics of a weight update DeltaW relative to W_old."""
    DeltaW = np.asarray(DeltaW, dtype=np.float64)
    W_old = np.asarray(W_old, dtype=np.float64)
    s = np.linalg.svd(DeltaW, compute_uv=False)
    s_old = np.linalg.svd(W_old, compute_uv=False)
    total = float((s ** 2).sum())
    cum = (np.cumsum(s ** 2) / total) if total > 0 else np.zeros_like(s)
    fro_delta = float(np.linalg.norm(DeltaW))
    fro_old = float(np.linalg.norm(W_old))
    spec_delta = float(s[0]) if s.size else 0.0
    spec_old = float(s_old[0]) if s_old.size else 0.0
    rankr = {}
    for r in ranks:
        if r <= s.size:
            energy = float(cum[r - 1])
            tail = float(np.sqrt(max(0.0, 1.0 - energy)))  # rel. error of best rank-r approx
        else:
            energy, tail = 1.0, 0.0
        rankr[int(r)] = {"energy_captured": energy, "approx_rel_error": tail}
    return {
        "singular_values": s.tolist(),
        "normalized_spectrum": (s / s[0]).tolist() if s.size and s[0] > 0 else s.tolist(),
        "cumulative_energy": cum.tolist(),
        "numerical_rank": numerical_rank_sv(s, rtol),
        "numerical_rank_rtol": rtol,
        "effective_rank": effective_rank_sv(s),
        "stable_rank": stable_rank_sv(s),
        "rank90": rank_for_energy(s, 0.90),
        "rank95": rank_for_energy(s, 0.95),
        "fro_ratio_delta_over_old": (fro_delta / fro_old) if fro_old > 0 else float("nan"),
        "spectral_ratio_delta_over_old": (spec_delta / spec_old) if spec_old > 0 else float("nan"),
        "rankr_approx": rankr,
    }


def compare_updates(dA: np.ndarray, dB: np.ndarray, r: int) -> Dict[str, object]:
    """Compare two updates (e.g. DeltaW_FFT vs DeltaW_LoRA) via top-r subspaces."""
    dA = np.asarray(dA, dtype=np.float64)
    dB = np.asarray(dB, dtype=np.float64)
    kk = int(min(r, min(dA.shape), min(dB.shape)))
    la = principal_angles(top_left_subspace(dA, kk), top_left_subspace(dB, kk))
    ra = principal_angles(top_right_subspace(dA, kk), top_right_subspace(dB, kk))
    out = {
        "r": kk,
        "left_subspace_overlap": subspace_similarity(top_left_subspace(dA, kk),
                                                     top_left_subspace(dB, kk)),
        "right_subspace_overlap": subspace_similarity(top_right_subspace(dA, kk),
                                                      top_right_subspace(dB, kk)),
        "left_principal_angle_cos": np.cos(la).tolist(),
        "right_principal_angle_cos": np.cos(ra).tolist(),
    }
    if dA.shape == dB.shape:
        denom = np.linalg.norm(dA) * np.linalg.norm(dB)
        out["frobenius_cosine"] = float((dA * dB).sum() / denom) if denom > 0 else float("nan")
    return out


def compare_layer(W_before: np.ndarray, W_after: np.ndarray, k: int = 4,
                  taus: Sequence[float] = (0.3, 0.5, 0.7)) -> Dict[str, object]:
    """Bundle spectral comparison for one layer (both left and right subspaces)."""
    Sb, Sa = singular_values(W_before), singular_values(W_after)
    kk = int(min(k, Sb.shape[0], Sa.shape[0]))
    left_angles = principal_angles(top_left_subspace(W_before, kk),
                                   top_left_subspace(W_after, kk))
    right_angles = principal_angles(top_right_subspace(W_before, kk),
                                    top_right_subspace(W_after, kk))
    return {
        "singular_values_before": Sb.tolist(),
        "singular_values_after": Sa.tolist(),
        "spectral_l2_change": float(np.linalg.norm(Sb[:kk] - Sa[:kk])),
        "weight_fro_change": float(np.linalg.norm(np.asarray(W_after) - np.asarray(W_before))),
        "k": kk,
        "left_singular_alignment": singular_vector_alignment(W_before, W_after, kk, "left"),
        "right_singular_alignment": singular_vector_alignment(W_before, W_after, kk, "right"),
        "left_principal_angles_rad": left_angles.tolist(),
        "right_principal_angles_rad": right_angles.tolist(),
        "left_subspace_similarity": subspace_similarity(
            top_left_subspace(W_before, kk), top_left_subspace(W_after, kk)),
        "right_subspace_similarity": subspace_similarity(
            top_right_subspace(W_before, kk), top_right_subspace(W_after, kk)),
        "left_intruders": intruder_dimensions(W_before, W_after, kk, kk, 0.5, "left"),
        "left_intruder_sensitivity": intruder_sensitivity(W_before, W_after, kk, kk, taus, "left"),
    }
