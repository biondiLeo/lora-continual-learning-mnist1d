"""Query-gallery retrieval with L2-normalized cosine similarity.

Three DISTINCT similarity types are reported separately (same-class items are
NOT "negatives" in semantic retrieval):
  * same-sample positive : identical sample (same stable ID);
  * same-class non-self  : different sample, same class;
  * different-class      : different class (the true negatives).

Two protocols:
  * self_match : the gallery contains the query's own sample; same-sample Top-1
    can legitimately be 100% (a feature's nearest neighbour is itself);
  * leave_one_out : the query's own sample is removed from the gallery, so
    same-sample Top-1 is no longer meaningful; report same-class Top-1, class
    Recall@K, and the intra/inter similarity structure instead.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def cosine_sim_matrix(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    return l2_normalize(query) @ l2_normalize(gallery).T


def _mean(a: np.ndarray) -> float:
    return float(a.mean()) if a.size else float("nan")


def query_gallery_eval(query_feats: np.ndarray, query_ids: np.ndarray,
                       query_labels: np.ndarray,
                       gallery_feats: np.ndarray, gallery_ids: np.ndarray,
                       gallery_labels: np.ndarray,
                       protocol: str = "self_match", k: int = 5,
                       return_matrix: bool = False,
                       return_groups: bool = False,
                       group_subsample: int = 20000, group_seed: int = 0
                       ) -> Dict[str, object]:
    """Retrieval metrics for one (query, gallery) pair under a protocol."""
    if protocol not in ("self_match", "leave_one_out"):
        raise ValueError(f"Unknown protocol: {protocol}")

    sim = cosine_sim_matrix(query_feats, gallery_feats)      # (Q, G)
    Q, G = sim.shape

    q_ids = np.asarray(query_ids)
    g_ids = np.asarray(gallery_ids)
    q_lab = np.asarray(query_labels)
    g_lab = np.asarray(gallery_labels)

    same_id = q_ids[:, None] == g_ids[None, :]              # (Q, G)
    same_lab = q_lab[:, None] == g_lab[None, :]             # (Q, G)

    # Retrieval matrix: for LOO, mask out the query's own sample from gallery.
    ret = sim.copy()
    if protocol == "leave_one_out":
        ret[same_id] = -np.inf
    valid_row = np.isfinite(ret).any(axis=1)

    order = np.argsort(-ret, axis=1)                        # descending
    top1 = order[:, 0]
    topk = order[:, :k]

    same_sample_top1 = _mean((g_ids[top1] == q_ids)[valid_row].astype(float))
    same_class_top1 = _mean((g_lab[top1] == q_lab)[valid_row].astype(float))
    # class Recall@k: any of top-k has the same class.
    recall_k = _mean(np.any(g_lab[topk] == q_lab[:, None], axis=1)[valid_row].astype(float))

    # -- similarity structure (uses the raw sim; self excluded where noted) --
    ss_mask = same_id                                       # same-sample
    scn_mask = same_lab & ~same_id                          # same-class non-self
    dc_mask = ~same_lab                                     # different-class
    mean_same_sample = _mean(sim[ss_mask])
    mean_same_class_nonself = _mean(sim[scn_mask])
    mean_diff_class = _mean(sim[dc_mask])

    # nearest-neighbour similarity (cosine of the retrieved top-1), valid rows.
    nn_sim = ret[np.arange(Q), top1]
    nn_mean_similarity = _mean(nn_sim[valid_row])
    top1_same_class = (g_lab[top1] == q_lab)

    out: Dict[str, object] = {
        "protocol": protocol,
        "k": k,
        "n_query": int(Q),
        "n_gallery": int(G),
        "same_sample_top1": same_sample_top1,
        "same_class_top1": same_class_top1,
        f"recall@{k}_class": recall_k,
        "mean_same_sample_cosine": mean_same_sample,
        "mean_same_class_nonself_cosine": mean_same_class_nonself,
        "mean_diff_class_cosine": mean_diff_class,
        "margin_same_sample_vs_diff": (mean_same_sample - mean_diff_class),
        "margin_same_class_vs_diff": (mean_same_class_nonself - mean_diff_class),
        "nn_mean_similarity": nn_mean_similarity,
        "n_valid_query_rows": int(valid_row.sum()),
    }
    if protocol == "leave_one_out":
        out["note_same_sample"] = ("same_sample_top1 not meaningful under "
                                   "leave_one_out (self removed from gallery)")
    if return_matrix:
        out["similarity_matrix"] = sim
        out["nn_index"] = top1
        out["top1_same_class"] = top1_same_class
        out["nn_similarity_per_query"] = nn_sim
        out["valid_row"] = valid_row
    if return_groups:
        rng = np.random.default_rng(group_seed)

        def _sub(a):
            a = np.asarray(a)
            if a.size > group_subsample:
                idx = rng.choice(a.size, size=group_subsample, replace=False)
                return a[idx]
            return a
        out["sim_same_sample"] = _sub(sim[ss_mask])
        out["sim_same_class_nonself"] = _sub(sim[scn_mask])
        out["sim_diff_class"] = _sub(sim[dc_mask])
    return out


def retrieval_error_analysis(query_labels: np.ndarray, gallery_labels: np.ndarray,
                             nn_index: np.ndarray, valid_row: np.ndarray,
                             num_classes: int = 10) -> Dict[str, object]:
    """Analyse same-class retrieval errors: which classes get confused.

    Returns the overall class-error rate and a confusion of query-class ->
    retrieved-neighbour-class among the errors.
    """
    q_lab = np.asarray(query_labels)
    g_lab = np.asarray(gallery_labels)
    nn_lab = g_lab[nn_index]
    valid = np.asarray(valid_row, dtype=bool)
    wrong = valid & (nn_lab != q_lab)
    conf = {}
    for qc in sorted(np.unique(q_lab).tolist()):
        m = wrong & (q_lab == qc)
        if m.any():
            vals, cnts = np.unique(nn_lab[m], return_counts=True)
            conf[int(qc)] = {int(v): int(c) for v, c in zip(vals, cnts)}
    return {
        "class_error_rate": float(wrong[valid].mean()) if valid.any() else float("nan"),
        "n_errors": int(wrong.sum()),
        "n_valid": int(valid.sum()),
        "error_confusion_query_to_nn": conf,
    }


def audit_query_gallery(query_ids: np.ndarray, gallery_ids: np.ndarray,
                        query_feats: np.ndarray, gallery_feats: np.ndarray,
                        eps: float = 1e-6) -> Dict[str, object]:
    """Sanity audit for retrieval, especially around 100% results."""
    g_ids = np.asarray(gallery_ids)
    q_ids = np.asarray(query_ids)
    unique_g, counts_g = np.unique(g_ids, return_counts=True)
    q_norm = np.linalg.norm(query_feats, axis=1)
    g_norm = np.linalg.norm(gallery_feats, axis=1)
    aligned = bool(len(q_ids) == len(g_ids) and np.array_equal(q_ids, g_ids))
    return {
        "n_query": int(len(q_ids)),
        "n_gallery": int(len(g_ids)),
        "gallery_has_duplicate_ids": bool((counts_g > 1).any()),
        "n_duplicate_gallery_ids": int((counts_g > 1).sum()),
        "n_query_ids_in_gallery": int(np.isin(q_ids, g_ids).sum()),
        "n_zero_norm_query": int((q_norm < eps).sum()),
        "n_zero_norm_gallery": int((g_norm < eps).sum()),
        "query_gallery_same_length": bool(len(q_ids) == len(g_ids)),
        "query_gallery_ids_aligned": aligned,
    }


def audit_hundred_percent(result: Dict[str, object], query_ids: np.ndarray,
                          gallery_ids: np.ndarray, atol: float = 1e-9
                          ) -> Dict[str, object]:
    """Audit a (self_match) result whose same-sample Top-1 is ~100%.

    Verifies the result is a legitimate self-match: each query's retrieved
    neighbour is the gallery entry with the SAME id, and there are no gallery
    duplicates. Requires ``result`` produced with return_matrix=True.
    """
    q_ids = np.asarray(query_ids)
    g_ids = np.asarray(gallery_ids)
    nn = np.asarray(result["nn_index"])
    valid = np.asarray(result.get("valid_row", np.ones(len(q_ids), bool)), bool)
    retrieved_ids = g_ids[nn]
    self_match_frac = float((retrieved_ids[valid] == q_ids[valid]).mean()) if valid.any() else float("nan")
    _u, counts = np.unique(g_ids, return_counts=True)
    return {
        "same_sample_top1": float(result.get("same_sample_top1", float("nan"))),
        "self_match_fraction_by_id": self_match_frac,
        "is_legitimate_self_match": bool(abs(self_match_frac - 1.0) <= 1e-6),
        "gallery_has_duplicates": bool((counts > 1).any()),
        "note": ("same-sample Top-1 == 100% is EXPECTED when self-match is "
                 "allowed (a feature's nearest neighbour is itself); it is NOT "
                 "evidence of class separation or generalization."),
    }
