"""Tests #12-#15, #26: query-gallery, self-match, LOO, duplicates, 3 sim types."""

import numpy as np

from mnist1d_cl.querygallery import (query_gallery_eval, audit_query_gallery,
                                     audit_hundred_percent, retrieval_error_analysis)


def _two_class_features(n=30, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    c0 = np.array([3.0] + [0.0] * (dim - 1))
    c1 = np.array([0.0, 3.0] + [0.0] * (dim - 2))
    x0 = c0 + 0.1 * rng.normal(size=(n, dim))
    x1 = c1 + 0.1 * rng.normal(size=(n, dim))
    feats = np.concatenate([x0, x1]).astype(np.float64)
    labels = np.array([0] * n + [1] * n)
    ids = np.array([f"s_{i:04d}" for i in range(2 * n)])
    return feats, ids, labels


def test_self_match_old_vs_old_100pct():
    feats, ids, labels = _two_class_features()
    res = query_gallery_eval(feats, ids, labels, feats, ids, labels,
                             protocol="self_match", k=5)
    # #13: identical query==gallery, self is nearest -> same-sample Top-1 = 100%
    assert res["same_sample_top1"] == 1.0
    assert abs(res["mean_same_sample_cosine"] - 1.0) < 1e-6
    # #26: same-class (non-self) more similar than different-class
    assert res["mean_same_class_nonself_cosine"] > res["mean_diff_class_cosine"]
    assert res["margin_same_class_vs_diff"] > 0
    assert res["margin_same_sample_vs_diff"] > 0


def test_leave_one_out():
    feats, ids, labels = _two_class_features()
    res = query_gallery_eval(feats, ids, labels, feats, ids, labels,
                             protocol="leave_one_out", k=5)
    # #14: self removed; same-class retrieval still strong for separable classes
    assert res["same_class_top1"] > 0.9
    assert "note_same_sample" in res
    assert res["n_valid_query_rows"] == len(ids)


def test_duplicate_detection():
    feats, ids, labels = _two_class_features(n=5)
    dup_ids = ids.copy()
    dup_ids[1] = dup_ids[0]  # inject duplicate gallery id
    audit = audit_query_gallery(ids, dup_ids, feats, feats)
    assert audit["gallery_has_duplicate_ids"] is True
    assert audit["n_zero_norm_gallery"] == 0


def test_keys_present():
    feats, ids, labels = _two_class_features(n=6)
    res = query_gallery_eval(feats, ids, labels, feats, ids, labels)
    for key in ["same_sample_top1", "same_class_top1", "recall@5_class",
                "mean_same_sample_cosine", "mean_same_class_nonself_cosine",
                "mean_diff_class_cosine", "nn_mean_similarity"]:
        assert key in res


def test_hundred_percent_audit_old_vs_old():
    feats, ids, labels = _two_class_features()
    res = query_gallery_eval(feats, ids, labels, feats, ids, labels,
                             protocol="self_match", return_matrix=True)
    audit = audit_hundred_percent(res, ids, ids)
    assert audit["is_legitimate_self_match"] is True
    assert audit["gallery_has_duplicates"] is False
    assert abs(audit["self_match_fraction_by_id"] - 1.0) < 1e-9


def test_ids_aligned_audit():
    feats, ids, labels = _two_class_features(n=5)
    a = audit_query_gallery(ids, ids, feats, feats)
    assert a["query_gallery_ids_aligned"] is True
    b = audit_query_gallery(ids[::-1], ids, feats, feats)
    assert b["query_gallery_ids_aligned"] is False


def test_retrieval_error_analysis_low_on_separable():
    feats, ids, labels = _two_class_features()
    res = query_gallery_eval(feats, ids, labels, feats, ids, labels,
                             protocol="leave_one_out", return_matrix=True)
    err = retrieval_error_analysis(labels, labels, res["nn_index"], res["valid_row"])
    assert err["class_error_rate"] < 0.1
    assert err["n_valid"] == len(ids)
