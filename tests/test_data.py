"""Tests #3, #4, #25: class splits, sample-ID stability, normalization no-leakage."""

import pickle
from pathlib import Path

import numpy as np

from mnist1d_cl.constants import TASK_A_CLASSES, TASK_B_CLASSES, OOD_CLASSES
from mnist1d_cl.data import build_datasets
from mnist1d_cl.data.dataset import Normalizer, _stratified_val_indices


def test_class_split_correct(datasets):
    a = datasets.task("train", TASK_A_CLASSES)
    b = datasets.task("train", TASK_B_CLASSES)
    ood = datasets.task("train", OOD_CLASSES)
    assert set(np.unique(a.y)).issubset(set(TASK_A_CLASSES))
    assert set(np.unique(b.y)).issubset(set(TASK_B_CLASSES))
    assert set(np.unique(ood.y)).issubset(set(OOD_CLASSES))
    # all ten classes appear across the full train split
    assert set(np.unique(datasets.train.y)) == set(range(10))
    # validation is stratified: every class present
    assert set(np.unique(datasets.val.y)) == set(range(10))
    # train and val are disjoint by ID
    assert len(set(datasets.train.ids) & set(datasets.val.ids)) == 0


def test_sample_id_stability(datasets):
    ds2 = build_datasets(data_dir="data", data_seed=1234, val_fraction=0.15)
    assert np.array_equal(datasets.train.ids, ds2.train.ids)
    assert np.array_equal(datasets.val.ids, ds2.val.ids)
    assert np.array_equal(datasets.test.ids, ds2.test.ids)
    # IDs unique within each split and correctly formatted
    for split in (datasets.train, datasets.val, datasets.test):
        assert len(set(split.ids)) == len(split.ids)
    assert all(i.startswith("test_") for i in datasets.test.ids)


def test_normalization_no_leakage(datasets):
    # Normalizer statistics must come ONLY from Task A train samples.
    raw = pickle.load(open(Path("data") / "mnist1d_data.pkl", "rb"))
    x_tr = np.asarray(raw["x"], dtype=np.float64)
    y_tr = np.asarray(raw["y"], dtype=np.int64)
    val_idx = _stratified_val_indices(y_tr, datasets.val_fraction, datasets.data_seed)
    val_mask = np.zeros(x_tr.shape[0], dtype=bool)
    val_mask[val_idx] = True
    taskA_train_mask = (~val_mask) & np.isin(y_tr, TASK_A_CLASSES)
    ref = Normalizer.fit(x_tr[taskA_train_mask])
    assert np.allclose(ref.mean, datasets.normalizer.mean)
    assert np.allclose(ref.std, datasets.normalizer.std)

    # Task A train features (normalized) have ~zero mean / unit std by construction.
    a_train = datasets.task("train", TASK_A_CLASSES).x
    assert np.abs(a_train.mean(axis=0)).max() < 1e-4
    assert np.abs(a_train.std(axis=0) - 1.0).max() < 1e-3
