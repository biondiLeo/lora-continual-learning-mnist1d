"""Shared fixtures for the test suite."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from mnist1d_cl.models import MLP, MLPConfig
from mnist1d_cl.data.dataset import SplitData


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def small_config() -> MLPConfig:
    return MLPConfig(hidden1=16, hidden2=8)


@pytest.fixture
def base_model(small_config) -> MLP:
    return MLP(small_config)


@pytest.fixture
def cloned_model(base_model):
    """A deep copy of ``base_model`` with identical weights."""
    return copy.deepcopy(base_model)


def make_synthetic_split(n_per_class: int = 20, classes=range(10),
                         dim: int = 40, seed: int = 0,
                         split: str = "train") -> SplitData:
    rng = np.random.default_rng(seed)
    xs, ys, ids = [], [], []
    counter = 0
    for c in classes:
        # class-dependent mean so that classes are somewhat separable
        center = rng.normal(size=dim) * 2.0
        x = center[None, :] + rng.normal(size=(n_per_class, dim))
        xs.append(x)
        ys.append(np.full(n_per_class, c, dtype=np.int64))
        ids.append(np.array([f"{split}_{counter + i:04d}" for i in range(n_per_class)]))
        counter += n_per_class
    return SplitData(x=np.concatenate(xs).astype(np.float32),
                     y=np.concatenate(ys),
                     ids=np.concatenate(ids), split=split)


@pytest.fixture
def synthetic_split() -> SplitData:
    return make_synthetic_split()


@pytest.fixture
def make_split():
    """Factory fixture returning ``make_synthetic_split``."""
    return make_synthetic_split


@pytest.fixture(scope="session")
def datasets():
    """Real MNIST-1D datasets (downloads once). Skips if offline."""
    from mnist1d_cl.data import build_datasets
    try:
        return build_datasets(data_dir="data", data_seed=1234, val_fraction=0.15)
    except Exception as e:  # pragma: no cover - network dependent
        pytest.skip(f"MNIST-1D dataset unavailable: {e}")
