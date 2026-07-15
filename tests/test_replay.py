"""Replay buffer: balanced construction and sampling."""

import numpy as np
import torch

from mnist1d_cl.constants import TASK_A_CLASSES
from mnist1d_cl.replay import ReplayBuffer


def test_balanced_buffer(make_split):
    split = make_split(n_per_class=30, classes=TASK_A_CLASSES, seed=3, split="train")
    buf = ReplayBuffer.build_balanced(split, TASK_A_CLASSES, per_class=10, seed=0)
    assert len(buf) == 10 * len(TASK_A_CLASSES)
    _, counts = np.unique(buf.y, return_counts=True)
    assert set(counts.tolist()) == {10}
    assert set(np.unique(buf.y)).issubset(set(TASK_A_CLASSES))


def test_buffer_sampling_shape(make_split):
    split = make_split(n_per_class=30, classes=TASK_A_CLASSES, seed=3, split="train")
    buf = ReplayBuffer.build_balanced(split, TASK_A_CLASSES, per_class=10, seed=0)
    g = torch.Generator().manual_seed(0)
    x, y = buf.sample(16, g, torch.device("cpu"))
    assert x.shape == (16, 40)
    assert y.shape == (16,)


def test_buffer_determinism(make_split):
    split = make_split(n_per_class=30, classes=TASK_A_CLASSES, seed=3, split="train")
    b1 = ReplayBuffer.build_balanced(split, TASK_A_CLASSES, per_class=8, seed=42)
    b2 = ReplayBuffer.build_balanced(split, TASK_A_CLASSES, per_class=8, seed=42)
    assert np.array_equal(b1.ids, b2.ids)
