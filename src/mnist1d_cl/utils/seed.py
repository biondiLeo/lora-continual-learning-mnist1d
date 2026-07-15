"""Reproducibility helpers: seed every RNG and (optionally) force determinism."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_all(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA).

    When ``deterministic`` is True we also configure cuDNN for deterministic
    behaviour. We do NOT call ``torch.use_deterministic_algorithms(True)``
    globally because a few CUDA kernels lack deterministic implementations and
    would raise; cuDNN determinism + fixed seeds is enough for the tiny MLPs
    used here to give reproducible metrics within numerical tolerance.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:  # pragma: no cover - used by DataLoader
    """Worker init fn so that DataLoader workers are seeded deterministically."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
