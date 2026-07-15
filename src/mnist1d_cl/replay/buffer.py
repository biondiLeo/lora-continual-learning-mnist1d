"""Fixed, class-balanced replay buffer.

The buffer is sampled ONCE (deterministically) from the Task A training split,
``per_class`` samples per class, and stored by sample ID. It is fixed (not
updated during training). Mini-batches during the Task B update draw uniformly
from the buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch

from ..data.dataset import SplitData


@dataclass
class ReplayBuffer:
    x: np.ndarray          # (M, 40) float32, normalized
    y: np.ndarray          # (M,) int64
    ids: np.ndarray        # (M,) str
    per_class: int
    classes: List[int]

    # cached tensors
    def __post_init__(self) -> None:
        self._xt = torch.from_numpy(np.ascontiguousarray(self.x)).float()
        self._yt = torch.from_numpy(np.ascontiguousarray(self.y)).long()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @classmethod
    def build_balanced(cls, data: SplitData, classes: Sequence[int],
                       per_class: int, seed: int = 0) -> "ReplayBuffer":
        rng = np.random.default_rng(seed)
        xs, ys, ids = [], [], []
        for c in sorted(classes):
            idx_c = np.where(data.y == c)[0]
            idx_c = np.sort(idx_c)
            if idx_c.shape[0] < per_class:
                raise ValueError(
                    f"Class {c} has only {idx_c.shape[0]} samples < per_class={per_class}")
            chosen = idx_c[rng.permutation(idx_c.shape[0])[:per_class]]
            chosen = np.sort(chosen)
            xs.append(data.x[chosen]); ys.append(data.y[chosen]); ids.append(data.ids[chosen])
        return cls(x=np.concatenate(xs), y=np.concatenate(ys), ids=np.concatenate(ids),
                   per_class=per_class, classes=sorted(classes))

    def as_split(self) -> SplitData:
        return SplitData(x=self.x.copy(), y=self.y.copy(), ids=self.ids.copy(),
                         split="replay")

    def sample(self, n: int, generator: torch.Generator,
               device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Uniformly sample ``n`` items (with replacement if n > len)."""
        m = len(self)
        replacement = n > m
        idx = torch.randint(0, m, (n,), generator=generator) if replacement else \
            torch.randperm(m, generator=generator)[:n]
        return self._xt[idx].to(device), self._yt[idx].to(device)
