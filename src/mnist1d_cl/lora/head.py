"""Classification-head training strategies for the Task B update.

Modes:
  * 'frozen'  : head fully frozen.
  * 'full'    : head fully trainable.
  * 'partial' : only rows of ``trainable_rows`` (Task B classes 5,6,7) are
    trainable; all other rows (0-4 and 8-9) stay frozen.

For 'partial' we (a) zero the gradient of frozen rows via tensor hooks AND
(b) restore the frozen rows from a snapshot after every optimizer step, because
weight decay / Adam moments can move parameters even when their gradient is
zero. ``PartialHeadController`` is callable and is passed as ``post_step_hook``.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


class PartialHeadController:
    """Keeps frozen head rows identical to their snapshot after each step."""

    def __init__(self, head: nn.Linear, trainable_rows: Sequence[int]):
        num_rows = head.out_features
        self.head = head
        self.trainable_rows: List[int] = sorted(set(int(r) for r in trainable_rows))
        self.frozen_rows: List[int] = [i for i in range(num_rows)
                                       if i not in set(self.trainable_rows)]

        head.weight.requires_grad_(True)
        if head.bias is not None:
            head.bias.requires_grad_(True)

        # Snapshot frozen rows (from the initial/base checkpoint).
        self._frozen_w = head.weight.detach()[self.frozen_rows].clone()
        self._frozen_b = (head.bias.detach()[self.frozen_rows].clone()
                          if head.bias is not None else None)

        # Zero gradients of frozen rows.
        head.weight.register_hook(self._mask_weight_grad)
        if head.bias is not None:
            head.bias.register_hook(self._mask_bias_grad)

    def _mask_weight_grad(self, grad: torch.Tensor) -> torch.Tensor:
        grad = grad.clone()
        grad[self.frozen_rows] = 0.0
        return grad

    def _mask_bias_grad(self, grad: torch.Tensor) -> torch.Tensor:
        grad = grad.clone()
        grad[self.frozen_rows] = 0.0
        return grad

    @torch.no_grad()
    def restore(self) -> None:
        self.head.weight[self.frozen_rows] = self._frozen_w.to(self.head.weight.device)
        if self._frozen_b is not None and self.head.bias is not None:
            self.head.bias[self.frozen_rows] = self._frozen_b.to(self.head.bias.device)

    def frozen_rows_unchanged(self, atol: float = 1e-8) -> bool:
        w_ok = torch.allclose(self.head.weight.detach()[self.frozen_rows].cpu(),
                              self._frozen_w.cpu(), atol=atol)
        b_ok = True
        if self._frozen_b is not None and self.head.bias is not None:
            b_ok = torch.allclose(self.head.bias.detach()[self.frozen_rows].cpu(),
                                  self._frozen_b.cpu(), atol=atol)
        return bool(w_ok and b_ok)

    def __call__(self, model: nn.Module | None = None) -> None:
        # post_step_hook signature: (model) -> None
        self.restore()


def configure_head(head: nn.Linear, mode: str,
                   trainable_rows: Sequence[int]) -> PartialHeadController | None:
    """Apply the requested head strategy. Returns a controller for 'partial'."""
    if mode == "frozen":
        head.weight.requires_grad_(False)
        if head.bias is not None:
            head.bias.requires_grad_(False)
        return None
    if mode == "full":
        head.weight.requires_grad_(True)
        if head.bias is not None:
            head.bias.requires_grad_(True)
        return None
    if mode == "partial":
        return PartialHeadController(head, trainable_rows)
    raise ValueError(f"Unknown head mode: {mode}")
