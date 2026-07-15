"""LoRA-adapted linear layer.

For a base linear ``y = x W^T + b`` with ``W in R^{out x in}``, LoRA adds a
low-rank update ``dW = (alpha/r) * B @ A`` with ``A in R^{r x in}`` and
``B in R^{out x r}``. ``B`` is initialized to zero so ``dW = 0`` at start and the
adapted layer reproduces the base layer exactly. Only ``A`` and ``B`` are
trainable; the base ``W`` and ``b`` are frozen.

Constraint: ``r <= min(in, out)``. When ``r == min(in, out)`` the update is no
longer strictly low-rank for that layer; this is flagged in ``is_low_rank``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float,
                 init: str = "kaiming"):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got r={r}")
        in_f, out_f = base.in_features, base.out_features
        max_r = min(in_f, out_f)
        if r > max_r:
            raise ValueError(
                f"LoRA rank r={r} exceeds min(in={in_f}, out={out_f})={max_r}")

        self.in_features = in_f
        self.out_features = out_f
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / r
        self.is_low_rank = r < max_r  # False => full-rank update for this layer

        # Frozen base layer (reuses the provided module's parameters).
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        # Trainable low-rank factors.
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        self._reset_lora(init)

    def _reset_lora(self, init: str) -> None:
        if init == "kaiming":
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        elif init == "normal":
            nn.init.normal_(self.lora_A, std=1.0 / self.r)
        else:
            raise ValueError(f"Unknown LoRA init: {init}")
        nn.init.zeros_(self.lora_B)  # dW = 0 at initialization

    def delta_weight(self) -> torch.Tensor:
        """The LoRA weight update dW = scaling * B @ A  (shape out x in)."""
        return self.scaling * (self.lora_B @ self.lora_A)

    def effective_weight(self) -> torch.Tensor:
        """W_eff = W_base + dW."""
        return self.base.weight.detach() + self.delta_weight()

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        # (x @ A^T) @ B^T  == x @ (B A)^T
        delta = (x @ self.lora_A.t()) @ self.lora_B.t()
        return out + self.scaling * delta
