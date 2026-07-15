"""Inject LoRA adapters into an MLP backbone and configure the head strategy."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch.nn as nn

from ..constants import TASK_B_CLASSES
from .layers import LoRALinear
from .head import configure_head, PartialHeadController
from .checks import count_trainable_params


def max_rank_for(model: nn.Module, target_names: Sequence[str] = ("fc1", "fc2")
                 ) -> Dict[str, int]:
    """Per-layer maximum admissible LoRA rank = min(in, out)."""
    out: Dict[str, int] = {}
    for name in target_names:
        layer = getattr(model, name)
        base = layer.base if isinstance(layer, LoRALinear) else layer
        out[name] = min(base.in_features, base.out_features)
    return out


def apply_lora(model: nn.Module, r: int, alpha: float,
               target_names: Sequence[str] = ("fc1", "fc2"),
               init: str = "kaiming") -> Dict[str, LoRALinear]:
    """Replace each target Linear in-place with a LoRALinear. Returns them."""
    injected: Dict[str, LoRALinear] = {}
    for name in target_names:
        base = getattr(model, name)
        if isinstance(base, LoRALinear):
            raise ValueError(f"Layer {name} already has LoRA applied")
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Target {name} is not nn.Linear: {type(base)}")
        lora = LoRALinear(base, r=r, alpha=alpha, init=init)
        setattr(model, name, lora)
        injected[name] = lora
    return injected


def setup_lora_model(model: nn.Module, r: int, alpha: float, head_mode: str,
                     target_names: Sequence[str] = ("fc1", "fc2"),
                     trainable_head_rows: Sequence[int] = tuple(TASK_B_CLASSES),
                     init: str = "kaiming",
                     ) -> Tuple[nn.Module, PartialHeadController | None, Dict[str, object]]:
    """Full setup: freeze backbone, inject LoRA, configure head.

    The model is expected to already carry the Task A weights (LoRA reuses them
    as the frozen base, so it starts from the exact same outputs).
    Returns ``(model, head_controller_or_None, info)``.
    """
    injected = apply_lora(model, r=r, alpha=alpha, target_names=target_names, init=init)

    # Freeze everything that is neither a LoRA factor nor the head; the head is
    # then configured according to ``head_mode``.
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad_(True)
        elif name.startswith("head."):
            continue  # handled by configure_head
        else:
            p.requires_grad_(False)

    controller = configure_head(model.head, head_mode, trainable_head_rows)

    tp = count_trainable_params(model)
    low_rank_flags = {name: bool(lin.is_low_rank) for name, lin in injected.items()}
    info: Dict[str, object] = {
        "rank": r,
        "alpha": alpha,
        "scaling": alpha / r,
        "head_mode": head_mode,
        "target_names": list(target_names),
        "trainable_head_rows": list(trainable_head_rows) if head_mode == "partial" else None,
        "max_rank_per_layer": max_rank_for(model, target_names),
        "is_low_rank_per_layer": low_rank_flags,
        "n_trainable": tp["n_trainable"],
        "n_total": tp["n_total"],
        "pct_trainable": tp["pct_trainable"],
        "trainable_names": tp["trainable_names"],
    }
    return model, controller, info
