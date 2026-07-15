"""Correctness checks for LoRA models (used by experiments and unit tests)."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .layers import LoRALinear


def count_trainable_params(model: nn.Module) -> Dict[str, object]:
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    return {
        "n_trainable": int(n_train),
        "n_total": int(n_total),
        "pct_trainable": 100.0 * n_train / max(n_total, 1),
        "trainable_names": names,
    }


def lora_modules(model: nn.Module) -> List[LoRALinear]:
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def delta_is_zero(model: nn.Module, atol: float = 1e-8) -> bool:
    """True iff every LoRA delta weight is (numerically) zero."""
    mods = lora_modules(model)
    if not mods:
        return False
    return all(torch.allclose(m.delta_weight().detach().cpu(),
                              torch.zeros_like(m.delta_weight().cpu()), atol=atol)
               for m in mods)


@torch.no_grad()
def outputs_equivalent(model_a: nn.Module, model_b: nn.Module, x: torch.Tensor,
                       atol: float = 1e-6) -> bool:
    """True iff two models produce the same logits on ``x``."""
    was_a, was_b = model_a.training, model_b.training
    model_a.eval(); model_b.eval()
    ya = model_a(x)
    yb = model_b(x)
    model_a.train(was_a); model_b.train(was_b)
    return bool(torch.allclose(ya, yb, atol=atol))


def base_frozen(model: nn.Module) -> bool:
    """True iff all frozen base weights of LoRA modules are non-trainable."""
    for m in lora_modules(model):
        for p in m.base.parameters():
            if p.requires_grad:
                return False
    return True


def snapshot_frozen(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Clone all currently-frozen parameters (requires_grad == False)."""
    return {n: p.detach().clone().cpu()
            for n, p in model.named_parameters() if not p.requires_grad}


def frozen_unchanged(model: nn.Module, snapshot: Dict[str, torch.Tensor],
                     atol: float = 1e-8) -> bool:
    """True iff every snapshotted frozen parameter is unchanged."""
    for n, p in model.named_parameters():
        if n in snapshot:
            if not torch.allclose(p.detach().cpu(), snapshot[n], atol=atol):
                return False
    return True


def effective_linear_weight(module: nn.Module) -> torch.Tensor:
    """Effective weight for a Linear or LoRALinear (W_base + dW)."""
    if isinstance(module, LoRALinear):
        return module.effective_weight().detach().cpu()
    if isinstance(module, nn.Linear):
        return module.weight.detach().cpu()
    raise TypeError(f"Unsupported module type: {type(module)}")
