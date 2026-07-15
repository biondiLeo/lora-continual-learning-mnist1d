"""LoRA: low-rank adapters, injection, head strategies and correctness checks."""

from .layers import LoRALinear
from .head import configure_head, PartialHeadController
from .inject import setup_lora_model, apply_lora, max_rank_for
from .checks import (
    count_trainable_params,
    lora_modules,
    delta_is_zero,
    outputs_equivalent,
    base_frozen,
    snapshot_frozen,
    frozen_unchanged,
    effective_linear_weight,
)

__all__ = [
    "LoRALinear",
    "configure_head",
    "PartialHeadController",
    "setup_lora_model",
    "apply_lora",
    "max_rank_for",
    "count_trainable_params",
    "lora_modules",
    "delta_is_zero",
    "outputs_equivalent",
    "base_frozen",
    "snapshot_frozen",
    "frozen_unchanged",
    "effective_linear_weight",
]
