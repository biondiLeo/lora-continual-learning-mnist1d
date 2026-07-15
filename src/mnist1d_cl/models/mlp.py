"""Configurable MLP backbone: 40 -> fc1(h1) -> fc2(h2) -> head(10).

The "feature" of the network is the penultimate representation, i.e. the input
to the classification ``head`` (output of ``fc2`` after its optional activation).

Two regimes:
  * standard:  penultimate activation ON (ReLU after fc2).
  * low-dim:   penultimate activation OFF (linear embedding) so that 2-D
    features can spread over the whole plane for interpretable visualization.
    This is a disclosed architectural choice, identical for FFT and LoRA.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..constants import INPUT_DIM, NUM_CLASSES


def _activation(name: str) -> nn.Module:
    name = (name or "identity").lower()
    return {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "identity": nn.Identity(),
        "none": nn.Identity(),
    }[name]


@dataclass
class MLPConfig:
    hidden1: int
    hidden2: int
    input_dim: int = INPUT_DIM
    num_classes: int = NUM_CLASSES
    activation: str = "relu"            # activation after fc1 (and fc2 if enabled)
    penultimate_activation: bool = True  # False => linear penultimate (low-dim)
    dropout: float = 0.0
    norm: str = "none"                  # 'none' | 'layernorm'

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "MLPConfig":
        fields = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        return cls(**fields)


class MLP(nn.Module):
    """Two-hidden-layer MLP with a 10-way head and feature extraction."""

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.input_dim, config.hidden1)
        self.norm1 = (nn.LayerNorm(config.hidden1)
                      if config.norm == "layernorm" else nn.Identity())
        self.act1 = _activation(config.activation)
        self.dropout = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.hidden1, config.hidden2)
        self.act2 = (_activation(config.activation)
                     if config.penultimate_activation else nn.Identity())
        self.head = nn.Linear(config.hidden2, config.num_classes)

    # -- feature = input to the head ------------------------------------- #
    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.norm1(h)
        h = self.act1(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.act2(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.features(x)
        logits = self.head(feats)
        return logits, feats

    @property
    def feature_dim(self) -> int:
        return self.config.hidden2


def build_mlp(config: MLPConfig | Dict[str, object]) -> MLP:
    if isinstance(config, dict):
        config = MLPConfig.from_dict(config)
    return MLP(config)
