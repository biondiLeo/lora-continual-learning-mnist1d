"""Small 1D CNN backbone for MNIST-1D: Conv1d stack -> linear projection -> head.

Mirrors the MLP interface so the rest of the pipeline is architecture-agnostic:
  * input is ``(B, input_dim)``; it is reshaped to ``(B, 1, input_dim)`` internally;
  * the "feature" is the penultimate representation, i.e. the output of the linear
    projection ``proj`` (the input to the classification ``head``);
  * ``penultimate_activation`` toggles a ReLU after ``proj`` (linear embedding when
    False), exactly like :class:`MLP`;
  * ``proj`` and ``head`` are plain ``nn.Linear`` so LoRA can be injected on
    ``target_names=("proj",)`` and the head strategy reused unchanged.

A CNN adds the translation-related inductive bias that an MLP lacks, which is the
whole point of comparing the two on MNIST-1D (a dataset built with random shifts).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..constants import INPUT_DIM, NUM_CLASSES


def _activation(name: str) -> nn.Module:
    name = (name or "identity").lower()
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh(),
            "identity": nn.Identity(), "none": nn.Identity()}[name]


@dataclass
class CNNConfig:
    feature_dim: int                     # dimension of the penultimate projection
    channels: int = 16                   # base number of conv channels (C)
    in_channels: int = 1
    input_dim: int = INPUT_DIM
    num_classes: int = NUM_CLASSES
    activation: str = "relu"             # activation inside the conv stack
    penultimate_activation: bool = True  # ReLU after proj (False => linear embedding)
    dropout: float = 0.0
    kind: str = "cnn"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "CNNConfig":
        fields = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        return cls(**fields)


class Conv1D(nn.Module):
    """Three Conv1d blocks + global average pool + linear projection + 10-way head."""

    def __init__(self, config: CNNConfig):
        super().__init__()
        self.config = config
        C = config.channels
        act = config.activation
        self.features_net = nn.Sequential(
            nn.Conv1d(config.in_channels, C, kernel_size=5, padding=2), _activation(act), nn.MaxPool1d(2),
            nn.Conv1d(C, 2 * C, kernel_size=3, padding=1), _activation(act), nn.MaxPool1d(2),
            nn.Conv1d(2 * C, 2 * C, kernel_size=3, padding=1), _activation(act),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.proj = nn.Linear(2 * C, config.feature_dim)
        self.act_proj = (_activation(act) if config.penultimate_activation else nn.Identity())
        self.head = nn.Linear(config.feature_dim, config.num_classes)

    # -- feature = input to the head ------------------------------------- #
    def features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)                      # (B, 40) -> (B, 1, 40)
        h = self.features_net(x).flatten(1)         # (B, 2C)
        h = self.dropout(h)
        h = self.proj(h)
        h = self.act_proj(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.features(x)
        logits = self.head(feats)
        return logits, feats

    @property
    def feature_dim(self) -> int:
        return self.config.feature_dim


def build_cnn(config: CNNConfig | Dict[str, object]) -> Conv1D:
    if isinstance(config, dict):
        config = CNNConfig.from_dict(config)
    return Conv1D(config)
