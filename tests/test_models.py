"""Tests #1, #2: output and feature shapes."""

import torch

from mnist1d_cl.models import MLP, MLPConfig, build_mlp


def test_output_shape(base_model):
    x = torch.randn(7, 40)
    logits = base_model(x)
    assert logits.shape == (7, 10)


def test_feature_shape(base_model):
    x = torch.randn(7, 40)
    feats = base_model.features(x)
    assert feats.shape == (7, base_model.feature_dim)
    logits, feats2 = base_model.forward_features(x)
    assert logits.shape == (7, 10)
    assert torch.allclose(feats, feats2)


def test_lowdim_linear_penultimate():
    # low-dim config: no activation after fc2 -> features can be negative
    cfg = MLPConfig(hidden1=8, hidden2=2, penultimate_activation=False)
    model = build_mlp(cfg)
    x = torch.randn(64, 40)
    feats = model.features(x)
    assert feats.shape == (64, 2)
    assert (feats < 0).any()  # linear embedding spans the plane
