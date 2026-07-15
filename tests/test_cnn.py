"""Unit tests for the Conv1D backbone and LoRA-on-proj injection."""

import copy

import torch

from mnist1d_cl.models import CNNConfig, build_cnn
from mnist1d_cl.lora import setup_lora_model, delta_is_zero, outputs_equivalent, base_frozen


def _x(n=16):
    torch.manual_seed(0)
    return torch.randn(n, 40)


def test_output_shape():
    m = build_cnn(CNNConfig(feature_dim=64))
    assert m(_x()).shape == (16, 10)


def test_feature_shape_and_dim():
    for fd in (8, 64):
        m = build_cnn(CNNConfig(feature_dim=fd))
        logits, feats = m.forward_features(_x())
        assert feats.shape == (16, fd)
        assert logits.shape == (16, 10)
        assert m.feature_dim == fd


def test_penultimate_activation_controls_sign():
    """ReLU penultimate => non-negative features; linear => negatives allowed."""
    relu = build_cnn(CNNConfig(feature_dim=32, penultimate_activation=True))
    lin = build_cnn(CNNConfig(feature_dim=32, penultimate_activation=False))
    lin.load_state_dict(relu.state_dict())  # same weights, only activation differs
    fr = relu.features(_x())
    fl = lin.features(_x())
    assert torch.all(fr >= 0)          # ReLU output is non-negative
    assert (fl < 0).any()              # linear embedding spreads to negatives


def test_lora_on_proj_invariants():
    cfg = CNNConfig(feature_dim=64, penultimate_activation=False)
    base = build_cnn(cfg)
    model = build_cnn(cfg)
    model.load_state_dict(copy.deepcopy(base.state_dict()))
    model, controller, info = setup_lora_model(
        model, r=4, alpha=8.0, head_mode="full", target_names=("proj",))
    x = _x()
    # LoRA starts as an identity correction: output must equal the backbone's.
    assert delta_is_zero(model)
    assert outputs_equivalent(base, model, x)
    assert base_frozen(model)
    # conv stack is frozen; only proj LoRA factors (+ head) are trainable.
    conv_grads = [p.requires_grad for n, p in model.named_parameters() if n.startswith("features_net")]
    assert not any(conv_grads)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any("proj" in n and ("lora_A" in n or "lora_B" in n) for n in trainable)
    assert any(n.startswith("head.") for n in trainable)  # head=full


def test_lora_base_frozen_after_step():
    cfg = CNNConfig(feature_dim=32, penultimate_activation=True)
    model = build_cnn(cfg)
    conv0 = copy.deepcopy(model.features_net.state_dict())
    model, controller, info = setup_lora_model(
        model, r=4, alpha=8.0, head_mode="full", target_names=("proj",))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        loss = model(_x()).pow(2).mean()
        loss.backward(); opt.step()
    # frozen conv weights must be byte-identical after optimization.
    for k, v in model.features_net.state_dict().items():
        assert torch.equal(v, conv0[k])
