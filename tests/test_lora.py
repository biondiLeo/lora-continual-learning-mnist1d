"""Tests #5-#10, #18: LoRA correctness and head strategies."""

import copy

import torch
import torch.nn.functional as F

from mnist1d_cl.constants import TASK_B_CLASSES
from mnist1d_cl.lora import (
    setup_lora_model, delta_is_zero, base_frozen, outputs_equivalent,
    count_trainable_params, LoRALinear,
)


def test_lora_init_equivalent_and_delta_zero(base_model, cloned_model):
    x = torch.randn(16, 40)
    lora_model, _ctrl, info = setup_lora_model(cloned_model, r=4, alpha=8,
                                               head_mode="full")
    # #6 delta zero at init
    assert delta_is_zero(lora_model)
    # #5 LoRA model reproduces the base model's outputs exactly at init
    assert outputs_equivalent(base_model, lora_model, x, atol=1e-6)
    # #7 base frozen
    assert base_frozen(lora_model)
    assert not lora_model.fc1.base.weight.requires_grad


def test_trainable_params_frozen_head(base_model):
    model = copy.deepcopy(base_model)
    lora_model, ctrl, info = setup_lora_model(model, r=4, alpha=8, head_mode="frozen")
    names = count_trainable_params(lora_model)["trainable_names"]
    # #8 only LoRA factors are trainable when the head is frozen
    assert all("lora_A" in n or "lora_B" in n for n in names)
    assert any("fc1" in n for n in names) and any("fc2" in n for n in names)
    assert ctrl is None


def test_rank_constraint_and_low_rank_flag(base_model):
    # fc2 is 16->8, so max rank there is 8; r=8 => not strictly low-rank.
    model = copy.deepcopy(base_model)
    _m, _c, info = setup_lora_model(model, r=8, alpha=8, head_mode="frozen")
    assert info["is_low_rank_per_layer"]["fc2"] is False  # r == min(16,8)
    assert info["is_low_rank_per_layer"]["fc1"] is True    # r=8 < min(40,16)=16


def test_partial_head_frozen_rows_unchanged_after_step(base_model):
    # #9 + #10: only rows 5,6,7 train; frozen rows stay identical even with WD.
    model = copy.deepcopy(base_model)
    lora_model, ctrl, info = setup_lora_model(model, r=4, alpha=8, head_mode="partial")
    assert ctrl is not None
    assert ctrl.trainable_rows == list(TASK_B_CLASSES)

    frozen_w0 = lora_model.head.weight.detach()[ctrl.frozen_rows].clone()
    train_w0 = lora_model.head.weight.detach()[ctrl.trainable_rows].clone()

    params = [p for p in lora_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=0.1, weight_decay=0.5)

    x = torch.randn(32, 40)
    y = torch.randint(5, 8, (32,))
    for _ in range(3):
        opt.zero_grad()
        loss = F.cross_entropy(lora_model(x), y)
        loss.backward()
        opt.step()
        ctrl.restore()  # post-step hook

    assert ctrl.frozen_rows_unchanged()
    assert torch.allclose(lora_model.head.weight.detach()[ctrl.frozen_rows], frozen_w0)
    # trainable rows actually moved
    assert not torch.allclose(lora_model.head.weight.detach()[ctrl.trainable_rows], train_w0)


def test_effective_weight_reconstruction():
    # #18: W_eff = W_base + scaling * B @ A, and forward matches x @ W_eff^T + b.
    base = torch.nn.Linear(12, 6)
    lora = LoRALinear(base, r=3, alpha=6)
    torch.nn.init.normal_(lora.lora_B, std=0.5)  # make delta non-zero
    x = torch.randn(10, 12)
    W_eff = lora.effective_weight()
    y = lora(x)
    y_manual = x @ W_eff.t() + base.bias
    assert torch.allclose(y, y_manual, atol=1e-5)
    # scaling * B @ A reconstruction
    delta = lora.scaling * (lora.lora_B @ lora.lora_A)
    assert torch.allclose(W_eff, base.weight.detach() + delta, atol=1e-6)
