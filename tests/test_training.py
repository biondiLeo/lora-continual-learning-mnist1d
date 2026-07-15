"""Tests #21, #22, #23: no test leakage, NaN guard, checkpoint save/load."""

import copy

import pytest
import torch

from mnist1d_cl.models import MLP, MLPConfig
from mnist1d_cl.training import TrainConfig, train_supervised, fit
from mnist1d_cl.data.dataset import make_loader
from mnist1d_cl.utils.io import save_checkpoint, load_checkpoint


def test_selection_uses_validation_only(make_split):
    train = make_split(n_per_class=15, classes=range(5), seed=1, split="train")
    val = make_split(n_per_class=5, classes=range(5), seed=2, split="val")
    model = MLP(MLPConfig(hidden1=16, hidden2=8))
    cfg = TrainConfig(epochs=3, early_stopping_patience=None, batch_size=16)
    out = train_supervised(model, train, val, cfg, torch.device("cpu"),
                           allowed_classes=list(range(5)))
    # #21: selection must be driven by the validation split, never 'test'.
    assert out["selection"]["val_split"] == "val"


def test_nan_guard(make_split):
    train = make_split(n_per_class=10, classes=range(3), seed=1, split="train")
    val = make_split(n_per_class=4, classes=range(3), seed=2, split="val")
    model = MLP(MLPConfig(hidden1=8, hidden2=4))
    loader = make_loader(train, batch_size=8, shuffle=False)

    def bad_closure(m, batch, device):
        x, y, _ = batch
        return torch.tensor(float("nan"), requires_grad=True), {}

    cfg = TrainConfig(epochs=2, early_stopping_patience=None)
    with pytest.raises(FloatingPointError):
        fit(model, loader, val, cfg, torch.device("cpu"), loss_closure=bad_closure)


def test_checkpoint_save_load(tmp_path):
    model = MLP(MLPConfig(hidden1=16, hidden2=8))
    state = {"model": model.state_dict(), "meta": {"seed": 0}}
    path = tmp_path / "ckpt.pt"
    save_checkpoint(state, path)
    loaded = load_checkpoint(path)
    model2 = MLP(MLPConfig(hidden1=16, hidden2=8))
    model2.load_state_dict(loaded["model"])
    x = torch.randn(4, 40)
    model.eval(); model2.eval()
    assert torch.allclose(model(x), model2(x))
    assert loaded["meta"]["seed"] == 0
