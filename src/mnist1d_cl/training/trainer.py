"""Training loop with validation-based checkpoint selection.

Selection criterion (fixed a priori, see plan §5.1):
  * primary  : maximum validation accuracy on the current task
               (argmax restricted to the task's ``allowed_classes`` -> "masked"
               task accuracy; computed on validation only, never on test);
  * tie-break : minimum validation loss (cross-entropy on the true labels).

Also provides: configurable early stopping (disable-able), gradient clipping,
NaN/Inf guards, and a ``post_step_hook`` used e.g. to restore frozen head rows.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.dataset import SplitData, make_loader

Batch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
LossClosure = Callable[[nn.Module, Batch, torch.device], Tuple[torch.Tensor, Dict[str, float]]]
PostStepHook = Callable[[nn.Module], None]


@dataclass
class TrainConfig:
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    optimizer: str = "adamw"
    scheduler: str = "none"            # 'none' | 'cosine'
    grad_clip: Optional[float] = None  # max-norm; None disables
    early_stopping_patience: Optional[int] = 15  # None disables
    min_epochs: int = 1
    shuffle_seed: int = 0
    log_every: int = 0                 # 0 => silent

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "TrainConfig":
        fields = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        return cls(**fields)


# --------------------------------------------------------------------------- #
# Evaluation helpers                                                            #
# --------------------------------------------------------------------------- #
def masked_argmax(logits: torch.Tensor, allowed_classes: Optional[Sequence[int]]) -> torch.Tensor:
    """Argmax over all classes, or restricted to ``allowed_classes`` (masked)."""
    if allowed_classes is None:
        return logits.argmax(dim=1)
    mask = torch.full_like(logits, float("-inf"))
    idx = torch.as_tensor(list(allowed_classes), device=logits.device, dtype=torch.long)
    mask[:, idx] = 0.0
    return (logits + mask).argmax(dim=1)


@torch.no_grad()
def evaluate(model: nn.Module, data: SplitData, device: torch.device,
             allowed_classes: Optional[Sequence[int]] = None,
             batch_size: int = 256) -> Dict[str, float]:
    """Return masked accuracy (over ``allowed_classes``) and mean CE loss."""
    model.eval()
    loader = make_loader(data, batch_size=batch_size, shuffle=False)
    total, correct, loss_sum = 0, 0, 0.0
    for x, y, _idx in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        pred = masked_argmax(logits, allowed_classes)
        correct += (pred == y).sum().item()
        total += y.numel()
    return {"accuracy": correct / max(total, 1), "loss": loss_sum / max(total, 1),
            "n": total}


# --------------------------------------------------------------------------- #
# Loss closures                                                                 #
# --------------------------------------------------------------------------- #
def supervised_closure() -> LossClosure:
    """Standard cross-entropy on (x, y)."""
    def closure(model: nn.Module, batch: Batch, device: torch.device):
        x, y, _idx = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        return loss, {"ce": float(loss.detach())}
    return closure


# --------------------------------------------------------------------------- #
# Core loop                                                                      #
# --------------------------------------------------------------------------- #
def _build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def fit(model: nn.Module,
        train_loader: DataLoader,
        val_data: SplitData,
        cfg: TrainConfig,
        device: torch.device,
        loss_closure: Optional[LossClosure] = None,
        allowed_classes: Optional[Sequence[int]] = None,
        post_step_hook: Optional[PostStepHook] = None,
        monitor: Optional[Dict[str, Tuple[SplitData, Optional[Sequence[int]]]]] = None,
        ) -> Dict[str, object]:
    """Train ``model`` and select the best checkpoint on validation.

    ``monitor`` optionally maps a name -> (SplitData, allowed_classes) that is
    evaluated each epoch and logged, but NEVER used for selection (e.g. Task A
    validation while training on Task B).
    """
    loss_closure = loss_closure or supervised_closure()
    model.to(device)
    optimizer = _build_optimizer(model, cfg)
    scheduler = None
    if cfg.scheduler.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    history: List[Dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_acc, best_val_loss, best_epoch = -math.inf, math.inf, -1
    epochs_no_improve = 0
    stopped_early = False

    for epoch in range(cfg.epochs):
        model.train()
        run_loss, n_batches = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _logs = loss_closure(model, batch, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}: {loss.item()}")
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
            optimizer.step()
            if post_step_hook is not None:
                post_step_hook(model)
            run_loss += float(loss.detach())
            n_batches += 1
        if scheduler is not None:
            scheduler.step()

        val = evaluate(model, val_data, device, allowed_classes)
        rec: Dict[str, float] = {
            "epoch": epoch,
            "train_loss": run_loss / max(n_batches, 1),
            "val_acc": val["accuracy"],
            "val_loss": val["loss"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        if monitor:
            for name, (mdata, mallowed) in monitor.items():
                m = evaluate(model, mdata, device, mallowed)
                rec[f"monitor_{name}_acc"] = m["accuracy"]
                rec[f"monitor_{name}_loss"] = m["loss"]
        history.append(rec)
        if cfg.log_every and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1):
            print(f"[epoch {epoch:03d}] train_loss={rec['train_loss']:.4f} "
                  f"val_acc={rec['val_acc']:.4f} val_loss={rec['val_loss']:.4f}")

        # ---- selection: max val acc, tie-break min val loss --------------- #
        improved = (val["accuracy"] > best_val_acc + 1e-9) or (
            abs(val["accuracy"] - best_val_acc) <= 1e-9 and val["loss"] < best_val_loss - 1e-9)
        if improved:
            best_val_acc, best_val_loss, best_epoch = val["accuracy"], val["loss"], epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (cfg.early_stopping_patience is not None
                and epoch + 1 >= cfg.min_epochs
                and epochs_no_improve >= cfg.early_stopping_patience):
            stopped_early = True
            break

    last_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)  # leave model at best checkpoint

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_state": best_state,
        "last_state": last_state,
        "stopped_early": stopped_early,
        "selection": {
            "criterion": "max val accuracy (masked to task), tie-break min val loss",
            "allowed_classes": list(allowed_classes) if allowed_classes else None,
            # Recorded so a unit test can assert the test split was never used
            # for checkpoint selection (must be 'val').
            "val_split": val_data.split,
        },
    }


def train_supervised(model: nn.Module, train_data: SplitData, val_data: SplitData,
                     cfg: TrainConfig, device: torch.device,
                     allowed_classes: Optional[Sequence[int]] = None,
                     monitor: Optional[Dict[str, Tuple[SplitData, Optional[Sequence[int]]]]] = None,
                     ) -> Dict[str, object]:
    """Convenience wrapper: standard CE training from a SplitData."""
    generator = torch.Generator().manual_seed(cfg.shuffle_seed)
    train_loader = make_loader(train_data, batch_size=cfg.batch_size, shuffle=True,
                               generator=generator)
    return fit(model, train_loader, val_data, cfg, device,
               loss_closure=supervised_closure(), allowed_classes=allowed_classes,
               monitor=monitor)
