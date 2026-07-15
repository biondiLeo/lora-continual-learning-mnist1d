"""Shared orchestration for the continual-learning experiments.

Provides: Task A backbone training, the four Task B method runners (FFT, LoRA,
KD-LoRA, KD-LoRA+Replay), evaluation (masked/non-masked, forgetting, OOD) and
run-directory persistence. All checkpoint selection happens on validation; final
metrics are reported on the test split.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import (TASK_A_CLASSES, TASK_B_CLASSES, OOD_CLASSES,
                         SEEN_CLASSES, NUM_CLASSES)
from ..data import MNIST1DData, make_loader
from ..models import MLP, MLPConfig, build_mlp, extract
from ..training import TrainConfig, fit, train_supervised, supervised_closure
from ..lora import setup_lora_model, delta_is_zero, outputs_equivalent, base_frozen
from ..losses import kd_loss
from ..replay import ReplayBuffer
from ..metrics import full_report, accuracy, mean_seen_accuracy, softmax_np
from ..utils import (seed_all, get_device, save_json, save_checkpoint,
                     make_run_dir, capture_environment, append_csv_row)


# --------------------------------------------------------------------------- #
# Default configurations (standard-model config is chosen by select_config)     #
# --------------------------------------------------------------------------- #
DEFAULT_STD_CONFIG = MLPConfig(hidden1=64, hidden2=64, activation="relu",
                               penultimate_activation=True, dropout=0.0, norm="none")

DEFAULT_TASKA_TRAIN = TrainConfig(epochs=80, lr=1e-3, weight_decay=1e-4,
                                  batch_size=128, scheduler="cosine",
                                  grad_clip=5.0, early_stopping_patience=15)
DEFAULT_TASKB_TRAIN = TrainConfig(epochs=60, lr=1e-3, weight_decay=1e-4,
                                  batch_size=128, scheduler="cosine",
                                  grad_clip=5.0, early_stopping_patience=15)


def load_selected_config(path: str | Path = "configs/selected_config.json",
                         default: MLPConfig = DEFAULT_STD_CONFIG) -> MLPConfig:
    p = Path(path)
    if p.exists():
        from ..utils import load_json
        return MLPConfig.from_dict(load_json(p)["config"])
    return default


# --------------------------------------------------------------------------- #
# Task A backbone                                                               #
# --------------------------------------------------------------------------- #
def train_task_a(config: MLPConfig, datasets: MNIST1DData, device: torch.device,
                 train_cfg: TrainConfig = DEFAULT_TASKA_TRAIN, seed: int = 0
                 ) -> Dict[str, object]:
    seed_all(seed)
    model = build_mlp(config).to(device)
    train = datasets.task("train", TASK_A_CLASSES)
    val = datasets.task("val", TASK_A_CLASSES)
    res = train_supervised(model, train, val, train_cfg, device,
                           allowed_classes=TASK_A_CLASSES)
    return {"model": model, "train_result": res,
            "state": copy.deepcopy(model.state_dict())}


def taskA_ckpt_path(out_base: str | Path, seed: int) -> Path:
    return Path(out_base) / "taskA" / f"seed{seed}" / "checkpoint.pt"


def ensure_task_a_state(config: MLPConfig, datasets: MNIST1DData, device: torch.device,
                        out_base: str | Path, seed: int,
                        train_cfg: TrainConfig = DEFAULT_TASKA_TRAIN) -> Dict:
    """Load the cached Task A backbone for ``seed`` or train+save it once.

    All Task B methods must initialize from this identical checkpoint, so caching
    guarantees a fair comparison and lets each experiment run independently.
    """
    from ..utils import load_checkpoint
    path = taskA_ckpt_path(out_base, seed)
    if path.exists():
        return load_checkpoint(path)["model"]
    res = train_task_a(config, datasets, device, train_cfg, seed=seed)
    save_checkpoint({"model": res["state"], "config": config.to_dict(), "seed": seed}, path)
    save_json(res["train_result"]["history"], path.parent / "history.json")
    return res["state"]


def make_teacher(config: MLPConfig, state: Dict, device: torch.device) -> MLP:
    teacher = build_mlp(config).to(device)
    teacher.load_state_dict(state)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def fresh_from_state(config: MLPConfig, state: Dict, device: torch.device) -> MLP:
    model = build_mlp(config).to(device)
    model.load_state_dict(copy.deepcopy(state))
    return model


# --------------------------------------------------------------------------- #
# Task B method runners                                                          #
# --------------------------------------------------------------------------- #
def _taskB_monitor(datasets: MNIST1DData):
    return {"taskA": (datasets.task("val", TASK_A_CLASSES), TASK_A_CLASSES)}


def run_fft(config: MLPConfig, state: Dict, datasets: MNIST1DData,
            device: torch.device, train_cfg: TrainConfig = DEFAULT_TASKB_TRAIN
            ) -> Dict[str, object]:
    model = fresh_from_state(config, state, device)
    for p in model.parameters():
        p.requires_grad_(True)
    train = datasets.task("train", TASK_B_CLASSES)
    val = datasets.task("val", TASK_B_CLASSES)
    gen = torch.Generator().manual_seed(train_cfg.shuffle_seed)
    loader = make_loader(train, batch_size=train_cfg.batch_size, shuffle=True, generator=gen)
    res = fit(model, loader, val, train_cfg, device, supervised_closure(),
              allowed_classes=TASK_B_CLASSES, monitor=_taskB_monitor(datasets))
    return {"model": model, "train_result": res, "method": "fft",
            "trainable": {"n_trainable": sum(p.numel() for p in model.parameters()),
                          "pct_trainable": 100.0}}


def run_lora(config: MLPConfig, state: Dict, datasets: MNIST1DData,
             device: torch.device, rank: int, alpha: float, head_mode: str,
             train_cfg: TrainConfig = DEFAULT_TASKB_TRAIN) -> Dict[str, object]:
    model = fresh_from_state(config, state, device)
    base_ref = fresh_from_state(config, state, device)  # for equivalence check
    model, controller, info = setup_lora_model(model, r=rank, alpha=alpha,
                                               head_mode=head_mode)
    model = model.to(device)  # move freshly-created LoRA params onto the device
    # runtime invariants
    x0 = torch.from_numpy(datasets.task("val", TASK_A_CLASSES).x[:32]).float().to(device)
    checks = {
        "delta_zero_init": bool(delta_is_zero(model)),
        "output_equiv_init": bool(outputs_equivalent(base_ref, model, x0)),
        "base_frozen": bool(base_frozen(model)),
    }
    train = datasets.task("train", TASK_B_CLASSES)
    val = datasets.task("val", TASK_B_CLASSES)
    gen = torch.Generator().manual_seed(train_cfg.shuffle_seed)
    loader = make_loader(train, batch_size=train_cfg.batch_size, shuffle=True, generator=gen)
    res = fit(model, loader, val, train_cfg, device, supervised_closure(),
              allowed_classes=TASK_B_CLASSES, post_step_hook=controller,
              monitor=_taskB_monitor(datasets))
    if controller is not None:
        checks["frozen_head_rows_unchanged"] = bool(controller.frozen_rows_unchanged())
    return {"model": model, "train_result": res, "method": "lora",
            "lora_info": info, "checks": checks, "trainable": info}


def _kd_lora_closure(teacher: MLP, temperature: float, lambda_kd: float,
                     old_classes: Sequence[int]):
    def closure(model, batch, device):
        x, y, _ = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        ce = F.cross_entropy(logits, y)
        with torch.no_grad():
            t_logits = teacher(x)
        kd = kd_loss(logits, t_logits, temperature, classes=old_classes)
        loss = ce + lambda_kd * kd
        return loss, {"ce": float(ce.detach()), "kd": float(kd.detach())}
    return closure


def run_kd_lora(config: MLPConfig, state: Dict, datasets: MNIST1DData,
                device: torch.device, rank: int, alpha: float, head_mode: str,
                temperature: float, lambda_kd: float,
                train_cfg: TrainConfig = DEFAULT_TASKB_TRAIN) -> Dict[str, object]:
    model = fresh_from_state(config, state, device)
    teacher = make_teacher(config, state, device)
    model, controller, info = setup_lora_model(model, r=rank, alpha=alpha,
                                               head_mode=head_mode)
    model = model.to(device)
    train = datasets.task("train", TASK_B_CLASSES)
    val = datasets.task("val", TASK_B_CLASSES)
    gen = torch.Generator().manual_seed(train_cfg.shuffle_seed)
    loader = make_loader(train, batch_size=train_cfg.batch_size, shuffle=True, generator=gen)
    closure = _kd_lora_closure(teacher, temperature, lambda_kd, TASK_A_CLASSES)
    res = fit(model, loader, val, train_cfg, device, closure,
              allowed_classes=TASK_B_CLASSES, post_step_hook=controller,
              monitor=_taskB_monitor(datasets))
    return {"model": model, "train_result": res, "method": "kd_lora",
            "lora_info": info, "kd": {"temperature": temperature, "lambda_kd": lambda_kd,
                                      "support": "Task B samples (LwF)",
                                      "distilled_classes": TASK_A_CLASSES},
            "trainable": info}


def _kd_replay_closure(teacher: MLP, buffer: ReplayBuffer, temperature: float,
                       lambda_kd: float, old_classes: Sequence[int],
                       replay_bs: int, gen: torch.Generator):
    def closure(model, batch, device):
        x, y, _ = batch
        x, y = x.to(device), y.to(device)
        logits_b = model(x)
        ce_b = F.cross_entropy(logits_b, y)
        xr, yr = buffer.sample(replay_bs, gen, device)
        logits_r = model(xr)
        ce_r = F.cross_entropy(logits_r, yr)
        with torch.no_grad():
            t_logits_r = teacher(xr)
        kd_r = kd_loss(logits_r, t_logits_r, temperature, classes=old_classes)
        loss = ce_b + ce_r + lambda_kd * kd_r
        return loss, {"ce_b": float(ce_b.detach()), "ce_r": float(ce_r.detach()),
                      "kd_r": float(kd_r.detach())}
    return closure


def run_kd_lora_replay(config: MLPConfig, state: Dict, datasets: MNIST1DData,
                       device: torch.device, rank: int, alpha: float, head_mode: str,
                       temperature: float, lambda_kd: float, per_class: int,
                       replay_bs: int = 128, buffer_seed: int = 0,
                       train_cfg: TrainConfig = DEFAULT_TASKB_TRAIN) -> Dict[str, object]:
    model = fresh_from_state(config, state, device)
    teacher = make_teacher(config, state, device)
    buffer = ReplayBuffer.build_balanced(datasets.task("train", TASK_A_CLASSES),
                                          TASK_A_CLASSES, per_class=per_class, seed=buffer_seed)
    model, controller, info = setup_lora_model(model, r=rank, alpha=alpha, head_mode=head_mode)
    model = model.to(device)
    train = datasets.task("train", TASK_B_CLASSES)
    val = datasets.task("val", TASK_B_CLASSES)
    gen = torch.Generator().manual_seed(train_cfg.shuffle_seed)
    loader = make_loader(train, batch_size=train_cfg.batch_size, shuffle=True, generator=gen)
    rgen = torch.Generator().manual_seed(train_cfg.shuffle_seed + 999)
    closure = _kd_replay_closure(teacher, buffer, temperature, lambda_kd,
                                 TASK_A_CLASSES, replay_bs, rgen)
    res = fit(model, loader, val, train_cfg, device, closure,
              allowed_classes=TASK_B_CLASSES, post_step_hook=controller,
              monitor=_taskB_monitor(datasets))
    return {"model": model, "train_result": res, "method": "kd_lora_replay",
            "lora_info": info, "replay": {"per_class": per_class, "buffer_size": len(buffer),
                                          "replay_bs": replay_bs},
            "kd": {"temperature": temperature, "lambda_kd": lambda_kd,
                   "support_ce": "Task B + replay(Task A)",
                   "support_kd": "replay(Task A) over old classes"},
            "trainable": info}


# --------------------------------------------------------------------------- #
# Evaluation                                                                    #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_model(model: nn.Module, datasets: MNIST1DData, device: torch.device,
                   split: str = "test") -> Dict[str, object]:
    """Full metric bundle on a split: Task A, Task B, seen (0-7)."""
    outA = extract(model, datasets.task(split, TASK_A_CLASSES), device)
    outB = extract(model, datasets.task(split, TASK_B_CLASSES), device)
    outAB = extract(model, datasets.task(split, SEEN_CLASSES), device)

    repA = full_report(outA["logits"], outA["labels"], TASK_A_CLASSES)
    repB = full_report(outB["logits"], outB["labels"], TASK_B_CLASSES)
    # Class-incremental accuracy over SEEN classes 0-7 (argmax restricted to
    # trained classes). This is the primary CL metric: it removes the untrained
    # 8/9 rows (whose arbitrary logits otherwise destabilize the all-10 argmax)
    # while still fully capturing forgetting / new-class bias.
    repA["accuracy_seen07"] = accuracy(outA["logits"], outA["labels"], SEEN_CLASSES)
    repB["accuracy_seen07"] = accuracy(outB["logits"], outB["labels"], SEEN_CLASSES)
    return {
        "split": split,
        "taskA": repA,
        "taskB": repB,
        "seen": {
            "accuracy_seen07": accuracy(outAB["logits"], outAB["labels"], SEEN_CLASSES),
            "accuracy_all10": accuracy(outAB["logits"], outAB["labels"], None),
            "mean_seen_accuracy_seen07": mean_seen_accuracy(
                outAB["logits"], outAB["labels"], SEEN_CLASSES, SEEN_CLASSES),
            "n": int(outAB["labels"].size),
        },
    }


def forgetting_report(base_metrics_taskA: Dict, updated_metrics_taskA: Dict) -> Dict[str, float]:
    return {
        "acc_A_before_masked": base_metrics_taskA["accuracy_masked"],
        "acc_A_after_masked": updated_metrics_taskA["accuracy_masked"],
        "acc_A_before_seen07": base_metrics_taskA["accuracy_seen07"],
        "acc_A_after_seen07": updated_metrics_taskA["accuracy_seen07"],
        "acc_A_before_all10": base_metrics_taskA["accuracy_non_masked"],
        "acc_A_after_all10": updated_metrics_taskA["accuracy_non_masked"],
        "forgetting_masked": base_metrics_taskA["accuracy_masked"]
        - updated_metrics_taskA["accuracy_masked"],
        "forgetting_seen07": base_metrics_taskA["accuracy_seen07"]
        - updated_metrics_taskA["accuracy_seen07"],
        "forgetting_all10": base_metrics_taskA["accuracy_non_masked"]
        - updated_metrics_taskA["accuracy_non_masked"],
    }


# --------------------------------------------------------------------------- #
# Persistence                                                                    #
# --------------------------------------------------------------------------- #
def compact_metrics(ev: Dict, base_taskA: Dict) -> Dict[str, float]:
    """Compact per-run metrics (updated model ``ev`` vs base Task A metrics)."""
    fr = forgetting_report(base_taskA, ev["taskA"])
    return {
        "accA_masked": ev["taskA"]["accuracy_masked"],
        "accA_seen07": ev["taskA"]["accuracy_seen07"],
        "forget_masked": fr["forgetting_masked"],
        "forget_seen07": fr["forgetting_seen07"],
        "accB_masked": ev["taskB"]["accuracy_masked"],
        "accB_seen07": ev["taskB"]["accuracy_seen07"],
        "seen07_overall": ev["seen"]["accuracy_seen07"],
    }


def aggregate_over_seeds(per_seed: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Mean/std/per-seed for each metric key across a list of per-seed dicts."""
    if not per_seed:
        return {}
    keys = per_seed[0].keys()
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [float(d[k]) for d in per_seed]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                  "per_seed": vals}
    return out


def alpha_for(rank: int, scaling: float = 2.0) -> float:
    """LoRA alpha such that alpha/rank == scaling (default scaling = 2)."""
    return scaling * rank


def save_run(out_base: str | Path, experiment: str, seed: int, tag: str,
             config: MLPConfig, metrics: Dict, history: Optional[List] = None,
             state: Optional[Dict] = None, extra: Optional[Dict] = None) -> Path:
    run_dir = make_run_dir(out_base, experiment, seed=seed, tag=tag)
    save_json({"config": config.to_dict(), "seed": seed, "tag": tag,
               "experiment": experiment}, run_dir / "config.json")
    save_json(metrics, run_dir / "metrics.json")
    save_json(capture_environment(include_pip_freeze=False), run_dir / "env.json")
    if history is not None:
        save_json(history, run_dir / "history.json")
    if state is not None:
        save_checkpoint({"model": state, "config": config.to_dict(), "seed": seed},
                        run_dir / "checkpoint.pt")
    if extra is not None:
        save_json(extra, run_dir / "extra.json")
    return run_dir
