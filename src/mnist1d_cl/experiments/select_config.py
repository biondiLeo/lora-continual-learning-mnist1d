"""Preliminary hyperparameter selection on VALIDATION (never on test).

1) Select the standard MLP hidden config among candidates by Task A validation
   accuracy (masked to Task A), preferring the smallest config within a margin
   of the best. Writes ``configs/selected_config.json``.
2) Report a joint-training upper bound (train on classes 0-7) for the selected
   config, as a reference for the underfitting analysis.
3) Run a single-seed preview of the four Task B methods with the selected config
   and report validation metrics (retention of Task A vs learning of Task B).

Run:  python -m mnist1d_cl.experiments.select_config
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ..constants import TASK_A_CLASSES, SEEN_CLASSES
from ..data import build_datasets
from ..models import MLPConfig, build_mlp
from ..training import TrainConfig, train_supervised
from ..utils import get_device, seed_all, save_json, load_json
from . import common

CANDIDATES = [(32, 32), (64, 32), (64, 64), (100, 100), (128, 64)]
SELECT_MARGIN = 0.01  # prefer smaller config within 1% of best val accuracy


def _count_params(cfg: MLPConfig) -> int:
    return sum(p.numel() for p in build_mlp(cfg).parameters())


def select_standard_config(datasets, device, sel_cfg: TrainConfig, seed: int = 0):
    trainA = datasets.task("train", TASK_A_CLASSES)
    valA = datasets.task("val", TASK_A_CLASSES)
    rows = []
    for (h1, h2) in CANDIDATES:
        cfg = MLPConfig(hidden1=h1, hidden2=h2)
        seed_all(seed)
        model = build_mlp(cfg).to(device)
        res = train_supervised(model, trainA, valA, sel_cfg, device,
                               allowed_classes=TASK_A_CLASSES)
        rows.append({
            "hidden1": h1, "hidden2": h2,
            "val_accA_masked": res["best_val_acc"],
            "val_loss": res["best_val_loss"],
            "best_epoch": res["best_epoch"],
            "n_params": _count_params(cfg),
        })
        print(f"  (h1={h1:3d}, h2={h2:3d})  val_accA={res['best_val_acc']:.4f}  "
              f"params={rows[-1]['n_params']:6d}  best_epoch={res['best_epoch']}")

    best = max(r["val_accA_masked"] for r in rows)
    eligible = [r for r in rows if r["val_accA_masked"] >= best - SELECT_MARGIN]
    chosen = min(eligible, key=lambda r: r["n_params"])
    return chosen, rows, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--preview", action="store_true", default=True)
    parser.add_argument("--no-preview", dest="preview", action="store_false")
    parser.add_argument("--reuse-config", action="store_true",
                        help="skip the candidate sweep and load configs/selected_config.json")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"[select] device={device}")
    datasets = build_datasets(data_dir=args.data_dir)

    sel_cfg = TrainConfig(epochs=args.epochs, lr=1e-3, weight_decay=1e-4,
                          batch_size=128, scheduler="cosine", grad_clip=5.0,
                          early_stopping_patience=12)

    if args.reuse_config:
        cfg = MLPConfig.from_dict(load_json("configs/selected_config.json")["config"])
        print(f"[select] reusing config (h1={cfg.hidden1}, h2={cfg.hidden2})")
    else:
        print("[select] standard-config candidates (Task A validation, masked):")
        chosen, rows, best = select_standard_config(datasets, device, sel_cfg)
        cfg = MLPConfig(hidden1=chosen["hidden1"], hidden2=chosen["hidden2"])
        print(f"[select] chosen standard config: (h1={cfg.hidden1}, h2={cfg.hidden2}) "
              f"val_accA={chosen['val_accA_masked']:.4f} (best={best:.4f})")

        # Joint upper bound (train on 0-7) for reference.
        seed_all(0)
        jmodel = build_mlp(cfg).to(device)
        jres = train_supervised(jmodel, datasets.task("train", SEEN_CLASSES),
                                datasets.task("val", SEEN_CLASSES), sel_cfg, device,
                                allowed_classes=SEEN_CLASSES)
        print(f"[select] joint(0-7) upper bound: val_acc={jres['best_val_acc']:.4f}")

        save_json({
            "config": cfg.to_dict(),
            "selection_metric": "Task A validation accuracy (masked to 0-4)",
            "select_margin": SELECT_MARGIN,
            "best_val_accA": best,
            "chosen": chosen,
            "candidates": rows,
            "joint_upper_bound_val_acc": jres["best_val_acc"],
            "selection_train_config": sel_cfg.to_dict(),
        }, "configs/selected_config.json")
        print("[select] wrote configs/selected_config.json")

    if not args.preview:
        return

    # ---- single-seed preview of the four Task B methods (VALIDATION) ---- #
    print("\n[select] preview of Task B methods (single seed, VALIDATION metrics):")
    tb_cfg = TrainConfig(epochs=args.epochs, lr=1e-3, weight_decay=1e-4,
                         batch_size=128, scheduler="cosine", grad_clip=5.0,
                         early_stopping_patience=12)
    seed_all(0)
    a = common.train_task_a(cfg, datasets, device, sel_cfg, seed=0)
    base_state = a["state"]
    base_val = common.evaluate_model(a["model"], datasets, device, split="val")
    print(f"  Task A backbone: val masked accA={base_val['taskA']['accuracy_masked']:.3f} "
          f"seen07={base_val['taskA']['accuracy_seen07']:.3f}")

    runners = {
        "fft": lambda: common.run_fft(cfg, base_state, datasets, device, tb_cfg),
        "lora(r8,partial)": lambda: common.run_lora(cfg, base_state, datasets, device,
                                                    rank=8, alpha=16, head_mode="partial",
                                                    train_cfg=tb_cfg),
        "kd_lora(r8)": lambda: common.run_kd_lora(cfg, base_state, datasets, device,
                                                 rank=8, alpha=16, head_mode="partial",
                                                 temperature=2.0, lambda_kd=1.0, train_cfg=tb_cfg),
        "kd_replay(r8,50)": lambda: common.run_kd_lora_replay(cfg, base_state, datasets, device,
                                                             rank=8, alpha=16, head_mode="partial",
                                                             temperature=2.0, lambda_kd=1.0,
                                                             per_class=50, replay_bs=128, train_cfg=tb_cfg),
    }
    print(f"\n  {'method':18s} {'accA_m':>7s} {'accA_s07':>9s} {'forgA_s07':>10s} "
          f"{'accB_m':>7s} {'accB_s07':>9s}")
    preview_rows = []
    for name, fn in runners.items():
        r = fn()
        ev = common.evaluate_model(r["model"], datasets, device, split="val")
        fr = common.forgetting_report(base_val["taskA"], ev["taskA"])
        preview_rows.append({"method": name,
                             "accA_masked": ev["taskA"]["accuracy_masked"],
                             "accA_seen07": ev["taskA"]["accuracy_seen07"],
                             "forgA_seen07": fr["forgetting_seen07"],
                             "accB_masked": ev["taskB"]["accuracy_masked"],
                             "accB_seen07": ev["taskB"]["accuracy_seen07"]})
        print(f"  {name:18s} {ev['taskA']['accuracy_masked']:7.3f} "
              f"{ev['taskA']['accuracy_seen07']:9.3f} {fr['forgetting_seen07']:10.3f} "
              f"{ev['taskB']['accuracy_masked']:7.3f} {ev['taskB']['accuracy_seen07']:9.3f}")

    save_json({"base_taskA_val": base_val["taskA"], "preview": preview_rows,
               "config": cfg.to_dict()}, "configs/preview_methods.json")
    print("\n[select] wrote configs/preview_methods.json")


if __name__ == "__main__":
    main()
