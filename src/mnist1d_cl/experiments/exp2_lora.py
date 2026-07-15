"""Exp2 - LoRA grid on Task B: rank in {2,4,8,16} x head in {frozen,full,partial}.

  python -m mnist1d_cl.experiments.exp2_lora --seeds 0 1 2
"""

from __future__ import annotations

import argparse

import torch

from ..data import build_datasets
from ..training import TrainConfig
from ..utils import get_device, save_csv_rows
from . import common

RANKS = [2, 4, 8, 16]
HEADS = ["frozen", "full", "partial"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="outputs")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    p.add_argument("--heads", nargs="+", default=HEADS)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    datasets = build_datasets(data_dir=args.data_dir)
    cfg = common.load_selected_config()
    tb = TrainConfig(epochs=args.epochs, scheduler="cosine", grad_clip=5.0,
                     early_stopping_patience=15)

    rows = []
    for seed in args.seeds:
        state = common.ensure_task_a_state(cfg, datasets, device, args.out, seed)
        base = common.evaluate_model(common.fresh_from_state(cfg, state, device),
                                     datasets, device, "test")
        for r in args.ranks:
            for h in args.heads:
                run = common.run_lora(cfg, state, datasets, device, rank=r,
                                      alpha=common.alpha_for(r), head_mode=h, train_cfg=tb)
                ev = common.evaluate_model(run["model"], datasets, device, "test")
                cm = common.compact_metrics(ev, base["taskA"])
                cm.update({"seed": seed, "rank": r, "head": h,
                           "pct_trainable": run["lora_info"]["pct_trainable"],
                           "checks_ok": all(run["checks"].values())})
                rows.append(cm)
                print(f"[exp2] seed{seed} r{r}/{h}: accA_m={cm['accA_masked']:.3f} "
                      f"accB_m={cm['accB_masked']:.3f} forgetB_s07={cm['forget_seen07']:.3f} "
                      f"checks={cm['checks_ok']}")

    out_csv = common.save_run(args.out, "exp2_lora", args.seeds[0], "grid", cfg,
                              metrics={"n_rows": len(rows)}).parent
    save_csv_rows(rows, out_csv / "lora_grid.csv")
    print(f"[exp2] saved grid ({len(rows)} rows) under {out_csv}")


if __name__ == "__main__":
    main()
