"""Exp1 - Full Fine-Tuning on Task B (from the shared Task A backbone).

  python -m mnist1d_cl.experiments.exp1_fft --seeds 0 1 2
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ..data import build_datasets
from ..training import TrainConfig
from ..utils import get_device
from . import common


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="outputs")
    p.add_argument("--epochs", type=int, default=60)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    datasets = build_datasets(data_dir=args.data_dir)
    cfg = common.load_selected_config()
    tb = TrainConfig(epochs=args.epochs, scheduler="cosine", grad_clip=5.0,
                     early_stopping_patience=15)

    per_seed = []
    for seed in args.seeds:
        state = common.ensure_task_a_state(cfg, datasets, device, args.out, seed)
        base = common.evaluate_model(common.fresh_from_state(cfg, state, device),
                                     datasets, device, "test")
        r = common.run_fft(cfg, state, datasets, device, tb)
        ev = common.evaluate_model(r["model"], datasets, device, "test")
        cm = common.compact_metrics(ev, base["taskA"])
        per_seed.append(cm)
        common.save_run(args.out, "exp1_fft", seed, "fft", cfg,
                        metrics={"eval": ev, "forgetting": common.forgetting_report(base["taskA"], ev["taskA"])},
                        history=r["train_result"]["history"], state=r["model"].state_dict())
        print(f"[exp1] seed {seed}: accA_m={cm['accA_masked']:.3f} "
              f"forgetB_s07={cm['forget_seen07']:.3f} accB_m={cm['accB_masked']:.3f}")

    agg = common.aggregate_over_seeds(per_seed)
    print(f"[exp1] FFT accA_m={agg['accA_masked']['mean']:.3f}±{agg['accA_masked']['std']:.3f} "
          f"accB_m={agg['accB_masked']['mean']:.3f}±{agg['accB_masked']['std']:.3f}")


if __name__ == "__main__":
    main()
