"""Exp3 - KD-LoRA on Task B (teacher = frozen Task A model; LwF on Task B data).

Base LoRA config: rank 8, partial head. T/lambda default to (2.0, 1.0) or read
from a kd_selection.json produced by run_sequential.

  python -m mnist1d_cl.experiments.exp3_kdlora --seeds 0 1 2 --T 2 --lambda-kd 1
"""

from __future__ import annotations

import argparse

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
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--head", default="partial")
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--lambda-kd", type=float, default=1.0)
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
        r = common.run_kd_lora(cfg, state, datasets, device, rank=args.rank,
                               alpha=common.alpha_for(args.rank), head_mode=args.head,
                               temperature=args.T, lambda_kd=args.lambda_kd, train_cfg=tb)
        ev = common.evaluate_model(r["model"], datasets, device, "test")
        cm = common.compact_metrics(ev, base["taskA"])
        per_seed.append(cm)
        common.save_run(args.out, "exp3_kdlora", seed, f"r{args.rank}_{args.head}", cfg,
                        metrics={"eval": ev, "kd": r["kd"],
                                 "forgetting": common.forgetting_report(base["taskA"], ev["taskA"])},
                        history=r["train_result"]["history"], state=r["model"].state_dict())
        print(f"[exp3] seed {seed}: accA_m={cm['accA_masked']:.3f} "
              f"forgetB_s07={cm['forget_seen07']:.3f} accB_m={cm['accB_masked']:.3f}")

    agg = common.aggregate_over_seeds(per_seed)
    print(f"[exp3] KD-LoRA accA_m={agg['accA_masked']['mean']:.3f}±{agg['accA_masked']['std']:.3f} "
          f"accB_m={agg['accB_masked']['mean']:.3f}±{agg['accB_masked']['std']:.3f}")


if __name__ == "__main__":
    main()
