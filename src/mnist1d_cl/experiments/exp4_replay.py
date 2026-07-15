"""Exp4 - KD-LoRA + Replay on Task B; per_class in {20,50,100}.

Base LoRA config: rank 8, partial head. Replay buffer is fixed and balanced;
CE on Task B + replay(Task A); KD on replay over old classes.

  python -m mnist1d_cl.experiments.exp4_replay --seeds 0 1 2
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
    p.add_argument("--per-class", type=int, nargs="+", default=[20, 50, 100])
    args = p.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    datasets = build_datasets(data_dir=args.data_dir)
    cfg = common.load_selected_config()
    tb = TrainConfig(epochs=args.epochs, scheduler="cosine", grad_clip=5.0,
                     early_stopping_patience=15)

    for pc in args.per_class:
        per_seed = []
        for seed in args.seeds:
            state = common.ensure_task_a_state(cfg, datasets, device, args.out, seed)
            base = common.evaluate_model(common.fresh_from_state(cfg, state, device),
                                         datasets, device, "test")
            r = common.run_kd_lora_replay(cfg, state, datasets, device, rank=args.rank,
                                          alpha=common.alpha_for(args.rank), head_mode=args.head,
                                          temperature=args.T, lambda_kd=args.lambda_kd,
                                          per_class=pc, replay_bs=128, buffer_seed=seed, train_cfg=tb)
            ev = common.evaluate_model(r["model"], datasets, device, "test")
            cm = common.compact_metrics(ev, base["taskA"])
            per_seed.append(cm)
            common.save_run(args.out, "exp4_replay", seed, f"pc{pc}", cfg,
                            metrics={"eval": ev, "replay": r["replay"], "kd": r["kd"],
                                     "forgetting": common.forgetting_report(base["taskA"], ev["taskA"])},
                            history=r["train_result"]["history"], state=r["model"].state_dict())
        agg = common.aggregate_over_seeds(per_seed)
        print(f"[exp4] pc{pc}: accA_m={agg['accA_masked']['mean']:.3f}±{agg['accA_masked']['std']:.3f} "
              f"accA_s07={agg['accA_seen07']['mean']:.3f} "
              f"accB_m={agg['accB_masked']['mean']:.3f} forgetB_s07={agg['forget_seen07']['mean']:.3f}")


if __name__ == "__main__":
    main()
