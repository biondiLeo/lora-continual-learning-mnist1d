"""End-to-end smoke test: tiny training of every method + invariant checks.

Run:  python -m mnist1d_cl.experiments.smoke
Fast, reduced-scale; verifies wiring, invariants, finiteness and run-dir saving
BEFORE any full experiment. Not a scientific result.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ..constants import TASK_A_CLASSES
from ..data import build_datasets
from ..models import MLPConfig
from ..training import TrainConfig
from ..utils import get_device, seed_all
from . import common


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the CL pipeline")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"[smoke] device={device}")
    datasets = build_datasets(data_dir=args.data_dir)

    config = MLPConfig(hidden1=32, hidden2=16)
    ta_cfg = TrainConfig(epochs=args.epochs, early_stopping_patience=None,
                         scheduler="none", batch_size=128)
    tb_cfg = TrainConfig(epochs=args.epochs, early_stopping_patience=None,
                         scheduler="none", batch_size=128)

    seed = 0
    seed_all(seed)
    a = common.train_task_a(config, datasets, device, ta_cfg, seed=seed)
    base_state = a["state"]
    base_eval = common.evaluate_model(a["model"], datasets, device, split="test")
    accA0 = base_eval["taskA"]["accuracy_masked"]
    print(f"[smoke] Task A backbone: test masked accA={accA0:.3f} "
          f"(non-masked {base_eval['taskA']['accuracy_non_masked']:.3f})")
    assert np.isfinite(accA0)

    results = {}

    fft = common.run_fft(config, base_state, datasets, device, tb_cfg)
    results["fft"] = fft

    lora = common.run_lora(config, base_state, datasets, device, rank=4, alpha=8,
                           head_mode="partial", train_cfg=tb_cfg)
    assert lora["checks"]["delta_zero_init"], "LoRA delta not zero at init"
    assert lora["checks"]["output_equiv_init"], "LoRA output != backbone at init"
    assert lora["checks"]["base_frozen"], "LoRA base not frozen"
    assert lora["checks"].get("frozen_head_rows_unchanged", True), "frozen head rows moved"
    print(f"[smoke] LoRA checks OK; trainable={lora['lora_info']['n_trainable']} "
          f"({lora['lora_info']['pct_trainable']:.2f}%)")
    results["lora"] = lora

    kd = common.run_kd_lora(config, base_state, datasets, device, rank=4, alpha=8,
                            head_mode="partial", temperature=2.0, lambda_kd=1.0,
                            train_cfg=tb_cfg)
    results["kd_lora"] = kd

    rp = common.run_kd_lora_replay(config, base_state, datasets, device, rank=4, alpha=8,
                                   head_mode="partial", temperature=2.0, lambda_kd=1.0,
                                   per_class=20, replay_bs=64, train_cfg=tb_cfg)
    results["kd_lora_replay"] = rp

    print(f"\n{'method':16s} {'accA_nm':>8s} {'accA_m':>8s} {'forgetA':>8s} "
          f"{'accB_nm':>8s} {'accB_m':>8s}")
    for name, r in results.items():
        ev = common.evaluate_model(r["model"], datasets, device, split="test")
        fr = common.forgetting_report(base_eval["taskA"], ev["taskA"])
        for v in [ev["taskA"]["accuracy_non_masked"], ev["taskB"]["accuracy_masked"]]:
            assert np.isfinite(v)
        print(f"{name:16s} {ev['taskA']['accuracy_non_masked']:8.3f} "
              f"{ev['taskA']['accuracy_masked']:8.3f} {fr['forgetting_non_masked']:8.3f} "
              f"{ev['taskB']['accuracy_non_masked']:8.3f} {ev['taskB']['accuracy_masked']:8.3f}")

    # Save one small run dir to verify persistence.
    run_dir = common.save_run(args.out, "smoke", seed, "demo", config,
                              metrics={"base_taskA": base_eval["taskA"],
                                       "lora_checks": lora["checks"]},
                              history=lora["train_result"]["history"],
                              state=lora["model"].state_dict())
    print(f"\n[smoke] saved run dir: {run_dir}")
    print("[smoke] OK")


if __name__ == "__main__":
    main()
