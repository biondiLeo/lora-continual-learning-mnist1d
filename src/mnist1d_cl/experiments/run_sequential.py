"""Experiments 0-4: sequential class-incremental study, multi-seed.

Per seed s: train ONE Task A backbone, save it, and initialize every Task B
method from that identical checkpoint (fair comparison). Runs:
  Exp0  joint(0-7) baseline (diagnostic upper bound)
  Exp1  Full Fine-Tuning
  Exp2  LoRA grid: rank in {2,4,8,16} x head in {frozen,full,partial}
  Exp3  KD-LoRA   (base = LoRA rank8/partial; T,lambda selected on validation)
  Exp4  KD-LoRA + Replay (per_class in {20,50,100})

Checkpoint/hyperparameter selection uses validation only; final metrics are on
test. seen-0-7 is the primary class-incremental accuracy.

Run:  python -m mnist1d_cl.experiments.run_sequential --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from ..constants import TASK_A_CLASSES, SEEN_CLASSES
from ..data import build_datasets
from ..models import MLPConfig, build_mlp
from ..training import TrainConfig, train_supervised
from ..utils import get_device, seed_all, save_json, save_csv_rows, make_run_dir
from . import common

RANKS = [2, 4, 8, 16]
HEADS = ["frozen", "full", "partial"]
REPLAY_SIZES = [20, 50, 100]
KD_BASE_RANK, KD_BASE_HEAD = 8, "partial"
KD_T_GRID = [2.0, 4.0]
KD_LAMBDA_GRID = [0.5, 1.0, 2.0]


def eval_both(model, datasets, device):
    return {"val": common.evaluate_model(model, datasets, device, "val"),
            "test": common.evaluate_model(model, datasets, device, "test")}


def select_kd_hparams(cfg, base_state, datasets, device, tb_cfg):
    """Select (T, lambda) on validation for LoRA rank8/partial (seed 0)."""
    best, best_score = None, -1.0
    trials = []
    for T in KD_T_GRID:
        for lam in KD_LAMBDA_GRID:
            r = common.run_kd_lora(cfg, base_state, datasets, device,
                                   rank=KD_BASE_RANK, alpha=common.alpha_for(KD_BASE_RANK),
                                   head_mode=KD_BASE_HEAD, temperature=T, lambda_kd=lam,
                                   train_cfg=tb_cfg)
            ev = common.evaluate_model(r["model"], datasets, device, "val")
            # balance retention (masked A) and plasticity (masked B) on validation
            score = 0.5 * (ev["taskA"]["accuracy_masked"] + ev["taskB"]["accuracy_masked"])
            trials.append({"T": T, "lambda_kd": lam,
                           "val_accA_masked": ev["taskA"]["accuracy_masked"],
                           "val_accB_masked": ev["taskB"]["accuracy_masked"],
                           "score": score})
            if score > best_score:
                best_score, best = score, {"temperature": T, "lambda_kd": lam}
    return best, trials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--taskA-epochs", type=int, default=80)
    parser.add_argument("--taskB-epochs", type=int, default=60)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"[seq] device={device} seeds={args.seeds}")
    datasets = build_datasets(data_dir=args.data_dir)
    cfg = common.load_selected_config()
    print(f"[seq] standard config: (h1={cfg.hidden1}, h2={cfg.hidden2})")

    ta_cfg = TrainConfig(epochs=args.taskA_epochs, lr=1e-3, weight_decay=1e-4,
                         batch_size=128, scheduler="cosine", grad_clip=5.0,
                         early_stopping_patience=15)
    tb_cfg = TrainConfig(epochs=args.taskB_epochs, lr=1e-3, weight_decay=1e-4,
                         batch_size=128, scheduler="cosine", grad_clip=5.0,
                         early_stopping_patience=15)

    run_dir = make_run_dir(args.out, "sequential", tag="exp0-4")
    print(f"[seq] run dir: {run_dir}")
    t0 = time.time()

    # ---- KD hyperparameter selection (validation, seed 0) ------------- #
    seed_all(0)
    a0 = common.train_task_a(cfg, datasets, device, ta_cfg, seed=0)
    kd_hp, kd_trials = select_kd_hparams(cfg, a0["state"], datasets, device, tb_cfg)
    print(f"[seq] selected KD hparams (val): {kd_hp}")
    save_json({"selected": kd_hp, "trials": kd_trials,
               "base_lora": {"rank": KD_BASE_RANK, "head": KD_BASE_HEAD}},
              run_dir / "kd_selection.json")

    # ---- per-seed runs ------------------------------------------------ #
    per_seed = {seed: {} for seed in args.seeds}
    base_test_taskA = {}
    lora_rows = []
    for seed in args.seeds:
        print(f"\n[seq] ===== seed {seed} =====")
        # Exp0 joint
        seed_all(seed)
        jmodel = build_mlp(cfg).to(device)
        jres = train_supervised(jmodel, datasets.task("train", SEEN_CLASSES),
                                datasets.task("val", SEEN_CLASSES), ta_cfg, device,
                                allowed_classes=SEEN_CLASSES)
        jtest = common.evaluate_model(jmodel, datasets, device, "test")
        per_seed[seed]["joint"] = {"test_seen07": jtest["seen"]["accuracy_seen07"],
                                   "val_seen07": jres["best_val_acc"]}

        # Task A backbone (shared by all Task B methods)
        a = a0 if seed == 0 else common.train_task_a(cfg, datasets, device, ta_cfg, seed=seed)
        state = a["state"]
        base_eval = eval_both(a["model"], datasets, device)
        base_test_taskA[seed] = base_eval["test"]["taskA"]
        per_seed[seed]["taskA"] = {
            "test_accA_masked": base_eval["test"]["taskA"]["accuracy_masked"],
            "test_accA_seen07": base_eval["test"]["taskA"]["accuracy_seen07"],
            "val_accA_masked": base_eval["val"]["taskA"]["accuracy_masked"],
        }
        print(f"  taskA test masked accA={base_eval['test']['taskA']['accuracy_masked']:.3f}")

        # Exp1 FFT
        fft = common.run_fft(cfg, state, datasets, device, tb_cfg)
        fft_ev = eval_both(fft["model"], datasets, device)
        per_seed[seed]["fft"] = common.compact_metrics(fft_ev["test"], base_test_taskA[seed])

        # Exp2 LoRA grid
        per_seed[seed]["lora"] = {}
        for r in RANKS:
            for h in HEADS:
                lr = common.run_lora(cfg, state, datasets, device, rank=r,
                                     alpha=common.alpha_for(r), head_mode=h, train_cfg=tb_cfg)
                lev = eval_both(lr["model"], datasets, device)
                cm = common.compact_metrics(lev["test"], base_test_taskA[seed])
                cm["val_accB_masked"] = lev["val"]["taskB"]["accuracy_masked"]
                cm["pct_trainable"] = lr["lora_info"]["pct_trainable"]
                cm["checks_ok"] = all(lr["checks"].values())
                per_seed[seed]["lora"][f"r{r}_{h}"] = cm
                lora_rows.append({"seed": seed, "rank": r, "head": h, **cm})

        # Exp3 KD-LoRA (base r8/partial + selected hparams)
        kd = common.run_kd_lora(cfg, state, datasets, device, rank=KD_BASE_RANK,
                                alpha=common.alpha_for(KD_BASE_RANK), head_mode=KD_BASE_HEAD,
                                temperature=kd_hp["temperature"], lambda_kd=kd_hp["lambda_kd"],
                                train_cfg=tb_cfg)
        kd_ev = eval_both(kd["model"], datasets, device)
        per_seed[seed]["kd_lora"] = common.compact_metrics(kd_ev["test"], base_test_taskA[seed])

        # Exp4 KD-LoRA + Replay
        per_seed[seed]["replay"] = {}
        for pc in REPLAY_SIZES:
            rp = common.run_kd_lora_replay(cfg, state, datasets, device, rank=KD_BASE_RANK,
                                           alpha=common.alpha_for(KD_BASE_RANK), head_mode=KD_BASE_HEAD,
                                           temperature=kd_hp["temperature"], lambda_kd=kd_hp["lambda_kd"],
                                           per_class=pc, replay_bs=128, buffer_seed=seed, train_cfg=tb_cfg)
            rev = eval_both(rp["model"], datasets, device)
            per_seed[seed]["replay"][f"pc{pc}"] = common.compact_metrics(rev["test"], base_test_taskA[seed])

    # ---- aggregate ---------------------------------------------------- #
    seeds = args.seeds
    agg = {}
    agg["joint_test_seen07"] = common.aggregate_over_seeds(
        [{"seen07": per_seed[s]["joint"]["test_seen07"]} for s in seeds])
    agg["taskA_test_masked"] = common.aggregate_over_seeds(
        [{"accA_masked": per_seed[s]["taskA"]["test_accA_masked"]} for s in seeds])
    agg["fft"] = common.aggregate_over_seeds([per_seed[s]["fft"] for s in seeds])
    agg["kd_lora"] = common.aggregate_over_seeds([per_seed[s]["kd_lora"] for s in seeds])
    agg["lora"] = {}
    for r in RANKS:
        for h in HEADS:
            key = f"r{r}_{h}"
            agg["lora"][key] = common.aggregate_over_seeds(
                [per_seed[s]["lora"][key] for s in seeds])
    agg["replay"] = {}
    for pc in REPLAY_SIZES:
        agg["replay"][f"pc{pc}"] = common.aggregate_over_seeds(
            [per_seed[s]["replay"][f"pc{pc}"] for s in seeds])

    # primary LoRA selection by mean validation Task B masked accuracy (plan §5.1)
    primary = max(
        [(r, h) for r in RANKS for h in HEADS],
        key=lambda rh: agg["lora"][f"r{rh[0]}_{rh[1]}"]["val_accB_masked"]["mean"])
    primary_key = f"r{primary[0]}_{primary[1]}"

    save_json({"config": cfg.to_dict(), "seeds": seeds, "kd_hparams": kd_hp,
               "per_seed": per_seed, "aggregate": agg,
               "primary_lora": {"rank": primary[0], "head": primary[1], "key": primary_key,
                                "criterion": "max mean val Task B masked accuracy (plan 5.1)"}},
              run_dir / "summary.json")
    save_csv_rows(lora_rows, run_dir / "lora_grid.csv")

    # ---- print summary ------------------------------------------------ #
    def line(name, m):
        return (f"{name:24s} accA_m={m['accA_masked']['mean']:.3f}±{m['accA_masked']['std']:.3f} "
                f"accA_s07={m['accA_seen07']['mean']:.3f} forgetB_s07={m['forget_seen07']['mean']:.3f} "
                f"accB_m={m['accB_masked']['mean']:.3f}±{m['accB_masked']['std']:.3f} "
                f"accB_s07={m['accB_seen07']['mean']:.3f}")

    print("\n===== AGGREGATE (test, mean±std over seeds) =====")
    print(f"joint(0-7) seen07     = {agg['joint_test_seen07']['seen07']['mean']:.3f}"
          f"±{agg['joint_test_seen07']['seen07']['std']:.3f}")
    print(f"taskA masked accA     = {agg['taskA_test_masked']['accA_masked']['mean']:.3f}"
          f"±{agg['taskA_test_masked']['accA_masked']['std']:.3f}")
    print(line("FFT", agg["fft"]))
    print(f"[primary LoRA by val = {primary_key}]")
    for r in RANKS:
        for h in HEADS:
            print(line(f"LoRA r{r}/{h}", agg["lora"][f"r{r}_{h}"]))
    print(line(f"KD-LoRA (r8/part, T={kd_hp['temperature']},l={kd_hp['lambda_kd']})", agg["kd_lora"]))
    for pc in REPLAY_SIZES:
        print(line(f"KD-LoRA+Replay pc{pc}", agg["replay"][f"pc{pc}"]))

    print(f"\n[seq] elapsed {time.time() - t0:.1f}s. saved to {run_dir}")


if __name__ == "__main__":
    main()
