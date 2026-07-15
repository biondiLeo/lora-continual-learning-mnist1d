"""Exp6 - OOD detection (classes 8, 9 held out; used ONLY here).

Scores are computed EXCLUSIVELY from the seen-class logits (0-4 before Task B;
0-7 after Task B); the unsupervised 8/9 rows never enter the scores. Every score
is oriented so higher == more OOD, and the direction is checked automatically.

Frames:
  A (before Task B): Old backbone, ID = classes 0-4, OOD = 8-9.
  B (after Task B) : FFT, LoRA, KD-LoRA, KD-LoRA+Replay; ID = 0-7, OOD = 8-9.

Methods use the frozen main configs (LoRA base = rank 8 / partial head; KD T=2,
lambda=2; Replay per_class=100). Nothing is selected on the OOD test.

  python -m mnist1d_cl.experiments.exp6_ood --seeds 0 1 2
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import roc_curve

from ..constants import TASK_A_CLASSES, SEEN_CLASSES, OOD_CLASSES
from ..data import build_datasets
from ..models import extract
from ..training import TrainConfig
from ..ood import seen_ood_scores, evaluate_ood
from ..plotting import plot_score_distributions, plot_roc
from ..utils import get_device, save_json, save_csv_rows, make_run_dir
from . import common

SCORES = ["MSP", "MaxLogit", "Energy_T1", "Entropy"]
# frozen main configs
LORA_RANK, LORA_HEAD, KD_T, KD_LAMBDA, REPLAY_PC = 8, "partial", 2.0, 2.0, 100


def _logits(model, datasets, device, classes):
    return extract(model, datasets.task("test", classes), device)["logits"]


def eval_ood_for_model(model, datasets, device, seen):
    id_log = _logits(model, datasets, device, seen)
    ood_log = _logits(model, datasets, device, OOD_CLASSES)
    ood8 = _logits(model, datasets, device, [8])
    ood9 = _logits(model, datasets, device, [9])
    id_s = seen_ood_scores(id_log, seen)
    ood_s = seen_ood_scores(ood_log, seen)
    ood8_s = seen_ood_scores(ood8, seen)
    ood9_s = seen_ood_scores(ood9, seen)
    out = {}
    for s in SCORES:
        out[s] = {"all": evaluate_ood(id_s[s], ood_s[s]),
                  "ood8": evaluate_ood(id_s[s], ood8_s[s]),
                  "ood9": evaluate_ood(id_s[s], ood9_s[s])}
    return out, id_s, ood_s, (len(id_log), len(ood_log))


def compact(m):
    return {"auroc": m["auroc"], "aupr_in": m["aupr_in"], "aupr_out": m["aupr_out"],
            "fpr95": m["fpr@95tpr"], "direction_ok": m["direction_ok"]}


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
    out_dir = make_run_dir(args.out, "exp6_ood", tag="ood_detection")
    print(f"[ood] device={device} config=({cfg.hidden1},{cfg.hidden2}) out={out_dir}")

    # results[frame][method][score]['all'|'ood8'|'ood9'] = list over seeds of compact
    frames = {"before": ["Old"], "after": ["FFT", "LoRA", "KD-LoRA", "KD-LoRA+Replay"]}
    results = {fr: {m: {s: {"all": [], "ood8": [], "ood9": []} for s in SCORES}
                    for m in ms} for fr, ms in frames.items()}
    counts = {}
    csv_rows = []
    plot_cache = {}  # (frame, method) -> (id_scores, ood_scores) at seed0

    for seed in args.seeds:
        print(f"[ood] seed {seed}: regenerating sequential models from cached Old")
        state = common.ensure_task_a_state(cfg, datasets, device, args.out, seed)
        old_model = common.fresh_from_state(cfg, state, device)
        models = {
            "before": {"Old": (old_model, TASK_A_CLASSES)},
            "after": {
                "FFT": (common.run_fft(cfg, state, datasets, device, tb)["model"], SEEN_CLASSES),
                "LoRA": (common.run_lora(cfg, state, datasets, device, LORA_RANK,
                                         common.alpha_for(LORA_RANK), LORA_HEAD, tb)["model"], SEEN_CLASSES),
                "KD-LoRA": (common.run_kd_lora(cfg, state, datasets, device, LORA_RANK,
                                               common.alpha_for(LORA_RANK), LORA_HEAD, KD_T, KD_LAMBDA, tb)["model"], SEEN_CLASSES),
                "KD-LoRA+Replay": (common.run_kd_lora_replay(cfg, state, datasets, device, LORA_RANK,
                                    common.alpha_for(LORA_RANK), LORA_HEAD, KD_T, KD_LAMBDA,
                                    REPLAY_PC, replay_bs=128, buffer_seed=seed, train_cfg=tb)["model"], SEEN_CLASSES),
            },
        }
        for fr, ms in models.items():
            for method, (model, seen) in ms.items():
                res, id_s, ood_s, (n_id, n_ood) = eval_ood_for_model(model, datasets, device, seen)
                counts[(fr, method)] = {"n_id": n_id, "n_ood": n_ood, "seen": list(seen)}
                if seed == args.seeds[0]:
                    plot_cache[(fr, method)] = (id_s, ood_s)
                for s in SCORES:
                    for subset in ("all", "ood8", "ood9"):
                        results[fr][method][s][subset].append(compact(res[s][subset]))
                        c = res[s][subset]
                        csv_rows.append({"seed": seed, "frame": fr, "method": method,
                                         "score": s, "subset": subset, **compact(c),
                                         "mean_id": c["mean_id_score"], "mean_ood": c["mean_ood_score"],
                                         "n_id": c["n_id"], "n_ood": c["n_ood"]})

    # aggregate
    agg = {fr: {m: {s: {sub: common.aggregate_over_seeds(results[fr][m][s][sub])
                        for sub in ("all", "ood8", "ood9")}
                    for s in SCORES} for m in ms} for fr, ms in frames.items()}

    # best score (after frame) by mean AUROC averaged over after methods
    best_score = max(SCORES, key=lambda s: np.mean(
        [agg["after"][m][s]["all"]["auroc"]["mean"] for m in frames["after"]]))

    save_json({"seeds": args.seeds, "config": cfg.to_dict(),
               "frozen_configs": {"lora_rank": LORA_RANK, "lora_head": LORA_HEAD,
                                  "kd_T": KD_T, "kd_lambda": KD_LAMBDA, "replay_per_class": REPLAY_PC},
               "score_orientation": "higher == more OOD (scores from seen-class logits only)",
               "best_score_after": best_score, "counts": {f"{k[0]}/{k[1]}": v for k, v in counts.items()},
               "aggregate": agg, "per_seed": results}, out_dir / "summary.json")
    save_csv_rows(csv_rows, out_dir / "ood_metrics.csv")

    # report-ready synthetic table (AUROC mean±std, 'all' subset)
    rr = ["# OOD detection - AUROC (mean±std over seeds), scores from seen logits\n"]
    rr.append(f"Best score (after): {best_score}\n")
    for fr, ms in frames.items():
        rr.append(f"\n## Frame {fr} (ID={'0-4' if fr=='before' else '0-7'}, OOD=8-9)\n")
        rr.append("method | " + " | ".join(SCORES))
        rr.append("---|" + "|".join(["---"] * len(SCORES)))
        for m in ms:
            cells = [f"{agg[fr][m][s]['all']['auroc']['mean']:.3f}±{agg[fr][m][s]['all']['auroc']['std']:.3f}"
                     for s in SCORES]
            rr.append(f"{m} | " + " | ".join(cells))
    (out_dir / "report_table.md").write_text("\n".join(rr), encoding="utf-8")

    # plots for the best score (seed0)
    # distributions: Old(before) and KD-LoRA+Replay(after) as representatives
    for fr, method in [("before", "Old"), ("after", "KD-LoRA+Replay")]:
        id_s, ood_s = plot_cache[(fr, method)]
        plot_score_distributions(id_s[best_score], ood_s[best_score],
                                 out_dir / f"scoredist_{fr}_{method}_{best_score}.png",
                                 title=f"{fr} {method} | {best_score} (seed{args.seeds[0]})",
                                 score_name=best_score)
    # ROC overlay of after methods for the best score (seed0)
    curves = {}
    for m in frames["after"]:
        id_s, ood_s = plot_cache[("after", m)]
        scores = np.concatenate([id_s[best_score], ood_s[best_score]])
        is_ood = np.concatenate([np.zeros(len(id_s[best_score])), np.ones(len(ood_s[best_score]))])
        fpr, tpr, _ = roc_curve(is_ood, scores)
        au = agg["after"][m][best_score]["all"]["auroc"]["mean"]
        curves[m] = (fpr, tpr, au)
    plot_roc(curves, out_dir / f"roc_after_{best_score}.png",
             title=f"ROC (after Task B) | {best_score} | seed{args.seeds[0]}")

    # print
    print(f"\n[ood] best score (after) = {best_score}")
    for fr, ms in frames.items():
        print(f"\n== {fr} (ID={'0-4' if fr=='before' else '0-7'}, OOD=8-9) AUROC mean±std ==")
        for m in ms:
            cells = " ".join(f"{s}={agg[fr][m][s]['all']['auroc']['mean']:.3f}±{agg[fr][m][s]['all']['auroc']['std']:.3f}"
                             for s in SCORES)
            dirs = all(all(cc["direction_ok"] for cc in results[fr][m][s]["all"]) for s in SCORES)
            print(f"  {m:16s} {cells}  [dir_ok={dirs}]")
    print(f"\n[ood] counts: " + "; ".join(f"{k[0]}/{k[1]}:ID={v['n_id']},OOD={v['n_ood']}"
                                          for k, v in counts.items()))
    print(f"[ood] saved to {out_dir}")


if __name__ == "__main__":
    main()
