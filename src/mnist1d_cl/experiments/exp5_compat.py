"""Exp5 - Representation-compatibility via query-gallery (SEPARATE from the
sequential CL protocol).

  Old       : backbone trained ONLY on Task A (the shared Task A checkpoint).
  FFT-joint : from Old, full fine-tuning on the JOINT classes 0-7.
  LoRA-joint: from Old, LoRA (rank 8, head full) on the JOINT classes 0-7.

Updated models are trained on 0-7 so representation compatibility is isolated
from catastrophic forgetting. Gallery = Old features, Query = updated features,
cosine on L2-normalized features.

Main comparison: FFT vs Old, LoRA vs Old. Old vs Old is kept ONLY as an internal
sanity check (audit of the 100% self-match). Two evaluation sets (primary Task A;
diagnostic Task A+B) x two protocols (self_match; leave_one_out).

  python -m mnist1d_cl.experiments.exp5_compat --seeds 0 1 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..constants import TASK_A_CLASSES, SEEN_CLASSES
from ..data import build_datasets, make_loader
from ..models import extract
from ..training import TrainConfig, train_supervised, fit, supervised_closure
from ..lora import setup_lora_model, delta_is_zero, outputs_equivalent, base_frozen
from ..querygallery import (query_gallery_eval, retrieval_error_analysis,
                            audit_query_gallery, audit_hundred_percent)
from ..plotting import plot_similarity_distributions
from ..utils import get_device, seed_all, save_json, save_csv_rows, make_run_dir
from . import common

RANK, HEAD = 8, "full"
K = 5
EVAL_SETS = {"taskA": TASK_A_CLASSES, "taskAB": SEEN_CLASSES}


def train_joint_fft(cfg, state, datasets, device, tcfg):
    model = common.fresh_from_state(cfg, state, device)
    for p in model.parameters():
        p.requires_grad_(True)
    train_supervised(model, datasets.task("train", SEEN_CLASSES),
                     datasets.task("val", SEEN_CLASSES), tcfg, device,
                     allowed_classes=SEEN_CLASSES)
    return model


def train_joint_lora(cfg, state, datasets, device, tcfg):
    model = common.fresh_from_state(cfg, state, device)
    base_ref = common.fresh_from_state(cfg, state, device)
    model, ctrl, info = setup_lora_model(model, r=RANK, alpha=common.alpha_for(RANK),
                                         head_mode=HEAD)
    model = model.to(device)
    x0 = torch.from_numpy(datasets.task("val", TASK_A_CLASSES).x[:32]).float().to(device)
    checks = {"delta_zero_init": bool(delta_is_zero(model)),
              "output_equiv_init": bool(outputs_equivalent(base_ref, model, x0)),
              "base_frozen": bool(base_frozen(model))}
    gen = torch.Generator().manual_seed(tcfg.shuffle_seed)
    loader = make_loader(datasets.task("train", SEEN_CLASSES), batch_size=tcfg.batch_size,
                         shuffle=True, generator=gen)
    fit(model, loader, datasets.task("val", SEEN_CLASSES), tcfg, device,
        supervised_closure(), allowed_classes=SEEN_CLASSES, post_step_hook=ctrl)
    return model, info, checks


def features(model, datasets, device, classes):
    out = extract(model, datasets.task("test", classes), device)
    return out["features"], out["ids"], out["labels"]


def compact(res, err):
    return {
        "same_sample_top1": res["same_sample_top1"],
        "same_class_top1": res["same_class_top1"],
        f"recall@{K}_class": res[f"recall@{K}_class"],
        "nn_mean_similarity": res["nn_mean_similarity"],
        "mean_same_sample_cosine": res["mean_same_sample_cosine"],
        "mean_same_class_nonself_cosine": res["mean_same_class_nonself_cosine"],
        "mean_diff_class_cosine": res["mean_diff_class_cosine"],
        "margin_same_sample_vs_diff": res["margin_same_sample_vs_diff"],
        "margin_same_class_vs_diff": res["margin_same_class_vs_diff"],
        "class_error_rate": err["class_error_rate"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="outputs")
    p.add_argument("--epochs", type=int, default=80)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    datasets = build_datasets(data_dir=args.data_dir)
    cfg = common.load_selected_config()
    tcfg = TrainConfig(epochs=args.epochs, scheduler="cosine", grad_clip=5.0,
                       early_stopping_patience=15)
    out_dir = make_run_dir(args.out, "exp5_compat", tag="query_gallery")
    print(f"[exp5] device={device} config=({cfg.hidden1},{cfg.hidden2}) out={out_dir}")

    # results[pair][eval_set][protocol] = list over seeds of compact dict
    MAIN_PAIRS = ["fft_vs_old", "lora_vs_old"]
    results = {pair: {es: {pr: [] for pr in ("self_match", "leave_one_out")}
                      for es in EVAL_SETS} for pair in MAIN_PAIRS + ["old_vs_old"]}
    audits = []
    lora_checks_all = []
    csv_rows = []

    for seed in args.seeds:
        print(f"[exp5] seed {seed}: training Old / FFT-joint / LoRA-joint (0-7)")
        old_state = common.ensure_task_a_state(cfg, datasets, device, args.out, seed)
        old_model = common.fresh_from_state(cfg, old_state, device)
        fft_model = train_joint_fft(cfg, old_state, datasets, device, tcfg)
        lora_model, lora_info, lora_checks = train_joint_lora(cfg, old_state, datasets, device, tcfg)
        lora_checks_all.append({"seed": seed, **lora_checks})

        for es_name, classes in EVAL_SETS.items():
            old_f, old_ids, old_lab = features(old_model, datasets, device, classes)
            fft_f, fft_ids, fft_lab = features(fft_model, datasets, device, classes)
            lora_f, lora_ids, lora_lab = features(lora_model, datasets, device, classes)

            # audit (uses Old as gallery; query ids identical -> aligned)
            aud = audit_query_gallery(fft_ids, old_ids, fft_f, old_f)
            aud.update({"seed": seed, "eval_set": es_name})
            audits.append(aud)

            queries = {"fft_vs_old": (fft_f, fft_ids, fft_lab),
                       "lora_vs_old": (lora_f, lora_ids, lora_lab),
                       "old_vs_old": (old_f, old_ids, old_lab)}
            for pair, (qf, qi, ql) in queries.items():
                for protocol in ("self_match", "leave_one_out"):
                    want_groups = (seed == args.seeds[0] and es_name == "taskA"
                                   and protocol == "self_match")
                    res = query_gallery_eval(qf, qi, ql, old_f, old_ids, old_lab,
                                             protocol=protocol, k=K, return_matrix=True,
                                             return_groups=want_groups)
                    err = retrieval_error_analysis(ql, old_lab, res["nn_index"],
                                                   res["valid_row"])
                    cm = compact(res, err)
                    results[pair][es_name][protocol].append(cm)
                    csv_rows.append({"seed": seed, "pair": pair, "eval_set": es_name,
                                     "protocol": protocol, **cm})

                    # 100% audit for old_vs_old self_match
                    if pair == "old_vs_old" and protocol == "self_match":
                        h = audit_hundred_percent(res, old_ids, old_ids)
                        h.update({"seed": seed, "eval_set": es_name})
                        audits.append({"hundred_percent_audit": h})

                    # similarity-distribution plots (seed0, taskA, self_match, main pairs)
                    if want_groups and pair in MAIN_PAIRS:
                        plot_similarity_distributions(
                            res["sim_same_sample"], res["sim_same_class_nonself"],
                            res["sim_diff_class"], out_dir / f"simdist_{pair}_{es_name}.png",
                            title=f"{pair} | {es_name} | self-match (seed{seed})")

    # aggregate
    agg = {}
    for pair in results:
        agg[pair] = {}
        for es in EVAL_SETS:
            agg[pair][es] = {}
            for pr in ("self_match", "leave_one_out"):
                agg[pair][es][pr] = common.aggregate_over_seeds(results[pair][es][pr])

    save_json({"seeds": args.seeds, "config": cfg.to_dict(),
               "setup": {"old": "Task A only", "updated": "joint 0-7 from Old",
                         "lora_rank": RANK, "lora_head": HEAD, "k": K},
               "aggregate": agg, "per_seed": results,
               "lora_checks": lora_checks_all, "audits": audits},
              out_dir / "summary.json")
    save_csv_rows(csv_rows, out_dir / "retrieval_metrics.csv")

    # ---- print (main = FFT vs Old, LoRA vs Old; Old vs Old only as sanity) ---- #
    def ms(a, k):
        return f"{a[k]['mean']:.3f}±{a[k]['std']:.3f}"

    for es in EVAL_SETS:
        tag = "PRIMARIA (Task A)" if es == "taskA" else "DIAGNOSTICA (Task A+B, Old mai visto 5-7)"
        print(f"\n===== {tag} — self_match (mean±std) =====")
        print(f"{'pair':13s} {'ss_top1':>12s} {'sc_top1':>12s} {'R@5':>12s} "
              f"{'m_ss':>12s} {'m_scn':>12s} {'m_dc':>12s} {'marg_sc-dc':>12s}")
        for pair in MAIN_PAIRS:
            a = agg[pair][es]["self_match"]
            print(f"{pair:13s} {ms(a,'same_sample_top1'):>12s} {ms(a,'same_class_top1'):>12s} "
                  f"{ms(a,'recall@5_class'):>12s} {ms(a,'mean_same_sample_cosine'):>12s} "
                  f"{ms(a,'mean_same_class_nonself_cosine'):>12s} {ms(a,'mean_diff_class_cosine'):>12s} "
                  f"{ms(a,'margin_same_class_vs_diff'):>12s}")
        print(f"\n===== {tag} — leave_one_out (mean±std; same_sample NON riportato) =====")
        print(f"{'pair':13s} {'sc_top1':>12s} {'R@5':>12s} {'nn_sim':>12s} "
              f"{'intra(scn)':>12s} {'inter(dc)':>12s} {'err_rate':>12s}")
        for pair in MAIN_PAIRS:
            a = agg[pair][es]["leave_one_out"]
            print(f"{pair:13s} {ms(a,'same_class_top1'):>12s} {ms(a,'recall@5_class'):>12s} "
                  f"{ms(a,'nn_mean_similarity'):>12s} {ms(a,'mean_same_class_nonself_cosine'):>12s} "
                  f"{ms(a,'mean_diff_class_cosine'):>12s} {ms(a,'class_error_rate'):>12s}")

    # sanity block (Old vs Old)
    a = agg["old_vs_old"]["taskA"]["self_match"]
    print("\n----- SANITY (Old vs Old, self_match, Task A) -----")
    print(f"same_sample_top1 = {ms(a,'same_sample_top1')} (atteso 1.0; self-match, "
          f"NON prova di separazione)")
    print(f"[exp5] LoRA init checks: {lora_checks_all}")
    print(f"[exp5] saved to {out_dir}")


if __name__ == "__main__":
    main()
