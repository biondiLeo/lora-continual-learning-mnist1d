"""CNN vs MLP on MNIST-1D: accuracy, feature separation, forgetting and
representation compatibility.

Architecture is the variable of interest. Grid (3 seeds):
  arch in {mlp, cnn} x feature_dim in {8, 64} x penultimate in {linear, relu}
  x method in {FFT, LoRA r8/full}.  head = full;  for the CNN, LoRA is applied on
  the linear projection 'proj' (the conv stack is frozen).

Two protocols per (config, seed), identical to the MLP experiments so numbers are
directly comparable:
  * sequential (forgetting): Task A backbone -> FFT / LoRA on Task B. Reports
    accuracy (masked + seen-0-7), forgetting (masked + seen-0-7), feature
    separation, and CKA / Procrustes / cosine vs the backbone (reusing the same
    functions as exp_lowdim.run_arch).
  * joint (compatibility): Old = Task A backbone; FFT-joint / LoRA-joint on 0-7;
    query-gallery Old-vs-updated on Task A (self-match + leave-one-out), reusing
    the same functions as exp5_compat.

Everything else (data, splits, sample IDs, normalization, seen-0-7 primary metric,
validation-only selection, frozen test) is identical to the rest of the project.
NOTHING here touches the frozen MLP results or the report.

  python -m mnist1d_cl.experiments.exp_cnn_vs_mlp --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from ..constants import TASK_A_CLASSES, TASK_B_CLASSES, SEEN_CLASSES
from ..data import build_datasets, make_loader
from ..models import MLPConfig, build_mlp, CNNConfig, build_cnn, extract
from ..training import TrainConfig, train_supervised, fit, supervised_closure
from ..lora import setup_lora_model, delta_is_zero, outputs_equivalent, base_frozen
from ..metrics import accuracy
from ..represent import separation_metrics, representation_similarity
from ..querygallery import query_gallery_eval, retrieval_error_analysis
from ..utils import get_device, seed_all, save_json, save_csv_rows, make_run_dir
from . import common

MLP_H1 = 128            # first-layer width for the MLP (capable first stage)
CNN_C = 16              # base conv channels
RANK, HEAD, K = 8, "full", 5
CONFIGS = [(a, fd, pa) for a in ("mlp", "cnn") for fd in (8, 64) for pa in (False, True)]


def cfg_name(arch, fd, penult):
    return f"{arch}_fd{fd}_{'relu' if penult else 'lin'}"


def build_backbone(arch, fd, penult):
    if arch == "mlp":
        return build_mlp(MLPConfig(hidden1=MLP_H1, hidden2=fd, penultimate_activation=penult))
    return build_cnn(CNNConfig(feature_dim=fd, channels=CNN_C, penultimate_activation=penult))


def lora_targets(arch):
    return ("fc1", "fc2") if arch == "mlp" else ("proj",)


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def train_backbone(arch, fd, penult, datasets, device, seed, tcfg):
    seed_all(seed)
    model = build_backbone(arch, fd, penult).to(device)
    train_supervised(model, datasets.task("train", TASK_A_CLASSES),
                     datasets.task("val", TASK_A_CLASSES), tcfg, device,
                     allowed_classes=TASK_A_CLASSES)
    return model, copy.deepcopy(model.state_dict())


def run_fft(arch, fd, penult, state, datasets, device, classes, tcfg):
    model = build_backbone(arch, fd, penult).to(device)
    model.load_state_dict(copy.deepcopy(state))
    for p in model.parameters():
        p.requires_grad_(True)
    gen = torch.Generator().manual_seed(tcfg.shuffle_seed)
    loader = make_loader(datasets.task("train", classes), batch_size=tcfg.batch_size,
                         shuffle=True, generator=gen)
    fit(model, loader, datasets.task("val", classes), tcfg, device,
        supervised_closure(), allowed_classes=classes)
    return model


def run_lora(arch, fd, penult, state, datasets, device, classes, tcfg):
    model = build_backbone(arch, fd, penult).to(device)
    model.load_state_dict(copy.deepcopy(state))
    base_ref = build_backbone(arch, fd, penult).to(device)
    base_ref.load_state_dict(copy.deepcopy(state))
    model, ctrl, info = setup_lora_model(model, r=RANK, alpha=common.alpha_for(RANK),
                                         head_mode=HEAD, target_names=lora_targets(arch))
    model = model.to(device)
    x0 = torch.from_numpy(datasets.task("val", TASK_A_CLASSES).x[:32]).float().to(device)
    checks = {"delta_zero_init": bool(delta_is_zero(model)),
              "output_equiv_init": bool(outputs_equivalent(base_ref, model, x0)),
              "base_frozen": bool(base_frozen(model))}
    gen = torch.Generator().manual_seed(tcfg.shuffle_seed)
    loader = make_loader(datasets.task("train", classes), batch_size=tcfg.batch_size,
                         shuffle=True, generator=gen)
    fit(model, loader, datasets.task("val", classes), tcfg, device,
        supervised_closure(), allowed_classes=classes, post_step_hook=ctrl)
    return model, info, checks


def model_features(model, datasets, device):
    outA = extract(model, datasets.task("test", TASK_A_CLASSES), device)
    outB = extract(model, datasets.task("test", TASK_B_CLASSES), device)
    return {"featsA": outA["features"], "labelsA": outA["labels"], "idsA": outA["ids"],
            "accA_masked": accuracy(outA["logits"], outA["labels"], TASK_A_CLASSES),
            "accA_seen07": accuracy(outA["logits"], outA["labels"], SEEN_CLASSES),
            "accB_masked": accuracy(outB["logits"], outB["labels"], TASK_B_CLASSES)}


def compact_qg(res, err):
    return {"same_sample_top1": res["same_sample_top1"],
            "same_class_top1": res["same_class_top1"],
            "mean_same_sample_cosine": res["mean_same_sample_cosine"],
            "mean_same_class_nonself_cosine": res["mean_same_class_nonself_cosine"],
            "mean_diff_class_cosine": res["mean_diff_class_cosine"],
            "class_error_rate": err["class_error_rate"]}


def run_config(arch, fd, penult, datasets, device, seed, ta_cfg, tb_cfg, tj_cfg):
    model, state = train_backbone(arch, fd, penult, datasets, device, seed, ta_cfg)
    npar = n_params(model)
    back = model_features(model, datasets, device)

    # ---- sequential (Task B) ---- #
    fft_m = run_fft(arch, fd, penult, state, datasets, device, TASK_B_CLASSES, tb_cfg)
    lora_m, lora_info, lora_checks = run_lora(arch, fd, penult, state, datasets, device,
                                              TASK_B_CLASSES, tb_cfg)
    fft_f, lora_f = model_features(fft_m, datasets, device), model_features(lora_m, datasets, device)

    def block(x):
        return {"accA_masked": x["accA_masked"], "accA_seen07": x["accA_seen07"],
                "accB_masked": x["accB_masked"],
                "forget_A_masked": back["accA_masked"] - x["accA_masked"],
                "forget_A_seen07": back["accA_seen07"] - x["accA_seen07"],
                "separation_A_after": separation_metrics(np.asarray(x["featsA"]), np.asarray(x["labelsA"])),
                "similarity_to_backbone": representation_similarity(back["featsA"], x["featsA"])}

    # ---- joint (compatibility, query-gallery on Task A) ---- #
    old_A = extract(model, datasets.task("test", TASK_A_CLASSES), device)
    fftj = run_fft(arch, fd, penult, state, datasets, device, SEEN_CLASSES, tj_cfg)
    loraj, _, _ = run_lora(arch, fd, penult, state, datasets, device, SEEN_CLASSES, tj_cfg)
    qg = {}
    for pair, um in (("fft_vs_old", fftj), ("lora_vs_old", loraj)):
        uf = extract(um, datasets.task("test", TASK_A_CLASSES), device)
        qg[pair] = {}
        for protocol in ("self_match", "leave_one_out"):
            res = query_gallery_eval(uf["features"], uf["ids"], uf["labels"],
                                     old_A["features"], old_A["ids"], old_A["labels"],
                                     protocol=protocol, k=K, return_matrix=True)
            err = retrieval_error_analysis(uf["labels"], old_A["labels"],
                                           res["nn_index"], res["valid_row"])
            qg[pair][protocol] = compact_qg(res, err)

    return {"arch": arch, "feature_dim": fd, "penultimate": "relu" if penult else "linear",
            "seed": seed, "n_params": npar, "lora_rank": RANK,
            "lora_is_low_rank": lora_info["is_low_rank_per_layer"], "lora_checks": lora_checks,
            "backbone": {"accA_masked": back["accA_masked"], "accA_seen07": back["accA_seen07"],
                         "separation_A": separation_metrics(np.asarray(back["featsA"]),
                                                            np.asarray(back["labelsA"]))},
            "fft": block(fft_f), "lora": block(lora_f), "qg": qg}


def flatten(m):
    b, f, l, qg = m["backbone"], m["fft"], m["lora"], m["qg"]
    sb = b["separation_A"]
    out = {"n_params": m["n_params"],
           "backbone_accA_masked": b["accA_masked"], "backbone_accA_seen07": b["accA_seen07"],
           "backbone_sil": sb["silhouette"], "backbone_ratio": sb["inter_intra_ratio"],
           "backbone_effrank": sb["effective_rank"], "backbone_ncacc": sb["nearest_centroid_accuracy"]}
    for pfx, x in (("fft", f), ("lora", l)):
        sa, sim = x["separation_A_after"], x["similarity_to_backbone"]
        out.update({f"{pfx}_accA_masked": x["accA_masked"], f"{pfx}_accA_seen07": x["accA_seen07"],
                    f"{pfx}_accB_masked": x["accB_masked"],
                    f"{pfx}_forgetA_masked": x["forget_A_masked"],
                    f"{pfx}_forgetA_seen07": x["forget_A_seen07"],
                    f"{pfx}_sil_after": sa["silhouette"], f"{pfx}_ncacc_after": sa["nearest_centroid_accuracy"],
                    f"{pfx}_cka": sim["linear_cka"], f"{pfx}_procrustes": sim["procrustes_residual"],
                    f"{pfx}_cos": sim["mean_cosine_same_sample"]})
    for pair, pk in (("fft_vs_old", "qgfft"), ("lora_vs_old", "qglora")):
        sm, loo = qg[pair]["self_match"], qg[pair]["leave_one_out"]
        out.update({f"{pk}_ss_top1": sm["same_sample_top1"], f"{pk}_sc_top1": sm["same_class_top1"],
                    f"{pk}_cos_ss": sm["mean_same_sample_cosine"],
                    f"{pk}_loo_sc_top1": loo["same_class_top1"], f"{pk}_loo_err": loo["class_error_rate"]})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="outputs")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    datasets = build_datasets(data_dir=args.data_dir)
    ta_cfg = TrainConfig(epochs=80, lr=1e-3, weight_decay=1e-4, batch_size=128,
                         scheduler="cosine", grad_clip=5.0, early_stopping_patience=15)
    tb_cfg = TrainConfig(epochs=60, lr=1e-3, weight_decay=1e-4, batch_size=128,
                         scheduler="cosine", grad_clip=5.0, early_stopping_patience=15)
    tj_cfg = TrainConfig(epochs=80, lr=1e-3, weight_decay=1e-4, batch_size=128,
                         scheduler="cosine", grad_clip=5.0, early_stopping_patience=15)
    out_dir = make_run_dir(args.out, "cnn_vs_mlp", tag="grid")
    print(f"[cnn-vs-mlp] device={device} seeds={args.seeds} out={out_dir}")

    per_seed_all, csv_rows = {}, []
    for (arch, fd, penult) in CONFIGS:
        key = cfg_name(arch, fd, penult)
        flats = []
        for seed in args.seeds:
            print(f"[cnn-vs-mlp] {key} seed={seed}")
            m = run_config(arch, fd, penult, datasets, device, seed, ta_cfg, tb_cfg, tj_cfg)
            flats.append(flatten(m))
            csv_rows.append({"config": key, "arch": arch, "feature_dim": fd,
                             "penultimate": m["penultimate"], "seed": seed, **flatten(m)})
        per_seed_all[key] = flats

    aggregate = {key: {"arch": key.split("_")[0],
                       "feature_dim": int(key.split("fd")[1].split("_")[0]),
                       "penultimate": key.split("_")[-1],
                       "metrics": common.aggregate_over_seeds(per_seed_all[key])}
                 for key in per_seed_all}

    save_json({"seeds": args.seeds, "grid": [cfg_name(*c) for c in CONFIGS],
               "setup": {"mlp_h1": MLP_H1, "cnn_channels": CNN_C, "lora_rank": RANK,
                         "lora_head": HEAD, "lora_target_cnn": "proj (conv frozen)", "k": K},
               "aggregate": aggregate, "per_seed": per_seed_all},
              out_dir / "cnn_vs_mlp_summary.json")
    save_csv_rows(csv_rows, out_dir / "cnn_vs_mlp_grid.csv")

    # ---- compact print ---- #
    def ms(a, k):
        return f"{a[k]['mean']:.3f}±{a[k]['std']:.3f}"
    print("\n===== BACKBONE (accA masked / silhouette / nearest-centroid acc) =====")
    print(f"{'config':16s} {'params':>7s} {'accA':>13s} {'sil':>13s} {'nc-acc':>13s}")
    for key, d in aggregate.items():
        a = d["metrics"]
        print(f"{key:16s} {int(a['n_params']['mean']):7d} {ms(a,'backbone_accA_masked'):>13s} "
              f"{ms(a,'backbone_sil'):>13s} {ms(a,'backbone_ncacc'):>13s}")
    for method in ("fft", "lora"):
        print(f"\n===== {method.upper()} : forgetting & compatibility =====")
        print(f"{'config':16s} {'accA_s07':>13s} {'forgetA_s07':>13s} {'accB':>13s} "
              f"{'CKA':>13s} {'sil_after':>13s}")
        for key, d in aggregate.items():
            a = d["metrics"]
            print(f"{key:16s} {ms(a,method+'_accA_seen07'):>13s} {ms(a,method+'_forgetA_seen07'):>13s} "
                  f"{ms(a,method+'_accB_masked'):>13s} {ms(a,method+'_cka'):>13s} {ms(a,method+'_sil_after'):>13s}")
    print("\n===== QUERY-GALLERY (Task A): same-sample Top-1 / LOO same-class Top-1 =====")
    print(f"{'config':16s} {'FFT ss':>10s} {'LoRA ss':>10s} {'FFT loo-sc':>11s} {'LoRA loo-sc':>11s}")
    for key, d in aggregate.items():
        a = d["metrics"]
        print(f"{key:16s} {ms(a,'qgfft_ss_top1'):>10s} {ms(a,'qglora_ss_top1'):>10s} "
              f"{ms(a,'qgfft_loo_sc_top1'):>11s} {ms(a,'qglora_loo_sc_top1'):>11s}")
    print(f"\n[cnn-vs-mlp] saved to {out_dir}")


if __name__ == "__main__":
    main()
