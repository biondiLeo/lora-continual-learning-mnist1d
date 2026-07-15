"""Low-dimensional experiment: interpretable FFT-vs-LoRA feature comparison.

Both hidden dims < 10. Linear penultimate (no activation after fc2) so 2-D
features span the plane. For each architecture (h1, h2):
  1. train the backbone on Task A once;
  2. FFT and LoRA start from the SAME backbone and update on Task B (head 'full'
     in both, so the only difference is full vs low-rank backbone update);
  3. extract Task A features before/after B and the 0-7 features after B;
  4. quantify separation in the ORIGINAL h2 space + feature drift + null features;
  5. plot (direct coords when h2==2, common PCA when h2>2).

Separation metrics are computed in the original feature space; PCA is used only
for visualization. Run:
  python -m mnist1d_cl.experiments.exp_lowdim --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..constants import TASK_A_CLASSES, TASK_B_CLASSES, SEEN_CLASSES
from ..data import build_datasets
from ..models import MLPConfig, extract
from ..training import TrainConfig
from ..utils import get_device, seed_all, save_json, make_run_dir
from ..metrics import accuracy
from ..represent import (separation_metrics, drift_metrics, fit_common_pca,
                         pca_transform, representation_similarity)
from ..plotting import panel_scatter
from . import common

ARCHS = [(8, 2), (8, 4), (9, 6)]

# Role annotation for the screening requested by the professor.
ROLES = {
    (8, 2): "main-lowdim", (8, 4): "main-lowdim", (9, 6): "main-lowdim",
    (8, 8): "main", (9, 8): "main", (9, 9): "main",
    (16, 8): "diag-firstlayer", (32, 8): "diag-firstlayer", (64, 8): "diag-firstlayer",
    (32, 32): "capacity-ref", (64, 64): "capacity-ref",
    (128, 64): "standard", (256, 256): "capacity-ref-high",
}


def rank_for_arch(h1: int, h2: int, desired: int = 4) -> int:
    return int(min(desired, h1, h2))


def model_features(model, datasets, device, split="test"):
    outA = extract(model, datasets.task(split, TASK_A_CLASSES), device)
    outAB = extract(model, datasets.task(split, SEEN_CLASSES), device)
    outB = extract(model, datasets.task(split, TASK_B_CLASSES), device)
    return {
        "featsA": outA["features"], "labelsA": outA["labels"], "idsA": outA["ids"],
        "featsAB": outAB["features"], "labelsAB": outAB["labels"],
        "accA_masked": accuracy(outA["logits"], outA["labels"], TASK_A_CLASSES),
        "accB_masked": accuracy(outB["logits"], outB["labels"], TASK_B_CLASSES),
        "accA_seen07": accuracy(outA["logits"], outA["labels"], SEEN_CLASSES),
    }


def coords_for(feature_list, h2):
    """Direct 2-D coords when h2==2, else a common 2-comp PCA over the union."""
    if h2 == 2:
        return [np.asarray(f)[:, :2] for f in feature_list], None
    pca, evr = fit_common_pca(feature_list, n_components=2)
    return [pca_transform(pca, f) for f in feature_list], evr


def run_arch(h1, h2, datasets, device, seed, out_dir, rank_override=None):
    rank = int(min(rank_override, h1, h2)) if rank_override else rank_for_arch(h1, h2)
    cfg = MLPConfig(hidden1=h1, hidden2=h2, activation="relu",
                    penultimate_activation=False, dropout=0.0)
    ta_cfg = TrainConfig(epochs=120, lr=1e-3, weight_decay=1e-4, batch_size=128,
                         scheduler="cosine", grad_clip=1.0, early_stopping_patience=20)
    tb_cfg = TrainConfig(epochs=100, lr=1e-3, weight_decay=1e-4, batch_size=128,
                         scheduler="cosine", grad_clip=1.0, early_stopping_patience=20)

    # 1) backbone on Task A
    seed_all(seed)
    a = common.train_task_a(cfg, datasets, device, ta_cfg, seed=seed)
    state = a["state"]
    back = model_features(a["model"], datasets, device)

    # 2) FFT and LoRA from the same backbone (head 'full' in both)
    fft = common.run_fft(cfg, state, datasets, device, tb_cfg)
    lora = common.run_lora(cfg, state, datasets, device, rank=rank,
                           alpha=common.alpha_for(rank), head_mode="full", train_cfg=tb_cfg)
    fft_f = model_features(fft["model"], datasets, device)
    lora_f = model_features(lora["model"], datasets, device)

    # 3) quantitative metrics (original h2 space)
    def sep(feats, labels):
        return separation_metrics(np.asarray(feats), np.asarray(labels))

    def drift(before, after, ids):
        return drift_metrics(np.asarray(before), np.asarray(after),
                             ids_before=ids, ids_after=ids)

    metrics = {
        "arch": [h1, h2], "role": ROLES.get((h1, h2), ""), "seed": seed, "lora_rank": rank,
        "lora_is_low_rank_per_layer": lora["lora_info"]["is_low_rank_per_layer"],
        "backbone": {
            "accA_masked": back["accA_masked"],
            "separation_A": sep(back["featsA"], back["labelsA"]),
        },
        "fft": {
            "accA_masked": fft_f["accA_masked"], "accA_seen07": fft_f["accA_seen07"],
            "accB_masked": fft_f["accB_masked"],
            "forget_A_masked": back["accA_masked"] - fft_f["accA_masked"],
            "separation_A_after": sep(fft_f["featsA"], fft_f["labelsA"]),
            "separation_seen07_after": sep(fft_f["featsAB"], fft_f["labelsAB"]),
            "drift_A": drift(back["featsA"], fft_f["featsA"], back["idsA"]),
            "similarity_to_backbone": representation_similarity(back["featsA"], fft_f["featsA"]),
        },
        "lora": {
            "accA_masked": lora_f["accA_masked"], "accA_seen07": lora_f["accA_seen07"],
            "accB_masked": lora_f["accB_masked"],
            "forget_A_masked": back["accA_masked"] - lora_f["accA_masked"],
            "separation_A_after": sep(lora_f["featsA"], lora_f["labelsA"]),
            "separation_seen07_after": sep(lora_f["featsAB"], lora_f["labelsAB"]),
            "drift_A": drift(back["featsA"], lora_f["featsA"], back["idsA"]),
            "similarity_to_backbone": representation_similarity(back["featsA"], lora_f["featsA"]),
        },
    }

    # 4) plots
    arch_dir = Path(out_dir) / f"h{h1}_h{h2}_seed{seed}"
    # Figure 1: Task A before / after FFT / after LoRA
    coordsA, evrA = coords_for([back["featsA"], fft_f["featsA"], lora_f["featsA"]], h2)
    proj = "diretto (h2=2)" if h2 == 2 else f"PCA comune evr={np.round(evrA,3).tolist()}"
    panelsA = [
        (coordsA[0], back["labelsA"], f"backbone A (accA={back['accA_masked']:.2f})"),
        (coordsA[1], fft_f["labelsA"], f"A dopo FFT (accA={fft_f['accA_masked']:.2f})"),
        (coordsA[2], lora_f["labelsA"], f"A dopo LoRA r{rank} (accA={lora_f['accA_masked']:.2f})"),
    ]
    panel_scatter(panelsA, arch_dir / "taskA_before_after.png",
                  suptitle=f"(h1,h2)=({h1},{h2}) seed{seed} | Task A classi 0-4 | {proj}")

    # Figure 2: classes 0-7 after FFT vs after LoRA
    coordsAB, evrAB = coords_for([fft_f["featsAB"], lora_f["featsAB"]], h2)
    projB = "diretto (h2=2)" if h2 == 2 else f"PCA comune evr={np.round(evrAB,3).tolist()}"
    panelsAB = [
        (coordsAB[0], fft_f["labelsAB"], f"0-7 dopo FFT (accB={fft_f['accB_masked']:.2f})"),
        (coordsAB[1], lora_f["labelsAB"], f"0-7 dopo LoRA r{rank} (accB={lora_f['accB_masked']:.2f})"),
    ]
    panel_scatter(panelsAB, arch_dir / "seen07_after.png",
                  suptitle=f"(h1,h2)=({h1},{h2}) seed{seed} | classi 0-7 | {projB}")

    if evrA is not None:
        metrics["pca_explained_variance_taskA"] = list(np.round(evrA, 4))
        metrics["pca_explained_variance_seen07"] = list(np.round(evrAB, 4))
    save_json(metrics, arch_dir / "metrics.json")
    return metrics


def _parse_archs(items):
    return [tuple(int(v) for v in a.split(",")) for a in items]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="outputs")
    p.add_argument("--archs", nargs="+", default=["8,2", "8,4", "9,6"],
                   help="hidden dims as 'h1,h2' (e.g. 8,8 9,8 9,9)")
    p.add_argument("--rank", type=int, default=None,
                   help="override LoRA rank (clamped to min(h1,h2)); default min(4,h1,h2)")
    p.add_argument("--tag", default="fft_vs_lora")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    datasets = build_datasets(data_dir=args.data_dir)
    archs = _parse_archs(args.archs)
    out_dir = make_run_dir(args.out, "lowdim", seed=args.seed, tag=args.tag)
    print(f"[lowdim] device={device} out={out_dir}")

    all_metrics = {}
    for (h1, h2) in archs:
        role = ROLES.get((h1, h2), "")
        rank_used = int(min(args.rank, h1, h2)) if args.rank else rank_for_arch(h1, h2)
        print(f"\n[lowdim] arch (h1={h1}, h2={h2}) role={role} rank={rank_used}")
        m = run_arch(h1, h2, datasets, device, args.seed, out_dir, rank_override=args.rank)
        all_metrics[f"h{h1}_h{h2}"] = m
        b, f, l = m["backbone"], m["fft"], m["lora"]
        sb = b["separation_A"]
        print(f"  backbone: accA={b['accA_masked']:.3f} sil={sb['silhouette']:.3f} "
              f"intra={sb['mean_intra_class_distance']:.2f} inter={sb['mean_inter_class_distance']:.2f} "
              f"ratio={sb['inter_intra_ratio']:.2f} effrank={sb['effective_rank']:.2f}/{h2} "
              f"pct_nz={sb['pct_near_zero_features']:.1f}")
        for name, r in (("FFT", f), ("LoRA", l)):
            sa = r["separation_A_after"]
            sim = r["similarity_to_backbone"]
            print(f"  {name:4s}: accA={r['accA_masked']:.3f} accB={r['accB_masked']:.3f} "
                  f"forgetA_m={r['forget_A_masked']:.3f} sil={sa['silhouette']:.3f} "
                  f"effrank={sa['effective_rank']:.2f} | CKA={sim['linear_cka']:.3f} "
                  f"distcorr={sim['pairwise_distance_correlation']:.3f} "
                  f"procr={sim['procrustes_residual']:.3f} "
                  f"cos={sim['mean_cosine_same_sample']:.3f}")

    save_json(all_metrics, out_dir / "lowdim_summary.json")
    print(f"\n[lowdim] done. saved to {out_dir}")


if __name__ == "__main__":
    main()
