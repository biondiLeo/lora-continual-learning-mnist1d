"""Multi-seed capacity comparison for the low-dim / capacity-reference study.

Runs the FFT-vs-LoRA low-dim pipeline for several architectures over multiple
seeds and aggregates mean/std/per-seed for every scalar metric (accuracy,
forgetting, silhouette, intra/inter, effective rank, linear CKA, pairwise
distance correlation, Procrustes residual, cosine same-sample). Per-seed plots
are still saved by ``run_arch``.

  python -m mnist1d_cl.experiments.exp_lowdim_multiseed \
      --seeds 0 1 2 --rank 8 --archs 9,9 64,8 128,64 256,256
"""

from __future__ import annotations

import argparse

import torch

from ..data import build_datasets
from ..utils import get_device, save_json, make_run_dir
from . import common
from .exp_lowdim import run_arch, ROLES, _parse_archs, rank_for_arch


def flatten(m: dict) -> dict:
    b, f, l = m["backbone"], m["fft"], m["lora"]
    sb = b["separation_A"]

    def block(x, prefix):
        sa = x["separation_A_after"]
        sim = x["similarity_to_backbone"]
        return {
            f"{prefix}_accA": x["accA_masked"],
            f"{prefix}_accB": x["accB_masked"],
            f"{prefix}_forgetA": x["forget_A_masked"],
            f"{prefix}_sil_after": sa["silhouette"],
            f"{prefix}_effrank_after": sa["effective_rank"],
            f"{prefix}_cka": sim["linear_cka"],
            f"{prefix}_distcorr": sim["pairwise_distance_correlation"],
            f"{prefix}_procrustes": sim["procrustes_residual"],
            f"{prefix}_cos": sim["mean_cosine_same_sample"],
        }

    out = {
        "backbone_accA": b["accA_masked"],
        "backbone_sil": sb["silhouette"],
        "backbone_intra": sb["mean_intra_class_distance"],
        "backbone_inter": sb["mean_inter_class_distance"],
        "backbone_ratio": sb["inter_intra_ratio"],
        "backbone_effrank": sb["effective_rank"],
    }
    out.update(block(f, "fft"))
    out.update(block(l, "lora"))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="outputs")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--archs", nargs="+", default=["9,9", "64,8", "128,64", "256,256"])
    args = p.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    datasets = build_datasets(data_dir=args.data_dir)
    archs = _parse_archs(args.archs)
    out_dir = make_run_dir(args.out, "lowdim", tag="capacity_multiseed")
    print(f"[lowdim-ms] device={device} seeds={args.seeds} out={out_dir}")

    aggregate = {}
    per_seed_all = {}
    for (h1, h2) in archs:
        key = f"h{h1}_h{h2}"
        rank_used = int(min(args.rank, h1, h2))
        flats = []
        for seed in args.seeds:
            print(f"[lowdim-ms] {key} role={ROLES.get((h1,h2),'')} seed={seed} rank={rank_used}")
            m = run_arch(h1, h2, datasets, device, seed, out_dir, rank_override=args.rank)
            flats.append(flatten(m))
        agg = common.aggregate_over_seeds(flats)
        aggregate[key] = {"role": ROLES.get((h1, h2), ""), "rank": rank_used, "metrics": agg}
        per_seed_all[key] = flats

    save_json({"seeds": args.seeds, "aggregate": aggregate, "per_seed": per_seed_all},
              out_dir / "capacity_multiseed_summary.json")

    # print compact mean±std tables
    def ms(agg, k):
        return f"{agg[k]['mean']:.3f}±{agg[k]['std']:.3f}"

    print("\n===== BACKBONE (mean±std over seeds) =====")
    print(f"{'config':10s} {'accA':>13s} {'sil':>13s} {'effrank':>13s} {'ratio':>13s}")
    for key, d in aggregate.items():
        a = d["metrics"]
        print(f"{key:10s} {ms(a,'backbone_accA'):>13s} {ms(a,'backbone_sil'):>13s} "
              f"{ms(a,'backbone_effrank'):>13s} {ms(a,'backbone_ratio'):>13s}")

    for method in ("fft", "lora"):
        print(f"\n===== {method.upper()} after Task B (mean±std) =====")
        print(f"{'config':10s} {'accA':>13s} {'forgetA':>13s} {'accB':>13s} "
              f"{'CKA':>13s} {'procrustes':>13s} {'cos':>13s}")
        for key, d in aggregate.items():
            a = d["metrics"]
            print(f"{key:10s} {ms(a,method+'_accA'):>13s} {ms(a,method+'_forgetA'):>13s} "
                  f"{ms(a,method+'_accB'):>13s} {ms(a,method+'_cka'):>13s} "
                  f"{ms(a,method+'_procrustes'):>13s} {ms(a,method+'_cos'):>13s}")

    print(f"\n[lowdim-ms] saved to {out_dir}")


if __name__ == "__main__":
    main()
