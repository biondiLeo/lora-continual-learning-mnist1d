"""Exp7 - SVD analysis of the weight updates (sequential protocol).

Compares the structure of the Task B update produced by Full Fine-Tuning vs
LoRA, starting from the SAME Task A backbone per seed. fc1, fc2 and the
classification head are analysed SEPARATELY (the head is full-trainable even in
the LoRA models here, so it is reported only as a separate control and must not
be confused with the low-rank hidden-layer updates).

  DeltaW_FFT  = W_afterB - W_old
  DeltaW_LoRA = scaling * B @ A   (fc1, fc2);  head: W_afterB - W_old (full)

Uses the frozen Task A backbones; regenerates the FFT / LoRA updates
deterministically (no change to previous experiments). LoRA head = full.

  python -m mnist1d_cl.experiments.exp7_svd --seeds 0 1 2
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ..constants import SEEN_CLASSES
from ..data import build_datasets
from ..training import TrainConfig
from ..lora import LoRALinear
from ..svd import update_metrics, compare_updates, numerical_rank_sv, singular_values
from ..plotting import plot_lines
from ..utils import get_device, save_json, save_csv_rows, make_run_dir
from . import common

RANKS = [2, 4, 8, 16]
LAYERS = ["fc1", "fc2", "head"]
RTOL = 1e-6  # documented numerical-rank tolerance (relative to largest s.v.)


def W_np(module):
    m = module.base if isinstance(module, LoRALinear) else module
    return m.weight.detach().cpu().numpy()


def delta_np(model, name, W_old):
    layer = getattr(model, name)
    if isinstance(layer, LoRALinear):
        return layer.delta_weight().detach().cpu().numpy()
    return layer.weight.detach().cpu().numpy() - W_old


def scalar_um(um):
    return {"numerical_rank": um["numerical_rank"], "effective_rank": um["effective_rank"],
            "stable_rank": um["stable_rank"], "rank90": um["rank90"], "rank95": um["rank95"],
            "fro_ratio": um["fro_ratio_delta_over_old"],
            "spectral_ratio": um["spectral_ratio_delta_over_old"]}


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
    out_dir = make_run_dir(args.out, "exp7_svd", tag="weight_updates")
    print(f"[svd] device={device} config=({cfg.hidden1},{cfg.hidden2}) out={out_dir}")

    methods = ["FFT"] + [f"LoRA_r{r}" for r in RANKS]
    scalars = {ly: {mk: [] for mk in methods} for ly in LAYERS}
    fft_rankr = {ly: {r: [] for r in RANKS} for ly in LAYERS}
    overlap = {ly: {r: [] for r in RANKS} for ly in LAYERS}
    spectra0 = {ly: {} for ly in LAYERS}   # seed0 normalized spectra + cum energy
    verify = []
    csv_rows = []

    for seed in args.seeds:
        print(f"[svd] seed {seed}: regenerating FFT + LoRA(r=2,4,8,16) updates")
        state = common.ensure_task_a_state(cfg, datasets, device, args.out, seed)
        old_model = common.fresh_from_state(cfg, state, device)
        W_old = {ly: W_np(getattr(old_model, ly)) for ly in LAYERS}

        # FFT update
        fft_model = common.run_fft(cfg, state, datasets, device, tb)["model"]
        dFFT = {ly: delta_np(fft_model, ly, W_old[ly]) for ly in LAYERS}

        # LoRA updates per rank (head full)
        dLoRA = {}
        for r in RANKS:
            run = common.run_lora(cfg, state, datasets, device, r, common.alpha_for(r),
                                  head_mode="full", train_cfg=tb)
            lm = run["model"]
            dLoRA[r] = {ly: delta_np(lm, ly, W_old[ly]) for ly in LAYERS}

            # numerical verifications (seed0)
            if seed == args.seeds[0]:
                x = torch.from_numpy(datasets.task("test", SEEN_CLASSES).x[:16]).float().to(device)
                W_eff = lm.fc1.effective_weight()
                y = lm.fc1(x)
                y_manual = x @ W_eff.t() + lm.fc1.base.bias
                num_ranks = {ly: int(numerical_rank_sv(singular_values(dLoRA[r][ly]), RTOL))
                             for ly in ("fc1", "fc2")}
                verify.append({
                    "rank": r,
                    "delta_zero_init": run["checks"]["delta_zero_init"],
                    "output_equiv_init": run["checks"]["output_equiv_init"],
                    "base_frozen": run["checks"]["base_frozen"],
                    "effective_weight_matches_forward": bool(torch.allclose(y, y_manual, atol=1e-4)),
                    "numerical_rank_fc1": num_ranks["fc1"], "numerical_rank_fc2": num_ranks["fc2"],
                    "numerical_rank_le_nominal": bool(num_ranks["fc1"] <= r and num_ranks["fc2"] <= r),
                })

        # metrics per layer / method
        for ly in LAYERS:
            um_fft = update_metrics(dFFT[ly], W_old[ly], ranks=RANKS, rtol=RTOL)
            scalars[ly]["FFT"].append(scalar_um(um_fft))
            for r in RANKS:
                fft_rankr[ly][r].append(um_fft["rankr_approx"][r])
            if seed == args.seeds[0]:
                spectra0[ly]["FFT"] = {"spectrum": um_fft["normalized_spectrum"],
                                       "cum_energy": um_fft["cumulative_energy"]}
            for r in RANKS:
                um_l = update_metrics(dLoRA[r][ly], W_old[ly], ranks=RANKS, rtol=RTOL)
                scalars[ly][f"LoRA_r{r}"].append(scalar_um(um_l))
                overlap[ly][r].append(compare_updates(dFFT[ly], dLoRA[r][ly], r))
                if seed == args.seeds[0]:
                    spectra0[ly][f"LoRA_r{r}"] = {"spectrum": um_l["normalized_spectrum"],
                                                  "cum_energy": um_l["cumulative_energy"]}
            for mk in methods:
                sc = scalars[ly][mk][-1]
                csv_rows.append({"seed": seed, "layer": ly, "method": mk, **sc})

    # aggregate
    agg_scalars = {ly: {mk: common.aggregate_over_seeds(scalars[ly][mk]) for mk in methods}
                   for ly in LAYERS}
    agg_fft_rankr = {ly: {r: common.aggregate_over_seeds(fft_rankr[ly][r]) for r in RANKS}
                     for ly in LAYERS}
    agg_overlap = {ly: {r: common.aggregate_over_seeds(
        [{"left_overlap": o["left_subspace_overlap"], "right_overlap": o["right_subspace_overlap"],
          "frobenius_cosine": o.get("frobenius_cosine", float("nan"))} for o in overlap[ly][r]])
        for r in RANKS} for ly in LAYERS}

    save_json({"seeds": args.seeds, "config": cfg.to_dict(), "ranks": RANKS,
               "lora_head": "full", "numerical_rank_rtol": RTOL,
               "verifications": verify,
               "aggregate_scalars": agg_scalars, "fft_rankr_approx": agg_fft_rankr,
               "subspace_overlap_fft_lora": agg_overlap,
               "per_seed_scalars": scalars}, out_dir / "summary.json")
    save_csv_rows(csv_rows, out_dir / "svd_scalars.csv")

    # ---- plots (seed0): fc1 and fc2 separate; head separate control ---- #
    for ly in LAYERS:
        spec = {mk: spectra0[ly][mk]["spectrum"] for mk in methods}
        plot_lines(spec, out_dir / f"spectrum_{ly}.png",
                   title=f"DeltaW normalized spectrum - {ly} (seed{args.seeds[0]})",
                   xlabel="singular index", ylabel="s_i / s_1", logy=True)
        cume = {mk: spectra0[ly][mk]["cum_energy"] for mk in methods}
        plot_lines(cume, out_dir / f"cumenergy_{ly}.png",
                   title=f"DeltaW cumulative Frobenius energy - {ly} (seed{args.seeds[0]})",
                   xlabel="rank", ylabel="cumulative energy")

    # ---- report tables ---- #
    def ms(a, k):
        return f"{a[k]['mean']:.3f}±{a[k]['std']:.3f}"

    rr = ["# SVD of weight updates (mean±std over seeds)\n",
          f"numerical-rank tolerance rtol={RTOL}; LoRA head=full\n"]
    for ly in LAYERS:
        head_note = " (CONTROL — full head, not low-rank)" if ly == "head" else ""
        rr.append(f"\n## {ly}{head_note} — effective/stable/rank90/rank95/fro-ratio")
        rr.append("method | eff_rank | stable_rank | rank90 | rank95 | ||dW||F/||W||F | numrank")
        rr.append("---|---|---|---|---|---|---")
        for mk in methods:
            a = agg_scalars[ly][mk]
            rr.append(f"{mk} | {ms(a,'effective_rank')} | {ms(a,'stable_rank')} | "
                      f"{ms(a,'rank90')} | {ms(a,'rank95')} | {ms(a,'fro_ratio')} | {ms(a,'numerical_rank')}")
    rr.append("\n## FFT best rank-r approximation of DeltaW_FFT (energy / rel.error)")
    rr.append("layer | " + " | ".join(f"r{r}" for r in RANKS))
    rr.append("---|" + "|".join(["---"] * len(RANKS)))
    for ly in LAYERS:
        cells = [f"{agg_fft_rankr[ly][r]['energy_captured']['mean']:.2f}/"
                 f"{agg_fft_rankr[ly][r]['approx_rel_error']['mean']:.2f}" for r in RANKS]
        rr.append(f"{ly} | " + " | ".join(cells))
    rr.append("\n## Subspace overlap FFT vs LoRA (left/right, top-r)")
    rr.append("layer | " + " | ".join(f"r{r}" for r in RANKS))
    rr.append("---|" + "|".join(["---"] * len(RANKS)))
    for ly in LAYERS:
        cells = [f"{agg_overlap[ly][r]['left_overlap']['mean']:.2f}/"
                 f"{agg_overlap[ly][r]['right_overlap']['mean']:.2f}" for r in RANKS]
        rr.append(f"{ly} | " + " | ".join(cells))
    (out_dir / "report_table.md").write_text("\n".join(rr), encoding="utf-8")

    # ---- print ---- #
    print("\n[svd] verifications (seed0):")
    for v in verify:
        print(f"  r{v['rank']}: delta0={v['delta_zero_init']} equiv0={v['output_equiv_init']} "
              f"base_frozen={v['base_frozen']} eff==fwd={v['effective_weight_matches_forward']} "
              f"numrank(fc1,fc2)=({v['numerical_rank_fc1']},{v['numerical_rank_fc2']})<=r:{v['numerical_rank_le_nominal']}")
    for ly in LAYERS:
        print(f"\n[svd] {ly} eff_rank / stable_rank / rank90 (mean±std):")
        for mk in methods:
            a = agg_scalars[ly][mk]
            print(f"  {mk:10s} eff={ms(a,'effective_rank')} stable={ms(a,'stable_rank')} "
                  f"rank90={ms(a,'rank90')} froR={ms(a,'fro_ratio')}")
    print("\n[svd] subspace overlap FFT-LoRA (fc1/fc2, left, mean):")
    for ly in ("fc1", "fc2"):
        print(f"  {ly}: " + " ".join(f"r{r}={agg_overlap[ly][r]['left_overlap']['mean']:.2f}" for r in RANKS))
    print(f"[svd] saved to {out_dir}")


if __name__ == "__main__":
    main()
