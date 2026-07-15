"""Exp0 - Joint training baseline (diagnostic upper bound), classes 0-7.

NOT a continual-learning method. Run:
  python -m mnist1d_cl.experiments.exp0_joint --seeds 0 1 2
"""

from __future__ import annotations

import argparse

import torch

from ..constants import SEEN_CLASSES
from ..data import build_datasets
from ..models import build_mlp
from ..training import TrainConfig, train_supervised
from ..utils import get_device, seed_all
from . import common


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

    rows = []
    for seed in args.seeds:
        seed_all(seed)
        model = build_mlp(cfg).to(device)
        res = train_supervised(model, datasets.task("train", SEEN_CLASSES),
                               datasets.task("val", SEEN_CLASSES), tcfg, device,
                               allowed_classes=SEEN_CLASSES)
        ev = common.evaluate_model(model, datasets, device, "test")
        rows.append({"seed": seed, "test_seen07": ev["seen"]["accuracy_seen07"],
                     "val_seen07": res["best_val_acc"]})
        common.save_run(args.out, "exp0_joint", seed, "joint", cfg,
                        metrics=ev, history=res["history"], state=model.state_dict())
        print(f"[exp0] seed {seed}: test seen07={ev['seen']['accuracy_seen07']:.3f}")

    import numpy as np
    m = np.mean([r["test_seen07"] for r in rows]); s = np.std([r["test_seen07"] for r in rows])
    print(f"[exp0] joint(0-7) test seen07 = {m:.3f}±{s:.3f}")


if __name__ == "__main__":
    main()
