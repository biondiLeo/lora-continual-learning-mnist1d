"""Run a model over a split and collect features / logits / labels / IDs.

Order is preserved (shuffle=False), so the returned arrays are aligned with
``SplitData.ids`` row-by-row.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ..data.dataset import SplitData, make_loader


@torch.no_grad()
def extract(model: torch.nn.Module, data: SplitData, device: torch.device,
            batch_size: int = 256) -> Dict[str, np.ndarray]:
    model.eval()
    loader = make_loader(data, batch_size=batch_size, shuffle=False)
    feats_list, logits_list = [], []
    for x, _y, _idx in loader:
        x = x.to(device)
        logits, feats = model.forward_features(x) if hasattr(model, "forward_features") \
            else (model(x), None)
        logits_list.append(logits.detach().cpu().numpy())
        if feats is not None:
            feats_list.append(feats.detach().cpu().numpy())
    out: Dict[str, np.ndarray] = {
        "logits": np.concatenate(logits_list, axis=0),
        "labels": data.y.copy(),
        "ids": data.ids.copy(),
    }
    if feats_list:
        out["features"] = np.concatenate(feats_list, axis=0)
    return out
