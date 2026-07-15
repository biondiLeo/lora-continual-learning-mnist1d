"""Filesystem IO helpers: JSON/CSV, checkpoints and timestamped run dirs."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy scalars/arrays and torch tensors."""

    def default(self, obj: Any) -> Any:  # noqa: D401
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    return path


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv_rows(rows: Sequence[Dict[str, Any]], path: str | Path,
                  fieldnames: Sequence[str] | None = None) -> Path:
    """Write a list of dict rows to CSV. Union of keys used if no fieldnames."""
    path = Path(path)
    ensure_dir(path.parent)
    rows = list(rows)
    if fieldnames is None:
        seen: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.append(k)
        fieldnames = seen
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    return path


def append_csv_row(row: Dict[str, Any], path: str | Path,
                   fieldnames: Sequence[str] | None = None) -> Path:
    """Append a single row, writing a header if the file does not yet exist."""
    path = Path(path)
    ensure_dir(path.parent)
    is_new = not path.exists()
    if fieldnames is None:
        fieldnames = list(row.keys())
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    return path


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(state, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def make_run_dir(base: str | Path, experiment: str, seed: int | None = None,
                 tag: str | None = None) -> Path:
    """Create a unique timestamped run directory (never overwrites)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [ts]
    if seed is not None:
        parts.append(f"seed{seed}")
    if tag:
        parts.append(tag)
    name = "_".join(parts)
    run_dir = Path(base) / experiment / name
    # Guarantee uniqueness even if two runs start in the same second.
    suffix = 0
    unique = run_dir
    while unique.exists():
        suffix += 1
        unique = Path(str(run_dir) + f"_{suffix}")
    ensure_dir(unique)
    return unique
