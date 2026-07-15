"""Device selection and hardware/software introspection."""

from __future__ import annotations

import multiprocessing
import platform
from typing import Dict

import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return CUDA device if available and requested, else CPU."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_info(device: torch.device | None = None) -> Dict[str, object]:
    """Collect device + platform info for the run manifest."""
    if device is None:
        device = get_device()
    info: Dict[str, object] = {
        "device": str(device),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_count": multiprocessing.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        idx = device.index if device.type == "cuda" and device.index is not None else 0
        info["cuda_device_name"] = torch.cuda.get_device_name(idx)
        info["cuda_capability"] = ".".join(map(str, torch.cuda.get_device_capability(idx)))
        info["cuda_version"] = torch.version.cuda
    return info
