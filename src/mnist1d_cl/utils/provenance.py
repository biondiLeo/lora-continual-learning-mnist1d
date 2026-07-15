"""Provenance helpers: file hashing, environment capture, code version."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from .device import device_info


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 hex digest of a file, streaming in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _pip_freeze() -> List[str]:
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL, text=True, timeout=60,
        )
        return sorted(line.strip() for line in out.splitlines() if line.strip())
    except Exception:  # pragma: no cover - environment dependent
        return []


def code_version() -> Dict[str, str]:
    """Return a git commit hash if the project is under git, else 'unknown'."""
    info = {"git_commit": "unknown", "git_dirty": "unknown"}
    try:  # pragma: no cover - environment dependent
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, timeout=10
        ).strip()
        info["git_commit"] = commit
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True, timeout=10
        ).strip()
        info["git_dirty"] = "yes" if status else "no"
    except Exception:
        pass
    return info


def capture_environment(include_pip_freeze: bool = True) -> Dict[str, object]:
    """Snapshot of Python/lib versions, device, and (optionally) pip freeze."""
    env: Dict[str, object] = {
        "python_executable": sys.executable,
        "device_info": device_info(),
        "code_version": code_version(),
    }
    # Key library versions (best effort).
    versions: Dict[str, str] = {}
    for mod in ["numpy", "torch", "sklearn", "scipy", "matplotlib", "pandas", "yaml"]:
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "n/a")
        except Exception:
            versions[mod] = "missing"
    env["library_versions"] = versions
    if include_pip_freeze:
        env["pip_freeze"] = _pip_freeze()
    return env
