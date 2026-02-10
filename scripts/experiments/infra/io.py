"""I/O utilities for experiment outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch


def to_cpu(obj):
    """Recursively convert tensors to CPU numpy arrays."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


def save_array(path: Path, payload) -> None:
    """Save payload to numpy file, converting tensors to CPU."""
    np.save(path, np.array(to_cpu(payload), dtype=object))


def should_skip(outputs: Sequence[Path], force: bool) -> bool:
    """Check if outputs exist and should be skipped."""
    return (not force) and all(p.exists() for p in outputs)


def write_csv_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write list of dicts as CSV rows."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def noise_output_path(
    base: Path, noise_type: str, default_noise_type: str = "gaussian"
) -> Path:
    """Get output path for a specific noise type variant."""
    if noise_type == default_noise_type:
        return base
    return base.with_name(f"{base.stem}_{noise_type}{base.suffix}")
