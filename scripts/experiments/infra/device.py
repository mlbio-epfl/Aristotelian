"""Device resolution and multiprocessing context utilities."""

from __future__ import annotations

from multiprocessing import get_all_start_methods, get_context
from typing import Sequence

import torch


def resolve_device(device: str | None) -> str:
    """Resolve device string, defaulting to CUDA if available."""
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_devices(raw: str | None, default_device: str) -> Sequence[str]:
    """Parse comma-separated device list, filtering unavailable CUDA devices."""
    if not raw:
        return [default_device]
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if not devices:
        return [default_device]
    if torch.cuda.is_available():
        max_cuda = torch.cuda.device_count()
        filtered = []
        for dev in devices:
            if not dev.startswith("cuda:"):
                filtered.append(dev)
                continue
            try:
                idx = int(dev.split("cuda:")[1])
            except ValueError:
                continue
            if idx < max_cuda:
                filtered.append(dev)
        if filtered:
            return filtered
    return [default_device]


def mp_context(device: str, start_method: str | None):
    """Get appropriate multiprocessing context for device."""
    if start_method:
        return get_context(start_method)
    if device.startswith("cuda") and "spawn" in get_all_start_methods():
        return get_context("spawn")
    if "fork" in get_all_start_methods():
        return get_context("fork")
    return get_context("spawn")
