"""Multiprocessing utilities for parallel experiment execution."""

from __future__ import annotations

import multiprocessing as mp
import os

import torch

# Sections that support parallel execution
PARALLEL_SECTIONS = {
    "snr_sweep",
    "null_drift_gaussian",
    "null_drift_heavy",
    "type1_calibration",
    "prh_alignment",
    "exp_b_aggregator_calibration",
}


def mp_worker_init() -> None:
    """Initialize worker process with single-threaded settings."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass


def mp_limit_main_threads() -> None:
    """Limit threads in main process for parallel execution."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass


def mp_chunksize(total: int, num_workers: int) -> int:
    """Compute chunk size for parallel map operations."""
    return max(1, total // max(1, num_workers * 4))


class NonDaemonProcess(mp.Process):
    """Process that can spawn children (non-daemon)."""

    def __init__(self, *args, **kwargs):
        if args and isinstance(args[0], mp.context.BaseContext):
            args = args[1:]
        super().__init__(*args, **kwargs)

    @property
    def daemon(self) -> bool:
        return False

    @daemon.setter
    def daemon(self, value: bool) -> None:
        return None


def make_non_daemon_pool(ctx, processes: int):
    """Create pool with non-daemon processes that can spawn children."""
    return mp.pool.Pool(processes=processes, context=ctx, Process=NonDaemonProcess)
