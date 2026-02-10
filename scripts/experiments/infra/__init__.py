"""Infrastructure utilities for experiment runner."""

from .device import mp_context, parse_devices, resolve_device
from .io import noise_output_path, save_array, should_skip, to_cpu, write_csv_rows
from .parallel import (
    PARALLEL_SECTIONS,
    NonDaemonProcess,
    make_non_daemon_pool,
    mp_chunksize,
    mp_limit_main_threads,
    mp_worker_init,
)
from .stats import binomial_ci, bootstrap_spearman, mean_ci, rankdata, spearman

__all__ = [
    # device
    "resolve_device",
    "parse_devices",
    "mp_context",
    # io
    "to_cpu",
    "save_array",
    "should_skip",
    "write_csv_rows",
    "noise_output_path",
    # parallel
    "mp_worker_init",
    "mp_limit_main_threads",
    "mp_chunksize",
    "NonDaemonProcess",
    "make_non_daemon_pool",
    "PARALLEL_SECTIONS",
    # stats
    "mean_ci",
    "binomial_ci",
    "spearman",
    "rankdata",
    "bootstrap_spearman",
]
