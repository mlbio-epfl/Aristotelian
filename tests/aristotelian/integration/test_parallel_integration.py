#!/usr/bin/env python
"""Integration tests for multiprocessing speedup."""
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest


def _require_integration() -> None:
    if os.getenv("RUN_INTEGRATION", "") != "1":
        pytest.skip("Set RUN_INTEGRATION=1 to run integration tests")


pytestmark = pytest.mark.integration

MIN_SPEEDUP = float(os.getenv("MIN_SPEEDUP", "1.2"))
SPEEDUP_REPEATS = int(os.getenv("SPEEDUP_REPEATS", "2"))


def test_snr_sweep_speedup():
    _require_integration()
    from scripts.experiments.sections.snr import run_snr_sweep

    speedups = []
    for _ in range(SPEEDUP_REPEATS):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            device = "cpu"

            t0 = time.perf_counter()
            run_snr_sweep(tmpdir, device=device, force=True, seed=42, num_workers=1)
            t_seq = time.perf_counter() - t0
            seq_data = np.load(tmpdir / "snr_sweep.npy", allow_pickle=True).tolist()

            (tmpdir / "snr_sweep.npy").unlink()
            t0 = time.perf_counter()
            run_snr_sweep(tmpdir, device=device, force=True, seed=42, num_workers=4)
            t_par = time.perf_counter() - t0
            par_data = np.load(tmpdir / "snr_sweep.npy", allow_pickle=True).tolist()

            snr_out_seq = seq_data["mean_by_metric"]
            snr_out_par = par_data["mean_by_metric"]

            for metric in snr_out_seq.keys():
                seq_vals = np.array(snr_out_seq[metric])
                par_vals = np.array(snr_out_par[metric])
                max_diff = np.abs(seq_vals - par_vals).max()
                assert max_diff < 1e-5, f"Metric {metric}: max diff = {max_diff}"

            speedups.append(t_seq / t_par)

    speedup = float(np.median(speedups))
    assert (
        speedup > MIN_SPEEDUP
    ), f"Expected speedup > {MIN_SPEEDUP:.2f}x, got {speedup:.2f}x"


def test_null_drift_speedup():
    _require_integration()
    from scripts.experiments.sections.null_drift import run_null_drift_gaussian

    speedups = []
    for _ in range(SPEEDUP_REPEATS):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            device = "cpu"

            t0 = time.perf_counter()
            run_null_drift_gaussian(
                tmpdir,
                device=device,
                force=True,
                seed=42,
                num_workers=1,
                quick_test=True,
            )
            t_seq = time.perf_counter() - t0
            seq_data = np.load(
                tmpdir / "null_drift_gaussian.npy", allow_pickle=True
            ).tolist()

            (tmpdir / "null_drift_gaussian.npy").unlink()
            t0 = time.perf_counter()
            run_null_drift_gaussian(
                tmpdir,
                device=device,
                force=True,
                seed=42,
                num_workers=4,
                quick_test=True,
            )
            t_par = time.perf_counter() - t0
            par_data = np.load(
                tmpdir / "null_drift_gaussian.npy", allow_pickle=True
            ).tolist()

            _, _, vals_seq = seq_data
            _, _, vals_par = par_data

            for key in vals_seq:
                max_diff = np.abs(vals_seq[key] - vals_par[key]).max()
                assert max_diff < 1e-5, f"Key {key}: max diff = {max_diff}"

            speedups.append(t_seq / t_par)

    speedup = float(np.median(speedups))
    assert (
        speedup > MIN_SPEEDUP
    ), f"Expected speedup > {MIN_SPEEDUP:.2f}x, got {speedup:.2f}x"
