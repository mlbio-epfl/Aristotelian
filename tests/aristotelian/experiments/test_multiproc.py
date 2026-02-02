#!/usr/bin/env python
"""Tests for multiprocessing helpers in experiments sections."""
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

if os.getenv("RUN_INTEGRATION", "") != "1":
    pytest.skip(
        "Set RUN_INTEGRATION=1 to run integration tests", allow_module_level=True
    )

pytestmark = pytest.mark.integration


def test_snr_sweep_deterministic():
    """Test that SNR sweep helper function is deterministic."""
    from scripts.experiments.sections.snr import _process_snr_combination

    device = "cpu"

    result1 = _process_snr_combination(
        rank=10,
        sigma=0.5,
        trials=3,
        device=device,
        seed_offset=42,
        signal_strength=15.0,
        metric="sgcka_lin",
        noise_type="gaussian",
        num_permutations=50,
    )

    result2 = _process_snr_combination(
        rank=10,
        sigma=0.5,
        trials=3,
        device=device,
        seed_offset=42,
        signal_strength=15.0,
        metric="sgcka_lin",
        noise_type="gaussian",
        num_permutations=50,
    )

    for a, b in zip(result1, result2):
        assert abs(a - b) < 1e-6, f"Mismatch: {result1} vs {result2}"

    result3 = _process_snr_combination(
        rank=10,
        sigma=0.5,
        trials=3,
        device=device,
        seed_offset=99,
        signal_strength=15.0,
        metric="sgcka_lin",
        noise_type="gaussian",
        num_permutations=50,
    )

    print(f"result1={result1}, result2={result2}, result3={result3}")

    for a, b in zip(result1, result2):
        assert abs(a - b) < 1e-6, "Same seed should give identical results"
    print("SNR sweep function is deterministic")


def test_snr_sweep_correctness():
    """Test that parallel SNR sweep gives same results as sequential."""
    from scripts.experiments.sections.snr import run_snr_sweep

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        device = "cpu"

        print("Running sequential SNR sweep...")
        t0 = time.perf_counter()
        run_snr_sweep(tmpdir, device=device, force=True, seed=42, num_workers=1)
        t_seq = time.perf_counter() - t0
        seq_data = np.load(tmpdir / "snr_sweep.npy", allow_pickle=True).tolist()

        print("Running parallel SNR sweep...")
        (tmpdir / "snr_sweep.npy").unlink()
        t0 = time.perf_counter()
        run_snr_sweep(tmpdir, device=device, force=True, seed=42, num_workers=4)
        t_par = time.perf_counter() - t0
        par_data = np.load(tmpdir / "snr_sweep.npy", allow_pickle=True).tolist()

        noise_levels_seq = seq_data["noise_levels"]
        noise_levels_par = par_data["noise_levels"]
        snr_out_seq = seq_data["mean_by_metric"]
        snr_out_par = par_data["mean_by_metric"]

        assert np.allclose(
            noise_levels_seq, noise_levels_par
        ), "Noise levels should match"

        for metric in snr_out_seq:
            seq_vals = np.array(snr_out_seq[metric])
            par_vals = np.array(snr_out_par[metric])
            max_diff = np.abs(seq_vals - par_vals).max()
            assert max_diff < 1e-5, f"Metric {metric}: max diff = {max_diff}"

        speedup = t_seq / t_par
        print("\nSNR sweep correctness verified")
        print(f"  Sequential: {t_seq:.2f}s")
        print(f"  Parallel:   {t_par:.2f}s")
        print(f"  Speedup:    {speedup:.2f}x")


def test_null_drift_gaussian_deterministic():
    """Test that null drift helper function is deterministic."""
    from scripts.experiments.metrics import _metric_definitions
    from scripts.experiments.sections.null_drift import (
        _process_null_drift_nd_combination,
    )

    device = "cpu"  # Use CPU for deterministic testing
    metric_defs, multiq_helpers = _metric_definitions(
        num_permutations=50, device=device
    )

    metric_name, fn_raw, fn_gated = metric_defs[0]

    result1 = _process_null_drift_nd_combination(
        n=128,
        d=128,
        trials=2,
        metric_name=metric_name,
        fn_raw=fn_raw,
        fn_gated=fn_gated,
        quantiles=[0.95],
        multiq_helpers=multiq_helpers,
        num_permutations=50,
        device=device,
        seed_offset=42,
    )

    result2 = _process_null_drift_nd_combination(
        n=128,
        d=128,
        trials=2,
        metric_name=metric_name,
        fn_raw=fn_raw,
        fn_gated=fn_gated,
        quantiles=[0.95],
        multiq_helpers=multiq_helpers,
        num_permutations=50,
        device=device,
        seed_offset=42,
    )

    for key in result1:
        if isinstance(result1[key], (int, float)):
            assert abs(result1[key] - result2[key]) < 1e-6, f"{key} should match"
        elif isinstance(result1[key], dict):
            for k2 in result1[key]:
                assert abs(result1[key][k2] - result2[key][k2]) < 1e-6

    print("Null drift function is deterministic")


def test_null_drift_gaussian_correctness():
    """Test that parallel null drift gives same results as sequential."""
    from scripts.experiments.sections.null_drift import run_null_drift_gaussian

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        device = "cpu"

        print("Running sequential null drift...")
        t0 = time.perf_counter()
        run_null_drift_gaussian(
            tmpdir,
            device=device,
            force=True,
            seed=42,
            num_workers=1,
            quick_test=True,  # Use smaller params
        )
        t_seq = time.perf_counter() - t0
        seq_data = np.load(
            tmpdir / "null_drift_gaussian.npy", allow_pickle=True
        ).tolist()

        print("Running parallel null drift...")
        (tmpdir / "null_drift_gaussian.npy").unlink()
        t0 = time.perf_counter()
        run_null_drift_gaussian(
            tmpdir, device=device, force=True, seed=42, num_workers=4, quick_test=True
        )
        t_par = time.perf_counter() - t0
        par_data = np.load(
            tmpdir / "null_drift_gaussian.npy", allow_pickle=True
        ).tolist()

        n_list_seq, d_list_seq, vals_seq = seq_data
        n_list_par, d_list_par, vals_par = par_data

        assert n_list_seq == n_list_par
        assert d_list_seq == d_list_par

        for key in vals_seq:
            max_diff = np.abs(vals_seq[key] - vals_par[key]).max()
            assert max_diff < 1e-5, f"Key {key}: max diff = {max_diff}"

        speedup = t_seq / t_par
        print("\nNull drift correctness verified")
        print(f"  Sequential: {t_seq:.2f}s")
        print(f"  Parallel:   {t_par:.2f}s")
        print(f"  Speedup:    {speedup:.2f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING MULTIPROCESSING HELPERS")
    print("=" * 60)

    print("\n1. Testing SNR sweep...")
    test_snr_sweep_deterministic()

    print("\n2. Testing null drift...")
    test_null_drift_gaussian_deterministic()

    print("\n" + "=" * 60)
    print("TESTING FULL SECTIONS (with speedup)")
    print("=" * 60)

    print("\n1. Testing SNR sweep section...")
    test_snr_sweep_correctness()

    print("\n2. Testing null drift section...")
    test_null_drift_gaussian_correctness()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
