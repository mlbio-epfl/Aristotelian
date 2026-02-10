#!/usr/bin/env python
"""Generate minimal test data for plotting tests.

This script creates small .npy files that exercise the plotting functions
without requiring full experiment runs.

Usage:
    python -m tests.plotting.create_test_data [--output-dir tests/plotting/test_data]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class GatedResult:
    """Minimal result structure for perm_budget tests."""

    tau: float
    mean: float
    std: float = 0.0


def create_null_drift_data(n_list: list[int], d_list: list[int]) -> np.ndarray:
    """Create minimal null_drift_gaussian/heavy data."""
    vals: dict[tuple[str, str], np.ndarray] = {}
    rng = np.random.default_rng(42)

    for metric in ["CKA (lin)", "kNN"]:
        shape = (len(n_list), len(d_list))
        # Raw values increase with n and d
        raw = rng.random(shape) * 0.5 + 0.3
        vals[(metric, "raw")] = raw
        # Calibrated (q95) values are more stable
        vals[(metric, "q95")] = rng.random(shape) * 0.2 + 0.4
        # Null-centered values are around zero
        vals[(metric, "null_centered")] = rng.random(shape) * 0.4 - 0.2

    return np.array([n_list, d_list, vals], dtype=object)


def create_perm_budget_data() -> dict[str, Any]:
    """Create minimal perm_budget data."""
    budgets = [10, 50, 100, 200]
    results: dict[str, dict[int, GatedResult]] = {}

    rng = np.random.default_rng(42)
    for metric in ["CKA (lin)", "kNN"]:
        results[metric] = {}
        for b in budgets:
            # Tau decreases with more budget
            tau = 0.1 + 0.5 * np.exp(-b / 50)
            # Mean stabilizes
            mean = 0.5 + rng.random() * 0.1
            std = 0.1 / np.sqrt(b)
            results[metric][b] = GatedResult(tau=tau, mean=mean, std=std)

    return {"results": results, "num_trials": 50}


def create_type1_calibration_data() -> dict[str, Any]:
    """Create minimal type1_calibration data."""
    rng = np.random.default_rng(42)
    results: dict[str, Any] = {}

    for metric in ["CKA (lin)", "kNN"]:
        # p-values should be uniformly distributed under null
        p_vals = rng.random(100)
        results[metric] = {"p_values": p_vals, "alpha": 0.05, "type1_rate": 0.048}

    return results


def create_snr_sweep_data() -> dict[str, Any]:
    """Create minimal snr_sweep data matching expected format."""
    noise_levels = np.array([0.0, 0.1, 0.5, 1.0, 2.0])
    strengths = [0.5, 1.0, 2.0]
    ranks = [5, 10]
    num_trials = 50
    rng = np.random.default_rng(42)

    # Shape: (len(strengths), len(ranks), len(noise_levels))
    shape = (len(strengths), len(ranks), len(noise_levels))

    # Means increase with strength, decrease with noise
    mean = np.zeros(shape)
    std = np.zeros(shape)
    for s_idx, s in enumerate(strengths):
        for r_idx, r in enumerate(ranks):
            base = 0.3 + 0.3 * (s / max(strengths))
            mean[s_idx, r_idx, :] = (
                base * np.exp(-0.5 * noise_levels) + rng.random(len(noise_levels)) * 0.1
            )
            std[s_idx, r_idx, :] = rng.random(len(noise_levels)) * 0.05 + 0.02

    raw_mean = mean * 0.8 + rng.random(shape) * 0.1
    tau_mean = mean * 0.9 + rng.random(shape) * 0.05

    return {
        "noise_levels": noise_levels,
        "strengths": strengths,
        "ranks": ranks,
        "mean": mean,
        "std": std,
        "raw_mean": raw_mean,
        "tau_mean": tau_mean,
        "num_trials": num_trials,
    }


def create_phase_diagram_data() -> dict[str, Any]:
    """Create minimal phase_diagram data."""
    n_list = [50, 100, 200]
    d_list = [10, 50, 100]
    snr_list = [0.1, 0.5, 1.0]
    rng = np.random.default_rng(42)

    results: dict[str, Any] = {}
    for metric in ["CKA (lin)", "kNN"]:
        # 3D array: (n, d, snr)
        shape = (len(n_list), len(d_list), len(snr_list))
        results[metric] = {
            "raw": rng.random(shape) * 0.5 + 0.3,
            "q95": rng.random(shape) * 0.3 + 0.5,
        }

    return {
        "n_list": n_list,
        "d_list": d_list,
        "snr_list": snr_list,
        "results": results,
    }


def create_signal_fn_rate_data() -> dict[str, Any]:
    """Create minimal signal_fn_rate data."""
    snr_levels = np.array([0.0, 0.1, 0.5, 1.0, 2.0])
    rng = np.random.default_rng(42)
    results: dict[str, Any] = {}

    for metric in ["CKA (lin)", "kNN"]:
        # FN rate decreases with SNR
        fn_rates = 0.9 * np.exp(-2 * snr_levels)
        fn_stds = rng.random(len(snr_levels)) * 0.05

        results[metric] = {
            "fn_rate": {"mean": fn_rates, "std": fn_stds},
        }

    return {"snr_levels": snr_levels, "results": results, "num_trials": 50}


def create_prh_alignment_data() -> dict[str, Any]:
    """Create minimal prh_alignment data."""
    rng = np.random.default_rng(42)

    # Small matrix for testing
    n_lang, n_vis = 4, 5
    raw_matrix = rng.random((n_lang, n_vis)) * 0.5 + 0.3
    cal_matrix = rng.random((n_lang, n_vis)) * 0.3 + 0.5

    return {
        "metric": "mutual_knn",
        "raw": raw_matrix,
        "calibrated": cal_matrix,
        "llm_models": [f"llm_{i}" for i in range(n_lang)],
        "lvm_models": [f"lvm_{i}" for i in range(n_vis)],
    }


def create_exp_a_max_inflation_data() -> dict[str, Any]:
    """Create minimal exp_a data for aggregator inflation test."""
    rng = np.random.default_rng(42)
    n_samples = 50

    results: dict[str, Any] = {}
    for metric in ["CKA (lin)", "kNN"]:
        results[metric] = {
            "raw_max": rng.random(n_samples) * 0.3 + 0.6,
            "cal_max": rng.random(n_samples) * 0.2 + 0.5,
        }

    return {"results": results, "num_trials": n_samples}


def create_exp_b_aggregator_calibration_data() -> dict[str, Any]:
    """Create minimal exp_b data for aggregator calibration test."""
    rng = np.random.default_rng(42)
    aggregators = ["mean", "max", "median"]

    results: dict[str, Any] = {}
    for agg in aggregators:
        for metric in ["CKA (lin)", "kNN"]:
            key = f"{agg}_{metric}"
            results[key] = {
                "p_values": rng.random(100),
                "type1_rate": rng.random() * 0.1,
            }

    return {"results": results, "aggregators": aggregators}


def create_ranking_preservation_data() -> dict[str, Any]:
    """Create minimal ranking_preservation data."""
    rng = np.random.default_rng(42)
    n_trials = 30

    results: dict[str, Any] = {}
    for metric in ["CKA (lin)", "kNN"]:
        # Kendall tau values
        results[metric] = {
            "kendall_tau": rng.random(n_trials) * 0.4 + 0.5,
            "spearman_rho": rng.random(n_trials) * 0.3 + 0.6,
        }

    return {"results": results, "num_trials": n_trials}


def generate_all_test_data(output_dir: Path) -> None:
    """Generate all test data files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Null drift data (both gaussian and heavy use same structure)
    n_list = [50, 100, 200, 500]
    d_list = [10, 50, 100, 500]

    null_drift_gaussian = create_null_drift_data(n_list, d_list)
    np.save(output_dir / "null_drift_gaussian.npy", null_drift_gaussian)

    null_drift_heavy = create_null_drift_data(n_list, d_list)
    np.save(output_dir / "null_drift_heavy.npy", null_drift_heavy)

    # Perm budget
    perm_budget = create_perm_budget_data()
    np.save(output_dir / "perm_budget.npy", perm_budget)

    # Type 1 calibration
    type1_cal = create_type1_calibration_data()
    np.save(output_dir / "type1_calibration.npy", type1_cal)

    # SNR sweep
    snr_sweep = create_snr_sweep_data()
    np.save(output_dir / "snr_sweep.npy", snr_sweep)

    # Phase diagram
    phase_diagram = create_phase_diagram_data()
    np.save(output_dir / "phase_diagram.npy", phase_diagram)

    # Signal FN rate
    signal_fn_rate = create_signal_fn_rate_data()
    np.save(output_dir / "signal_fn_rate.npy", signal_fn_rate)

    # PRH alignment
    prh_alignment = create_prh_alignment_data()
    np.save(output_dir / "prh_alignment.npy", prh_alignment)

    # Exp A - max inflation
    exp_a = create_exp_a_max_inflation_data()
    np.save(output_dir / "exp_a_max_inflation.npy", exp_a)

    # Exp B - aggregator calibration
    exp_b = create_exp_b_aggregator_calibration_data()
    np.save(output_dir / "exp_b_aggregator_calibration.npy", exp_b)

    # Ranking preservation
    ranking = create_ranking_preservation_data()
    np.save(output_dir / "ranking_preservation.npy", ranking)

    print(f"Generated test data in {output_dir}")
    for f in sorted(output_dir.glob("*.npy")):
        print(f"  {f.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate test data for plotting tests"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "test_data",
        help="Output directory for test data files",
    )
    args = parser.parse_args()
    generate_all_test_data(args.output_dir)


if __name__ == "__main__":
    main()
