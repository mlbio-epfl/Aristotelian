"""CKA Estimator Comparison Experiments.

This script compares different CKA estimators:
1. Biased CKA (standard)
2. Debiased/Song CKA (Re-Align paper)
3. Dependent-columns CKA (arxiv 2502.15104)
4. Signal-gated CKA (our calibration approach)

Experiments:
1. Null drift: CKA on random matrices across varying n and d
2. Dim padding: Signal detection with increasing d (fixed signal subspace)
3. High d/n sweep: Explicit test of high-dimensional, low-sample regime

Usage:
    python -m scripts.analyses.cka_comparison --output-dir ./results/cka_comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from aristotelian import sg_cka_linear
from aristotelian.metrics.estimators import (
    cka_biased,
    cka_debiased,
    cka_depcols,
)
from scripts.experiments.generators import make_gen2_linear_signal, make_pure_noise

# =============================================================================
# Experiment 1: Null Drift (Random Matrices)
# =============================================================================


def run_null_drift_experiment(
    n_list: List[int],
    d_list: List[int],
    num_trials: int = 50,
    num_permutations: int = 200,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Run null drift experiment comparing CKA estimators on random matrices.

    Tests how each estimator behaves when there is no true alignment
    (X and Y are independent random matrices).

    Args:
        n_list: List of sample sizes to test.
        d_list: List of dimensions to test.
        num_trials: Number of trials per (n, d) combination.
        num_permutations: Number of permutations for signal-gated CKA.
        device: Device for computation.
        seed: Random seed.

    Returns:
        Dictionary with arrays of shape (len(n_list), len(d_list)) for each metric.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    results = {
        "biased": np.zeros((len(n_list), len(d_list))),
        "debiased": np.zeros((len(n_list), len(d_list))),
        "depcols": np.zeros((len(n_list), len(d_list))),
        "gated": np.zeros((len(n_list), len(d_list))),
        "pvalue": np.zeros((len(n_list), len(d_list))),
        "biased_std": np.zeros((len(n_list), len(d_list))),
        "debiased_std": np.zeros((len(n_list), len(d_list))),
        "depcols_std": np.zeros((len(n_list), len(d_list))),
        "gated_std": np.zeros((len(n_list), len(d_list))),
    }

    total = len(n_list) * len(d_list)
    with tqdm(total=total, desc="Null drift") as pbar:
        for i, n in enumerate(n_list):
            for j, d in enumerate(d_list):
                biased_vals = []
                debiased_vals = []
                depcols_vals = []
                gated_vals = []
                pvalue_vals = []

                for _ in range(num_trials):
                    X, Y = make_pure_noise(n, d, device=device)

                    biased_vals.append(cka_biased(X, Y))
                    debiased_vals.append(cka_debiased(X, Y))
                    depcols_vals.append(cka_depcols(X, Y))

                    res = sg_cka_linear(
                        X, Y, num_permutations=num_permutations, device=device
                    )
                    gated_vals.append(res.gated)
                    pvalue_vals.append(res.pvalue)

                results["biased"][i, j] = np.mean(biased_vals)
                results["debiased"][i, j] = np.mean(debiased_vals)
                results["depcols"][i, j] = np.mean(depcols_vals)
                results["gated"][i, j] = np.mean(gated_vals)
                results["pvalue"][i, j] = np.mean(pvalue_vals)
                results["biased_std"][i, j] = np.std(biased_vals)
                results["debiased_std"][i, j] = np.std(debiased_vals)
                results["depcols_std"][i, j] = np.std(depcols_vals)
                results["gated_std"][i, j] = np.std(gated_vals)

                pbar.update(1)

    results["n_list"] = np.array(n_list)
    results["d_list"] = np.array(d_list)

    return results


# =============================================================================
# Experiment 2: Dimension Padding with Signal
# =============================================================================


def run_dim_padding_experiment(
    n: int,
    d_list: List[int],
    signal_rank: int = 10,
    snr_list: List[float] = [0.0, 0.5, 1.0, 2.0],
    num_trials: int = 50,
    num_permutations: int = 200,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Run dimension padding experiment with varying signal strength.

    Tests how estimators behave as dimension increases, with and without
    true signal (shared low-rank structure).

    Args:
        n: Number of samples (fixed).
        d_list: List of dimensions to test.
        signal_rank: Rank of shared signal subspace.
        snr_list: List of signal-to-noise ratios (0.0 = no signal).
        num_trials: Number of trials per configuration.
        num_permutations: Number of permutations for signal-gated CKA.
        device: Device for computation.
        seed: Random seed.

    Returns:
        Dictionary with results arrays.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    results = {
        "biased": np.zeros((len(snr_list), len(d_list))),
        "debiased": np.zeros((len(snr_list), len(d_list))),
        "depcols": np.zeros((len(snr_list), len(d_list))),
        "gated": np.zeros((len(snr_list), len(d_list))),
        "pvalue": np.zeros((len(snr_list), len(d_list))),
        "rejection_rate": np.zeros((len(snr_list), len(d_list))),
    }

    total = len(snr_list) * len(d_list)
    with tqdm(total=total, desc="Dim padding") as pbar:
        for i, snr in enumerate(snr_list):
            for j, d in enumerate(d_list):
                biased_vals = []
                debiased_vals = []
                depcols_vals = []
                gated_vals = []
                pvalue_vals = []

                for _ in range(num_trials):
                    if snr == 0.0:
                        X, Y = make_pure_noise(n, d, device=device)
                    else:
                        X, Y, _ = make_gen2_linear_signal(
                            n=n,
                            d=d,
                            rank=signal_rank,
                            signal_strength=snr,
                            noise_std=1.0,
                            noise_type="gaussian",
                            device=device,
                        )

                    biased_vals.append(cka_biased(X, Y))
                    debiased_vals.append(cka_debiased(X, Y))
                    depcols_vals.append(cka_depcols(X, Y))

                    res = sg_cka_linear(
                        X, Y, num_permutations=num_permutations, device=device
                    )
                    gated_vals.append(res.gated)
                    pvalue_vals.append(res.pvalue)

                results["biased"][i, j] = np.mean(biased_vals)
                results["debiased"][i, j] = np.mean(debiased_vals)
                results["depcols"][i, j] = np.mean(depcols_vals)
                results["gated"][i, j] = np.mean(gated_vals)
                results["pvalue"][i, j] = np.mean(pvalue_vals)
                # Rejection rate at alpha=0.05
                results["rejection_rate"][i, j] = np.mean(
                    [p < 0.05 for p in pvalue_vals]
                )

                pbar.update(1)

    results["n"] = n
    results["d_list"] = np.array(d_list)
    results["snr_list"] = np.array(snr_list)
    results["signal_rank"] = signal_rank

    return results


# =============================================================================
# Experiment 3: High d/n Ratio Sweep
# =============================================================================


def run_high_dn_ratio_experiment(
    n: int,
    ratio_list: List[float],
    num_trials: int = 50,
    num_permutations: int = 200,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Run high d/n ratio experiment.

    Explicitly tests the regime where biased CKA fails: high d/n ratios.
    This replicates the key finding from the Re-Align paper.

    Args:
        n: Number of samples (fixed).
        ratio_list: List of d/n ratios to test.
        num_trials: Number of trials per ratio.
        num_permutations: Number of permutations for signal-gated CKA.
        device: Device for computation.
        seed: Random seed.

    Returns:
        Dictionary with results arrays.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    results = {
        "biased_mean": np.zeros(len(ratio_list)),
        "biased_std": np.zeros(len(ratio_list)),
        "debiased_mean": np.zeros(len(ratio_list)),
        "debiased_std": np.zeros(len(ratio_list)),
        "depcols_mean": np.zeros(len(ratio_list)),
        "depcols_std": np.zeros(len(ratio_list)),
        "gated_mean": np.zeros(len(ratio_list)),
        "gated_std": np.zeros(len(ratio_list)),
        "pvalue_mean": np.zeros(len(ratio_list)),
        "rejection_rate": np.zeros(len(ratio_list)),
    }

    for i, ratio in enumerate(tqdm(ratio_list, desc="High d/n sweep")):
        d = int(n * ratio)

        biased_vals = []
        debiased_vals = []
        depcols_vals = []
        gated_vals = []
        pvalue_vals = []

        for _ in range(num_trials):
            X, Y = make_pure_noise(n, d, device=device)

            biased_vals.append(cka_biased(X, Y))
            debiased_vals.append(cka_debiased(X, Y))
            depcols_vals.append(cka_depcols(X, Y))

            res = sg_cka_linear(X, Y, num_permutations=num_permutations, device=device)
            gated_vals.append(res.gated)
            pvalue_vals.append(res.pvalue)

        results["biased_mean"][i] = np.mean(biased_vals)
        results["biased_std"][i] = np.std(biased_vals)
        results["debiased_mean"][i] = np.mean(debiased_vals)
        results["debiased_std"][i] = np.std(debiased_vals)
        results["depcols_mean"][i] = np.mean(depcols_vals)
        results["depcols_std"][i] = np.std(depcols_vals)
        results["gated_mean"][i] = np.mean(gated_vals)
        results["gated_std"][i] = np.std(gated_vals)
        results["pvalue_mean"][i] = np.mean(pvalue_vals)
        results["rejection_rate"][i] = np.mean([p < 0.05 for p in pvalue_vals])

    results["n"] = n
    results["ratio_list"] = np.array(ratio_list)

    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run CKA estimator comparison experiments"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/cka_comparison",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu or cuda)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick version with fewer trials/configs",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=["null_drift", "dim_padding", "high_dn"],
        choices=["null_drift", "dim_padding", "high_dn"],
        help="Which experiments to run",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    seed = args.seed

    if args.quick:
        n_list = [64, 128, 256]
        d_list = [64, 128, 256, 512]
        num_trials = 3
        num_permutations = 100
        ratio_list = [0.5, 1.0, 2.0, 4.0, 8.0]
    else:
        n_list = [32, 48, 64, 96, 128, 192, 256, 384, 512]
        d_list = [
            32,
            48,
            64,
            96,
            128,
            192,
            256,
            384,
            512,
            768,
            1024,
            1536,
            2048,
            3072,
            4096,
            6144,
            8192,
        ]
        num_trials = 100
        num_permutations = 200
        # Harmonized with dim_padding: d_list / n where n=1024
        ratio_list = [d / 1024 for d in d_list]

    # Experiment 1: Null Drift
    if "null_drift" in args.experiments:
        logger.info("Running null drift experiment...")
        null_drift_results = run_null_drift_experiment(
            n_list=n_list,
            d_list=d_list,
            num_trials=num_trials,
            num_permutations=num_permutations,
            device=device,
            seed=seed,
        )
        output_path = output_dir / "null_drift_cka_comparison.npy"
        np.save(output_path, null_drift_results)
        logger.info(f"Saved null drift results to {output_path}")

    # Experiment 2: Dimension Padding
    if "dim_padding" in args.experiments:
        logger.info("Running dimension padding experiment...")
        dim_padding_results = run_dim_padding_experiment(
            n=1024,
            d_list=d_list,
            signal_rank=10,
            snr_list=[0.0, 0.5, 1.0, 2.0],
            num_trials=num_trials,
            num_permutations=num_permutations,
            device=device,
            seed=seed,
        )
        output_path = output_dir / "dim_padding_cka_comparison.npy"
        np.save(output_path, dim_padding_results)
        logger.info(f"Saved dimension padding results to {output_path}")

    # Experiment 3: High d/n Ratio
    if "high_dn" in args.experiments:
        logger.info("Running high d/n ratio experiment...")
        high_dn_results = run_high_dn_ratio_experiment(
            n=1024,
            ratio_list=ratio_list,
            num_trials=num_trials * 2,  # More trials for cleaner curves
            num_permutations=num_permutations,
            device=device,
            seed=seed,
        )
        output_path = output_dir / "high_dn_cka_comparison.npy"
        np.save(output_path, high_dn_results)
        logger.info(f"Saved high d/n results to {output_path}")

    logger.info("All experiments completed!")


if __name__ == "__main__":
    main()
