"""Calibration experiments for Type-I error and permutation budget."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Tuple

from loguru import logger

from aristotelian import run_permutation_budget
from aristotelian import run_type1_calibration as _run_type1_calibration_core

from ..infra.io import save_array, should_skip


def run_perm_budget(
    assets_dir: Path, *, device: str, force: bool, seed: int | None
) -> None:
    """Run permutation budget experiment varying number of permutations."""
    output = assets_dir / "perm_budget.npy"
    if should_skip([output], force):
        logger.info(f"Skipping perm_budget (output exists: {output})")
        return

    budgets = [1, 5, 10, 20, 50, 100, 200, 500]
    metrics = ["sgcka_lin", "sgcka_rbf", "sgknn", "sgrsa"]

    num_trials = 50
    budget_out = {}
    for metric in metrics:
        budget_out[metric] = run_permutation_budget(
            metric=metric,
            n=256,
            d=128,
            budgets=budgets,
            num_trials=num_trials,
            quantile=0.95,
            null_type="gaussian",
            seed=seed,
            device=device,
        )

    # Save with metadata
    output_data = {"results": budget_out, "num_trials": num_trials}
    save_array(output, output_data)


def _type1_calibration_single_config(
    config: Tuple[str, str, int, int, int, int, float, int | None, str],
) -> Tuple[str, str, int, int, float, int]:
    """Process a single type1 calibration configuration."""
    (
        null_type,
        metric,
        n,
        d,
        num_trials,
        num_permutations,
        alpha,
        seed,
        device,
    ) = config
    res = _run_type1_calibration_core(
        metric=metric,
        n=n,
        d=d,
        num_trials=num_trials,
        num_permutations=num_permutations,
        quantile=0.95,
        alpha=alpha,
        null_type=null_type,
        seed=seed,
        device=device,
        num_workers=1,  # Don't nest parallelism
    )
    return (null_type, metric, n, d, res.type1_rate, int(sum(res.positives)))


def run_type1_calibration(
    assets_dir: Path,
    *,
    device: str,
    force: bool,
    seed: int | None,
    num_workers: int = 1,
) -> None:
    """Run Type-I calibration experiment across metrics and null types."""
    output = assets_dir / "type1_calibration.npy"
    if should_skip([output], force):
        logger.info(f"Skipping type1_calibration (output exists: {output})")
        return

    metrics = ["sgcka_lin", "sgcka_rbf", "sgknn", "sgrsa"]
    ns = [128, 256, 512, 1024]
    ds = [64, 128, 256, 512]
    null_types = ["gaussian", "heavy", "shuffled"]
    alpha = 0.05
    num_trials = 500  # 500 trials for tighter Wilson CI bands
    num_permutations = 200

    # Build list of all configurations
    configs = [
        (null_type, metric, n, d, num_trials, num_permutations, alpha, seed, device)
        for null_type in null_types
        for metric in metrics
        for n in ns
        for d in ds
    ]
    logger.info(f"Running {len(configs)} type1 calibration configurations")

    # Run configurations (parallel for CPU, sequential for GPU)
    if num_workers > 1 and device == "cpu":
        logger.info(f"Using {num_workers} parallel workers for type1 calibration grid")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_type1_calibration_single_config, configs))
    else:
        results = [_type1_calibration_single_config(cfg) for cfg in configs]

    # Reconstruct the nested dictionary structure
    rates_by_null: Dict[str, Dict[str, list[list[float]]]] = {}
    positives_by_null: Dict[str, Dict[str, list[list[int]]]] = {}
    for null_type in null_types:
        rates_by_null[null_type] = {}
        positives_by_null[null_type] = {}
        for metric in metrics:
            rates_grid = []
            pos_grid = []
            for n in ns:
                rates_row = []
                pos_row = []
                for d in ds:
                    # Find matching result
                    for r_null, r_metric, r_n, r_d, rate, pos in results:
                        if (r_null, r_metric, r_n, r_d) == (null_type, metric, n, d):
                            rates_row.append(rate)
                            pos_row.append(pos)
                            break
                rates_grid.append(rates_row)
                pos_grid.append(pos_row)
            rates_by_null[null_type][metric] = rates_grid
            positives_by_null[null_type][metric] = pos_grid

    default_n = 256
    n_idx = ns.index(default_n)
    rates = {m: rates_by_null["gaussian"][m][n_idx] for m in metrics}
    positives = {m: positives_by_null["gaussian"][m][n_idx] for m in metrics}

    payload = {
        "ns": ns,
        "ds": ds,
        "null_types": null_types,
        "rates_by_null": rates_by_null,
        "positives_by_null": positives_by_null,
        "rates": rates,
        "positives": positives,
        "default_n": default_n,
        "num_trials": num_trials,
        "num_permutations": num_permutations,
        "alpha": alpha,
    }
    save_array(output, payload)
