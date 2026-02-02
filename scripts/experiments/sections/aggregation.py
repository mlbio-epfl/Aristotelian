"""Aggregation experiments: aggregator calibration."""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from aristotelian.experiments.layerwise_engine import (
    permutation_null_matrices_layerwise,
    run_batched_multi_aggregator_experiment,
    similarity_matrix_layerwise,
)
from aristotelian.metrics.aggregation import (
    SimpleMetric,
    agg_colmax_mean,
    agg_max,
    agg_rowmax_mean,
    agg_topk_mean,
    compute_null_summary,
    gated_rescaled,
)
from aristotelian.metrics.api import raw_cka_linear

from ..generators import (
    DEFAULT_NOISE_TYPE,
    NOISE_TYPES,
    make_random_layers,
    make_random_layers_batched,
    make_signal_layers,
    make_signal_layers_batched,
)
from ..infra.io import noise_output_path, save_array, should_skip


def _naive_calibrated_max(
    S: torch.Tensor,
    null_matrices: Sequence[torch.Tensor],
    alpha: float = 0.05,
    s_max: float = 1.0,
) -> float:
    """Compute naive per-cell calibration followed by max aggregation.

    This is the "wrong" approach: calibrate each cell independently, then take max.
    It doesn't properly handle multiple comparisons and will still show inflated
    values under null conditions.

    Args:
        S: Raw similarity matrix of shape (L, L).
        null_matrices: List of null similarity matrices (one per permutation).
        alpha: Significance level.
        s_max: Maximum possible score for rescaling.

    Returns:
        Max of per-cell gated values.
    """
    L1, L2 = S.shape

    # Stack null matrices for efficient computation
    null_stack = torch.stack(list(null_matrices))  # (num_perms, L1, L2)

    # Compute gated value for each cell
    gated_matrix = torch.zeros_like(S)

    for i in range(L1):
        for j in range(L2):
            raw_val = float(S[i, j].item())
            # Null samples for this cell
            null_samples = null_stack[:, i, j].cpu().numpy().tolist()

            # Compute summary statistics
            summary = compute_null_summary(null_samples, T_obs=raw_val, alpha=alpha)

            # Compute gated value
            g = gated_rescaled(raw_val, tau_alpha=summary["tau_alpha"], s_max=s_max)
            if summary["p_value"] > alpha:
                g = 0.0

            gated_matrix[i, j] = g

    # Return max of gated values
    return float(gated_matrix.max().item())


def _cka_metric() -> SimpleMetric:
    """Get CKA linear metric for aggregation experiments."""
    return SimpleMetric(
        name="cka_linear",
        max_value=1.0,
        compute=raw_cka_linear,
    )


def _layerwise_similarity(
    repsA_layers: Sequence[torch.Tensor],
    repsB_layers: Sequence[torch.Tensor],
    metric: SimpleMetric,
) -> torch.Tensor:
    """Compute layerwise similarity matrix."""
    return similarity_matrix_layerwise(
        repsA_layers, repsB_layers, metric, metric_name=metric.name
    )


def _permutation_null_matrices(
    repsA_layers: Sequence[torch.Tensor],
    repsB_layers: Sequence[torch.Tensor],
    metric: SimpleMetric,
    *,
    num_permutations: int,
    seed: int | None,
) -> list[torch.Tensor]:
    """Generate permutation null similarity matrices."""
    return permutation_null_matrices_layerwise(
        repsA_layers,
        repsB_layers,
        metric,
        metric_name=metric.name,
        num_permutations=num_permutations,
        seed=seed,
    )


def _aggregate_nulls_from_matrices(
    matrices: Sequence[torch.Tensor],
    agg_fn: Callable,
) -> list[float]:
    """Aggregate null matrices using given aggregation function."""
    out = []
    for S in matrices:
        res = agg_fn(S)
        value = res.value if hasattr(res, "value") else res
        out.append(float(value))
    return out


def _get_aggregator(name: str) -> Callable:
    """Get aggregator function by name (picklable for ProcessPoolExecutor)."""
    if name == "max":
        return lambda S: agg_max(S)
    elif name == "rowmax_mean":
        return lambda S: agg_rowmax_mean(S)
    elif name == "colmax_mean":
        return lambda S: agg_colmax_mean(S)
    elif name == "topk_5":
        return lambda S: agg_topk_mean(S, k=5)
    elif name == "topk_10":
        return lambda S: agg_topk_mean(S, k=10)
    else:
        raise ValueError(f"Unknown aggregator: {name}")


def _exp_b_trial_all_aggs(
    config: Tuple[int, int, int, int, int, int, float, float, str, int | None, str],
    aggregator_names: Sequence[str],
) -> list[Dict[str, object]]:
    """Process a single (L, trial) combination for exp_b across all aggregators."""
    L = config[0]
    trial = config[1]
    n = config[2]
    d = config[3]
    num_permutations = config[4]
    aligned_rank = config[5]
    aligned_signal_strength = config[6]
    aligned_noise_std = config[7]
    noise_type = config[8]
    seed = config[9]
    device = config[10]

    alpha = 0.05
    metric = _cka_metric()

    if seed is not None:
        torch.manual_seed(seed + L * 1000 + trial)

    with torch.no_grad():
        repsA = make_random_layers(n, d, L, device=device)
        repsB = make_random_layers(n, d, L, device=device)
        S_null = _layerwise_similarity(repsA, repsB, metric)
        null_mats = _permutation_null_matrices(
            repsA,
            repsB,
            metric,
            num_permutations=num_permutations,
            seed=None if seed is None else seed + trial,
        )

        repsA_sig, repsB_sig = make_signal_layers(
            n,
            d,
            L,
            rank=5,
            signal_strength=1.0,
            noise_std=1.0,
            noise_type=noise_type,
            device=device,
        )
        S_sig = _layerwise_similarity(repsA_sig, repsB_sig, metric)
        null_mats_sig = _permutation_null_matrices(
            repsA_sig,
            repsB_sig,
            metric,
            num_permutations=num_permutations,
            seed=None if seed is None else seed + 1000 + trial,
        )

        repsA_aligned, repsB_aligned = make_signal_layers(
            n,
            d,
            L,
            rank=aligned_rank,
            signal_strength=aligned_signal_strength,
            noise_std=aligned_noise_std,
            noise_type=noise_type,
            device=device,
        )
        S_aligned = _layerwise_similarity(repsA_aligned, repsB_aligned, metric)
        null_mats_aligned = _permutation_null_matrices(
            repsA_aligned,
            repsB_aligned,
            metric,
            num_permutations=num_permutations,
            seed=None if seed is None else seed + 2000 + trial,
        )

    # Compute naive correction once (only applicable to max aggregator concept)
    g_naive = _naive_calibrated_max(
        S_null, null_mats, alpha=alpha, s_max=metric.max_value
    )
    g_naive_sig = _naive_calibrated_max(
        S_sig, null_mats_sig, alpha=alpha, s_max=metric.max_value
    )
    g_naive_aligned = _naive_calibrated_max(
        S_aligned, null_mats_aligned, alpha=alpha, s_max=metric.max_value
    )

    results = []
    for agg_name in aggregator_names:
        agg_fn = _get_aggregator(agg_name)
        T_obs = agg_fn(S_null).value
        null_samples = _aggregate_nulls_from_matrices(null_mats, agg_fn)
        summary = compute_null_summary(null_samples, T_obs=T_obs, alpha=alpha)
        g = gated_rescaled(
            T_obs, tau_alpha=summary["tau_alpha"], s_max=metric.max_value
        )
        if summary["p_value"] > alpha:
            g = 0.0

        T_obs_sig = agg_fn(S_sig).value
        null_samples_sig = _aggregate_nulls_from_matrices(null_mats_sig, agg_fn)
        summary_sig = compute_null_summary(
            null_samples_sig, T_obs=T_obs_sig, alpha=alpha
        )
        g_sig = gated_rescaled(
            T_obs_sig, tau_alpha=summary_sig["tau_alpha"], s_max=metric.max_value
        )
        if summary_sig["p_value"] > alpha:
            g_sig = 0.0

        T_obs_aligned = agg_fn(S_aligned).value
        null_samples_aligned = _aggregate_nulls_from_matrices(null_mats_aligned, agg_fn)
        summary_aligned = compute_null_summary(
            null_samples_aligned, T_obs=T_obs_aligned, alpha=alpha
        )
        g_aligned = gated_rescaled(
            T_obs_aligned,
            tau_alpha=summary_aligned["tau_alpha"],
            s_max=metric.max_value,
        )
        if summary_aligned["p_value"] > alpha:
            g_aligned = 0.0

        # Naive correction only meaningful for max aggregator
        naive_val = g_naive if agg_name == "max" else float("nan")
        naive_val_sig = g_naive_sig if agg_name == "max" else float("nan")
        naive_val_aligned = g_naive_aligned if agg_name == "max" else float("nan")

        results.append(
            {
                "agg_name": agg_name,
                "L": L,
                "trial": trial,
                "noise_type": noise_type,
                "raw": T_obs,
                "gated": g,
                "naive": naive_val,
                "p_value": summary["p_value"],
                "tail": summary["tail_strength"],
                "raw_sig": T_obs_sig,
                "gated_sig": g_sig,
                "naive_sig": naive_val_sig,
                "p_value_sig": summary_sig["p_value"],
                "tail_sig": summary_sig["tail_strength"],
                "raw_aligned": T_obs_aligned,
                "gated_aligned": g_aligned,
                "naive_aligned": naive_val_aligned,
                "p_value_aligned": summary_aligned["p_value"],
                "tail_aligned": summary_aligned["tail_strength"],
            }
        )
    return results


def _run_exp_b_batched_for_layer(
    L: int,
    num_trials: int,
    n: int,
    d: int,
    num_permutations: int,
    aligned_rank: int,
    aligned_signal_strength: float,
    aligned_noise_std: float,
    noise_type: str,
    seed: int | None,
    device: str,
    aggregator_names: Sequence[str],
    alpha: float = 0.05,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run batched exp_b for a single layer count L across all trials and aggregators.

    Returns dictionary mapping aggregator names to result arrays.
    """
    if seed is not None:
        torch.manual_seed(seed + L * 1000)
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + L * 1000)

    with torch.no_grad():
        # Generate all null condition data at once
        repsA_null = make_random_layers_batched(num_trials, n, d, L, device=device)
        repsB_null = make_random_layers_batched(num_trials, n, d, L, device=device)

        # Generate all signal condition data at once
        repsA_sig, repsB_sig = make_signal_layers_batched(
            num_trials,
            n,
            d,
            L,
            rank=5,
            signal_strength=1.0,
            noise_std=1.0,
            noise_type=noise_type,
            device=device,
        )

        # Generate all aligned condition data at once
        repsA_aligned, repsB_aligned = make_signal_layers_batched(
            num_trials,
            n,
            d,
            L,
            rank=aligned_rank,
            signal_strength=aligned_signal_strength,
            noise_std=aligned_noise_std,
            noise_type=noise_type,
            device=device,
        )

        # Run batched multi-aggregator experiment for each condition
        null_results = run_batched_multi_aggregator_experiment(
            repsA_null,
            repsB_null,
            aggregator_names,
            num_permutations=num_permutations,
            alpha=alpha,
            seed=seed,
            show_progress=True,
        )

        sig_results = run_batched_multi_aggregator_experiment(
            repsA_sig,
            repsB_sig,
            aggregator_names,
            num_permutations=num_permutations,
            alpha=alpha,
            seed=seed + 1000 if seed else None,
            show_progress=True,
        )

        aligned_results = run_batched_multi_aggregator_experiment(
            repsA_aligned,
            repsB_aligned,
            aggregator_names,
            num_permutations=num_permutations,
            alpha=alpha,
            seed=seed + 2000 if seed else None,
            show_progress=True,
        )

    # Organize results by aggregator
    # Note: naive correction not computed in batched mode (would require per-trial matrix access)
    nan_arr = np.full(num_trials, np.nan)
    results = {}
    for agg_name in aggregator_names:
        results[agg_name] = {
            "raw": null_results[agg_name]["raw"].cpu().numpy(),
            "gated": null_results[agg_name]["gated"].cpu().numpy(),
            "naive": nan_arr.copy(),  # Not computed in batched mode
            "p_value": null_results[agg_name]["p_value"].cpu().numpy(),
            "tail": null_results[agg_name]["tail_strength"].cpu().numpy(),
            "raw_sig": sig_results[agg_name]["raw"].cpu().numpy(),
            "gated_sig": sig_results[agg_name]["gated"].cpu().numpy(),
            "naive_sig": nan_arr.copy(),  # Not computed in batched mode
            "p_value_sig": sig_results[agg_name]["p_value"].cpu().numpy(),
            "tail_sig": sig_results[agg_name]["tail_strength"].cpu().numpy(),
            "raw_aligned": aligned_results[agg_name]["raw"].cpu().numpy(),
            "gated_aligned": aligned_results[agg_name]["gated"].cpu().numpy(),
            "naive_aligned": nan_arr.copy(),  # Not computed in batched mode
            "p_value_aligned": aligned_results[agg_name]["p_value"].cpu().numpy(),
            "tail_aligned": aligned_results[agg_name]["tail_strength"].cpu().numpy(),
        }

    return results


def run_exp_b_aggregator_calibration(
    assets_dir: Path,
    *,
    device: str,
    force: bool,
    seed: int | None,
    num_workers: int = 1,
) -> None:
    """Run aggregator calibration experiment comparing different aggregators.

    Uses GPU-batched operations when device is cuda for significant speedup.
    Falls back to CPU parallelization when device is cpu and num_workers > 1.
    """
    # Use d/n = 8 to match PRH settings (high aspect ratio = more variance = stronger max inflation)
    n = 32
    d = 256
    layers_list = [2, 4, 8, 16, 32, 64, 128]
    num_trials = 50
    num_permutations = 200
    alpha = 0.05

    aligned_rank = 5
    aligned_signal_strength = 2.5
    aligned_noise_std = 0.5

    aggregator_names = ["max", "rowmax_mean", "colmax_mean", "topk_5", "topk_10"]

    base_output = assets_dir / "exp_b_aggregator_calibration.npy"

    # Use batched GPU mode when on CUDA
    use_batched = device != "cpu" and torch.cuda.is_available()

    for noise_type in NOISE_TYPES:
        output = noise_output_path(base_output, noise_type, DEFAULT_NOISE_TYPE)
        if should_skip([output], force):
            logger.info(
                f"Skipping exp_b_aggregator_calibration ({noise_type}) (output exists: {output})"
            )
            continue

        if use_batched:
            # GPU-batched mode
            logger.info(
                f"Running exp_b ({noise_type}) with GPU batching: "
                f"{len(layers_list)} layers × {num_trials} trials × {len(aggregator_names)} aggregators"
            )

            agg_results = {
                agg_name: {
                    "raw_mean": [],
                    "gated_mean": [],
                    "naive_mean": [],
                    "raw_mean_signal": [],
                    "gated_mean_signal": [],
                    "naive_mean_signal": [],
                    "raw_mean_aligned": [],
                    "gated_mean_aligned": [],
                    "naive_mean_aligned": [],
                    "raw_std": [],
                    "gated_std": [],
                    "naive_std": [],
                    "raw_std_signal": [],
                    "gated_std_signal": [],
                    "naive_std_signal": [],
                    "raw_std_aligned": [],
                    "gated_std_aligned": [],
                    "naive_std_aligned": [],
                    "p_value_mean": [],
                    "tail_strength_mean": [],
                    "p_value_mean_signal": [],
                    "tail_strength_mean_signal": [],
                    "p_value_mean_aligned": [],
                    "tail_strength_mean_aligned": [],
                }
                for agg_name in aggregator_names
            }

            for L in tqdm(layers_list, desc=f"exp_b ({noise_type})", unit="layer"):
                layer_results = _run_exp_b_batched_for_layer(
                    L=L,
                    num_trials=num_trials,
                    n=n,
                    d=d,
                    num_permutations=num_permutations,
                    aligned_rank=aligned_rank,
                    aligned_signal_strength=aligned_signal_strength,
                    aligned_noise_std=aligned_noise_std,
                    noise_type=noise_type,
                    seed=seed,
                    device=device,
                    aggregator_names=aggregator_names,
                    alpha=alpha,
                )

                for agg_name in aggregator_names:
                    ar = layer_results[agg_name]
                    agg_results[agg_name]["raw_mean"].append(float(np.mean(ar["raw"])))
                    agg_results[agg_name]["gated_mean"].append(
                        float(np.mean(ar["gated"]))
                    )
                    agg_results[agg_name]["naive_mean"].append(
                        float(np.nanmean(ar["naive"]))
                    )
                    agg_results[agg_name]["raw_mean_signal"].append(
                        float(np.mean(ar["raw_sig"]))
                    )
                    agg_results[agg_name]["gated_mean_signal"].append(
                        float(np.mean(ar["gated_sig"]))
                    )
                    agg_results[agg_name]["naive_mean_signal"].append(
                        float(np.nanmean(ar["naive_sig"]))
                    )
                    agg_results[agg_name]["raw_mean_aligned"].append(
                        float(np.mean(ar["raw_aligned"]))
                    )
                    agg_results[agg_name]["gated_mean_aligned"].append(
                        float(np.mean(ar["gated_aligned"]))
                    )
                    agg_results[agg_name]["naive_mean_aligned"].append(
                        float(np.nanmean(ar["naive_aligned"]))
                    )
                    agg_results[agg_name]["raw_std"].append(float(np.std(ar["raw"])))
                    agg_results[agg_name]["gated_std"].append(
                        float(np.std(ar["gated"]))
                    )
                    agg_results[agg_name]["naive_std"].append(
                        float(np.nanstd(ar["naive"]))
                    )
                    agg_results[agg_name]["raw_std_signal"].append(
                        float(np.std(ar["raw_sig"]))
                    )
                    agg_results[agg_name]["gated_std_signal"].append(
                        float(np.std(ar["gated_sig"]))
                    )
                    agg_results[agg_name]["naive_std_signal"].append(
                        float(np.nanstd(ar["naive_sig"]))
                    )
                    agg_results[agg_name]["raw_std_aligned"].append(
                        float(np.std(ar["raw_aligned"]))
                    )
                    agg_results[agg_name]["gated_std_aligned"].append(
                        float(np.std(ar["gated_aligned"]))
                    )
                    agg_results[agg_name]["naive_std_aligned"].append(
                        float(np.nanstd(ar["naive_aligned"]))
                    )
                    agg_results[agg_name]["p_value_mean"].append(
                        float(np.mean(ar["p_value"]))
                    )
                    agg_results[agg_name]["tail_strength_mean"].append(
                        float(np.mean(ar["tail"]))
                    )
                    agg_results[agg_name]["p_value_mean_signal"].append(
                        float(np.mean(ar["p_value_sig"]))
                    )
                    agg_results[agg_name]["tail_strength_mean_signal"].append(
                        float(np.mean(ar["tail_sig"]))
                    )
                    agg_results[agg_name]["p_value_mean_aligned"].append(
                        float(np.mean(ar["p_value_aligned"]))
                    )
                    agg_results[agg_name]["tail_strength_mean_aligned"].append(
                        float(np.mean(ar["tail_aligned"]))
                    )

                # Clear GPU cache after each layer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        else:
            # Original CPU-parallel mode
            configs = [
                (
                    L,
                    trial,
                    n,
                    d,
                    num_permutations,
                    aligned_rank,
                    aligned_signal_strength,
                    aligned_noise_std,
                    noise_type,
                    seed,
                    device,
                )
                for L in layers_list
                for trial in range(num_trials)
            ]
            logger.info(f"Running {len(configs)} exp_b trials ({noise_type})")

            if num_workers > 1 and device == "cpu":
                logger.info(f"Using {num_workers} parallel workers for exp_b")
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    results = list(
                        executor.map(
                            _exp_b_trial_all_aggs,
                            configs,
                            itertools.repeat(aggregator_names),
                        )
                    )
            else:
                results = [
                    _exp_b_trial_all_aggs(cfg, aggregator_names) for cfg in configs
                ]

            trial_results = [item for sublist in results for item in sublist]

            agg_results = {}
            for agg_name in aggregator_names:
                raw_means = []
                gated_means = []
                naive_means = []
                raw_means_signal = []
                gated_means_signal = []
                naive_means_signal = []
                raw_means_aligned = []
                gated_means_aligned = []
                naive_means_aligned = []
                raw_stds = []
                gated_stds = []
                naive_stds = []
                raw_stds_signal = []
                gated_stds_signal = []
                naive_stds_signal = []
                raw_stds_aligned = []
                gated_stds_aligned = []
                naive_stds_aligned = []
                p_means = []
                tail_means = []
                p_means_signal = []
                tail_means_signal = []
                p_means_aligned = []
                tail_means_aligned = []

                for L in layers_list:
                    layer_results = [
                        r
                        for r in trial_results
                        if r["agg_name"] == agg_name and r["L"] == L
                    ]
                    raw_trials = [r["raw"] for r in layer_results]
                    gated_trials = [r["gated"] for r in layer_results]
                    naive_trials = [r["naive"] for r in layer_results]
                    raw_trials_signal = [r["raw_sig"] for r in layer_results]
                    gated_trials_signal = [r["gated_sig"] for r in layer_results]
                    naive_trials_signal = [r["naive_sig"] for r in layer_results]
                    raw_trials_aligned = [r["raw_aligned"] for r in layer_results]
                    gated_trials_aligned = [r["gated_aligned"] for r in layer_results]
                    naive_trials_aligned = [r["naive_aligned"] for r in layer_results]
                    p_trials = [r["p_value"] for r in layer_results]
                    tail_trials = [r["tail"] for r in layer_results]
                    p_trials_signal = [r["p_value_sig"] for r in layer_results]
                    tail_trials_signal = [r["tail_sig"] for r in layer_results]
                    p_trials_aligned = [r["p_value_aligned"] for r in layer_results]
                    tail_trials_aligned = [r["tail_aligned"] for r in layer_results]

                    raw_means.append(float(np.mean(raw_trials)))
                    gated_means.append(float(np.mean(gated_trials)))
                    naive_means.append(float(np.nanmean(naive_trials)))
                    raw_means_signal.append(float(np.mean(raw_trials_signal)))
                    gated_means_signal.append(float(np.mean(gated_trials_signal)))
                    naive_means_signal.append(float(np.nanmean(naive_trials_signal)))
                    raw_means_aligned.append(float(np.mean(raw_trials_aligned)))
                    gated_means_aligned.append(float(np.mean(gated_trials_aligned)))
                    naive_means_aligned.append(float(np.nanmean(naive_trials_aligned)))
                    raw_stds.append(float(np.std(raw_trials)))
                    gated_stds.append(float(np.std(gated_trials)))
                    naive_stds.append(float(np.nanstd(naive_trials)))
                    raw_stds_signal.append(float(np.std(raw_trials_signal)))
                    gated_stds_signal.append(float(np.std(gated_trials_signal)))
                    naive_stds_signal.append(float(np.nanstd(naive_trials_signal)))
                    raw_stds_aligned.append(float(np.std(raw_trials_aligned)))
                    gated_stds_aligned.append(float(np.std(gated_trials_aligned)))
                    naive_stds_aligned.append(float(np.nanstd(naive_trials_aligned)))
                    p_means.append(float(np.mean(p_trials)))
                    tail_means.append(float(np.mean(tail_trials)))
                    p_means_signal.append(float(np.mean(p_trials_signal)))
                    tail_means_signal.append(float(np.mean(tail_trials_signal)))
                    p_means_aligned.append(float(np.mean(p_trials_aligned)))
                    tail_means_aligned.append(float(np.mean(tail_trials_aligned)))
                agg_results[agg_name] = {
                    "raw_mean": raw_means,
                    "gated_mean": gated_means,
                    "naive_mean": naive_means,
                    "raw_mean_signal": raw_means_signal,
                    "gated_mean_signal": gated_means_signal,
                    "naive_mean_signal": naive_means_signal,
                    "raw_std": raw_stds,
                    "gated_std": gated_stds,
                    "naive_std": naive_stds,
                    "raw_std_signal": raw_stds_signal,
                    "gated_std_signal": gated_stds_signal,
                    "naive_std_signal": naive_stds_signal,
                    "p_value_mean": p_means,
                    "tail_strength_mean": tail_means,
                    "p_value_mean_signal": p_means_signal,
                    "tail_strength_mean_signal": tail_means_signal,
                    "raw_mean_aligned": raw_means_aligned,
                    "gated_mean_aligned": gated_means_aligned,
                    "naive_mean_aligned": naive_means_aligned,
                    "raw_std_aligned": raw_stds_aligned,
                    "gated_std_aligned": gated_stds_aligned,
                    "naive_std_aligned": naive_stds_aligned,
                    "p_value_mean_aligned": p_means_aligned,
                    "tail_strength_mean_aligned": tail_means_aligned,
                }

        payload = {
            "noise_type": noise_type,
            "layers": layers_list,
            "results": agg_results,
            "num_trials": num_trials,
            "num_permutations": num_permutations,
            "alpha": alpha,
            "aligned_rank": aligned_rank,
            "aligned_signal_strength": aligned_signal_strength,
            "aligned_noise_std": aligned_noise_std,
        }
        save_array(output, payload)
