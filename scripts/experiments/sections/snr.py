"""SNR sweep and phase diagram experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from loguru import logger

from aristotelian import sg_cka_kernel, sg_cka_linear, sg_knn
from aristotelian.utils.logging import get_loguru_safe_tqdm

from ..generators import (
    DEFAULT_NOISE_TYPE,
    NOISE_TYPES,
    gen2_local,
    gen2_local_state,
    make_gen2_linear_signal,
)
from ..infra.device import mp_context
from ..infra.io import noise_output_path, save_array, should_skip
from ..infra.parallel import mp_chunksize, mp_limit_main_threads, mp_worker_init

tqdm = get_loguru_safe_tqdm()

# SNR sweep configuration - broader range, finer grid
SNR_NOISE_RANGE = (0.0, 3.0)  # Extended from (0.0, 2.0)
SNR_NOISE_POINTS = 17  # Finer grid (was 9)
SNR_STRENGTHS = (
    0.25,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
)  # Extended with intermediate values
SNR_STRENGTHS_FN = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)  # For FN rate (start at 0.5)
SNR_RANKS = (1, 5, 10)
SNR_PHASE_SIGMAS_POINTS = 13  # Finer grid for phase diagram (was 7)


def _process_snr_combination(
    rank: int,
    sigma: float,
    trials: int,
    device: str,
    seed_offset: int,
    signal_strength: float,
    metric: str,
    noise_type: str,
    n: int = 512,
    d: int = 256,
    num_permutations: int = 200,
) -> tuple[float, float, float, float, float, float]:
    """Process one (rank, sigma) combination for SNR sweep."""
    torch.manual_seed(seed_offset)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed_offset)
    np.random.seed(seed_offset)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed_offset)

    vals = []
    raw_vals = []
    tau_vals = []
    p_vals = []
    tail_vals = []
    if metric == "sgcka_lin":
        metric_fn = sg_cka_linear
        metric_kwargs = {}
    elif metric == "sgcka_rbf":
        metric_fn = sg_cka_kernel
        metric_kwargs = {}
    elif metric == "sgknn":
        metric_fn = sg_knn
        metric_kwargs = {"k": 10}
    else:
        raise ValueError(f"Unknown metric for SNR sweep: {metric}")
    with torch.no_grad():
        for _ in range(trials):
            if metric in {"sgcka_lin", "sgcka_rbf"}:
                X, Y, _ = make_gen2_linear_signal(
                    n=n,
                    d=d,
                    rank=rank,
                    signal_strength=float(signal_strength),
                    noise_std=float(sigma),
                    noise_type=noise_type,
                    device=device,
                )
            else:
                sep = float(signal_strength)
                noise = float(sigma)
                m = max(2, int(rank))
                state = gen2_local_state(
                    n, d, m, clusters=8, noise=noise, rng=rng, noise_type=noise_type
                )
                X, Y = gen2_local(
                    n=n,
                    d=d,
                    m=m,
                    sep=sep,
                    clusters=8,
                    noise=noise,
                    rng=rng,
                    noise_type=noise_type,
                    state=state,
                )
            res = metric_fn(
                X,
                Y,
                num_permutations=num_permutations,
                quantile=0.95,
                device=device,
                **metric_kwargs,
            )
            # Ensure we get a Python float
            vals.append(float(res.gated))
            raw_vals.append(float(res.raw))
            tau_vals.append(float(res.tau))
            p_vals.append(float(res.pvalue))
            tail_vals.append(float(res.tail_strength))
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals))
    raw_mean = float(np.mean(raw_vals))
    tau_mean = float(np.mean(tau_vals))
    p_mean = float(np.mean(p_vals)) if p_vals else float("nan")
    tail_mean = float(np.mean(tail_vals)) if tail_vals else float("nan")
    return mean_val, std_val, raw_mean, tau_mean, p_mean, tail_mean


def run_snr_sweep(
    assets_dir: Path,
    *,
    device: str,
    force: bool,
    seed: int | None,
    num_workers: int = 1,
    start_method: str | None = None,
) -> None:
    """Run SNR sweep experiment varying signal strength and noise."""
    if os.getenv("RUN_INTEGRATION", "") == "1":
        mp_limit_main_threads()

    noise_levels = np.linspace(SNR_NOISE_RANGE[0], SNR_NOISE_RANGE[1], SNR_NOISE_POINTS)
    strengths = list(SNR_STRENGTHS)
    trials = 50
    perms = 200
    ranks = list(SNR_RANKS)
    metrics = ["sgcka_lin", "sgcka_rbf", "sgknn"]

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    per_noise = len(metrics) * len(ranks) * len(strengths) * len(noise_levels)
    base_output = assets_dir / "snr_sweep.npy"
    for noise_idx, noise_type in enumerate(NOISE_TYPES):
        output = noise_output_path(base_output, noise_type, DEFAULT_NOISE_TYPE)
        if should_skip([output], force):
            logger.info(f"Skipping snr_sweep ({noise_type}) (output exists: {output})")
            continue

        combinations = []
        for m_idx, metric in enumerate(metrics):
            for r_idx, r in enumerate(ranks):
                for s_idx, strength in enumerate(strengths):
                    for n_idx, sigma in enumerate(noise_levels):
                        idx = (
                            noise_idx * per_noise
                            + m_idx * len(ranks) * len(strengths) * len(noise_levels)
                            + r_idx * len(strengths) * len(noise_levels)
                            + s_idx * len(noise_levels)
                            + n_idx
                        )
                        combinations.append((metric, r, sigma, strength, idx))

        if num_workers > 1:
            ctx = mp_context(device, start_method)
            chunksize = mp_chunksize(len(combinations), num_workers)
            with ctx.Pool(processes=num_workers, initializer=mp_worker_init) as pool:
                args = [
                    (
                        r,
                        float(sigma),
                        trials,
                        device,
                        (seed if seed else 0) + idx,
                        float(strength),
                        metric,
                        noise_type,
                        512,
                        256,
                        perms,
                    )
                    for metric, r, sigma, strength, idx in combinations
                ]

                results = list(
                    tqdm(
                        pool.starmap(
                            _process_snr_combination, args, chunksize=chunksize
                        ),
                        total=len(combinations),
                        desc=f"SNR combinations ({noise_type})",
                    )
                )
        else:
            args = [
                (
                    r,
                    float(sigma),
                    trials,
                    device,
                    (seed if seed else 0) + idx,
                    float(strength),
                    metric,
                    noise_type,
                    512,
                    256,
                    perms,
                )
                for metric, r, sigma, strength, idx in combinations
            ]
            results = []
            for entry in tqdm(args, desc=f"SNR combinations ({noise_type})"):
                results.append(_process_snr_combination(*entry))

        means: Dict[str, np.ndarray] = {}
        stds: Dict[str, np.ndarray] = {}
        raw_means: Dict[str, np.ndarray] = {}
        tau_means: Dict[str, np.ndarray] = {}
        p_means: Dict[str, np.ndarray] = {}
        tail_means: Dict[str, np.ndarray] = {}
        for metric in metrics:
            means[metric] = np.zeros((len(strengths), len(ranks), len(noise_levels)))
            stds[metric] = np.zeros_like(means[metric])
            raw_means[metric] = np.zeros_like(means[metric])
            tau_means[metric] = np.zeros_like(means[metric])
            p_means[metric] = np.zeros_like(means[metric])
            tail_means[metric] = np.zeros_like(means[metric])

        idx = 0
        for metric in metrics:
            for r_idx, _ in enumerate(ranks):
                for s_idx, _ in enumerate(strengths):
                    for n_idx, _ in enumerate(noise_levels):
                        mean_val, std_val, raw_mean, tau_mean, p_mean, tail_mean = (
                            results[idx]
                        )
                        means[metric][s_idx, r_idx, n_idx] = mean_val
                        stds[metric][s_idx, r_idx, n_idx] = std_val
                        raw_means[metric][s_idx, r_idx, n_idx] = raw_mean
                        tau_means[metric][s_idx, r_idx, n_idx] = tau_mean
                        p_means[metric][s_idx, r_idx, n_idx] = p_mean
                        tail_means[metric][s_idx, r_idx, n_idx] = tail_mean
                        idx += 1

        payload = {
            "noise_type": noise_type,
            "noise_levels": noise_levels,
            "strengths": strengths,
            "ranks": ranks,
            "metrics": metrics,
            "mean_by_metric": means,
            "std_by_metric": stds,
            "raw_mean_by_metric": raw_means,
            "tau_mean_by_metric": tau_means,
            "p_value_mean_by_metric": p_means,
            "tail_strength_mean_by_metric": tail_means,
            "mean": means["sgcka_lin"],
            "std": stds["sgcka_lin"],
            "raw_mean": raw_means["sgcka_lin"],
            "tau_mean": tau_means["sgcka_lin"],
            "p_value_mean": p_means["sgcka_lin"],
            "tail_strength_mean": tail_means["sgcka_lin"],
        }
        save_array(output, payload)


def run_phase_diagram(
    assets_dir: Path, *, device: str, force: bool, seed: int | None
) -> None:
    """Run phase diagram experiment varying noise and aspect ratio."""
    if seed is not None:
        torch.manual_seed(seed)

    sigmas = np.linspace(
        SNR_NOISE_RANGE[0], SNR_NOISE_RANGE[1], SNR_PHASE_SIGMAS_POINTS
    )
    d_list = [128, 256, 512]
    n_phase = 512
    trials = 50
    perms = 200
    base_output = assets_dir / "phase_diagram.npy"
    for noise_type in NOISE_TYPES:
        output = noise_output_path(base_output, noise_type, DEFAULT_NOISE_TYPE)
        if should_skip([output], force):
            logger.info(
                f"Skipping phase_diagram ({noise_type}) (output exists: {output})"
            )
            continue

        raw_grid = np.zeros((len(sigmas), len(d_list)))
        gated_grid = np.zeros_like(raw_grid)
        p_grid = np.zeros_like(raw_grid)
        tail_grid = np.zeros_like(raw_grid)

        for i, s in enumerate(tqdm(sigmas, desc=f"phase sigma ({noise_type})")):
            for j, d in enumerate(tqdm(d_list, desc="phase d", leave=False)):
                vals_raw, vals_gated = [], []
                vals_p, vals_tail = [], []
                for _ in range(trials):
                    X, Y, _ = make_gen2_linear_signal(
                        n_phase,
                        d,
                        rank=5,
                        signal_strength=1.0,
                        noise_std=float(s),
                        noise_type=noise_type,
                        device=device,
                    )
                    res = sg_cka_linear(
                        X, Y, num_permutations=perms, quantile=0.95, device=device
                    )
                    vals_raw.append(res.raw)
                    vals_gated.append(res.gated)
                    vals_p.append(res.pvalue)
                    vals_tail.append(res.tail_strength)
                raw_grid[i, j] = np.mean(vals_raw)
                gated_grid[i, j] = np.mean(vals_gated)
                p_grid[i, j] = np.mean(vals_p)
                tail_grid[i, j] = np.mean(vals_tail)

        aspect = [d / n_phase for d in d_list]
        payload = {
            "noise_type": noise_type,
            "sigmas": sigmas,
            "aspect": aspect,
            "raw_grid": raw_grid,
            "gated_grid": gated_grid,
            "p_value_grid": p_grid,
            "tail_strength_grid": tail_grid,
            "trials": trials,
            "perms": perms,
            "rank": 5,
            "signal_strength": 1.0,
        }
        save_array(output, payload)
