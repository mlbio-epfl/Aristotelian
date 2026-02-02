"""Null drift experiments measuring raw and gated scores under null distributions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Sequence

import numpy as np
import torch
from loguru import logger

from aristotelian.metrics.api import metric_definitions as registry_metric_definitions
from aristotelian.utils.logging import get_loguru_safe_tqdm

from ..generators import make_pure_noise
from ..infra.device import mp_context
from ..infra.io import save_array, should_skip
from ..infra.parallel import mp_chunksize, mp_limit_main_threads, mp_worker_init

tqdm = get_loguru_safe_tqdm()


def _process_null_drift_nd_combination(
    n: int,
    d: int,
    trials: int,
    metric_name: str,
    fn_raw: Callable,
    fn_gated: Callable | None,
    quantiles: Sequence[float],
    multiq_helpers: Dict,
    num_permutations: int,
    device: str,
    seed_offset: int,
) -> Dict[str, float]:
    """Process one (n, d) combination for null drift experiment."""
    torch.manual_seed(seed_offset)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed_offset)
    np.random.seed(seed_offset)

    scores_raw = []
    scores_q = {q: [] for q in quantiles}
    scores_variants = {"null_centered": [], "z": [], "ari": []}

    with torch.no_grad():
        for _ in range(trials):
            X, Y = make_pure_noise(n, d, device=device)
            raw_score = fn_raw(X, Y)
            scores_raw.append(
                float(raw_score) if not isinstance(raw_score, float) else raw_score
            )
            if fn_gated is not None:
                perms = torch.stack(
                    [torch.randperm(n, device=device) for _ in range(num_permutations)]
                )
                if metric_name in multiq_helpers:
                    res = multiq_helpers[metric_name](
                        X,
                        Y,
                        quantiles,
                        num_permutations=num_permutations,
                        device=device,
                        perms=perms,
                    )
                    for q in quantiles:
                        scores_q[q].append(float(res["gated"][q]))
                    scores_variants["null_centered"].append(
                        float(res["variants"].null_centered)
                    )
                    scores_variants["z"].append(float(res["variants"].z))
                    scores_variants["ari"].append(float(res["variants"].ari))
                else:
                    for q in quantiles:
                        res = fn_gated(X, Y, q, perms=perms)
                        scores_q[q].append(float(res.gated))

    def safe_mean(lst):
        mean_val = np.mean(lst)
        if isinstance(mean_val, np.ndarray):
            return float(mean_val.item())
        return float(mean_val)

    result = {"raw": safe_mean(scores_raw)}
    if fn_gated is not None:
        for q in quantiles:
            result[f"q{int(q*100)}"] = safe_mean(scores_q[q])
        if metric_name in multiq_helpers:
            result["null_centered"] = safe_mean(scores_variants["null_centered"])
            result["z"] = safe_mean(scores_variants["z"])
            result["ari"] = safe_mean(scores_variants["ari"])
    return result


def _process_null_drift_heavy_nd_combination(
    n: int,
    d: int,
    trials: int,
    metric_name: str,
    fn_raw: Callable,
    fn_gated: Callable | None,
    quantiles: Sequence[float],
    multiq_helpers: Dict,
    num_permutations: int,
    device: str,
    seed_offset: int,
) -> Dict[str, float]:
    """Process one (n, d) combination for heavy-tail null drift experiment."""
    torch.manual_seed(seed_offset)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed_offset)
    np.random.seed(seed_offset)

    scores_raw = []
    scores_q = {q: [] for q in quantiles}
    scores_variants = {"null_centered": [], "z": [], "ari": []}

    dist = torch.distributions.StudentT(df=3)
    with torch.no_grad():
        for _ in range(trials):
            X = dist.sample((n, d)).to(device)
            Y = dist.sample((n, d)).to(device)
            raw_score = fn_raw(X, Y)
            scores_raw.append(
                float(raw_score) if not isinstance(raw_score, float) else raw_score
            )
            if fn_gated is not None:
                perms = torch.stack(
                    [torch.randperm(n, device=device) for _ in range(num_permutations)]
                )
                if metric_name in multiq_helpers:
                    res = multiq_helpers[metric_name](
                        X,
                        Y,
                        quantiles,
                        num_permutations=num_permutations,
                        device=device,
                        perms=perms,
                    )
                    for q in quantiles:
                        scores_q[q].append(float(res["gated"][q]))
                    scores_variants["null_centered"].append(
                        float(res["variants"].null_centered)
                    )
                    scores_variants["z"].append(float(res["variants"].z))
                    scores_variants["ari"].append(float(res["variants"].ari))
                else:
                    for q in quantiles:
                        res = fn_gated(X, Y, q, perms=perms)
                        scores_q[q].append(float(res.gated))

    def safe_mean(lst):
        mean_val = np.mean(lst)
        if isinstance(mean_val, np.ndarray):
            return float(mean_val.item())
        return float(mean_val)

    result = {"raw": safe_mean(scores_raw)}
    if fn_gated is not None:
        for q in quantiles:
            result[f"q{int(q*100)}"] = safe_mean(scores_q[q])
        if metric_name in multiq_helpers:
            result["null_centered"] = safe_mean(scores_variants["null_centered"])
            result["z"] = safe_mean(scores_variants["z"])
            result["ari"] = safe_mean(scores_variants["ari"])
    return result


def run_null_drift_gaussian(
    assets_dir: Path,
    *,
    device: str,
    force: bool,
    seed: int | None,
    num_workers: int = 1,
    start_method: str | None = None,
    quick_test: bool = False,
) -> None:
    """Run null drift experiment with Gaussian noise.

    Set quick_test=True to use a smaller grid for integration tests.
    """
    if os.getenv("RUN_INTEGRATION", "") == "1":
        mp_limit_main_threads()
    output = assets_dir / "null_drift_gaussian.npy"
    checkpoint = assets_dir / "null_drift_gaussian_checkpoint.npy"
    if should_skip([output], force):
        logger.info(f"Skipping null_drift_gaussian (output exists: {output})")
        return

    if quick_test:
        n_list = [64, 128]
        d_list = [64, 128]
        trials = 5
        num_permutations = 50
    else:
        n_list = [128, 256, 512, 1024, 2048, 4096]
        d_list = [128, 256, 512, 1024, 2048]
        trials = 50
        num_permutations = 200

    quantiles = [0.90, 0.95]

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    metric_defs_all, multiq_helpers = registry_metric_definitions(
        num_permutations=num_permutations, device=device
    )

    # Load checkpoint if exists (for resuming)
    completed_metrics: set[str] = set()
    vals: Dict[tuple[str, str], np.ndarray] = {}
    if checkpoint.exists() and not force:
        try:
            ckpt_data = np.load(checkpoint, allow_pickle=True).item()
            vals = ckpt_data.get("vals", {})
            completed_metrics = set(ckpt_data.get("completed_metrics", []))
            logger.info(
                f"Resuming from checkpoint: {len(completed_metrics)} metrics completed"
            )
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}, starting fresh")

    if num_workers > 1:
        for metric_name, fn_raw, fn_gated in metric_defs_all:
            vals[(metric_name, "raw")] = np.zeros((len(n_list), len(d_list)))
            if fn_gated is not None:
                for q in quantiles:
                    vals[(metric_name, f"q{int(q*100)}")] = np.zeros(
                        (len(n_list), len(d_list))
                    )
                vals[(metric_name, "null_centered")] = np.zeros(
                    (len(n_list), len(d_list))
                )
                vals[(metric_name, "z")] = np.zeros((len(n_list), len(d_list)))
                vals[(metric_name, "ari")] = np.zeros((len(n_list), len(d_list)))

        args = []
        for metric_name, fn_raw, fn_gated in metric_defs_all:
            args.extend(
                [
                    (
                        n,
                        d,
                        trials,
                        metric_name,
                        fn_raw,
                        fn_gated,
                        quantiles,
                        multiq_helpers,
                        num_permutations,
                        device,
                        (seed if seed else 0) + i * len(d_list) + j,
                    )
                    for i, n in enumerate(n_list)
                    for j, d in enumerate(d_list)
                ]
            )

        ctx = mp_context(device, start_method)
        chunksize = mp_chunksize(len(args), num_workers)
        with ctx.Pool(processes=num_workers, initializer=mp_worker_init) as pool:
            results = list(
                tqdm(
                    pool.starmap(
                        _process_null_drift_nd_combination, args, chunksize=chunksize
                    ),
                    total=len(args),
                    desc="Null drift combinations",
                )
            )

        idx = 0
        for metric_name, _, fn_gated in metric_defs_all:
            for i in range(len(n_list)):
                for j in range(len(d_list)):
                    res = results[idx]
                    vals[(metric_name, "raw")][i, j] = res["raw"]
                    if fn_gated is not None:
                        for q in quantiles:
                            vals[(metric_name, f"q{int(q*100)}")][i, j] = res[
                                f"q{int(q*100)}"
                            ]
                        if metric_name in multiq_helpers:
                            vals[(metric_name, "null_centered")][i, j] = res[
                                "null_centered"
                            ]
                            vals[(metric_name, "z")][i, j] = res["z"]
                            vals[(metric_name, "ari")][i, j] = res["ari"]
                    idx += 1
    else:
        for metric_name, fn_raw, fn_gated in metric_defs_all:
            # Skip already completed metrics (from checkpoint)
            if metric_name in completed_metrics:
                logger.info(f"Skipping {metric_name} (already completed)")
                continue

            vals[(metric_name, "raw")] = np.zeros((len(n_list), len(d_list)))
            if fn_gated is not None:
                for q in quantiles:
                    vals[(metric_name, f"q{int(q*100)}")] = np.zeros(
                        (len(n_list), len(d_list))
                    )
                vals[(metric_name, "null_centered")] = np.zeros(
                    (len(n_list), len(d_list))
                )
                vals[(metric_name, "z")] = np.zeros((len(n_list), len(d_list)))
                vals[(metric_name, "ari")] = np.zeros((len(n_list), len(d_list)))

            args = [
                (
                    n,
                    d,
                    trials,
                    metric_name,
                    fn_raw,
                    fn_gated,
                    quantiles,
                    multiq_helpers,
                    num_permutations,
                    device,
                    (seed if seed else 0) + i * len(d_list) + j,
                )
                for i, n in enumerate(n_list)
                for j, d in enumerate(d_list)
            ]

            results = []
            for entry in tqdm(args, desc=f"Null drift {metric_name}"):
                results.append(_process_null_drift_nd_combination(*entry))

            idx = 0
            for i in range(len(n_list)):
                for j in range(len(d_list)):
                    res = results[idx]
                    vals[(metric_name, "raw")][i, j] = res["raw"]
                    if fn_gated is not None:
                        for q in quantiles:
                            vals[(metric_name, f"q{int(q*100)}")][i, j] = res[
                                f"q{int(q*100)}"
                            ]
                        if metric_name in multiq_helpers:
                            vals[(metric_name, "null_centered")][i, j] = res[
                                "null_centered"
                            ]
                            vals[(metric_name, "z")][i, j] = res["z"]
                            vals[(metric_name, "ari")][i, j] = res["ari"]
                    idx += 1

            # Save checkpoint after each metric
            completed_metrics.add(metric_name)
            ckpt_data = {"vals": vals, "completed_metrics": list(completed_metrics)}
            np.save(checkpoint, ckpt_data)
            logger.info(
                f"Checkpoint saved: {metric_name} completed ({len(completed_metrics)}/{len(metric_defs_all)})"
            )

    save_array(output, [n_list, d_list, vals])
    # Clean up checkpoint after successful completion
    if checkpoint.exists():
        checkpoint.unlink()
        logger.info("Checkpoint file removed (experiment completed)")


def run_null_drift_heavy(
    assets_dir: Path,
    *,
    device: str,
    force: bool,
    seed: int | None,
    num_workers: int = 1,
    start_method: str | None = None,
) -> None:
    """Run null drift experiment with heavy-tailed (Student-t) noise."""
    if os.getenv("RUN_INTEGRATION", "") == "1":
        mp_limit_main_threads()
    output = assets_dir / "null_drift_heavy.npy"
    checkpoint = assets_dir / "null_drift_heavy_checkpoint.npy"
    if should_skip([output], force):
        logger.info(f"Skipping null_drift_heavy (output exists: {output})")
        return

    n_list = [128, 256, 512, 1024, 2048, 4096]
    d_list = [128, 256, 512, 1024, 2048]
    trials = 50
    quantiles = [0.90, 0.95]
    num_permutations = 200

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    metric_defs_all, multiq_helpers = registry_metric_definitions(
        num_permutations=num_permutations, device=device
    )

    # Load checkpoint if exists (for resuming)
    completed_metrics: set[str] = set()
    vals_heavy: Dict[tuple[str, str], np.ndarray] = {}
    if checkpoint.exists() and not force:
        try:
            ckpt_data = np.load(checkpoint, allow_pickle=True).item()
            vals_heavy = ckpt_data.get("vals", {})
            completed_metrics = set(ckpt_data.get("completed_metrics", []))
            logger.info(
                f"Resuming from checkpoint: {len(completed_metrics)} metrics completed"
            )
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}, starting fresh")
    if num_workers > 1:
        for metric_name, fn_raw, fn_gated in metric_defs_all:
            vals_heavy[(metric_name, "raw")] = np.zeros((len(n_list), len(d_list)))
            if fn_gated is not None:
                for q in quantiles:
                    vals_heavy[(metric_name, f"q{int(q*100)}")] = np.zeros(
                        (len(n_list), len(d_list))
                    )
                vals_heavy[(metric_name, "null_centered")] = np.zeros(
                    (len(n_list), len(d_list))
                )
                vals_heavy[(metric_name, "z")] = np.zeros((len(n_list), len(d_list)))
                vals_heavy[(metric_name, "ari")] = np.zeros((len(n_list), len(d_list)))

        args = []
        for metric_name, fn_raw, fn_gated in metric_defs_all:
            args.extend(
                [
                    (
                        n,
                        d,
                        trials,
                        metric_name,
                        fn_raw,
                        fn_gated,
                        quantiles,
                        multiq_helpers,
                        num_permutations,
                        device,
                        (seed if seed else 0) + i * len(d_list) + j,
                    )
                    for i, n in enumerate(n_list)
                    for j, d in enumerate(d_list)
                ]
            )

        ctx = mp_context(device, start_method)
        chunksize = mp_chunksize(len(args), num_workers)
        with ctx.Pool(processes=num_workers, initializer=mp_worker_init) as pool:
            results = list(
                tqdm(
                    pool.starmap(
                        _process_null_drift_heavy_nd_combination,
                        args,
                        chunksize=chunksize,
                    ),
                    total=len(args),
                    desc="Null drift heavy combinations",
                )
            )

        idx = 0
        for metric_name, _, fn_gated in metric_defs_all:
            for i in range(len(n_list)):
                for j in range(len(d_list)):
                    res = results[idx]
                    vals_heavy[(metric_name, "raw")][i, j] = res["raw"]
                    if fn_gated is not None:
                        for q in quantiles:
                            vals_heavy[(metric_name, f"q{int(q*100)}")][i, j] = res[
                                f"q{int(q*100)}"
                            ]
                        if metric_name in multiq_helpers:
                            vals_heavy[(metric_name, "null_centered")][i, j] = res[
                                "null_centered"
                            ]
                            vals_heavy[(metric_name, "z")][i, j] = res["z"]
                            vals_heavy[(metric_name, "ari")][i, j] = res["ari"]
                    idx += 1
    else:
        with torch.no_grad():
            for metric_name, fn_raw, fn_gated in metric_defs_all:
                # Skip already completed metrics (from checkpoint)
                if metric_name in completed_metrics:
                    logger.info(f"Skipping {metric_name} (already completed)")
                    continue

                vals_heavy[(metric_name, "raw")] = np.zeros((len(n_list), len(d_list)))
                if fn_gated is not None:
                    for q in quantiles:
                        vals_heavy[(metric_name, f"q{int(q*100)}")] = np.zeros(
                            (len(n_list), len(d_list))
                        )
                    vals_heavy[(metric_name, "null_centered")] = np.zeros(
                        (len(n_list), len(d_list))
                    )
                    vals_heavy[(metric_name, "z")] = np.zeros(
                        (len(n_list), len(d_list))
                    )
                    vals_heavy[(metric_name, "ari")] = np.zeros(
                        (len(n_list), len(d_list))
                    )
                for i, n in enumerate(
                    tqdm(n_list, desc=f"heavy n values {metric_name}")
                ):
                    for j, d in enumerate(
                        tqdm(d_list, desc="heavy d values", leave=False)
                    ):
                        scores_raw = []
                        scores_q = {q: [] for q in quantiles}
                        scores_variants = {"null_centered": [], "z": [], "ari": []}
                        for _ in range(trials):
                            dist = torch.distributions.StudentT(df=3)
                            X = dist.sample((n, d)).to(device)
                            Y = dist.sample((n, d)).to(device)
                            scores_raw.append(fn_raw(X, Y))
                            if fn_gated is not None:
                                if metric_name in multiq_helpers:
                                    res = multiq_helpers[metric_name](
                                        X,
                                        Y,
                                        quantiles,
                                        num_permutations=num_permutations,
                                        device=device,
                                    )
                                    for q in quantiles:
                                        scores_q[q].append(res["gated"][q])
                                    scores_variants["null_centered"].append(
                                        res["variants"].null_centered
                                    )
                                    scores_variants["z"].append(res["variants"].z)
                                    scores_variants["ari"].append(res["variants"].ari)
                                else:
                                    for q in quantiles:
                                        res = fn_gated(X, Y, q)
                                        scores_q[q].append(res.gated)
                        vals_heavy[(metric_name, "raw")][i, j] = np.mean(scores_raw)
                        if fn_gated is not None:
                            for q in quantiles:
                                vals_heavy[(metric_name, f"q{int(q*100)}")][i, j] = (
                                    np.mean(scores_q[q])
                                )
                            if metric_name in multiq_helpers:
                                vals_heavy[(metric_name, "null_centered")][i, j] = (
                                    np.mean(scores_variants["null_centered"])
                                )
                                vals_heavy[(metric_name, "z")][i, j] = np.mean(
                                    scores_variants["z"]
                                )
                                vals_heavy[(metric_name, "ari")][i, j] = np.mean(
                                    scores_variants["ari"]
                                )

                # Save checkpoint after each metric
                completed_metrics.add(metric_name)
                ckpt_data = {
                    "vals": vals_heavy,
                    "completed_metrics": list(completed_metrics),
                }
                np.save(checkpoint, ckpt_data)
                logger.info(
                    f"Checkpoint saved: {metric_name} completed ({len(completed_metrics)}/{len(metric_defs_all)})"
                )

    save_array(output, [n_list, d_list, vals_heavy])
    # Clean up checkpoint after successful completion
    if checkpoint.exists():
        checkpoint.unlink()
        logger.info("Checkpoint file removed (experiment completed)")
