"""Permutation budget sweep utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from loguru import logger

from ..metrics.baselines import BaselineSummary, _aggregate
from ..metrics.utils import resolve_sg_metric
from .common import _sample_pair


def run_permutation_budget(
    *,
    metric: str,
    n: int,
    d: int,
    budgets: Iterable[int],
    num_trials: int = 50,
    quantile: float = 0.95,
    null_type: str = "gaussian",
    k_knn: int = 10,
    device: str = "cpu",
    seed: int | None = None,
    rsa_batch_size: int | None = 32,
) -> dict[int, BaselineSummary]:
    """Sweep permutation budgets and summarize null calibration stability."""
    budgets_list = list(budgets)
    logger.info(f"Running permutation budget sweep: metric={metric}, n={n}, d={d}")
    logger.info(f"Budgets: {budgets_list}, trials={num_trials}")

    metric_fn = resolve_sg_metric(metric)
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        logger.debug(f"Random seed set: {seed}")

    out: dict[int, BaselineSummary] = {}
    for budget in budgets_list:
        logger.debug(f"Testing budget: {budget} permutations")
        results = []
        for _ in range(num_trials):
            X, Y = _sample_pair(n, d, null_type=null_type, device=device, rng=rng)
            kwargs = dict(num_permutations=budget, quantile=quantile, device=device)
            if metric == "sgknn":
                kwargs["k"] = k_knn
            if metric == "sgrsa":
                kwargs["batch_size"] = rsa_batch_size
            res = metric_fn(X, Y, **kwargs)
            results.append(res)
        out[int(budget)] = _aggregate(results)
        logger.debug(f"Budget {budget}: mean gated = {out[budget].mean:.4f}")

    logger.success(f"Permutation budget sweep complete: tested {len(out)} budgets")
    return out
