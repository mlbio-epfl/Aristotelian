"""Statistical utilities for experiment analysis."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def mean_ci(values: Sequence[float], *, z: float = 1.96) -> tuple[float, float, float]:
    """Compute mean and confidence interval.

    Returns:
        Tuple of (mean, ci_low, ci_high).
    """
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, mean, mean
    se = float(arr.std(ddof=1)) / math.sqrt(len(arr))
    return mean, mean - z * se, mean + z * se


def binomial_ci(k: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score confidence interval for binomial proportion.

    Returns:
        Tuple of (ci_low, ci_high).
    """
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) / n) + (z**2) / (4.0 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def rankdata(x: np.ndarray) -> np.ndarray:
    """Compute ranks for array elements."""
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation."""
    rx = rankdata(x)
    ry = rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num_boot: int,
    seed_val: int,
) -> tuple[float, float, float]:
    """Bootstrap Spearman correlation with confidence interval.

    Returns:
        Tuple of (mean, ci_low, ci_high).
    """
    n = len(x)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed_val)
    vals = []
    for _ in range(num_boot):
        idx = rng.integers(0, n, n)
        vals.append(spearman(x[idx], y[idx]))
    vals = np.asarray(vals, dtype=float)
    return (
        float(vals.mean()),
        float(np.quantile(vals, 0.025)),
        float(np.quantile(vals, 0.975)),
    )
