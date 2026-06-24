"""Utilities for permutation null summaries and ARI-style normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .aggregation import gated_rescaled, tau_order_statistic
from .utils import EPS


@dataclass
class NullVariants:
    mean_null: float
    median_null: float
    std_null: float
    null_centered: float
    z: float
    ari: float


@dataclass
class CalibrationStats:
    """Statistics computed from null calibration."""

    tau: float
    gated: float
    pvalue: float
    tail_strength: float
    variants: NullVariants


def compute_null_variants(
    raw: float,
    null_samples: Sequence[float],
    *,
    min_score: float,
    max_score: float,
) -> NullVariants:
    """
    Compute null-centered, z-scored, and ARI-style adjusted variants.

    ARI-style adjustment mirrors Adjusted Rand Index semantics:
    scale by the distance to the expected null within the metric range.
    """
    if isinstance(null_samples, torch.Tensor):
        null_arr = null_samples.detach().cpu().numpy().astype(float, copy=False)
    else:
        null_arr = np.asarray(null_samples, dtype=float)
    mean_null = float(null_arr.mean())
    median_null = float(np.median(null_arr))
    std_null = float(null_arr.std()) if null_arr.size > 0 else 0.0

    null_centered = float(raw - mean_null)
    z = float((raw - mean_null) / (std_null + EPS))

    # ARI-style: (s - E[s]) / (s_max - E[s]) per the paper (one-sided)
    denom = max_score - mean_null
    ari = (raw - mean_null) / denom if denom > 0 else 0.0
    ari = float(max(min(ari, 1.0), -1.0))

    return NullVariants(
        mean_null=mean_null,
        median_null=median_null,
        std_null=std_null,
        null_centered=null_centered,
        z=z,
        ari=ari,
    )


def compute_calibration_stats(
    raw: float,
    null_samples: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    quantile: float,
    min_score: float = 0.0,
    max_score: float = 1.0,
) -> CalibrationStats:
    """Compute calibration statistics from raw score and null samples.

    This is the shared implementation used by torch-based metrics
    (via MetricResult).

    Args:
        raw: The observed raw metric value.
        null_samples: Null distribution samples from permutation testing.
        quantile: Quantile for computing tau threshold (e.g., 0.95).
        min_score: Minimum possible metric value.
        max_score: Maximum possible metric value.

    Returns:
        CalibrationStats containing tau, gated, pvalue, tail_strength, and variants.
    """
    # Convert to numpy array
    if isinstance(null_samples, torch.Tensor):
        null_arr = null_samples.detach().cpu().numpy().astype(float)
    else:
        null_arr = np.asarray(null_samples, dtype=float)

    # Exact permutation cutoff (shared helper; ceiling order statistic of null + obs).
    tau = tau_order_statistic(null_arr, quantile, obs=raw)

    # Compute gated score
    gated = gated_rescaled(raw, tau_alpha=tau, s_max=max_score)

    # Compute p-value (add-one estimator)
    num_exceed = int(np.sum(null_arr >= raw))
    num_total = len(null_arr)
    pvalue = float((num_exceed + 1.0) / (num_total + 1.0))

    # Compute tail strength
    alpha = 1.0 - float(quantile)
    if alpha <= 0:
        tail_strength = 0.0
    else:
        tail_strength = float(max(0.0, min(1.0, (alpha - pvalue) / alpha)))

    # Compute null variants (mean, median, std, z, ari, etc.)
    variants = compute_null_variants(
        raw, null_samples, min_score=min_score, max_score=max_score
    )

    return CalibrationStats(
        tau=tau,
        gated=gated,
        pvalue=pvalue,
        tail_strength=tail_strength,
        variants=variants,
    )
