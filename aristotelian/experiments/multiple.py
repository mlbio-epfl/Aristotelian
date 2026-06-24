"""Multiple comparison helpers (BH-FDR and max-statistic thresholds)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..metrics.aggregation import tau_order_statistic


def bh_fdr(pvalues: np.ndarray, *, alpha: float = 0.05) -> Tuple[float, np.ndarray]:
    """Benjamini-Hochberg FDR control. Returns (threshold, reject_mask)."""
    pvals = np.asarray(pvalues, dtype=float)
    if pvals.ndim != 1:
        raise ValueError("pvalues must be a 1D array")
    m = pvals.size
    if m == 0:
        return 0.0, np.zeros(0, dtype=bool)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    below = ranked <= thresh
    if not np.any(below):
        return 0.0, np.zeros_like(pvals, dtype=bool)
    k = np.max(np.where(below)[0])
    cutoff = ranked[k]
    reject = pvals <= cutoff
    return float(cutoff), reject


def max_stat_threshold(null_max: np.ndarray, *, alpha: float = 0.05) -> float:
    """Max-statistic threshold for FWER control using permutation maxima."""
    arr = np.asarray(null_max, dtype=float)
    if arr.size == 0:
        raise ValueError("null_max must be non-empty")
    return tau_order_statistic(arr, 1.0 - alpha)
