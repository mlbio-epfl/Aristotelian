"""Shared helpers for significance-gated extra metrics."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .aggregation import gated_rescaled
from .base import MetricResult
from .calibration import compute_calibration_stats, compute_null_variants


def _validate_perms(perms: np.ndarray, n: int) -> np.ndarray:
    perms = np.asarray(perms, dtype=int)
    if perms.ndim != 2 or perms.shape[1] != n:
        raise ValueError("perms must have shape (B, n)")
    return perms


def _sg_metric(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    num_permutations: int,
    quantile: float,
    perms: np.ndarray | None,
    min_score: float,
    max_score: float,
) -> MetricResult:
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    n = X.shape[0]
    raw = float(metric_fn(X, Y))

    if perms is None:
        if num_permutations <= 0:
            raise ValueError("num_permutations must be positive")
        perms = np.stack([np.random.permutation(n) for _ in range(num_permutations)])
    else:
        perms = _validate_perms(perms, n)
    null_samples = [float(metric_fn(X, Y[p])) for p in perms]

    # Use shared calibration stats computation
    stats = compute_calibration_stats(
        raw, null_samples, quantile=quantile, min_score=min_score, max_score=max_score
    )

    return MetricResult(
        raw=raw,
        gated=stats.gated,
        tau=stats.tau,
        pvalue=stats.pvalue,
        tail_strength=stats.tail_strength,
        null_samples=null_samples,
        mean_null=stats.variants.mean_null,
        median_null=stats.variants.median_null,
        std_null=stats.variants.std_null,
        null_centered=stats.variants.null_centered,
        z=stats.variants.z,
        ari=stats.variants.ari,
    )


def _sg_metric_multiq(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    num_permutations: int,
    quantiles: Sequence[float],
    perms: np.ndarray | None,
    min_score: float,
    max_score: float,
) -> dict:
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    for q in quantiles:
        if not 0.0 <= float(q) <= 1.0:
            raise ValueError("quantiles must be in [0, 1]")
    n = X.shape[0]
    raw = float(metric_fn(X, Y))

    if perms is None:
        if num_permutations <= 0:
            raise ValueError("num_permutations must be positive")
        perms = np.stack([np.random.permutation(n) for _ in range(num_permutations)])
    else:
        perms = _validate_perms(perms, n)

    null_samples = [float(metric_fn(X, Y[p])) for p in perms]
    pvalue = (sum(s >= raw for s in null_samples) + 1.0) / (len(null_samples) + 1.0)

    # Include observed value in the permutation distribution for tau
    full_samples = null_samples + [raw]

    gated = {}
    tau = {}
    tail_strength = {}
    for q in quantiles:
        alpha = 1.0 - float(q)
        tau_q = float(np.quantile(full_samples, q))
        gated[q] = gated_rescaled(raw, tau_alpha=tau_q, s_max=1.0)
        tau[q] = tau_q
        if alpha <= 0:
            tail_strength[q] = 0.0
        else:
            tail_strength[q] = float(max(0.0, min(1.0, (alpha - pvalue) / alpha)))

    variants = compute_null_variants(
        raw, null_samples, min_score=min_score, max_score=max_score
    )
    return {
        "raw": raw,
        "gated": gated,
        "tau": tau,
        "p_value": pvalue,
        "tail_strength": tail_strength,
        "null_samples": null_samples,
        "variants": variants,
    }
