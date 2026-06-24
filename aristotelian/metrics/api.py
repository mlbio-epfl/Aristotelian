"""Convenience API for accessing similarity metrics.

This module provides a clean API for accessing metrics through the registry.
All functions delegate to aristotelian.metrics.registry.MetricRegistry.

Usage:
    from aristotelian.metrics.api import raw_cka_linear, gated_cka_linear

    # Raw computation
    score = raw_cka_linear(X, Y)

    # With null calibration
    result = gated_cka_linear(X, Y, q=0.95, num_permutations=200, device="cpu")
    print(result.raw, result.gated, result.pvalue)

For new code, prefer using the registry directly:
    from aristotelian.metrics import MetricRegistry, MetricConfig

    result = MetricRegistry.compute("cka_linear", X, Y)
    result = MetricRegistry.compute("cka_linear", X, Y,
                                     MetricConfig(calibrate=True, num_permutations=200))
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Sequence

import torch

from .base import MetricConfig, MetricResult
from .registry import MetricRegistry

# =============================================================================
# Raw metric functions - simple wrappers around MetricRegistry.compute_raw()
# =============================================================================


def raw_cka_linear(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute raw linear CKA."""
    return MetricRegistry.compute_raw("cka_linear", X, Y)


def raw_cka_rbf(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute raw RBF CKA."""
    return MetricRegistry.compute_raw("cka_rbf", X, Y)


def raw_knn(X: torch.Tensor, Y: torch.Tensor, *, k: int) -> float:
    """Compute raw mutual kNN overlap."""
    return MetricRegistry.compute_raw("mutual_knn", X, Y, MetricConfig(topk=k))


def raw_rsa(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute raw RSA (Spearman correlation of RDMs)."""
    return MetricRegistry.compute_raw("rsa", X, Y)


def raw_cca(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute raw CCA mean correlation."""
    return MetricRegistry.compute_raw("cca", X, Y)


def raw_svcca(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute raw SVCCA."""
    return MetricRegistry.compute_raw("svcca", X, Y)


def raw_pwcca(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute raw PWCCA."""
    return MetricRegistry.compute_raw("pwcca", X, Y)


def raw_rv(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute raw RV coefficient."""
    return MetricRegistry.compute_raw("rv_coefficient", X, Y)


def raw_procrustes(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute raw Procrustes score."""
    return MetricRegistry.compute_raw("procrustes", X, Y)


# =============================================================================
# Gated (null-calibrated) metric functions
# =============================================================================


def gated_cka_linear(
    X: torch.Tensor,
    Y: torch.Tensor,
    q: float,
    *,
    num_permutations: int,
    device: str,
    perms: torch.Tensor | None = None,
) -> MetricResult:
    """Significance-gated linear CKA."""
    config = MetricConfig(
        calibrate=True,
        num_permutations=num_permutations,
        quantile=q,
        device=device,
        perms=perms,
    )
    return MetricRegistry.compute("cka_linear", X, Y, config)


def gated_cka_rbf(
    X: torch.Tensor,
    Y: torch.Tensor,
    q: float,
    *,
    num_permutations: int,
    device: str,
    perms: torch.Tensor | None = None,
) -> MetricResult:
    """Significance-gated RBF CKA."""
    config = MetricConfig(
        calibrate=True,
        num_permutations=num_permutations,
        quantile=q,
        device=device,
        perms=perms,
    )
    return MetricRegistry.compute("cka_rbf", X, Y, config)


def gated_knn(
    X: torch.Tensor,
    Y: torch.Tensor,
    q: float,
    *,
    k: int,
    num_permutations: int,
    device: str,
    perms: torch.Tensor | None = None,
) -> MetricResult:
    """Significance-gated mutual kNN overlap."""
    config = MetricConfig(
        topk=k,
        calibrate=True,
        num_permutations=num_permutations,
        quantile=q,
        device=device,
        perms=perms,
    )
    return MetricRegistry.compute("mutual_knn", X, Y, config)


def gated_rsa(
    X: torch.Tensor,
    Y: torch.Tensor,
    q: float,
    *,
    num_permutations: int,
    device: str,
    batch_size: int | None = 32,
    perms: torch.Tensor | None = None,
) -> MetricResult:
    """Significance-gated RSA."""
    config = MetricConfig(
        calibrate=True,
        num_permutations=num_permutations,
        quantile=q,
        device=device,
        batch_size=batch_size,
        perms=perms,
    )
    return MetricRegistry.compute("rsa", X, Y, config)


def gated_cca(
    X: torch.Tensor,
    Y: torch.Tensor,
    q: float,
    *,
    num_permutations: int,
    device: str,
    perms: torch.Tensor | None = None,
) -> MetricResult:
    """Significance-gated CCA mean."""
    config = MetricConfig(
        calibrate=True,
        num_permutations=num_permutations,
        quantile=q,
        device=device,
        perms=perms,
    )
    return MetricRegistry.compute("cca", X, Y, config)


def gated_svcca(
    X: torch.Tensor,
    Y: torch.Tensor,
    q: float,
    *,
    num_permutations: int,
    device: str,
    perms: torch.Tensor | None = None,
    k: int = 10,
) -> MetricResult:
    """Significance-gated SVCCA."""
    config = MetricConfig(
        cca_dim=k,
        calibrate=True,
        num_permutations=num_permutations,
        quantile=q,
        device=device,
        perms=perms,
    )
    return MetricRegistry.compute("svcca", X, Y, config)


def gated_pwcca(
    X: torch.Tensor,
    Y: torch.Tensor,
    q: float,
    *,
    num_permutations: int,
    device: str,
    perms: torch.Tensor | None = None,
) -> MetricResult:
    """Significance-gated PWCCA."""
    config = MetricConfig(
        calibrate=True,
        num_permutations=num_permutations,
        quantile=q,
        device=device,
        perms=perms,
    )
    return MetricRegistry.compute("pwcca", X, Y, config)


def gated_rv(
    X: torch.Tensor,
    Y: torch.Tensor,
    q: float,
    *,
    num_permutations: int,
    device: str,
    perms: torch.Tensor | None = None,
) -> MetricResult:
    """Significance-gated RV coefficient."""
    config = MetricConfig(
        calibrate=True,
        num_permutations=num_permutations,
        quantile=q,
        device=device,
        perms=perms,
    )
    return MetricRegistry.compute("rv_coefficient", X, Y, config)


def gated_procrustes(
    X: torch.Tensor,
    Y: torch.Tensor,
    q: float,
    *,
    num_permutations: int,
    device: str,
    perms: torch.Tensor | None = None,
) -> MetricResult:
    """Significance-gated Procrustes score."""
    config = MetricConfig(
        calibrate=True,
        num_permutations=num_permutations,
        quantile=q,
        device=device,
        perms=perms,
    )
    return MetricRegistry.compute("procrustes", X, Y, config)


# =============================================================================
# Multi-quantile functions for experiments
# =============================================================================


def _multiq_compute(
    metric_name: str,
    X: torch.Tensor,
    Y: torch.Tensor,
    quantiles: Sequence[float],
    *,
    num_permutations: int,
    device: str,
    perms: torch.Tensor | None = None,
    **kwargs,
) -> dict[str, float | dict[float, float]]:
    """Compute metric with multiple quantiles.

    Returns a dict with 'raw', 'gated', 'tau', 'p_value', 'tail_strength' keys.
    The gated/tau/tail_strength values are dicts mapping quantile -> value.
    """
    # First compute to get null samples
    config = MetricConfig(
        calibrate=True,
        num_permutations=num_permutations,
        quantile=quantiles[0],  # Use first quantile for initial computation
        device=device,
        perms=perms,
        **kwargs,
    )
    metric = MetricRegistry.get(metric_name)
    result = MetricRegistry.compute(metric_name, X, Y, config)

    # Now compute for each quantile using the same null samples
    import numpy as np

    from .aggregation import gated_rescaled, tau_order_statistic
    from .calibration import compute_null_variants

    null_arr = np.asarray(result.null_samples, dtype=float)
    raw = result.raw
    p_value = result.pvalue

    gated = {}
    tau = {}
    tail_strength = {}
    for q in quantiles:
        tau_q = tau_order_statistic(null_arr, q, obs=raw)
        alpha = 1.0 - float(q)
        # tau-gate only (consistent with the single-quantile compute_calibration_stats);
        # the tau threshold already encodes the alpha-level rejection, so no separate p-gate.
        g_val = gated_rescaled(raw, tau_alpha=tau_q, s_max=metric.max_score)
        gated[q] = g_val
        tau[q] = tau_q
        if alpha <= 0:
            tail_strength[q] = 0.0
        else:
            tail_strength[q] = float(max(0.0, min(1.0, (alpha - p_value) / alpha)))

    variants = compute_null_variants(
        raw, result.null_samples, min_score=metric.min_score, max_score=metric.max_score
    )
    return {
        "raw": raw,
        "gated": gated,
        "tau": tau,
        "p_value": p_value,
        "tail_strength": tail_strength,
        "variants": variants,
    }


def sg_cka_linear_multiq(
    X: torch.Tensor,
    Y: torch.Tensor,
    quantiles: Sequence[float],
    *,
    num_permutations: int,
    device: str,
    perms: torch.Tensor | None = None,
) -> dict[str, float | dict[float, float]]:
    """Linear CKA with multiple quantiles."""
    return _multiq_compute(
        "cka_linear",
        X,
        Y,
        quantiles,
        num_permutations=num_permutations,
        device=device,
        perms=perms,
    )


def sg_cka_kernel_multiq(
    X: torch.Tensor,
    Y: torch.Tensor,
    quantiles: Sequence[float],
    *,
    num_permutations: int,
    device: str,
    perms: torch.Tensor | None = None,
) -> dict[str, float | dict[float, float]]:
    """RBF CKA with multiple quantiles."""
    return _multiq_compute(
        "cka_rbf",
        X,
        Y,
        quantiles,
        num_permutations=num_permutations,
        device=device,
        perms=perms,
    )


def sg_knn_multiq(
    X: torch.Tensor,
    Y: torch.Tensor,
    quantiles: Sequence[float],
    *,
    k: int,
    num_permutations: int,
    device: str,
    perms: torch.Tensor | None = None,
) -> dict[str, float | dict[float, float]]:
    """Mutual kNN with multiple quantiles."""
    return _multiq_compute(
        "mutual_knn",
        X,
        Y,
        quantiles,
        num_permutations=num_permutations,
        device=device,
        perms=perms,
        topk=k,
    )


def sg_rsa_multiq(
    X: torch.Tensor,
    Y: torch.Tensor,
    quantiles: Sequence[float],
    *,
    num_permutations: int,
    device: str,
    batch_size: int | None = 32,
    perms: torch.Tensor | None = None,
) -> dict[str, float | dict[float, float]]:
    """RSA with multiple quantiles."""
    return _multiq_compute(
        "rsa",
        X,
        Y,
        quantiles,
        num_permutations=num_permutations,
        device=device,
        perms=perms,
        batch_size=batch_size,
    )


def sg_cca_multiq(
    X: torch.Tensor,
    Y: torch.Tensor,
    quantiles: Sequence[float],
    *,
    num_permutations: int,
    device: str,
    perms: torch.Tensor | None = None,
) -> dict[str, float | dict[float, float]]:
    """CCA with multiple quantiles."""
    return _multiq_compute(
        "cca",
        X,
        Y,
        quantiles,
        num_permutations=num_permutations,
        device=device,
        perms=perms,
    )


def sg_svcca_multiq(
    X: torch.Tensor,
    Y: torch.Tensor,
    quantiles: Sequence[float],
    *,
    num_permutations: int,
    device: str,
    perms: torch.Tensor | None = None,
    k: int = 10,
) -> dict[str, float | dict[float, float]]:
    """SVCCA with multiple quantiles."""
    return _multiq_compute(
        "svcca",
        X,
        Y,
        quantiles,
        num_permutations=num_permutations,
        device=device,
        perms=perms,
        cca_dim=k,
    )


def sg_pwcca_multiq(
    X: torch.Tensor,
    Y: torch.Tensor,
    quantiles: Sequence[float],
    *,
    num_permutations: int,
    device: str,
    perms: torch.Tensor | None = None,
) -> dict[str, float | dict[float, float]]:
    """PWCCA with multiple quantiles."""
    return _multiq_compute(
        "pwcca",
        X,
        Y,
        quantiles,
        num_permutations=num_permutations,
        device=device,
        perms=perms,
    )


# =============================================================================
# Helper functions for experiments
# =============================================================================


def metric_definitions(
    *, num_permutations: int, device: str
) -> tuple[Sequence[tuple[str, Callable, Callable | None]], dict[str, Callable]]:
    """Get metric definitions for experiments.

    Returns:
        metric_defs: List of (name, raw_fn, gated_fn) tuples
        multiq_helpers: Dict mapping metric name to multi-quantile function
    """
    metric_defs = [
        (
            "CKA (lin)",
            raw_cka_linear,
            partial(gated_cka_linear, num_permutations=num_permutations, device=device),
        ),
        (
            "CKA (rbf)",
            raw_cka_rbf,
            partial(gated_cka_rbf, num_permutations=num_permutations, device=device),
        ),
        (
            "kNN",
            partial(raw_knn, k=10),
            partial(gated_knn, k=10, num_permutations=num_permutations, device=device),
        ),
        (
            "RSA",
            raw_rsa,
            partial(
                gated_rsa,
                num_permutations=num_permutations,
                device=device,
                batch_size=32,
            ),
        ),
        (
            "CCA",
            raw_cca,
            partial(gated_cca, num_permutations=num_permutations, device=device),
        ),
        (
            "SVCCA",
            raw_svcca,
            partial(gated_svcca, num_permutations=num_permutations, device=device),
        ),
        (
            "PWCCA",
            raw_pwcca,
            partial(gated_pwcca, num_permutations=num_permutations, device=device),
        ),
        (
            "RV",
            raw_rv,
            partial(gated_rv, num_permutations=num_permutations, device=device),
        ),
        (
            "Procrustes",
            raw_procrustes,
            partial(gated_procrustes, num_permutations=num_permutations, device=device),
        ),
    ]

    multiq_helpers = {
        "CKA (lin)": partial(
            sg_cka_linear_multiq, num_permutations=num_permutations, device=device
        ),
        "CKA (rbf)": partial(
            sg_cka_kernel_multiq, num_permutations=num_permutations, device=device
        ),
        "kNN": partial(
            sg_knn_multiq, k=10, num_permutations=num_permutations, device=device
        ),
        "RSA": partial(
            sg_rsa_multiq,
            batch_size=32,
            num_permutations=num_permutations,
            device=device,
        ),
        "CCA": partial(sg_cca_multiq, num_permutations=num_permutations, device=device),
        "SVCCA": partial(
            sg_svcca_multiq, num_permutations=num_permutations, device=device
        ),
        "PWCCA": partial(
            sg_pwcca_multiq, num_permutations=num_permutations, device=device
        ),
    }

    return metric_defs, multiq_helpers


def prh_metric_spec(metric: str, *, topk: int) -> tuple[Callable, float]:
    """Get a metric function and its max value for PRH experiments.

    Returns:
        metric_fn: Callable taking (X, Y) tensors and returning a float
        max_value: Maximum possible value for this metric
    """
    metric_map = {
        "cycle_knn": ("cycle_knn", 1.0),
        "knn": ("mutual_knn", 1.0),
        "mutual_knn": ("mutual_knn", 1.0),
        "cka": ("cka_linear", 1.0),
        "cka_lin": ("cka_linear", 1.0),
        "cka_rbf": ("cka_rbf", 1.0),
        "unbiased_cka": ("cka_unbiased", 1.0),
        "cknna": ("cknna", 1.0),
        "svcca": ("svcca", 1.0),
        "pwcca": ("pwcca", 1.0),
        "procrustes": ("procrustes", 1.0),
        "cca": ("cca", 1.0),
        "rv_coefficient": ("rv_coefficient", 1.0),
        "rsa": ("rsa", 1.0),
    }

    if metric not in metric_map:
        raise ValueError(f"Unsupported metric {metric}")

    unified_name, max_val = metric_map[metric]

    def metric_fn(X: torch.Tensor, Y: torch.Tensor) -> float:
        # CKNNA uses unbiased=True by default (matching legacy prh_metrics.cknna)
        unbiased = unified_name == "cknna"
        config = MetricConfig(
            topk=topk,
            cca_dim=min(10, X.shape[1], Y.shape[1]),
            unbiased=unbiased,
        )
        return MetricRegistry.compute_raw(unified_name, X, Y, config)

    return metric_fn, max_val
