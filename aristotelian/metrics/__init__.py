"""Metric registry and implementations for representation similarity."""

from __future__ import annotations

# Import metric modules to trigger registration.
from . import cca as _cca  # noqa: F401
from . import cka as _cka  # noqa: F401
from . import knn as _knn  # noqa: F401
from . import other_metrics as _other  # noqa: F401
from . import rsa as _rsa  # noqa: F401
from .api import (
    gated_cca,
    gated_cka_linear,
    gated_cka_rbf,
    gated_knn,
    gated_procrustes,
    gated_pwcca,
    gated_rsa,
    gated_rv,
    gated_svcca,
    metric_definitions,
    prh_metric_spec,
    raw_cca,
    raw_cka_linear,
    raw_cka_rbf,
    raw_knn,
    raw_procrustes,
    raw_pwcca,
    raw_rsa,
    raw_rv,
    raw_svcca,
    sg_cca_multiq,
    sg_cka_kernel_multiq,
    sg_cka_linear_multiq,
    sg_knn_multiq,
    sg_pwcca_multiq,
    sg_rsa_multiq,
    sg_svcca_multiq,
)
from .base import BaseMetric, MetricConfig, MetricResult
from .registry import MetricRegistry, register_metric

__all__ = [
    "BaseMetric",
    "MetricConfig",
    "MetricResult",
    "MetricRegistry",
    "register_metric",
    "raw_cka_linear",
    "raw_cka_rbf",
    "raw_knn",
    "raw_rsa",
    "raw_cca",
    "raw_svcca",
    "raw_pwcca",
    "raw_rv",
    "raw_procrustes",
    "gated_cka_linear",
    "gated_cka_rbf",
    "gated_knn",
    "gated_rsa",
    "gated_cca",
    "gated_svcca",
    "gated_pwcca",
    "gated_rv",
    "gated_procrustes",
    "sg_cka_linear_multiq",
    "sg_cka_kernel_multiq",
    "sg_knn_multiq",
    "sg_rsa_multiq",
    "sg_cca_multiq",
    "sg_svcca_multiq",
    "sg_pwcca_multiq",
    "metric_definitions",
    "prh_metric_spec",
]
