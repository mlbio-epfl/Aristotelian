"""Reusable metrics and utilities for null-calibrated representation similarity.

The metrics system provides a single API for all metrics:

    from aristotelian import MetricRegistry, MetricConfig

    # Simple computation
    result = MetricRegistry.compute("mutual_knn", X, Y)
    print(result.raw)

    # With null calibration
    config = MetricConfig(calibrate=True, num_permutations=200)
    result = MetricRegistry.compute("cka_linear", X, Y, config)
    print(result.gated, result.pvalue)

    # List all metrics
    print(MetricRegistry.list_metrics())

Convenience functions are also available for common metrics:

    from aristotelian import sg_cka_linear, sg_knn, mutual_knn
    result = sg_cka_linear(X, Y, num_permutations=200)
"""

from __future__ import annotations

import numpy as np
import torch

# Unified API (the single source of truth)
from .metrics import BaseMetric, MetricConfig, MetricRegistry, MetricResult

# Internal utilities
from .metrics.utils import hsic_biased, hsic_unbiased

# CKA estimator variants (used by cka_comparison experiment)
from .metrics.estimators import (
    cka_biased,
    cka_debiased,
    cka_depcols,
    cka_estimators_all,
    cka_naive,
    cka_song,
    compare_cka_estimators,
)

# =============================================================================
# CKA convenience functions
# =============================================================================


def standard_cka(X: torch.Tensor, Y: torch.Tensor, mode: str = "linear") -> float:
    """Baseline CKA (linear or RBF kernel)."""
    metric = "cka_linear" if mode == "linear" else "cka_rbf"
    return MetricRegistry.compute_raw(metric, X, Y)


def sg_cka_linear(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    num_permutations: int = 200,
    quantile: float = 0.95,
    device: str = "cpu",
    perms: torch.Tensor | None = None,
) -> MetricResult:
    """Significance-gated linear CKA."""
    config = MetricConfig(
        calibrate=True,
        num_permutations=num_permutations,
        quantile=quantile,
        device=device,
        perms=perms,
    )
    return MetricRegistry.compute("cka_linear", X, Y, config)


def sg_cka_kernel(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    num_permutations: int = 200,
    quantile: float = 0.95,
    sigma: float | None = None,
    device: str = "cpu",
    perms: torch.Tensor | None = None,
) -> MetricResult:
    """Significance-gated RBF kernel CKA."""
    config = MetricConfig(
        calibrate=True,
        num_permutations=num_permutations,
        quantile=quantile,
        sigma=sigma,
        device=device,
        perms=perms,
    )
    return MetricRegistry.compute("cka_rbf", X, Y, config)


# =============================================================================
# kNN convenience functions
# =============================================================================


def mutual_knn_overlap(X: torch.Tensor, Y: torch.Tensor, k: int = 10) -> float:
    """Mutual-kNN overlap (PRH metric)."""
    return MetricRegistry.compute_raw("mutual_knn", X, Y, MetricConfig(topk=k))


def sg_knn(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    k: int = 10,
    num_permutations: int = 200,
    quantile: float = 0.95,
    device: str = "cpu",
    perms: torch.Tensor | None = None,
) -> MetricResult:
    """Significance-gated mutual-kNN overlap."""
    config = MetricConfig(
        topk=k,
        calibrate=True,
        num_permutations=num_permutations,
        quantile=quantile,
        device=device,
        perms=perms,
    )
    return MetricRegistry.compute("mutual_knn", X, Y, config)


# =============================================================================
# RSA convenience functions
# =============================================================================


def rsa_vector(X: torch.Tensor) -> torch.Tensor:
    """Return the upper-triangular (vectorized) RDM using Euclidean distances."""
    return torch.pdist(X, p=2)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman correlation between two arrays."""
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).correlation)


def sg_rsa(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    num_permutations: int = 200,
    quantile: float = 0.95,
    device: str = "cpu",
    batch_size: int | None = 32,
    pair_samples: int | None = None,
    perms: torch.Tensor | None = None,
) -> MetricResult:
    """Significance-gated RSA (Spearman correlation of RDM vectors)."""
    config = MetricConfig(
        calibrate=True,
        num_permutations=num_permutations,
        quantile=quantile,
        device=device,
        batch_size=batch_size,
        pair_samples=pair_samples,
        perms=perms,
    )
    return MetricRegistry.compute("rsa", X, Y, config)


# =============================================================================
# PRH metric convenience functions
# =============================================================================


def mutual_knn(feats_A: torch.Tensor, feats_B: torch.Tensor, *, topk: int) -> float:
    """Mutual kNN overlap."""
    return MetricRegistry.compute_raw(
        "mutual_knn", feats_A, feats_B, MetricConfig(topk=topk)
    )


def cycle_knn(feats_A: torch.Tensor, feats_B: torch.Tensor, *, topk: int) -> float:
    """Cycle-consistent kNN accuracy."""
    return MetricRegistry.compute_raw(
        "cycle_knn", feats_A, feats_B, MetricConfig(topk=topk)
    )


def cka(
    feats_A: torch.Tensor,
    feats_B: torch.Tensor,
    *,
    kernel_metric: str = "ip",
    rbf_sigma: float = 1.0,
    unbiased: bool = False,
) -> float:
    """Centered Kernel Alignment."""
    config = MetricConfig(kernel=kernel_metric, rbf_sigma=rbf_sigma, unbiased=unbiased)
    return MetricRegistry.compute_raw("cka", feats_A, feats_B, config)


def unbiased_cka(feats_A: torch.Tensor, feats_B: torch.Tensor) -> float:
    """Unbiased CKA estimator."""
    return MetricRegistry.compute_raw("cka_unbiased", feats_A, feats_B)


def svcca(feats_A: torch.Tensor, feats_B: torch.Tensor, *, cca_dim: int = 10) -> float:
    """Singular Vector CCA."""
    return MetricRegistry.compute_raw(
        "svcca", feats_A, feats_B, MetricConfig(cca_dim=cca_dim)
    )


def pwcca(feats_A: torch.Tensor, feats_B: torch.Tensor) -> float:
    """Projection Weighted CCA."""
    return MetricRegistry.compute_raw("pwcca", feats_A, feats_B)


def procrustes(feats_A: torch.Tensor, feats_B: torch.Tensor) -> float:
    """Procrustes alignment similarity."""
    return MetricRegistry.compute_raw("procrustes", feats_A, feats_B)


def cknna(
    feats_A: torch.Tensor,
    feats_B: torch.Tensor,
    *,
    topk: int | None = None,
    distance_agnostic: bool = False,
    unbiased: bool = True,
) -> float:
    """Centered Kernel Neighborhood Nearest Neighbor Alignment (CKNNA)."""
    config = MetricConfig(
        topk=topk, distance_agnostic=distance_agnostic, unbiased=unbiased
    )
    return MetricRegistry.compute_raw("cknna", feats_A, feats_B, config)


# =============================================================================
# Utility functions (used by layerwise_engine and tests)
# =============================================================================


def compute_nearest_neighbors(feats: torch.Tensor, topk: int = 1) -> torch.Tensor:
    """Compute k nearest neighbors using cosine similarity."""
    if feats.ndim != 2:
        raise ValueError(f"Expected feats to be 2D, got {feats.ndim}")
    from .metrics.knn import _compute_knn_indices

    return _compute_knn_indices(feats, topk)


def compute_knn_accuracy(knn: torch.Tensor) -> torch.Tensor:
    """Compute kNN accuracy (cycle consistency)."""
    n = knn.shape[0]
    acc = knn == torch.arange(n, device=knn.device).view(-1, 1, 1)
    acc = acc.float().view(n, -1).max(dim=1).values.mean()
    return acc


# =============================================================================
# Other module imports (non-metric functionality)
# =============================================================================

from .experiments.experiments import (
    Type1Summary,
    run_permutation_budget,
    run_type1_calibration,
)
from .experiments.multiple import bh_fdr, max_stat_threshold
from .metrics.aggregation import (
    AggregationResult,
    BootstrapSummary,
    SimpleMetric,
    agg_colmax_mean,
    agg_hungarian_match_mean,
    agg_max,
    agg_rowmax_mean,
    agg_topk_mean,
    bootstrap_statistic,
    compute_null_summary,
    compute_similarity_matrix,
    gated_rescaled,
    permutation_null_aggregated,
)
from .metrics.baselines import (
    BaselineSummary,
    NullTypeSummary,
    run_null_baselines,
    run_null_type_ablation,
)
from .metrics.calibration import NullVariants
from .metrics.cca import (
    cca_mean,
    pwcca_mean,
    rv_coefficient,
    sg_cca_mean,
    sg_pwcca_mean,
    sg_rv_coefficient,
    sg_svcca_mean,
    svcca_mean,
)
from .metrics.nulls import spectrum_matched_view
from .metrics.other_metrics import procrustes_score, sg_procrustes_score
from .prh.preprocess import remove_outliers
from .prh.prh_data import iter_prh_samples, load_prh_dataset
from .prh.prh_experiment import (
    compute_alignment_gated,
    prepare_features,
    prh_alignment_filename,
    prh_feature_filename,
    run_prh_experiment,
)
from .prh.prh_models import get_models, load_text_model, load_vision_model
from .prh.prh_pipeline import collect_text_activations, collect_vision_activations

__all__ = [
    # Unified API
    "MetricRegistry",
    "MetricConfig",
    "MetricResult",
    "BaseMetric",
    # CKA functions
    "standard_cka",
    "sg_cka_linear",
    "sg_cka_kernel",
    "cka",
    "unbiased_cka",
    # kNN functions
    "mutual_knn_overlap",
    "sg_knn",
    "mutual_knn",
    "cycle_knn",
    # RSA functions
    "rsa_vector",
    "spearman_corr",
    "sg_rsa",
    # CCA functions
    "svcca",
    "pwcca",
    "cca_mean",
    "svcca_mean",
    "pwcca_mean",
    "sg_cca_mean",
    "sg_svcca_mean",
    "sg_pwcca_mean",
    # Other metrics
    "procrustes",
    "cknna",
    "rv_coefficient",
    "procrustes_score",
    "sg_rv_coefficient",
    "sg_procrustes_score",
    # Utilities
    "compute_nearest_neighbors",
    "compute_knn_accuracy",
    "hsic_biased",
    "hsic_unbiased",
    # Aggregation
    "SimpleMetric",
    "AggregationResult",
    "BootstrapSummary",
    "compute_similarity_matrix",
    "agg_max",
    "agg_rowmax_mean",
    "agg_colmax_mean",
    "agg_topk_mean",
    "agg_hungarian_match_mean",
    "permutation_null_aggregated",
    "compute_null_summary",
    "gated_rescaled",
    "bootstrap_statistic",
    # Baselines
    "run_null_baselines",
    "BaselineSummary",
    "run_null_type_ablation",
    "NullTypeSummary",
    # Experiments
    "run_permutation_budget",
    "run_type1_calibration",
    "Type1Summary",
    # PRH
    "load_prh_dataset",
    "iter_prh_samples",
    "get_models",
    "load_text_model",
    "load_vision_model",
    "collect_text_activations",
    "collect_vision_activations",
    "prh_feature_filename",
    "prh_alignment_filename",
    "remove_outliers",
    "prepare_features",
    "compute_alignment_gated",
    "run_prh_experiment",
    # CKA estimator variants
    "cka_biased",
    "cka_debiased",
    "cka_depcols",
    "cka_estimators_all",
    "cka_naive",
    "cka_song",
    "compare_cka_estimators",
    # Other
    "spectrum_matched_view",
    "NullVariants",
    "bh_fdr",
    "max_stat_threshold",
]
