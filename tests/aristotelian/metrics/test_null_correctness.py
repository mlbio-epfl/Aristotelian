"""Tests that every metric's _compute_null_distribution matches naive recomputation.

These tests guard against:
1. The cache-shallow-copy bug (dataclasses.replace shares cache dict)
2. Mathematical errors in optimized null implementations
3. Null scores being constant (all identical = stale cache)

For each metric with supports_calibration=True, we use fixed permutations
and compare the optimized null to a naive per-permutation _compute_raw call
with a fresh config (empty cache) each time.
"""

import numpy as np
import pytest
import torch

import aristotelian.metrics  # noqa: F401
from aristotelian.metrics.base import MetricConfig
from aristotelian.metrics.registry import MetricRegistry

N = 24
D = 8
NUM_PERMS = 5
SEED = 42
DEVICE = "cpu"


@pytest.fixture(scope="module")
def test_data():
    torch.manual_seed(SEED)
    X = torch.randn(N, D)
    Y = torch.randn(N, D)
    perms = torch.stack([torch.randperm(N) for _ in range(NUM_PERMS)])
    return X, Y, perms


def _naive_null(metric, X, Y, perms, extra_config):
    """Compute null the naive way: fresh config per permutation."""
    scores = []
    for perm in perms:
        cfg = MetricConfig(device=DEVICE, calibrate=False, **extra_config)
        scores.append(metric._compute_raw(X, Y[perm], cfg))
    return scores


def _optimized_null(metric, X, Y, perms, extra_config):
    """Compute null via the metric's own _compute_null_distribution."""
    cfg = MetricConfig(
        device=DEVICE,
        calibrate=True,
        num_permutations=perms.shape[0],
        perms=perms,
        **extra_config,
    )
    return metric._compute_null_distribution(X, Y, cfg)


# Each entry: (metric_name, extra_config, atol)
CALIBRATED_METRICS = [
    ("cca", {}, 1e-4),
    ("svcca", {"cca_dim": 5}, 1e-4),
    ("pwcca", {}, 2e-5),
    ("rv_coefficient", {}, 1e-5),
    ("cka_linear", {}, 1e-5),
    ("cka_rbf", {}, 1e-5),
    ("cka_unbiased", {}, 1e-5),
    ("procrustes", {}, 1e-4),
    ("rsa", {"batch_size": 32}, 1e-5),
    ("mutual_knn", {"topk": 5}, 0.0),
    ("cycle_knn", {"topk": 5}, 0.0),
    ("cknna", {"topk": 5}, 1e-5),
]


@pytest.mark.parametrize(
    "metric_name,extra_config,atol",
    CALIBRATED_METRICS,
    ids=[m[0] for m in CALIBRATED_METRICS],
)
def test_null_matches_naive(test_data, metric_name, extra_config, atol):
    """Optimized null must match naive per-permutation recomputation."""
    X, Y, perms = test_data
    metric = MetricRegistry.get(metric_name)

    opt = np.array(_optimized_null(metric, X, Y, perms, extra_config))
    nai = np.array(_naive_null(metric, X, Y, perms, extra_config))

    np.testing.assert_allclose(
        opt,
        nai,
        atol=atol,
        err_msg=f"{metric_name}: optimized null != naive null",
    )


@pytest.mark.parametrize(
    "metric_name,extra_config,atol",
    CALIBRATED_METRICS,
    ids=[m[0] for m in CALIBRATED_METRICS],
)
def test_null_scores_vary(test_data, metric_name, extra_config, atol):
    """Null scores must not all be identical (catches stale-cache bug)."""
    X, Y, perms = test_data
    metric = MetricRegistry.get(metric_name)

    scores = np.array(_optimized_null(metric, X, Y, perms, extra_config))
    assert np.std(scores) > 1e-8, (
        f"{metric_name}: all null scores identical ({scores[0]:.6f}) — "
        "likely stale cache or broken permutation"
    )


@pytest.mark.parametrize(
    "metric_name,extra_config,atol",
    CALIBRATED_METRICS,
    ids=[m[0] for m in CALIBRATED_METRICS],
)
def test_null_has_custom_implementation(test_data, metric_name, extra_config, atol):
    """Every calibrated metric should have a custom _compute_null_distribution.

    The base class default uses _compute_raw with shared cache (shallow copy),
    which is almost always wrong for metrics that use caching. This test ensures
    no metric accidentally falls back to the buggy default.
    """
    metric = MetricRegistry.get(metric_name)
    has_custom = "_compute_null_distribution" in type(metric).__dict__
    assert has_custom, (
        f"{metric_name}: uses base class _compute_null_distribution. "
        "This is likely buggy if the metric uses config.cache. "
        "Add a custom implementation."
    )


def test_noncalibrated_metrics_are_safe():
    """Metrics with supports_calibration=False don't need null distributions."""
    for name in MetricRegistry.list_metrics():
        metric = MetricRegistry.get(name)
        if not metric.supports_calibration:
            # These should be safe — they'll never have _compute_null_distribution called
            continue
        # If calibrated AND uses cache, must have custom null
        if metric.supports_caching and metric.cache_keys:
            has_custom = "_compute_null_distribution" in type(metric).__dict__
            assert has_custom, (
                f"{name}: supports_calibration=True AND uses cache "
                f"(keys={metric.cache_keys}) but has no custom "
                "_compute_null_distribution — this WILL produce wrong null scores"
            )
