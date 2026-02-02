"""Tests for knn alias and backward compatibility."""

import torch

from aristotelian.metrics import MetricConfig, MetricRegistry


def test_knn_alias_exists():
    """Test that knn is registered as an alias for mutual_knn."""
    assert MetricRegistry.has("knn")
    assert MetricRegistry.has("mutual_knn")


def test_knn_alias_returns_same_metric():
    """Test that knn and mutual_knn return the same metric instance."""
    knn_metric = MetricRegistry.get("knn")
    mutual_knn_metric = MetricRegistry.get("mutual_knn")

    # Should be the same instance (alias)
    assert knn_metric is mutual_knn_metric


def test_knn_alias_computes_same_value():
    """Test that knn and mutual_knn compute the same value."""
    torch.manual_seed(42)
    X = torch.randn(50, 32)
    Y = torch.randn(50, 32)

    config = MetricConfig(topk=5)

    knn_result = MetricRegistry.compute_raw("knn", X, Y, config)
    mutual_knn_result = MetricRegistry.compute_raw("mutual_knn", X, Y, config)

    assert knn_result == mutual_knn_result


def test_knn_alias_calibration_same():
    """Test that calibrated knn and mutual_knn give same results."""
    torch.manual_seed(42)
    X = torch.randn(50, 32)
    Y = torch.randn(50, 32)

    # Use same permutations for both
    perms = torch.stack([torch.randperm(50) for _ in range(20)])

    config = MetricConfig(
        topk=5,
        calibrate=True,
        num_permutations=20,
        quantile=0.95,
        perms=perms,
    )

    knn_result = MetricRegistry.compute("knn", X, Y, config)
    mutual_knn_result = MetricRegistry.compute("mutual_knn", X, Y, config)

    assert knn_result.raw == mutual_knn_result.raw
    assert knn_result.gated == mutual_knn_result.gated
    assert knn_result.pvalue == mutual_knn_result.pvalue
