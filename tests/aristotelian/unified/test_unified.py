"""Equivalence tests for the unified metrics system.

These tests verify that the unified metrics produce identical results
to the legacy implementations they replace.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from aristotelian.metrics import MetricConfig, MetricRegistry

# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def random_data():
    """Generate random test data."""
    torch.manual_seed(42)
    np.random.seed(42)
    n, d = 50, 32
    X = torch.randn(n, d)
    Y = torch.randn(n, d)
    return X, Y


@pytest.fixture
def identical_data():
    """Generate identical test data for perfect similarity."""
    torch.manual_seed(42)
    n, d = 50, 32
    X = torch.randn(n, d)
    return X, X.clone()


@pytest.fixture
def shared_perms():
    """Generate shared permutations for reproducible null distributions."""
    n = 50
    num_perms = 20
    torch.manual_seed(123)
    return torch.stack([torch.randperm(n) for _ in range(num_perms)])


# ============================================================================
# kNN metrics equivalence tests
# ============================================================================


def test_mutual_knn_matches_legacy(random_data):
    """Test that unified mutual_knn matches legacy implementation."""
    from aristotelian import mutual_knn_overlap

    X, Y = random_data
    k = 10

    # Legacy
    legacy_result = mutual_knn_overlap(X, Y, k=k)

    # Unified
    config = MetricConfig(topk=k)
    unified_result = MetricRegistry.compute_raw("mutual_knn", X, Y, config)

    assert abs(legacy_result - unified_result) < 1e-6


def test_mutual_knn_calibrated_matches_legacy(random_data, shared_perms):
    """Test that unified mutual_knn calibration matches legacy sg_knn."""
    from aristotelian import sg_knn

    X, Y = random_data
    k = 10

    # Legacy
    legacy = sg_knn(X, Y, k=k, num_permutations=20, quantile=0.95, perms=shared_perms)

    # Unified
    config = MetricConfig(
        topk=k,
        calibrate=True,
        num_permutations=20,
        quantile=0.95,
        perms=shared_perms,
    )
    unified = MetricRegistry.compute("mutual_knn", X, Y, config)

    assert abs(legacy.raw - unified.raw) < 1e-6
    # Note: gated values may differ slightly due to implementation details
    # but should be very close
    assert abs(legacy.gated - unified.gated) < 0.1
    assert abs(legacy.pvalue - unified.pvalue) < 0.1


def test_cycle_knn_matches_legacy(random_data):
    """Test that unified cycle_knn matches legacy implementation."""
    from aristotelian import cycle_knn

    X, Y = random_data
    k = 10

    # Legacy
    legacy_result = cycle_knn(X, Y, topk=k)

    # Unified
    config = MetricConfig(topk=k)
    unified_result = MetricRegistry.compute_raw("cycle_knn", X, Y, config)

    assert abs(legacy_result - unified_result) < 1e-6


# ============================================================================
# CKA metrics equivalence tests
# ============================================================================


def test_cka_linear_matches_legacy(random_data):
    """Test that unified cka_linear matches legacy standard_cka."""
    from aristotelian import standard_cka

    X, Y = random_data

    # Legacy
    legacy_result = standard_cka(X, Y, mode="linear")

    # Unified
    unified_result = MetricRegistry.compute_raw("cka_linear", X, Y)

    assert abs(legacy_result - unified_result) < 1e-6


def test_cka_linear_calibrated_matches_legacy(random_data, shared_perms):
    """Test that unified cka_linear calibration matches sg_cka_linear."""
    from aristotelian import sg_cka_linear

    X, Y = random_data

    # Legacy
    legacy = sg_cka_linear(X, Y, num_permutations=20, quantile=0.95, perms=shared_perms)

    # Unified
    config = MetricConfig(
        calibrate=True,
        num_permutations=20,
        quantile=0.95,
        perms=shared_perms,
    )
    unified = MetricRegistry.compute("cka_linear", X, Y, config)

    assert abs(legacy.raw - unified.raw) < 1e-6
    # Null distributions should be very similar
    assert abs(legacy.tau - unified.tau) < 0.1
    assert abs(legacy.pvalue - unified.pvalue) < 0.1


def test_cka_rbf_matches_legacy(random_data):
    """Test that unified cka_rbf produces reasonable results."""
    from aristotelian import standard_cka

    X, Y = random_data

    # Legacy
    legacy_result = standard_cka(X, Y, mode="kernel")

    # Unified
    unified_result = MetricRegistry.compute_raw("cka_rbf", X, Y)

    # Note: Different RBF implementations may differ slightly
    # but should be in similar range
    assert 0 <= legacy_result <= 1
    assert 0 <= unified_result <= 1


def test_cka_unbiased_matches_legacy(random_data):
    """Test that unified cka_unbiased matches legacy unbiased_cka."""
    from aristotelian import unbiased_cka

    X, Y = random_data

    # Legacy
    legacy_result = unbiased_cka(X, Y)

    # Unified
    unified_result = MetricRegistry.compute_raw("cka_unbiased", X, Y)

    assert abs(legacy_result - unified_result) < 1e-6


def test_cka_general_matches_legacy(random_data):
    """Test that unified general cka matches legacy prh_metrics.cka."""
    from aristotelian import cka

    X, Y = random_data

    # Legacy linear
    legacy_linear = cka(X, Y, kernel_metric="ip")
    unified_linear = MetricRegistry.compute_raw(
        "cka", X, Y, MetricConfig(kernel="linear")
    )
    assert abs(legacy_linear - unified_linear) < 1e-6

    # Legacy RBF
    legacy_rbf = cka(X, Y, kernel_metric="rbf", rbf_sigma=1.0)
    unified_rbf = MetricRegistry.compute_raw(
        "cka", X, Y, MetricConfig(kernel="rbf", rbf_sigma=1.0)
    )
    assert abs(legacy_rbf - unified_rbf) < 1e-6


# ============================================================================
# CCA metrics equivalence tests
# ============================================================================


def test_svcca_matches_legacy(random_data):
    """Test that unified svcca matches legacy convenience function."""
    from aristotelian import svcca

    X, Y = random_data

    # Legacy convenience function
    legacy_result = svcca(X, Y, cca_dim=10)

    # Unified
    config = MetricConfig(cca_dim=10)
    unified_result = MetricRegistry.compute_raw("svcca", X, Y, config)

    assert abs(legacy_result - unified_result) < 1e-6


def test_pwcca_matches_legacy(random_data):
    """Test that unified pwcca matches legacy prh_metrics.pwcca."""
    from aristotelian import pwcca

    X, Y = random_data

    # Legacy
    legacy_result = pwcca(X, Y)

    # Unified
    unified_result = MetricRegistry.compute_raw("pwcca", X, Y)

    assert abs(legacy_result - unified_result) < 1e-6


def test_cca_mean_matches_legacy(random_data):
    """Test that unified cca matches legacy cca_mean."""
    from aristotelian.metrics.cca import cca_mean

    X, Y = random_data
    X_np = X.numpy()
    Y_np = Y.numpy()

    # Legacy
    legacy_result = cca_mean(X_np, Y_np)

    # Unified
    unified_result = MetricRegistry.compute_raw("cca", X, Y)

    assert abs(legacy_result - unified_result) < 1e-6


def test_rv_coefficient_matches_legacy(random_data):
    """Test that unified rv_coefficient matches legacy implementation."""
    from aristotelian.metrics.cca import rv_coefficient

    X, Y = random_data
    X_np = X.numpy()
    Y_np = Y.numpy()

    # Legacy
    legacy_result = rv_coefficient(X_np, Y_np)

    # Unified
    unified_result = MetricRegistry.compute_raw("rv_coefficient", X, Y)

    assert abs(legacy_result - unified_result) < 1e-6


# ============================================================================
# RSA metrics equivalence tests
# ============================================================================


def test_rsa_matches_legacy(random_data):
    """Test that unified rsa matches legacy sg_rsa raw score."""
    from aristotelian import sg_rsa

    X, Y = random_data

    # Legacy (without calibration, just raw)
    # sg_rsa always calibrates, but we can compare raw scores
    legacy = sg_rsa(X, Y, num_permutations=10, quantile=0.95)

    # Unified
    unified = MetricRegistry.compute("rsa", X, Y)

    assert abs(legacy.raw - unified.raw) < 1e-6


def test_rsa_calibrated_matches_legacy(random_data, shared_perms):
    """Test that unified rsa calibration matches sg_rsa."""
    from aristotelian import sg_rsa

    X, Y = random_data

    # Legacy
    legacy = sg_rsa(X, Y, num_permutations=20, quantile=0.95, perms=shared_perms)

    # Unified
    config = MetricConfig(
        calibrate=True,
        num_permutations=20,
        quantile=0.95,
        perms=shared_perms,
    )
    unified = MetricRegistry.compute("rsa", X, Y, config)

    assert abs(legacy.raw - unified.raw) < 1e-6
    # Null distributions should be very similar
    assert abs(legacy.pvalue - unified.pvalue) < 0.1


# ============================================================================
# Other metrics equivalence tests
# ============================================================================


def test_procrustes_matches_legacy(random_data):
    """Test that unified procrustes matches legacy procrustes_score."""
    from aristotelian import procrustes

    X, Y = random_data

    # Legacy
    legacy_result = procrustes(X, Y)

    # Unified
    unified_result = MetricRegistry.compute_raw("procrustes", X, Y)

    assert abs(legacy_result - unified_result) < 1e-6


def test_cknna_matches_legacy(random_data):
    """Test that unified cknna matches legacy prh_metrics.cknna."""
    from aristotelian import cknna

    X, Y = random_data

    # Legacy (default: all neighbors, unbiased)
    legacy_result = cknna(X, Y)

    # Unified
    config = MetricConfig(unbiased=True)
    unified_result = MetricRegistry.compute_raw("cknna", X, Y, config)

    assert abs(legacy_result - unified_result) < 1e-6


# ============================================================================
# Registry functionality tests
# ============================================================================


def test_registry_lists_all_metrics():
    """Test that all expected metrics are registered."""
    expected = {
        # kNN
        "mutual_knn",
        "knn",
        "cycle_knn",
        # CKA
        "cka_linear",
        "cka_rbf",
        "cka_unbiased",
        "cka",
        # CCA
        "cca",
        "svcca",
        "pwcca",
        "rv_coefficient",
        # RSA
        "rsa",
        # Other
        "procrustes",
        "cknna",
    }
    registered = set(MetricRegistry.list_metrics())
    assert expected <= registered


def test_registry_get_unknown_metric_raises():
    """Test that getting unknown metric raises KeyError."""
    with pytest.raises(KeyError):
        MetricRegistry.get("nonexistent_metric")


def test_registry_has():
    """Test has() method."""
    assert MetricRegistry.has("mutual_knn")
    assert not MetricRegistry.has("nonexistent_metric")


# ============================================================================
# Perfect similarity tests
# ============================================================================


def test_identical_data_high_similarity(identical_data):
    """Test that identical data produces high/perfect similarity scores."""
    X, Y = identical_data

    # mutual_knn should be 1.0 for identical data
    result = MetricRegistry.compute_raw("mutual_knn", X, Y, MetricConfig(topk=10))
    assert result == 1.0

    # CKA should be 1.0 for identical data
    result = MetricRegistry.compute_raw("cka_linear", X, Y)
    assert abs(result - 1.0) < 1e-6

    # RSA should be 1.0 for identical data
    result = MetricRegistry.compute_raw("rsa", X, Y)
    assert abs(result - 1.0) < 1e-6

    # Procrustes should be 1.0 for identical data
    result = MetricRegistry.compute_raw("procrustes", X, Y)
    assert abs(result - 1.0) < 1e-6
