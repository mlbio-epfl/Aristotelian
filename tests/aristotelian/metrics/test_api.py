"""Tests for aristotelian/metrics/api.py wrapper functions.

These tests ensure all API wrapper functions work correctly and return expected types.
"""

import pytest
import torch

from aristotelian.metrics.api import (  # Raw functions; Gated functions; Multi-quantile functions; Helper functions
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
from aristotelian.metrics.base import MetricResult

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    torch.manual_seed(42)
    X = torch.randn(30, 16)
    Y = torch.randn(30, 16)
    return X, Y


@pytest.fixture
def small_data():
    """Small data for faster CCA tests."""
    torch.manual_seed(42)
    X = torch.randn(20, 8)
    Y = torch.randn(20, 8)
    return X, Y


# =============================================================================
# Raw Function Tests
# =============================================================================


class TestRawFunctions:
    """Tests for raw_* wrapper functions."""

    def test_raw_cka_linear(self, sample_data):
        """raw_cka_linear should return a float in [0, 1]."""
        X, Y = sample_data
        score = raw_cka_linear(X, Y)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_raw_cka_rbf(self, sample_data):
        """raw_cka_rbf should return a float in [0, 1]."""
        X, Y = sample_data
        score = raw_cka_rbf(X, Y)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_raw_knn(self, sample_data):
        """raw_knn should return a float in [0, 1]."""
        X, Y = sample_data
        score = raw_knn(X, Y, k=5)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_raw_rsa(self, sample_data):
        """raw_rsa should return a float in [-1, 1]."""
        X, Y = sample_data
        score = raw_rsa(X, Y)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_raw_cca(self, small_data):
        """raw_cca should return a float in [0, 1]."""
        X, Y = small_data
        score = raw_cca(X, Y)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_raw_svcca(self, small_data):
        """raw_svcca should return a float in [0, 1]."""
        X, Y = small_data
        score = raw_svcca(X, Y)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_raw_pwcca(self, small_data):
        """raw_pwcca should return a float in [0, 1]."""
        X, Y = small_data
        score = raw_pwcca(X, Y)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_raw_rv(self, sample_data):
        """raw_rv should return a float in [0, 1]."""
        X, Y = sample_data
        score = raw_rv(X, Y)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_raw_procrustes(self, sample_data):
        """raw_procrustes should return a float."""
        X, Y = sample_data
        score = raw_procrustes(X, Y)
        assert isinstance(score, float)

    def test_raw_identical_data_high_score(self, sample_data):
        """Raw metrics on identical data should return high scores."""
        X, _ = sample_data

        assert raw_cka_linear(X, X) > 0.99
        assert raw_cka_rbf(X, X) > 0.99
        assert raw_knn(X, X, k=5) > 0.99
        assert raw_rsa(X, X) > 0.99
        assert raw_rv(X, X) > 0.99


# =============================================================================
# Gated Function Tests
# =============================================================================


class TestGatedFunctions:
    """Tests for gated_* wrapper functions."""

    def test_gated_cka_linear(self, sample_data):
        """gated_cka_linear should return MetricResult."""
        X, Y = sample_data
        result = gated_cka_linear(X, Y, q=0.95, num_permutations=20, device="cpu")

        assert isinstance(result, MetricResult)
        assert isinstance(result.raw, float)
        assert isinstance(result.gated, float)
        assert isinstance(result.pvalue, float)
        assert 0.0 <= result.gated <= 1.0
        assert 0.0 < result.pvalue <= 1.0

    def test_gated_cka_rbf(self, sample_data):
        """gated_cka_rbf should return MetricResult."""
        X, Y = sample_data
        result = gated_cka_rbf(X, Y, q=0.95, num_permutations=20, device="cpu")

        assert isinstance(result, MetricResult)
        assert 0.0 <= result.gated <= 1.0

    def test_gated_knn(self, sample_data):
        """gated_knn should return MetricResult."""
        X, Y = sample_data
        result = gated_knn(X, Y, q=0.95, k=5, num_permutations=20, device="cpu")

        assert isinstance(result, MetricResult)
        assert 0.0 <= result.gated <= 1.0

    def test_gated_rsa(self, sample_data):
        """gated_rsa should return MetricResult."""
        X, Y = sample_data
        result = gated_rsa(
            X, Y, q=0.95, num_permutations=20, device="cpu", batch_size=16
        )

        assert isinstance(result, MetricResult)
        assert 0.0 <= result.gated <= 1.0

    def test_gated_cca(self, small_data):
        """gated_cca should return MetricResult."""
        X, Y = small_data
        result = gated_cca(X, Y, q=0.95, num_permutations=20, device="cpu")

        assert isinstance(result, MetricResult)
        assert 0.0 <= result.gated <= 1.0

    def test_gated_svcca(self, small_data):
        """gated_svcca should return MetricResult."""
        X, Y = small_data
        result = gated_svcca(X, Y, q=0.95, num_permutations=20, device="cpu", k=5)

        assert isinstance(result, MetricResult)
        assert 0.0 <= result.gated <= 1.0

    def test_gated_pwcca(self, small_data):
        """gated_pwcca should return MetricResult."""
        X, Y = small_data
        result = gated_pwcca(
            X, Y, q=0.95, num_permutations=20, device="cpu"
        )

        assert isinstance(result, MetricResult)
        assert 0.0 <= result.gated <= 1.0

    def test_gated_rv(self, sample_data):
        """gated_rv should return MetricResult."""
        X, Y = sample_data
        result = gated_rv(X, Y, q=0.95, num_permutations=20, device="cpu")

        assert isinstance(result, MetricResult)
        assert 0.0 <= result.gated <= 1.0

    def test_gated_procrustes(self, sample_data):
        """gated_procrustes should return MetricResult."""
        X, Y = sample_data
        result = gated_procrustes(X, Y, q=0.95, num_permutations=20, device="cpu")

        assert isinstance(result, MetricResult)
        assert 0.0 <= result.gated <= 1.0

    def test_gated_with_perms(self, sample_data):
        """Gated functions should accept pre-generated permutations."""
        X, Y = sample_data
        n = X.shape[0]
        perms = torch.stack([torch.randperm(n) for _ in range(20)])

        result = gated_cka_linear(
            X, Y, q=0.95, num_permutations=20, device="cpu", perms=perms
        )

        assert isinstance(result, MetricResult)

    def test_gated_identical_data_low_pvalue(self, sample_data):
        """Gated metrics on identical data should have low p-value."""
        X, _ = sample_data

        result = gated_cka_linear(X, X, q=0.95, num_permutations=50, device="cpu")

        assert result.pvalue <= 0.05
        assert result.gated > 0.5


# =============================================================================
# Multi-Quantile Function Tests
# =============================================================================


class TestMultiqFunctions:
    """Tests for sg_*_multiq wrapper functions."""

    def test_sg_cka_linear_multiq(self, sample_data):
        """sg_cka_linear_multiq should return dict with expected keys."""
        X, Y = sample_data
        quantiles = [0.9, 0.95]

        result = sg_cka_linear_multiq(
            X, Y, quantiles, num_permutations=20, device="cpu"
        )

        assert isinstance(result, dict)
        assert "raw" in result
        assert "gated" in result
        assert "tau" in result
        assert "p_value" in result
        assert "tail_strength" in result
        assert "variants" in result

        assert isinstance(result["raw"], float)
        assert isinstance(result["gated"], dict)
        assert all(q in result["gated"] for q in quantiles)

    def test_sg_cka_kernel_multiq(self, sample_data):
        """sg_cka_kernel_multiq should return dict with expected keys."""
        X, Y = sample_data
        quantiles = [0.9, 0.95]

        result = sg_cka_kernel_multiq(
            X, Y, quantiles, num_permutations=20, device="cpu"
        )

        assert isinstance(result, dict)
        assert "raw" in result
        assert isinstance(result["gated"], dict)

    def test_sg_knn_multiq(self, sample_data):
        """sg_knn_multiq should return dict with expected keys."""
        X, Y = sample_data
        quantiles = [0.9, 0.95]

        result = sg_knn_multiq(X, Y, quantiles, k=5, num_permutations=20, device="cpu")

        assert isinstance(result, dict)
        assert "raw" in result

    def test_sg_rsa_multiq(self, sample_data):
        """sg_rsa_multiq should return dict with expected keys."""
        X, Y = sample_data
        quantiles = [0.9, 0.95]

        result = sg_rsa_multiq(
            X, Y, quantiles, num_permutations=20, device="cpu", batch_size=16
        )

        assert isinstance(result, dict)
        assert "raw" in result

    def test_sg_cca_multiq(self, small_data):
        """sg_cca_multiq should return dict with expected keys."""
        X, Y = small_data
        quantiles = [0.9, 0.95]

        result = sg_cca_multiq(
            X, Y, quantiles, num_permutations=20, device="cpu"
        )

        assert isinstance(result, dict)
        assert "raw" in result

    def test_sg_svcca_multiq(self, small_data):
        """sg_svcca_multiq should return dict with expected keys."""
        X, Y = small_data
        quantiles = [0.9, 0.95]

        result = sg_svcca_multiq(
            X, Y, quantiles, num_permutations=20, device="cpu", k=5
        )

        assert isinstance(result, dict)
        assert "raw" in result

    def test_sg_pwcca_multiq(self, small_data):
        """sg_pwcca_multiq should return dict with expected keys."""
        X, Y = small_data
        quantiles = [0.9, 0.95]

        result = sg_pwcca_multiq(
            X, Y, quantiles, num_permutations=20, device="cpu"
        )

        assert isinstance(result, dict)
        assert "raw" in result

    def test_multiq_gated_values_bounded(self, sample_data):
        """All gated values should be bounded in [0, 1]."""
        X, Y = sample_data
        quantiles = [0.9, 0.95, 0.99]

        result = sg_cka_linear_multiq(
            X, Y, quantiles, num_permutations=20, device="cpu"
        )

        for q in quantiles:
            assert 0.0 <= result["gated"][q] <= 1.0
            assert 0.0 <= result["tail_strength"][q] <= 1.0


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_metric_definitions_returns_correct_structure(self):
        """metric_definitions should return correct structure."""
        metric_defs, multiq_helpers = metric_definitions(
            num_permutations=10, device="cpu"
        )

        assert isinstance(metric_defs, list)
        assert len(metric_defs) == 9  # 9 metrics defined

        for name, raw_fn, gated_fn in metric_defs:
            assert isinstance(name, str)
            assert callable(raw_fn)
            assert gated_fn is None or callable(gated_fn)

        assert isinstance(multiq_helpers, dict)
        assert "CKA (lin)" in multiq_helpers
        assert "RSA" in multiq_helpers

    def test_metric_definitions_functions_work(self, sample_data):
        """Functions from metric_definitions should work."""
        X, Y = sample_data
        metric_defs, multiq_helpers = metric_definitions(
            num_permutations=10, device="cpu"
        )

        # Test first metric (CKA linear)
        name, raw_fn, gated_fn = metric_defs[0]

        raw_score = raw_fn(X, Y)
        assert isinstance(raw_score, float)

        gated_result = gated_fn(X, Y, q=0.95)
        assert isinstance(gated_result, MetricResult)

    def test_prh_metric_spec_valid_metrics(self, sample_data):
        """prh_metric_spec should return valid functions for known metrics."""
        X, Y = sample_data

        valid_metrics = [
            "cycle_knn",
            "knn",
            "mutual_knn",
            "cka",
            "cka_lin",
            "cka_rbf",
            "svcca",
            "pwcca",
            "procrustes",
        ]

        for metric_name in valid_metrics:
            metric_fn, max_val = prh_metric_spec(metric_name, topk=5)

            assert callable(metric_fn)
            assert isinstance(max_val, float)

            score = metric_fn(X, Y)
            assert isinstance(score, float)

    def test_prh_metric_spec_invalid_metric_raises(self):
        """prh_metric_spec should raise for unknown metrics."""
        with pytest.raises(ValueError, match="Unsupported metric"):
            prh_metric_spec("invalid_metric", topk=5)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_small_k_for_knn(self):
        """Verify kNN works with small k values."""
        torch.manual_seed(42)
        X = torch.randn(20, 8)
        Y = torch.randn(20, 8)

        score = raw_knn(X, Y, k=1)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_high_dimensional_data(self):
        """Metrics should work with high-dimensional data (d > n)."""
        torch.manual_seed(42)
        X = torch.randn(20, 100)
        Y = torch.randn(20, 100)

        assert isinstance(raw_cka_linear(X, Y), float)
        assert isinstance(raw_rv(X, Y), float)

    def test_different_feature_dims_cca(self):
        """CCA metrics should work with different feature dimensions."""
        torch.manual_seed(42)
        X = torch.randn(30, 16)
        Y = torch.randn(30, 24)

        # CCA should handle different dims
        score = raw_cca(X, Y)
        assert isinstance(score, float)

    def test_single_quantile_multiq(self, sample_data):
        """Multiq functions should work with single quantile."""
        X, Y = sample_data
        quantiles = [0.95]

        result = sg_cka_linear_multiq(
            X, Y, quantiles, num_permutations=10, device="cpu"
        )

        assert 0.95 in result["gated"]


# =============================================================================
# Regression tests for fixed bugs
# =============================================================================


class TestMultiqRegressions:
    """Regression tests for _multiq_compute bugs."""

    def test_multiq_tau_includes_observed_value(self, sample_data):
        """Multiq tau must include the observed value in the distribution,
        matching the behavior of compute_calibration_stats (regression:
        tau was computed from null_arr only, excluding raw)."""
        import numpy as np

        from aristotelian.metrics import MetricConfig, MetricRegistry

        X, Y = sample_data
        config = MetricConfig(calibrate=True, num_permutations=50, quantile=0.95)
        result = MetricRegistry.compute("cka_linear", X, Y, config)

        from aristotelian.metrics.aggregation import tau_order_statistic

        null_arr = np.asarray(result.null_samples, dtype=float)

        q = 0.95
        sg_cka_linear_multiq(X, Y, [q], num_permutations=50, device="cpu")

        # tau is the exact permutation cutoff (ceiling order statistic) including the
        # observed value, via the shared helper used by both single and multiq paths.
        assert result.tau == tau_order_statistic(null_arr, q, obs=result.raw)

    def test_multiq_rsa_ari_uses_correct_bounds(self):
        """Multiq RSA ARI must use min_score=-1.0 (regression:
        min_score=0.0 was hardcoded, making ARI wrong for RSA)."""
        import numpy as np

        torch.manual_seed(99)
        X = torch.randn(30, 10)
        Y = torch.randn(30, 10)

        result = sg_rsa_multiq(X, Y, [0.95], num_permutations=50, device="cpu")
        variants = result["variants"]

        # Recompute ARI with the paper's one-sided bound: (s - E[s]) / (s_max - E[s])
        raw = result["raw"]
        mean_null = variants.mean_null
        denom = 1.0 - mean_null  # max_score=1.0

        if denom > 0:
            expected_ari = (raw - mean_null) / denom
            expected_ari = max(min(expected_ari, 1.0), -1.0)
        else:
            expected_ari = 0.0

        assert np.isclose(
            variants.ari, expected_ari, atol=1e-6
        ), f"ARI={variants.ari}, expected={expected_ari}"
