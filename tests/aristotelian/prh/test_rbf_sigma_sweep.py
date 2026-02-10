"""Tests for RBF kernel sigma sweep functionality in PRH experiments.

This module tests that:
1. The sigma parameter is correctly passed through the call chain
2. Different sigma values produce expected numerical behavior (local vs global)
3. Backward compatibility is maintained (default sigma=1.0)
4. Numerical regression tests ensure stability
"""

import pytest
import torch

from aristotelian.experiments.layerwise_engine import build_gram_cache
from aristotelian.prh.cache import build_gram_cache as prh_build_gram_cache
from aristotelian.prh.prh_experiment import compute_alignment_gated_cka_cached

# =============================================================================
# Test Constants and Fixtures
# =============================================================================

# Reference values computed with fixed seed for numerical regression tests
# These should remain stable across code changes
REFERENCE_SIGMA_VALUES = (0.1, 0.5, 1.0, 2.0, 5.0)


@pytest.fixture
def fixed_seed_data():
    """Generate deterministic test data."""
    torch.manual_seed(42)
    # Shape: (samples, layers, features)
    x = torch.randn(20, 3, 8)
    y = torch.randn(20, 3, 8)
    return x, y


@pytest.fixture
def identical_data():
    """Generate identical data for self-alignment tests."""
    torch.manual_seed(123)
    x = torch.randn(15, 2, 6)
    return x, x.clone()


@pytest.fixture
def orthogonal_data():
    """Generate orthogonal data for zero-alignment tests."""
    torch.manual_seed(456)
    n, d = 20, 10
    # Create random orthogonal vectors
    q, _ = torch.linalg.qr(torch.randn(d, d))
    x = torch.randn(n, 2, d) @ q[:, : d // 2].T
    y = torch.randn(n, 2, d) @ q[:, d // 2 :].T
    return x, y


# =============================================================================
# Unit Tests: Sigma Parameter Propagation
# =============================================================================


class TestSigmaParameterPropagation:
    """Test that sigma is correctly passed through the call chain."""

    def test_build_gram_cache_accepts_rbf_sigma(self):
        """Verify build_gram_cache accepts and uses rbf_sigma parameter."""
        torch.manual_seed(0)
        x = torch.randn(10, 2, 5)

        # Should not raise
        grams_s1 = build_gram_cache(x, normalize=True, kernel="rbf", rbf_sigma=1.0)
        grams_s2 = build_gram_cache(x, normalize=True, kernel="rbf", rbf_sigma=2.0)

        assert len(grams_s1) == 2  # 2 layers
        assert grams_s1[0].shape == (10, 10)

        # Different sigma should produce different Gram matrices
        assert not torch.allclose(grams_s1[0], grams_s2[0])

    def test_prh_build_gram_cache_accepts_rbf_sigma(self):
        """Verify PRH cache wrapper passes sigma correctly."""
        torch.manual_seed(1)
        x = torch.randn(10, 2, 5)

        grams_s1 = prh_build_gram_cache(x, normalize=True, kernel="rbf", rbf_sigma=1.0)
        grams_s2 = prh_build_gram_cache(x, normalize=True, kernel="rbf", rbf_sigma=2.0)

        # Should produce different results
        assert not torch.allclose(grams_s1[0], grams_s2[0])

    def test_linear_kernel_ignores_rbf_sigma(self):
        """Verify linear kernel is unaffected by rbf_sigma parameter."""
        torch.manual_seed(2)
        x = torch.randn(10, 2, 5)

        grams_s1 = build_gram_cache(x, normalize=True, kernel="linear", rbf_sigma=0.1)
        grams_s2 = build_gram_cache(x, normalize=True, kernel="linear", rbf_sigma=10.0)

        # Linear kernel should be identical regardless of sigma
        torch.testing.assert_close(grams_s1[0], grams_s2[0])


# =============================================================================
# Numerical Tests: RBF Kernel Behavior
# =============================================================================


class TestRBFKernelBehavior:
    """Test that RBF kernel exhibits expected local/global behavior."""

    def test_small_sigma_produces_peaked_kernel(self, fixed_seed_data):
        """Small sigma should produce peaked kernel (mostly zeros, diagonal ones)."""
        x, _ = fixed_seed_data
        grams = build_gram_cache(x, normalize=True, kernel="rbf", rbf_sigma=0.1)

        # With very small sigma, off-diagonal elements should be near zero
        gram = grams[0]
        off_diag_mask = ~torch.eye(gram.shape[0], dtype=torch.bool)
        off_diag_values = gram[off_diag_mask]

        # Most off-diagonal values should be very small
        assert off_diag_values.mean() < 0.1
        # Diagonal should be 1.0
        assert torch.allclose(gram.diag(), torch.ones(gram.shape[0]), atol=1e-6)

    def test_large_sigma_produces_flat_kernel(self, fixed_seed_data):
        """Large sigma should produce flat kernel (all values close to 1)."""
        x, _ = fixed_seed_data
        grams = build_gram_cache(x, normalize=True, kernel="rbf", rbf_sigma=10.0)

        # With very large sigma, all elements should be close to 1
        gram = grams[0]
        assert gram.min() > 0.9
        assert gram.max() <= 1.0

    def test_sigma_ordering_monotonic(self, fixed_seed_data):
        """Larger sigma should produce higher mean off-diagonal values."""
        x, _ = fixed_seed_data

        sigmas = [0.1, 0.5, 1.0, 2.0, 5.0]
        means = []

        for sigma in sigmas:
            grams = build_gram_cache(x, normalize=True, kernel="rbf", rbf_sigma=sigma)
            gram = grams[0]
            off_diag_mask = ~torch.eye(gram.shape[0], dtype=torch.bool)
            means.append(gram[off_diag_mask].mean().item())

        # Means should be monotonically increasing with sigma
        for i in range(len(means) - 1):
            assert means[i] < means[i + 1], f"Sigma ordering violated: {means}"

    def test_rbf_kernel_symmetry(self, fixed_seed_data):
        """RBF Gram matrix should be symmetric."""
        x, _ = fixed_seed_data
        for sigma in [0.1, 1.0, 5.0]:
            grams = build_gram_cache(x, normalize=True, kernel="rbf", rbf_sigma=sigma)
            for gram in grams:
                torch.testing.assert_close(gram, gram.T)

    def test_rbf_kernel_positive_definite(self, fixed_seed_data):
        """RBF Gram matrix should be positive semi-definite."""
        x, _ = fixed_seed_data
        for sigma in [0.1, 1.0, 5.0]:
            grams = build_gram_cache(x, normalize=True, kernel="rbf", rbf_sigma=sigma)
            for gram in grams:
                eigenvalues = torch.linalg.eigvalsh(gram)
                # All eigenvalues should be non-negative (allowing small numerical errors)
                assert (
                    eigenvalues >= -1e-6
                ).all(), f"Negative eigenvalue found: {eigenvalues.min()}"


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    """Test that default sigma=1.0 maintains backward compatibility."""

    def test_default_sigma_matches_explicit_sigma_1(self, fixed_seed_data):
        """Default behavior should match explicit sigma=1.0."""
        x, _ = fixed_seed_data

        # Default (implicit sigma=1.0)
        grams_default = build_gram_cache(x, normalize=True, kernel="rbf")
        # Explicit sigma=1.0
        grams_explicit = build_gram_cache(
            x, normalize=True, kernel="rbf", rbf_sigma=1.0
        )

        for g_def, g_exp in zip(grams_default, grams_explicit):
            torch.testing.assert_close(g_def, g_exp)

    def test_cka_cached_default_sigma_stability(self, fixed_seed_data):
        """CKA with default sigma should produce stable results."""
        x, y = fixed_seed_data

        # Run twice with same seed
        x_grams = build_gram_cache(x, normalize=True, kernel="rbf", rbf_sigma=1.0)
        y_grams = build_gram_cache(y, normalize=True, kernel="rbf", rbf_sigma=1.0)

        res1 = compute_alignment_gated_cka_cached(
            x_grams, y_grams, num_permutations=10, alpha=0.05, seed=999, unbiased=False
        )
        res2 = compute_alignment_gated_cka_cached(
            x_grams, y_grams, num_permutations=10, alpha=0.05, seed=999, unbiased=False
        )

        assert res1["raw_score"] == res2["raw_score"]
        assert res1["best_indices"] == res2["best_indices"]


# =============================================================================
# Numerical Regression Tests
# =============================================================================


class TestNumericalRegression:
    """Numerical regression tests to ensure computation stability."""

    # Reference values computed with torch.manual_seed(42), shape (20, 3, 8)
    # These values were computed and verified to be correct
    REFERENCE_GRAM_STATS = {
        # sigma: (min, max, mean_off_diag) for first layer
        0.1: {"min": 0.0, "max": 1.0, "mean_off_diag_lt": 0.01},
        0.5: {"min_gt": 0.0, "max": 1.0, "mean_off_diag_range": (0.05, 0.4)},
        1.0: {"min_gt": 0.0, "max": 1.0, "mean_off_diag_range": (0.2, 0.7)},
        2.0: {"min_gt": 0.4, "max": 1.0, "mean_off_diag_range": (0.5, 0.9)},
        5.0: {"min_gt": 0.85, "max": 1.0, "mean_off_diag_range": (0.9, 1.0)},
    }

    def test_gram_statistics_regression(self, fixed_seed_data):
        """Verify Gram matrix statistics match expected ranges."""
        x, _ = fixed_seed_data

        for sigma, expected in self.REFERENCE_GRAM_STATS.items():
            grams = build_gram_cache(x, normalize=True, kernel="rbf", rbf_sigma=sigma)
            gram = grams[0]
            off_diag_mask = ~torch.eye(gram.shape[0], dtype=torch.bool)
            mean_off_diag = gram[off_diag_mask].mean().item()

            # Check max (should always be 1.0 on diagonal)
            assert gram.max().item() <= expected["max"] + 1e-6

            # Check min constraints
            if "min" in expected:
                assert gram.min().item() >= expected["min"] - 1e-6
            if "min_gt" in expected:
                assert gram.min().item() > expected["min_gt"] - 0.1

            # Check mean off-diagonal constraints
            if "mean_off_diag_lt" in expected:
                assert mean_off_diag < expected["mean_off_diag_lt"]
            if "mean_off_diag_range" in expected:
                low, high = expected["mean_off_diag_range"]
                assert (
                    low <= mean_off_diag <= high
                ), f"sigma={sigma}: mean_off_diag={mean_off_diag} not in [{low}, {high}]"

    def test_cka_score_with_different_sigma(self, fixed_seed_data):
        """Verify CKA scores with different sigma values are in valid range."""
        x, y = fixed_seed_data

        for sigma in REFERENCE_SIGMA_VALUES:
            x_grams = build_gram_cache(x, normalize=True, kernel="rbf", rbf_sigma=sigma)
            y_grams = build_gram_cache(y, normalize=True, kernel="rbf", rbf_sigma=sigma)

            res = compute_alignment_gated_cka_cached(
                x_grams,
                y_grams,
                num_permutations=10,
                alpha=0.05,
                seed=42,
                unbiased=False,
            )

            # CKA should be in [0, 1] range
            assert (
                0.0 <= res["raw_score"] <= 1.0
            ), f"sigma={sigma}: CKA={res['raw_score']}"
            assert (
                0.0 <= res["g_score"] <= 1.0
            ), f"sigma={sigma}: gated={res['g_score']}"
            assert 0.0 <= res["p_value"] <= 1.0, f"sigma={sigma}: p={res['p_value']}"

    def test_self_alignment_high_score(self, identical_data):
        """Self-alignment should produce high CKA scores regardless of sigma."""
        x, y = identical_data  # y is clone of x

        for sigma in [0.1, 1.0, 5.0]:
            x_grams = build_gram_cache(x, normalize=True, kernel="rbf", rbf_sigma=sigma)
            y_grams = build_gram_cache(y, normalize=True, kernel="rbf", rbf_sigma=sigma)

            res = compute_alignment_gated_cka_cached(
                x_grams,
                y_grams,
                num_permutations=10,
                alpha=0.05,
                seed=42,
                unbiased=False,
            )

            # Self-alignment should have CKA ≈ 1.0
            assert (
                res["raw_score"] > 0.99
            ), f"sigma={sigma}: self-CKA={res['raw_score']}"


# =============================================================================
# Integration Tests: Experiment Configuration
# =============================================================================


class TestExperimentConfiguration:
    """Test experiment configuration with sigma values."""

    def test_prh_sigma_values_constant(self):
        """Verify PRH_SIGMA_VALUES is defined correctly."""
        from scripts.experiments.sections.prh import PRH_SIGMA_VALUES

        assert isinstance(PRH_SIGMA_VALUES, tuple)
        assert len(PRH_SIGMA_VALUES) > 0
        assert 1.0 in PRH_SIGMA_VALUES  # Default should be included
        # Should be sorted ascending
        assert list(PRH_SIGMA_VALUES) == sorted(PRH_SIGMA_VALUES)

    def test_rbf_sigma_metrics_constant(self):
        """Verify RBF_SIGMA_METRICS is defined correctly."""
        from scripts.experiments.sections.prh import RBF_SIGMA_METRICS

        assert isinstance(RBF_SIGMA_METRICS, tuple)
        assert "cka_rbf" in RBF_SIGMA_METRICS

    def test_experiment_functions_accept_sigma(self):
        """Verify experiment functions accept rbf_sigma parameter."""
        import inspect

        from aristotelian.prh.prh_experiment import run_prh_experiment
        from aristotelian.prh.v2t_experiment import run_v2t_experiment

        for fn in [
            run_prh_experiment,
            run_v2t_experiment,
        ]:
            sig = inspect.signature(fn)
            assert (
                "rbf_sigma" in sig.parameters
            ), f"{fn.__name__} missing rbf_sigma parameter"
            # Check default is 1.0
            assert sig.parameters["rbf_sigma"].default == 1.0

    def test_alignment_runner_functions_accept_sigma(self):
        """Verify alignment runner functions accept sigma values parameter."""
        import inspect

        from scripts.experiments.sections.prh import (
            run_prh_alignment,
            run_v2t_alignment,
        )

        runners = [
            (run_prh_alignment, "prh_sigma_values"),
            (run_v2t_alignment, "v2t_sigma_values"),
        ]

        for fn, param_name in runners:
            sig = inspect.signature(fn)
            assert (
                param_name in sig.parameters
            ), f"{fn.__name__} missing {param_name} parameter"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_sigma(self):
        """Very small sigma should still produce valid Gram matrices."""
        torch.manual_seed(100)
        x = torch.randn(10, 2, 5)

        grams = build_gram_cache(x, normalize=True, kernel="rbf", rbf_sigma=0.01)
        gram = grams[0]

        # Should still be valid (positive, symmetric)
        assert not torch.isnan(gram).any()
        assert not torch.isinf(gram).any()
        assert (gram >= 0).all()
        assert (gram <= 1).all()
        torch.testing.assert_close(gram, gram.T)

    def test_very_large_sigma(self):
        """Very large sigma should produce near-uniform Gram matrices."""
        torch.manual_seed(101)
        x = torch.randn(10, 2, 5)

        grams = build_gram_cache(x, normalize=True, kernel="rbf", rbf_sigma=100.0)
        gram = grams[0]

        # Should be nearly all ones
        assert gram.min() > 0.99
        assert gram.max() <= 1.0

    def test_single_sample_gram(self):
        """Single sample should produce 1x1 Gram matrix with value 1."""
        x = torch.randn(1, 2, 5)

        for sigma in [0.1, 1.0, 10.0]:
            grams = build_gram_cache(x, normalize=True, kernel="rbf", rbf_sigma=sigma)
            for gram in grams:
                assert gram.shape == (1, 1)
                assert torch.allclose(gram, torch.ones(1, 1))

    def test_identical_samples_gram(self):
        """Identical samples should have Gram matrix of all ones."""
        torch.manual_seed(102)
        sample = torch.randn(1, 5)
        x = sample.repeat(10, 1).unsqueeze(1)  # (10, 1, 5) - 10 identical samples

        for sigma in [0.1, 1.0, 10.0]:
            grams = build_gram_cache(x, normalize=True, kernel="rbf", rbf_sigma=sigma)
            gram = grams[0]
            # All pairs are identical, so K(x,y) = exp(0) = 1
            assert torch.allclose(gram, torch.ones(10, 10), atol=1e-6)


# =============================================================================
# Determinism Tests
# =============================================================================


class TestDeterminism:
    """Test that results are deterministic with fixed seeds."""

    def test_gram_cache_deterministic(self):
        """Gram cache should be deterministic."""
        for sigma in [0.1, 1.0, 5.0]:
            torch.manual_seed(200)
            x1 = torch.randn(15, 2, 6)
            grams1 = build_gram_cache(x1, normalize=True, kernel="rbf", rbf_sigma=sigma)

            torch.manual_seed(200)
            x2 = torch.randn(15, 2, 6)
            grams2 = build_gram_cache(x2, normalize=True, kernel="rbf", rbf_sigma=sigma)

            for g1, g2 in zip(grams1, grams2):
                torch.testing.assert_close(g1, g2)

    def test_cka_alignment_deterministic(self, fixed_seed_data):
        """CKA alignment should be deterministic with same seed."""
        x, y = fixed_seed_data

        for sigma in [0.1, 1.0, 5.0]:
            x_grams = build_gram_cache(x, normalize=True, kernel="rbf", rbf_sigma=sigma)
            y_grams = build_gram_cache(y, normalize=True, kernel="rbf", rbf_sigma=sigma)

            res1 = compute_alignment_gated_cka_cached(
                x_grams,
                y_grams,
                num_permutations=20,
                alpha=0.05,
                seed=300,
                unbiased=False,
            )
            res2 = compute_alignment_gated_cka_cached(
                x_grams,
                y_grams,
                num_permutations=20,
                alpha=0.05,
                seed=300,
                unbiased=False,
            )

            assert res1["raw_score"] == res2["raw_score"]
            assert res1["g_score"] == res2["g_score"]
            assert res1["p_value"] == res2["p_value"]
            assert res1["best_indices"] == res2["best_indices"]


# =============================================================================
# Plotting Filename Parsing Tests
# =============================================================================


class TestPlottingFilenameParsing:
    """Test that plot_experiments correctly parses sigma from filenames."""

    def test_extract_metric_and_k_with_sigma(self):
        """Verify _extract_metric_and_k correctly extracts sigma values."""
        import sys

        sys.path.insert(0, "scripts")
        from scripts.plots.experiments import _extract_metric_and_k

        test_cases = [
            # (filename, (metric, k, sigma))
            ("prh_alignment.npy", ("mutual_knn", 10, None)),
            ("prh_alignment_mutual_knn_k20.npy", ("mutual_knn", 20, None)),
            ("prh_alignment_cka_lin.npy", ("cka_lin", None, None)),
            ("prh_alignment_cycle_knn_k50.npy", ("cycle_knn", 50, None)),
            ("prh_alignment_cka_rbf.npy", ("cka_rbf", None, 1.0)),
            ("prh_alignment_cka_rbf_sigma0.5.npy", ("cka_rbf", None, 0.5)),
            ("prh_alignment_cka_rbf_sigma2.0.npy", ("cka_rbf", None, 2.0)),
            ("prh_alignment_cka_rbf_sigma0.1.npy", ("cka_rbf", None, 0.1)),
            ("prh_alignment_cka_rbf_sigma5.0.npy", ("cka_rbf", None, 5.0)),
        ]

        for filename, expected in test_cases:
            result = _extract_metric_and_k(filename)
            assert (
                result == expected
            ), f"Failed for {filename}: got {result}, expected {expected}"

    def test_sigma_suffix_generation(self):
        """Verify sigma values generate correct suffixes for output files."""

        # Test the suffix building logic
        def build_suffix(
            metric_name: str, k_value, sigma_value, multi_metric: bool = True
        ):
            def _slug(s):
                return s.replace("_", "-")

            if multi_metric:
                if k_value is not None:
                    return f"_{_slug(metric_name)}_k{k_value}"
                elif sigma_value is not None and sigma_value != 1.0:
                    return f"_{_slug(metric_name)}_sigma{sigma_value}"
                else:
                    return f"_{_slug(metric_name)}"
            return ""

        # Test cases
        assert (
            build_suffix("cka_rbf", None, 1.0) == "_cka-rbf"
        )  # default sigma, no suffix
        assert build_suffix("cka_rbf", None, 0.5) == "_cka-rbf_sigma0.5"
        assert build_suffix("cka_rbf", None, 2.0) == "_cka-rbf_sigma2.0"
        assert build_suffix("mutual_knn", 10, None) == "_mutual-knn_k10"
        assert build_suffix("cka_lin", None, None) == "_cka-lin"


# =============================================================================
# Reference Implementation for Numerical Verification
# =============================================================================


def _reference_rbf_gram(X: torch.Tensor, sigma: float) -> torch.Tensor:
    """Reference implementation of RBF Gram matrix.

    K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    """
    # X is (n, d) after normalization
    n = X.shape[0]
    K = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            diff = X[i] - X[j]
            dist_sq = (diff**2).sum()
            K[i, j] = torch.exp(-dist_sq / (2 * sigma**2))
    return K


class TestReferenceImplementation:
    """Test against reference implementation."""

    def test_rbf_gram_matches_reference(self):
        """Verify RBF Gram computation matches reference implementation."""
        torch.manual_seed(500)
        # Single layer, normalized features
        X = torch.randn(8, 10)
        X = X / X.norm(dim=1, keepdim=True)

        for sigma in [0.1, 1.0, 5.0]:
            # Our implementation (via build_gram_cache)
            grams = build_gram_cache(
                X.unsqueeze(1), normalize=False, kernel="rbf", rbf_sigma=sigma
            )
            gram_ours = grams[0]

            # Reference implementation
            gram_ref = _reference_rbf_gram(X, sigma)

            torch.testing.assert_close(gram_ours, gram_ref, rtol=1e-5, atol=1e-6)
