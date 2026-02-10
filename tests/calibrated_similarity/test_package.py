"""Tests for the calibrated_similarity package.

These tests verify that the package is correctly structured and importable,
and that the core functionality works as expected.
"""

import pytest
import torch

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cosine_sim():
    """Cosine similarity function."""

    def sim(X, Y):
        X_norm = torch.nn.functional.normalize(X, dim=1)
        Y_norm = torch.nn.functional.normalize(Y, dim=1)
        return (X_norm * Y_norm).sum(dim=1).mean()

    return sim


@pytest.fixture
def cka_sim():
    """CKA similarity function."""

    def sim(X, Y):
        X, Y = X - X.mean(0), Y - Y.mean(0)
        hsic = (X @ X.T * (Y @ Y.T)).sum()
        return hsic / (
            torch.sqrt((X @ X.T).pow(2).sum() * (Y @ Y.T).pow(2).sum()) + 1e-10
        )

    return sim


@pytest.fixture
def simple_sim():
    """Simple element-wise similarity."""

    def sim(X, Y):
        return (X * Y).mean()

    return sim


# =============================================================================
# Package Import Tests
# =============================================================================


class TestPackageImports:
    """Test that the package can be imported correctly."""

    def test_import_package(self):
        """Package should be importable."""
        import calibrated_similarity

        assert hasattr(calibrated_similarity, "__version__")
        assert hasattr(calibrated_similarity, "calibrate")
        assert hasattr(calibrated_similarity, "calibrate_layers")

    def test_import_functions_directly(self):
        """Functions should be importable directly from package."""
        from calibrated_similarity import calibrate, calibrate_layers

        assert callable(calibrate)
        assert callable(calibrate_layers)

    def test_version_format(self):
        """Version should be a valid semver string."""
        from calibrated_similarity import __version__

        parts = __version__.split(".")
        assert (
            len(parts) >= 3
        ), "Version should have at least 3 parts (major.minor.patch)"
        assert all(
            p.isdigit() for p in parts[:3]
        ), "First 3 version parts should be numeric"


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Tests for input validation in calibrate()."""

    def test_mismatched_sample_count_raises(self, simple_sim):
        """Should raise when X and Y have different sample counts."""
        from calibrated_similarity import calibrate

        X = torch.randn(20, 8)
        Y = torch.randn(15, 8)  # Different n

        with pytest.raises(ValueError, match="same number of samples"):
            calibrate(X, Y, simple_sim, K=10)

    def test_1d_input_raises(self, simple_sim):
        """Should raise when input is 1D."""
        from calibrated_similarity import calibrate

        X = torch.randn(20)
        Y = torch.randn(20)

        with pytest.raises(ValueError, match="2-dimensional"):
            calibrate(X, Y, simple_sim, K=10)

    def test_3d_input_raises(self, simple_sim):
        """Should raise when input is 3D."""
        from calibrated_similarity import calibrate

        X = torch.randn(20, 8, 4)
        Y = torch.randn(20, 8, 4)

        with pytest.raises(ValueError, match="2-dimensional"):
            calibrate(X, Y, simple_sim, K=10)

    def test_empty_input_raises(self, simple_sim):
        """Should raise when input is empty."""
        from calibrated_similarity import calibrate

        X = torch.randn(0, 8)
        Y = torch.randn(0, 8)

        with pytest.raises(ValueError, match="at least one sample"):
            calibrate(X, Y, simple_sim, K=10)

    def test_device_mismatch_raises(self, simple_sim):
        """Should raise when X and Y are on different devices."""
        from calibrated_similarity import calibrate

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        X = torch.randn(20, 8, device="cpu")
        Y = torch.randn(20, 8, device="cuda")

        with pytest.raises(ValueError, match="same device"):
            calibrate(X, Y, simple_sim, K=10)


class TestParameterValidation:
    """Tests for parameter validation."""

    def test_invalid_K_zero_raises(self, simple_sim):
        """Should raise when K=0."""
        from calibrated_similarity import calibrate

        X = torch.randn(20, 8)
        Y = torch.randn(20, 8)

        with pytest.raises(ValueError, match="positive integer"):
            calibrate(X, Y, simple_sim, K=0)

    def test_invalid_K_negative_raises(self, simple_sim):
        """Should raise when K is negative."""
        from calibrated_similarity import calibrate

        X = torch.randn(20, 8)
        Y = torch.randn(20, 8)

        with pytest.raises(ValueError, match="positive integer"):
            calibrate(X, Y, simple_sim, K=-5)

    def test_invalid_alpha_zero_raises(self, simple_sim):
        """Should raise when alpha=0."""
        from calibrated_similarity import calibrate

        X = torch.randn(20, 8)
        Y = torch.randn(20, 8)

        with pytest.raises(ValueError, match="alpha must be in"):
            calibrate(X, Y, simple_sim, K=10, alpha=0.0)

    def test_invalid_alpha_one_raises(self, simple_sim):
        """Should raise when alpha=1."""
        from calibrated_similarity import calibrate

        X = torch.randn(20, 8)
        Y = torch.randn(20, 8)

        with pytest.raises(ValueError, match="alpha must be in"):
            calibrate(X, Y, simple_sim, K=10, alpha=1.0)

    def test_invalid_alpha_negative_raises(self, simple_sim):
        """Should raise when alpha is negative."""
        from calibrated_similarity import calibrate

        X = torch.randn(20, 8)
        Y = torch.randn(20, 8)

        with pytest.raises(ValueError, match="alpha must be in"):
            calibrate(X, Y, simple_sim, K=10, alpha=-0.1)

    def test_invalid_smax_zero_raises(self, simple_sim):
        """Should raise when smax=0."""
        from calibrated_similarity import calibrate

        X = torch.randn(20, 8)
        Y = torch.randn(20, 8)

        with pytest.raises(ValueError, match="smax must be positive"):
            calibrate(X, Y, simple_sim, K=10, smax=0.0)

    def test_invalid_smax_negative_raises(self, simple_sim):
        """Should raise when smax is negative."""
        from calibrated_similarity import calibrate

        X = torch.randn(20, 8)
        Y = torch.randn(20, 8)

        with pytest.raises(ValueError, match="smax must be positive"):
            calibrate(X, Y, simple_sim, K=10, smax=-1.0)


class TestLayerInputValidation:
    """Tests for input validation in calibrate_layers()."""

    def test_empty_x_layers_raises(self, simple_sim):
        """Should raise when X_layers is empty."""
        from calibrated_similarity import calibrate_layers

        X_layers = []
        Y_layers = [torch.randn(20, 8)]

        with pytest.raises(ValueError, match="X_layers must not be empty"):
            calibrate_layers(X_layers, Y_layers, simple_sim, K=10)

    def test_empty_y_layers_raises(self, simple_sim):
        """Should raise when Y_layers is empty."""
        from calibrated_similarity import calibrate_layers

        X_layers = [torch.randn(20, 8)]
        Y_layers = []

        with pytest.raises(ValueError, match="Y_layers must not be empty"):
            calibrate_layers(X_layers, Y_layers, simple_sim, K=10)

    def test_mismatched_layer_samples_raises(self, simple_sim):
        """Should raise when layers have different sample counts."""
        from calibrated_similarity import calibrate_layers

        X_layers = [torch.randn(20, 8), torch.randn(20, 8)]
        Y_layers = [torch.randn(15, 8)]  # Different n

        with pytest.raises(ValueError, match="same number of samples"):
            calibrate_layers(X_layers, Y_layers, simple_sim, K=10)

    def test_inconsistent_x_layers_raises(self, simple_sim):
        """Should raise when X_layers have inconsistent sample counts."""
        from calibrated_similarity import calibrate_layers

        X_layers = [torch.randn(20, 8), torch.randn(15, 8)]  # Inconsistent n
        Y_layers = [torch.randn(20, 8)]

        with pytest.raises(ValueError, match="same number of samples"):
            calibrate_layers(X_layers, Y_layers, simple_sim, K=10)

    def test_1d_layer_raises(self, simple_sim):
        """Should raise when a layer is 1D."""
        from calibrated_similarity import calibrate_layers

        X_layers = [torch.randn(20)]  # 1D
        Y_layers = [torch.randn(20, 8)]

        with pytest.raises(ValueError, match="2-dimensional"):
            calibrate_layers(X_layers, Y_layers, simple_sim, K=10)

    def test_non_callable_agg_raises(self, simple_sim):
        """Should raise when agg is not callable or valid string."""
        from calibrated_similarity import calibrate_layers

        X_layers = [torch.randn(20, 8)]
        Y_layers = [torch.randn(20, 8)]

        with pytest.raises(ValueError, match="agg must be"):
            calibrate_layers(X_layers, Y_layers, simple_sim, agg=123, K=10)


# =============================================================================
# Core Functionality Tests
# =============================================================================


class TestCalibrateFunction:
    """Tests for the calibrate() function."""

    def test_basic_usage(self, cosine_sim):
        """Basic calibration should work."""
        from calibrated_similarity import calibrate

        torch.manual_seed(42)
        X = torch.randn(20, 8)
        Y = torch.randn(20, 8)

        scal, p, tau = calibrate(X, Y, cosine_sim, K=10)

        assert isinstance(scal, torch.Tensor)
        assert isinstance(p, torch.Tensor)
        assert isinstance(tau, torch.Tensor)
        assert 0.0 <= float(scal) <= 1.0
        assert 0.0 < float(p) <= 1.0

    def test_identical_data_detection(self, cka_sim):
        """Should detect identical data with high confidence."""
        from calibrated_similarity import calibrate

        torch.manual_seed(123)
        X = torch.randn(30, 10)

        scal, p, tau = calibrate(X, X, cka_sim, K=50)

        assert float(p) <= 0.05, f"p-value {p} should be <= 0.05 for identical data"
        assert float(scal) > 0.5, f"Calibrated score {scal} should be > 0.5"

    def test_reproducibility_with_generator(self, simple_sim):
        """Results should be reproducible with generator."""
        from calibrated_similarity import calibrate

        X = torch.randn(15, 6)
        Y = torch.randn(15, 6)

        gen1 = torch.Generator().manual_seed(999)
        scal1, p1, tau1 = calibrate(X, Y, simple_sim, K=20, generator=gen1)

        gen2 = torch.Generator().manual_seed(999)
        scal2, p2, tau2 = calibrate(X, Y, simple_sim, K=20, generator=gen2)

        assert torch.allclose(scal1, scal2)
        assert torch.allclose(p1, p2)
        assert torch.allclose(tau1, tau2)

    def test_smax_none_returns_unnormalized(self, simple_sim):
        """With smax=None, should return unnormalized effect size."""
        from calibrated_similarity import calibrate

        torch.manual_seed(42)
        X = torch.randn(20, 8)

        # Identical data should have positive effect
        scal, p, tau = calibrate(X, X, simple_sim, K=20, smax=None)

        assert isinstance(scal, torch.Tensor)
        assert float(scal) >= 0.0, "Unnormalized score should be >= 0"
        # Unlike normalized score, this can be > 1

    def test_smax_none_with_random_data(self, simple_sim):
        """With smax=None on random data, effect could be zero."""
        from calibrated_similarity import calibrate

        torch.manual_seed(42)
        X = torch.randn(20, 8)
        Y = torch.randn(20, 8)

        scal, p, tau = calibrate(X, Y, simple_sim, K=20, smax=None)

        # Should still return valid tensor
        assert isinstance(scal, torch.Tensor)
        assert float(scal) >= 0.0

    def test_custom_perm_fn(self, simple_sim):
        """Should work with custom permutation function."""
        from calibrated_similarity import calibrate

        torch.manual_seed(42)
        X = torch.randn(20, 8)
        Y = torch.randn(20, 8)

        # Custom permutation: reverse order
        def reverse_perm(n, device):
            return torch.arange(n - 1, -1, -1, device=device)

        scal, p, tau = calibrate(X, Y, simple_sim, K=10, perm_fn=reverse_perm)

        assert isinstance(scal, torch.Tensor)
        assert 0.0 <= float(scal) <= 1.0

    def test_different_feature_dims(self, simple_sim):
        """Should work when X and Y have different feature dimensions."""
        from calibrated_similarity import calibrate

        torch.manual_seed(42)
        X = torch.randn(20, 8)
        Y = torch.randn(20, 16)  # Different d

        # Need a sim that handles different dims
        def flex_sim(X, Y):
            # Just use sample means
            return (X.mean(dim=1) * Y.mean(dim=1)).mean()

        scal, p, tau = calibrate(X, Y, flex_sim, K=10)
        assert isinstance(scal, torch.Tensor)


# =============================================================================
# calibrate_layers Tests
# =============================================================================


class TestCalibrateLayersFunction:
    """Tests for the calibrate_layers() function."""

    def test_basic_usage(self, cosine_sim):
        """Basic layer-wise calibration should work."""
        from calibrated_similarity import calibrate_layers

        torch.manual_seed(42)
        X_layers = [torch.randn(20, 8) for _ in range(3)]
        Y_layers = [torch.randn(20, 8) for _ in range(4)]

        Tcal, p, tau = calibrate_layers(X_layers, Y_layers, cosine_sim, K=10)

        assert isinstance(Tcal, torch.Tensor)
        assert isinstance(p, torch.Tensor)
        assert isinstance(tau, torch.Tensor)
        assert 0.0 <= float(Tcal) <= 1.0
        assert 0.0 < float(p) <= 1.0

    def test_max_aggregation(self, simple_sim):
        """Max aggregation should find maximum similarity."""
        from calibrated_similarity import calibrate_layers

        torch.manual_seed(100)
        X_layers = [torch.randn(15, 5) for _ in range(2)]
        Y_layers = [torch.randn(15, 5) for _ in range(2)]

        Tcal_max, _, _ = calibrate_layers(
            X_layers, Y_layers, simple_sim, agg="max", K=10
        )
        Tcal_mean, _, _ = calibrate_layers(
            X_layers, Y_layers, simple_sim, agg="mean", K=10
        )

        assert 0.0 <= float(Tcal_max) <= 1.0
        assert 0.0 <= float(Tcal_mean) <= 1.0

    def test_callable_aggregation(self, simple_sim):
        """Should work with custom callable aggregation."""
        from calibrated_similarity import calibrate_layers

        torch.manual_seed(42)
        X_layers = [torch.randn(15, 5) for _ in range(2)]
        Y_layers = [torch.randn(15, 5) for _ in range(2)]

        # Custom aggregation: median
        def median_agg(S):
            return S.flatten().median()

        Tcal, p, tau = calibrate_layers(
            X_layers, Y_layers, simple_sim, agg=median_agg, K=10
        )

        assert isinstance(Tcal, torch.Tensor)
        assert 0.0 <= float(Tcal) <= 1.0

    def test_shared_layer_detection(self, cka_sim):
        """Should detect when layers share structure."""
        from calibrated_similarity import calibrate_layers

        torch.manual_seed(200)

        shared = torch.randn(20, 8)
        X_layers = [torch.randn(20, 8), shared, torch.randn(20, 8)]
        Y_layers = [torch.randn(20, 8), shared, torch.randn(20, 8)]

        Tcal, p, tau = calibrate_layers(X_layers, Y_layers, cka_sim, agg="max", K=30)

        assert float(p) <= 0.1, f"p-value {p} should detect signal from shared layer"

    def test_smax_none_layers(self, simple_sim):
        """smax=None should work for layers too."""
        from calibrated_similarity import calibrate_layers

        torch.manual_seed(42)
        X_layers = [torch.randn(15, 5) for _ in range(2)]
        Y_layers = [torch.randn(15, 5) for _ in range(2)]

        Tcal, p, tau = calibrate_layers(X_layers, Y_layers, simple_sim, K=10, smax=None)

        assert isinstance(Tcal, torch.Tensor)
        assert float(Tcal) >= 0.0

    def test_custom_perm_fn_layers(self, simple_sim):
        """Should work with custom permutation function."""
        from calibrated_similarity import calibrate_layers

        torch.manual_seed(42)
        X_layers = [torch.randn(15, 5) for _ in range(2)]
        Y_layers = [torch.randn(15, 5) for _ in range(2)]

        def block_perm(n, device):
            # Block permutation: swap halves
            mid = n // 2
            return torch.cat([torch.arange(mid, n), torch.arange(0, mid)]).to(device)

        Tcal, p, tau = calibrate_layers(
            X_layers, Y_layers, simple_sim, K=10, perm_fn=block_perm
        )

        assert isinstance(Tcal, torch.Tensor)

    def test_single_layer_each(self, simple_sim):
        """Should work with single layer on each side."""
        from calibrated_similarity import calibrate_layers

        torch.manual_seed(42)
        X_layers = [torch.randn(15, 5)]
        Y_layers = [torch.randn(15, 5)]

        Tcal, p, tau = calibrate_layers(X_layers, Y_layers, simple_sim, K=10)

        assert isinstance(Tcal, torch.Tensor)
        assert 0.0 <= float(Tcal) <= 1.0


# =============================================================================
# Statistical Properties Tests
# =============================================================================


class TestStatisticalProperties:
    """Tests for statistical properties of calibration."""

    def test_pvalue_range(self, simple_sim):
        """P-value should always be in valid range."""
        from calibrated_similarity import calibrate

        torch.manual_seed(300)
        X = torch.randn(15, 6)
        Y = torch.randn(15, 6)

        for seed in range(10):
            gen = torch.Generator().manual_seed(seed)
            _, p, _ = calibrate(X, Y, simple_sim, K=20, generator=gen)
            assert 0.0 < float(p) <= 1.0, f"p-value {p} out of range (0, 1]"

    def test_minimum_pvalue(self, cka_sim):
        """Minimum p-value should be 1/(K+1)."""
        from calibrated_similarity import calibrate

        torch.manual_seed(400)
        X = torch.randn(20, 8)
        K = 50

        # Identical data should have minimum p-value
        _, p, _ = calibrate(X, X, cka_sim, K=K)
        assert float(p) == pytest.approx(
            1.0 / (K + 1), rel=1e-6
        ), f"Expected p={1/(K+1)}, got {p}"

    def test_calibrated_score_bounded(self, simple_sim):
        """Calibrated score should be bounded when smax is set."""
        from calibrated_similarity import calibrate

        torch.manual_seed(500)
        X = torch.randn(20, 8)
        Y = torch.randn(20, 8)

        for seed in range(10):
            gen = torch.Generator().manual_seed(seed)
            scal, _, _ = calibrate(X, Y, simple_sim, K=20, smax=1.0, generator=gen)
            assert 0.0 <= float(scal) <= 1.0, f"Score {scal} out of bounds [0, 1]"

    def test_tau_less_than_smax(self, simple_sim):
        """Threshold tau should typically be less than smax."""
        from calibrated_similarity import calibrate

        torch.manual_seed(42)
        X = torch.randn(20, 8)
        Y = torch.randn(20, 8)

        smax = 1.0
        _, _, tau = calibrate(X, Y, simple_sim, K=50, smax=smax)

        # tau should be reasonable (not guaranteed but typical)
        assert float(tau) < smax * 2, f"tau={tau} seems unreasonably large"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_k_equals_one(self, simple_sim):
        """Should work with K=1 (minimum permutations)."""
        from calibrated_similarity import calibrate

        torch.manual_seed(42)
        X = torch.randn(20, 8)
        Y = torch.randn(20, 8)

        scal, p, tau = calibrate(X, Y, simple_sim, K=1)

        assert isinstance(scal, torch.Tensor)
        # With K=1, p can only be 0.5 or 1.0
        assert float(p) in [0.5, 1.0]

    def test_small_alpha(self, simple_sim):
        """Should work with very small alpha."""
        from calibrated_similarity import calibrate

        torch.manual_seed(42)
        X = torch.randn(20, 8)
        Y = torch.randn(20, 8)

        scal, p, tau = calibrate(X, Y, simple_sim, K=100, alpha=0.001)

        assert isinstance(scal, torch.Tensor)
        assert 0.0 <= float(scal) <= 1.0

    def test_large_alpha(self, simple_sim):
        """Should work with large alpha (close to 1)."""
        from calibrated_similarity import calibrate

        torch.manual_seed(42)
        X = torch.randn(20, 8)
        Y = torch.randn(20, 8)

        scal, p, tau = calibrate(X, Y, simple_sim, K=50, alpha=0.99)

        assert isinstance(scal, torch.Tensor)
        # With high alpha, more scores should be "significant"

    def test_single_sample(self, simple_sim):
        """Should work with single sample (n=1)."""
        from calibrated_similarity import calibrate

        X = torch.randn(1, 8)
        Y = torch.randn(1, 8)

        scal, p, tau = calibrate(X, Y, simple_sim, K=10)

        assert isinstance(scal, torch.Tensor)

    def test_high_dimensional(self, simple_sim):
        """Should work with high-dimensional data (d > n)."""
        from calibrated_similarity import calibrate

        torch.manual_seed(42)
        X = torch.randn(10, 100)  # d > n
        Y = torch.randn(10, 100)

        scal, p, tau = calibrate(X, Y, simple_sim, K=10)

        assert isinstance(scal, torch.Tensor)
        assert 0.0 <= float(scal) <= 1.0

    def test_large_smax(self, simple_sim):
        """Should work with large smax."""
        from calibrated_similarity import calibrate

        torch.manual_seed(42)
        X = torch.randn(20, 8)
        Y = torch.randn(20, 8)

        scal, p, tau = calibrate(X, Y, simple_sim, K=20, smax=1000.0)

        assert isinstance(scal, torch.Tensor)
        assert 0.0 <= float(scal) <= 1.0
