"""Unit tests for experiment data generators."""

import pytest
import torch

from scripts.experiments.generators.common import (
    center_norm_features,
    knn_indicator,
    projection_matrix,
    projection_matrix_no_rng,
)
from scripts.experiments.generators.gen1 import gen1_mixture_invariance
from scripts.experiments.generators.gen2 import (
    gen2_geometry,
    gen2_geometry_state,
    gen2_linear,
    gen2_linear_from_state,
    gen2_linear_shared_state,
    gen2_linear_state,
    gen2_local,
    gen2_local_state,
    make_gen2_linear_signal,
)
from scripts.experiments.generators.layerwise import (
    make_random_layers,
    make_random_layers_with_rng,
    make_signal_layers,
)
from scripts.experiments.generators.low_rank import (
    make_low_rank_signal,
    make_low_rank_signal_unitvar,
    make_pure_noise,
)
from scripts.experiments.generators.noise import (
    DEFAULT_NOISE_TYPE,
    NOISE_TYPES,
    sample_noise,
    strength_from_snr,
)


class TestNoiseGenerators:
    def test_noise_types_contains_defaults(self):
        assert "gaussian" in NOISE_TYPES
        assert DEFAULT_NOISE_TYPE in NOISE_TYPES

    def test_strength_from_snr_increases(self):
        # Higher signal relative to noise = higher strength
        strength1 = strength_from_snr(1.0, 1.0)  # equal
        strength2 = strength_from_snr(2.0, 1.0)  # more signal
        assert strength2 > strength1

    def test_strength_from_snr_range(self):
        # Output should be in [0, 1]
        assert 0.0 <= strength_from_snr(1.0, 1.0) <= 1.0
        assert 0.0 <= strength_from_snr(0.0, 1.0) <= 1.0
        assert 0.0 <= strength_from_snr(10.0, 0.0) <= 1.0

    def test_sample_noise_gaussian_shape(self):
        noise = sample_noise(100, 64, 1.0, "gaussian", device="cpu")
        assert noise.shape == (100, 64)

    def test_sample_noise_student_t_shape(self):
        noise = sample_noise(100, 64, 1.0, "student_t", device="cpu")
        assert noise.shape == (100, 64)

    def test_sample_noise_laplace_shape(self):
        noise = sample_noise(100, 64, 1.0, "laplace", device="cpu")
        assert noise.shape == (100, 64)

    def test_sample_noise_mixture_shape(self):
        noise = sample_noise(100, 64, 1.0, "mixture", device="cpu")
        assert noise.shape == (100, 64)

    @pytest.mark.parametrize("noise_type", NOISE_TYPES)
    def test_sample_noise_all_types_run(self, noise_type):
        noise = sample_noise(50, 32, 1.0, noise_type, device="cpu")
        assert noise.shape == (50, 32)

    def test_sample_noise_zero_std(self):
        # Zero noise_std should return zeros
        noise = sample_noise(10, 5, 0.0, "gaussian", device="cpu")
        assert torch.allclose(noise, torch.zeros(10, 5))


class TestLowRankGenerators:
    def test_make_pure_noise_shape(self):
        X, Y = make_pure_noise(100, 64, device="cpu")
        assert X.shape == (100, 64)
        assert Y.shape == (100, 64)

    def test_make_pure_noise_different(self):
        torch.manual_seed(42)
        X, Y = make_pure_noise(100, 64, device="cpu")
        # X and Y should be independent
        assert not torch.allclose(X, Y)

    def test_make_low_rank_signal_shape(self):
        torch.manual_seed(42)
        X, Y = make_low_rank_signal(100, 64, rank=5, device="cpu")
        assert X.shape == (100, 64)
        assert Y.shape == (100, 64)

    def test_make_low_rank_signal_unitvar_shape(self):
        torch.manual_seed(42)
        X, Y = make_low_rank_signal_unitvar(
            100, 64, rank=5, signal_strength=1.0, noise_std=0.5, device="cpu"
        )
        assert X.shape == (100, 64)
        assert Y.shape == (100, 64)


class TestCommonUtilities:
    def test_projection_matrix_no_rng_orthonormal(self):
        torch.manual_seed(42)
        P = projection_matrix_no_rng(32, 64, "cpu")
        # Check orthonormal: P @ P.T should be identity
        PPt = P @ P.T
        assert torch.allclose(PPt, torch.eye(32), atol=1e-5)

    def test_projection_matrix_shape(self):
        rng = torch.Generator(device="cpu").manual_seed(42)
        P = projection_matrix(rng, 32, 64)
        assert P.shape == (32, 64)

    def test_center_norm_features(self):
        X = torch.randn(100, 64)
        X_norm = center_norm_features(X)
        # Should be centered
        assert torch.allclose(X_norm.mean(dim=0), torch.zeros(64), atol=1e-5)
        # Should have approximately unit Frobenius norm
        assert torch.norm(X_norm, p="fro").item() == pytest.approx(1.0, abs=1e-5)

    def test_knn_indicator_shape(self):
        # Create fake knn indices
        n = 100
        k = 10
        knn_idx = torch.randint(0, n, (n, k))
        indicator = knn_indicator(knn_idx, n)
        assert indicator.shape == (n, n)
        # Diagonal should be False
        assert not indicator.diag().any()


class TestGen2Linear:
    def test_gen2_linear_shared_state_shapes(self):
        torch.manual_seed(42)
        state = gen2_linear_shared_state(n=100, d=64, r=5, device="cpu")
        assert state["Z"].shape == (100, 5)
        assert state["A"].shape == (5, 64)
        assert state["B"].shape == (5, 64)

    def test_gen2_linear_from_state_shape(self):
        torch.manual_seed(42)
        state = gen2_linear_shared_state(n=100, d=64, r=5, device="cpu")
        X, Y = gen2_linear_from_state(state, strength=0.5, noise_type="gaussian")
        assert X.shape == (100, 64)
        assert Y.shape == (100, 64)

    def test_gen2_linear_state_shapes(self):
        rng = torch.Generator(device="cpu").manual_seed(42)
        state = gen2_linear_state(n=100, d=64, r=5, rng=rng)
        assert state["Z"].shape == (100, 5)
        assert state["A"].shape == (5, 64)
        assert state["B"].shape == (5, 64)
        assert state["Nx"].shape == (100, 64)
        assert state["Ny"].shape == (100, 64)

    def test_gen2_linear_shape(self):
        rng = torch.Generator(device="cpu").manual_seed(42)
        X, Y = gen2_linear(n=100, d=64, r=5, strength=0.5, rng=rng)
        assert X.shape == (100, 64)
        assert Y.shape == (100, 64)

    def test_gen2_linear_deterministic(self):
        rng1 = torch.Generator(device="cpu").manual_seed(42)
        rng2 = torch.Generator(device="cpu").manual_seed(42)
        X1, Y1 = gen2_linear(n=100, d=64, r=5, strength=0.5, rng=rng1)
        X2, Y2 = gen2_linear(n=100, d=64, r=5, strength=0.5, rng=rng2)
        assert torch.equal(X1, X2)
        assert torch.equal(Y1, Y2)

    def test_make_gen2_linear_signal(self):
        torch.manual_seed(42)
        X, Y, state = make_gen2_linear_signal(
            n=100,
            d=64,
            rank=5,
            signal_strength=1.0,
            noise_std=0.5,
            noise_type="gaussian",
            device="cpu",
        )
        assert X.shape == (100, 64)
        assert Y.shape == (100, 64)
        assert "Z" in state


class TestGen2Geometry:
    def test_gen2_geometry_state_shapes(self):
        rng = torch.Generator(device="cpu").manual_seed(42)
        state = gen2_geometry_state(n=100, d=64, m=5, rng=rng, noise_type="gaussian")
        assert state["U"].shape == (100, 5)
        assert state["P"].shape == (5, 64)
        assert state["E1"].shape == (100, 64)
        assert state["E2"].shape == (100, 64)

    def test_gen2_geometry_shape(self):
        rng = torch.Generator(device="cpu").manual_seed(42)
        X, Y = gen2_geometry(
            n=100, d=64, m=5, sigma=0.5, rng=rng, noise_type="gaussian"
        )
        assert X.shape == (100, 64)
        assert Y.shape == (100, 64)


class TestGen2Local:
    def test_gen2_local_state_shapes(self):
        rng = torch.Generator(device="cpu").manual_seed(42)
        state = gen2_local_state(
            n=100, d=64, m=5, clusters=4, noise=0.5, rng=rng, noise_type="gaussian"
        )
        assert state["centers_dir"].shape == (4, 5)
        assert state["labels"].shape == (100,)
        assert state["P"].shape == (5, 64)

    def test_gen2_local_shape(self):
        rng = torch.Generator(device="cpu").manual_seed(42)
        X, Y = gen2_local(
            n=100,
            d=64,
            m=5,
            sep=1.0,
            clusters=4,
            noise=0.5,
            rng=rng,
            noise_type="gaussian",
        )
        assert X.shape == (100, 64)
        assert Y.shape == (100, 64)


class TestGen1Mixture:
    def test_gen1_mixture_invariance_shape(self):
        rng = torch.Generator(device="cpu").manual_seed(42)
        X, Y = gen1_mixture_invariance(
            n=100, d=64, eta=0.5, rng=rng, intrinsic_dim=None
        )
        assert X.shape == (100, 64)
        assert Y.shape == (100, 64)

    def test_gen1_mixture_invariance_low_dim(self):
        rng = torch.Generator(device="cpu").manual_seed(42)
        X, Y = gen1_mixture_invariance(n=100, d=64, eta=0.5, rng=rng, intrinsic_dim=10)
        assert X.shape == (100, 64)
        assert Y.shape == (100, 64)


class TestLayerwiseGenerators:
    def test_make_random_layers_shape(self):
        layers = make_random_layers(n=100, d=64, num_layers=5, device="cpu")
        assert len(layers) == 5
        for layer in layers:
            assert layer.shape == (100, 64)

    def test_make_random_layers_with_rng_deterministic(self):
        rng1 = torch.Generator(device="cpu").manual_seed(42)
        rng2 = torch.Generator(device="cpu").manual_seed(42)
        layers1 = make_random_layers_with_rng(
            n=100, d=64, num_layers=5, rng=rng1, device="cpu"
        )
        layers2 = make_random_layers_with_rng(
            n=100, d=64, num_layers=5, rng=rng2, device="cpu"
        )
        for l1, l2 in zip(layers1, layers2):
            assert torch.equal(l1, l2)

    def test_make_signal_layers_shape(self):
        torch.manual_seed(42)
        layersA, layersB = make_signal_layers(
            n=100,
            d=64,
            num_layers=5,
            rank=3,
            signal_strength=1.0,
            noise_std=0.5,
            noise_type="gaussian",
            device="cpu",
        )
        assert len(layersA) == 5
        assert len(layersB) == 5
        for la, lb in zip(layersA, layersB):
            assert la.shape == (100, 64)
            assert lb.shape == (100, 64)
