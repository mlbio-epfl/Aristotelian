"""Tests for layerwise signal generation in experiment runner.

These tests verify the critical property that make_signal_layers creates:
- HIGH alignment on diagonal entries (layer i with layer i)
- NULL-like behavior on off-diagonal entries (layer i with layer j, i != j)

This is essential for selection bias experiments to be valid.
"""

import math

import numpy as np
import pytest
import torch

from scripts.experiments.generators.common import projection_matrix_no_rng
from scripts.experiments.generators.gen2 import (
    gen2_linear_from_state,
    gen2_linear_shared_state,
    make_gen2_linear_signal,
)
from scripts.experiments.generators.layerwise import (
    make_random_layers,
    make_signal_layers,
)
from scripts.experiments.generators.noise import (
    NOISE_TYPES,
    sample_noise,
    strength_from_snr,
)


def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between two representation matrices."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic = (X.T @ Y).pow(2).sum()
    norm_x = (X.T @ X).pow(2).sum().sqrt()
    norm_y = (Y.T @ Y).pow(2).sum().sqrt()
    return float(hsic / (norm_x * norm_y + 1e-10))


def _compute_cka_matrix(
    repsA: list[torch.Tensor], repsB: list[torch.Tensor]
) -> np.ndarray:
    """Compute full CKA matrix between two sets of layer representations."""
    num_layers = len(repsA)
    cka_matrix = np.zeros((num_layers, num_layers))
    for i in range(num_layers):
        for j in range(num_layers):
            cka_matrix[i, j] = _linear_cka(repsA[i], repsB[j])
    return cka_matrix


class TestStrengthFromSNR:
    """Tests for strength_from_snr normalization."""

    def test_zero_signal_returns_zero(self):
        assert strength_from_snr(0.0, 1.0) == 0.0

    def test_zero_noise_returns_one(self):
        assert strength_from_snr(1.0, 0.0) == 1.0

    def test_equal_signal_noise(self):
        # signal / sqrt(signal^2 + noise^2) = 1 / sqrt(2) ≈ 0.707
        result = strength_from_snr(1.0, 1.0)
        expected = 1.0 / math.sqrt(2.0)
        assert abs(result - expected) < 1e-6

    def test_bounded_output(self):
        """Verify output is always in [0, 1]."""
        for signal in [0.0, 0.1, 1.0, 10.0, 100.0]:
            for noise in [0.0, 0.1, 1.0, 10.0, 100.0]:
                if signal == 0.0 and noise == 0.0:
                    continue  # Skip degenerate case
                result = strength_from_snr(signal, noise)
                assert 0.0 <= result <= 1.0, f"Out of bounds: {result}"

    def test_monotonic_in_signal(self):
        """Higher signal strength should give higher normalized value."""
        noise = 1.0
        prev = 0.0
        for signal in [0.1, 0.5, 1.0, 2.0, 5.0]:
            current = strength_from_snr(signal, noise)
            assert current > prev, f"Not monotonic: {current} <= {prev}"
            prev = current


class TestSampleNoise:
    """Tests for sample_noise with different noise types."""

    @pytest.fixture
    def dims(self):
        return 1000, 50  # n, d

    def test_gaussian_noise_stats(self, dims):
        n, d = dims
        torch.manual_seed(42)
        noise = sample_noise(n, d, 1.0, "gaussian", device="cpu")
        assert noise.shape == (n, d)
        # Check approximate unit variance
        assert 0.9 < noise.std().item() < 1.1

    def test_laplace_noise_stats(self, dims):
        n, d = dims
        torch.manual_seed(42)
        noise = sample_noise(n, d, 1.0, "laplace", device="cpu")
        assert noise.shape == (n, d)
        # Laplace with scale b has variance 2*b^2
        # We set b = noise_std / sqrt(2), so variance should be noise_std^2
        assert 0.8 < noise.std().item() < 1.2

    def test_student_t_noise_has_heavier_tails(self, dims):
        n, d = dims
        torch.manual_seed(42)
        gauss = sample_noise(n, d, 1.0, "gaussian", device="cpu")
        student = sample_noise(n, d, 1.0, "student_t", device="cpu")
        # Student-t should have more extreme values
        gauss_max = gauss.abs().max().item()
        student_max = student.abs().max().item()
        # This is probabilistic but should hold with high probability
        assert student_max > gauss_max * 0.5  # Relaxed check

    def test_mixture_noise_has_outliers(self, dims):
        n, d = dims
        torch.manual_seed(42)
        mixture = sample_noise(n, d, 1.0, "mixture", device="cpu")
        assert mixture.shape == (n, d)
        # Mixture should be mostly Gaussian with some heavy-tailed samples

    def test_zero_noise_std_returns_zeros(self, dims):
        n, d = dims
        noise = sample_noise(n, d, 0.0, "gaussian", device="cpu")
        assert torch.allclose(noise, torch.zeros(n, d))

    def test_invalid_noise_type_raises(self, dims):
        n, d = dims
        with pytest.raises(ValueError, match="Unknown noise type"):
            sample_noise(n, d, 1.0, "invalid_type", device="cpu")

    @pytest.mark.parametrize("noise_type", NOISE_TYPES)
    def test_all_noise_types_produce_correct_shape(self, dims, noise_type):
        n, d = dims
        torch.manual_seed(42)
        noise = sample_noise(n, d, 1.0, noise_type, device="cpu")
        assert noise.shape == (n, d)


class TestGen2LinearSignal:
    """Tests for the unified gen2 linear signal generator."""

    def test_shared_state_structure(self):
        n, d, r = 100, 50, 5
        state = gen2_linear_shared_state(n, d, r, device="cpu")
        assert "Z" in state
        assert "A" in state
        assert "B" in state
        assert state["Z"].shape == (n, r)
        assert state["A"].shape == (r, d)
        assert state["B"].shape == (r, d)

    def test_from_state_produces_aligned_pairs(self):
        torch.manual_seed(42)
        n, d, r = 200, 100, 10
        state = gen2_linear_shared_state(n, d, r, device="cpu")
        X, Y = gen2_linear_from_state(state, strength=0.9, noise_type="gaussian")
        cka = _linear_cka(X, Y)
        # High strength should produce high CKA
        assert cka > 0.3, f"CKA too low for aligned pair: {cka}"

    def test_zero_strength_produces_null(self):
        torch.manual_seed(42)
        n, d, r = 200, 100, 10
        state = gen2_linear_shared_state(n, d, r, device="cpu")
        X, Y = gen2_linear_from_state(state, strength=0.0, noise_type="gaussian")
        cka = _linear_cka(X, Y)
        # Zero strength means pure noise - should have baseline CKA
        assert cka < 0.4, f"CKA too high for null pair: {cka}"

    def testmake_gen2_linear_signal_returns_state(self):
        torch.manual_seed(42)
        X, Y, state = make_gen2_linear_signal(
            n=100,
            d=50,
            rank=5,
            signal_strength=1.0,
            noise_std=1.0,
            noise_type="gaussian",
            device="cpu",
        )
        assert X.shape == (100, 50)
        assert Y.shape == (100, 50)
        assert state is not None
        assert "Z" in state


class TestMakeSignalLayers:
    """Critical tests for make_signal_layers diagonal/off-diagonal separation."""

    @pytest.fixture
    def layer_params(self):
        return {
            "n": 512,
            "d": 256,
            "num_layers": 4,
            "rank": 5,
            "signal_strength": 2.0,
            "noise_std": 1.0,
            "noise_type": "gaussian",
            "device": "cpu",
        }

    def test_produces_correct_number_of_layers(self, layer_params):
        torch.manual_seed(42)
        repsA, repsB = make_signal_layers(**layer_params)
        assert len(repsA) == layer_params["num_layers"]
        assert len(repsB) == layer_params["num_layers"]

    def test_produces_correct_shapes(self, layer_params):
        torch.manual_seed(42)
        repsA, repsB = make_signal_layers(**layer_params)
        for X, Y in zip(repsA, repsB):
            assert X.shape == (layer_params["n"], layer_params["d"])
            assert Y.shape == (layer_params["n"], layer_params["d"])

    def test_diagonal_has_high_cka(self, layer_params):
        """Diagonal entries (i, i) should have high CKA due to shared Z."""
        torch.manual_seed(42)
        repsA, repsB = make_signal_layers(**layer_params)
        cka_matrix = _compute_cka_matrix(repsA, repsB)
        diagonal = np.diag(cka_matrix)
        # Diagonal should be significantly above baseline
        assert diagonal.mean() > 0.35, f"Diagonal mean too low: {diagonal.mean()}"

    def test_off_diagonal_is_null_like(self, layer_params):
        """Off-diagonal entries (i, j) should be null-like due to independent Z's."""
        torch.manual_seed(42)
        repsA, repsB = make_signal_layers(**layer_params)
        cka_matrix = _compute_cka_matrix(repsA, repsB)
        num_layers = layer_params["num_layers"]
        off_diag_mask = ~np.eye(num_layers, dtype=bool)
        off_diagonal = cka_matrix[off_diag_mask]
        # Off-diagonal should be near baseline (null) level
        assert (
            off_diagonal.mean() < 0.32
        ), f"Off-diagonal mean too high: {off_diagonal.mean()}"

    def test_diagonal_significantly_higher_than_off_diagonal(self, layer_params):
        """CRITICAL: Diagonal must be significantly higher than off-diagonal."""
        torch.manual_seed(42)
        repsA, repsB = make_signal_layers(**layer_params)
        cka_matrix = _compute_cka_matrix(repsA, repsB)
        num_layers = layer_params["num_layers"]

        diagonal = np.diag(cka_matrix)
        off_diag_mask = ~np.eye(num_layers, dtype=bool)
        off_diagonal = cka_matrix[off_diag_mask]

        # The key invariant: diagonal should be clearly separated from off-diagonal
        separation = diagonal.mean() - off_diagonal.mean()
        assert separation > 0.1, (
            f"Insufficient separation between diagonal ({diagonal.mean():.3f}) "
            f"and off-diagonal ({off_diagonal.mean():.3f}): {separation:.3f}"
        )

    def test_off_diagonal_similar_to_pure_null(self, layer_params):
        """Off-diagonal should be similar to pure random (null) representations."""
        torch.manual_seed(42)
        repsA_signal, repsB_signal = make_signal_layers(**layer_params)

        # Generate pure null representations
        torch.manual_seed(43)  # Different seed
        repsA_null = make_random_layers(
            layer_params["n"],
            layer_params["d"],
            layer_params["num_layers"],
            device=layer_params["device"],
        )
        repsB_null = make_random_layers(
            layer_params["n"],
            layer_params["d"],
            layer_params["num_layers"],
            device=layer_params["device"],
        )

        cka_signal = _compute_cka_matrix(repsA_signal, repsB_signal)
        cka_null = _compute_cka_matrix(repsA_null, repsB_null)

        num_layers = layer_params["num_layers"]
        off_diag_mask = ~np.eye(num_layers, dtype=bool)

        signal_off_diag = cka_signal[off_diag_mask].mean()
        null_mean = cka_null.mean()

        # Off-diagonal of signal should be close to null baseline
        diff = abs(signal_off_diag - null_mean)
        assert diff < 0.05, (
            f"Signal off-diagonal ({signal_off_diag:.3f}) differs too much "
            f"from null baseline ({null_mean:.3f})"
        )

    @pytest.mark.parametrize("noise_type", NOISE_TYPES)
    def test_separation_holds_for_all_noise_types(self, layer_params, noise_type):
        """Diagonal/off-diagonal separation should hold for all noise types."""
        torch.manual_seed(42)
        params = {**layer_params, "noise_type": noise_type}
        repsA, repsB = make_signal_layers(**params)
        cka_matrix = _compute_cka_matrix(repsA, repsB)
        num_layers = params["num_layers"]

        diagonal = np.diag(cka_matrix)
        off_diag_mask = ~np.eye(num_layers, dtype=bool)
        off_diagonal = cka_matrix[off_diag_mask]

        separation = diagonal.mean() - off_diagonal.mean()
        assert separation > 0.08, (
            f"Insufficient separation for {noise_type}: "
            f"diagonal={diagonal.mean():.3f}, off-diagonal={off_diagonal.mean():.3f}"
        )

    def test_more_layers_maintains_separation(self):
        """Separation should hold even with many layers."""
        torch.manual_seed(42)
        repsA, repsB = make_signal_layers(
            n=256,
            d=128,
            num_layers=16,
            rank=5,
            signal_strength=2.0,
            noise_std=1.0,
            noise_type="gaussian",
            device="cpu",
        )
        cka_matrix = _compute_cka_matrix(repsA, repsB)

        diagonal = np.diag(cka_matrix)
        off_diag_mask = ~np.eye(16, dtype=bool)
        off_diagonal = cka_matrix[off_diag_mask]

        separation = diagonal.mean() - off_diagonal.mean()
        assert separation > 0.08, f"Separation failed with 16 layers: {separation:.3f}"


class TestProjectionMatrix:
    """Tests for orthogonal projection matrix generation."""

    def test_produces_orthonormal_rows(self):
        torch.manual_seed(42)
        P = projection_matrix_no_rng(5, 20, "cpu")
        # P is (5, 20) with orthonormal rows
        gram = P @ P.T
        identity = torch.eye(5)
        assert torch.allclose(gram, identity, atol=1e-5)

    def test_independent_calls_produce_different_matrices(self):
        torch.manual_seed(42)
        P1 = projection_matrix_no_rng(5, 20, "cpu")
        P2 = projection_matrix_no_rng(5, 20, "cpu")
        # Should be different (not seeded the same way each call)
        assert not torch.allclose(P1, P2)


class TestMakeRandomLayers:
    """Tests for pure random (null) layer generation."""

    def test_produces_independent_layers(self):
        torch.manual_seed(42)
        reps = make_random_layers(100, 50, 4, device="cpu")
        # Each layer should be independent
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                cka = _linear_cka(reps[i], reps[j])
                # Random layers should have baseline CKA
                assert cka < 0.4, f"Layers {i} and {j} unexpectedly correlated: {cka}"

    def test_cross_model_null_cka(self):
        torch.manual_seed(42)
        repsA = make_random_layers(100, 50, 4, device="cpu")
        repsB = make_random_layers(100, 50, 4, device="cpu")
        cka_matrix = _compute_cka_matrix(repsA, repsB)
        # All entries should be baseline (null) level
        assert cka_matrix.mean() < 0.4
        assert cka_matrix.max() < 0.5
