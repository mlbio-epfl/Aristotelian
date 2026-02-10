"""Tests for batched layerwise CKA implementation equivalence.

These tests verify that the GPU-batched implementations produce
numerically equivalent results to the original sequential implementations.
"""

import pytest
import torch

from aristotelian import standard_cka
from aristotelian.experiments.layerwise_engine import (
    _center_gram_batched,
    permutation_null_batched,
    permutation_null_matrices_batched,
    run_batched_gating_experiment,
    run_batched_multi_aggregator_experiment,
    similarity_matrix_cka_batched,
    similarity_matrix_layerwise,
)
from aristotelian.metrics.aggregation import SimpleMetric


def _cka_metric() -> SimpleMetric:
    return SimpleMetric(
        name="cka_linear",
        max_value=1.0,
        compute=lambda X, Y: standard_cka(X, Y, mode="linear"),
    )


class TestCenterGramBatched:
    """Tests for batched Gram matrix centering."""

    def test_center_gram_batched_single(self):
        """Test centering a single Gram matrix matches manual centering."""
        torch.manual_seed(42)
        n = 10
        X = torch.randn(n, 5)
        K = X @ X.T

        # Manual centering
        n = K.shape[0]
        row_mean = K.mean(dim=0, keepdim=True)
        col_mean = K.mean(dim=1, keepdim=True)
        grand_mean = K.mean()
        K_centered_ref = K - row_mean - col_mean + grand_mean

        # Batched centering (add batch dims and squeeze back)
        K_batched = K.unsqueeze(0)
        K_centered_batched = _center_gram_batched(K_batched).squeeze(0)

        assert torch.allclose(K_centered_ref, K_centered_batched, atol=1e-6)

    def test_center_gram_batched_multiple(self):
        """Test centering multiple Gram matrices in batch."""
        torch.manual_seed(43)
        T, L, n, d = 3, 2, 10, 5
        reps = torch.randn(T, L, n, d)

        # Compute Gram matrices
        grams = torch.bmm(
            reps.view(T * L, n, d),
            reps.view(T * L, n, d).transpose(-1, -2),
        ).view(T, L, n, n)

        # Center in batch
        centered = _center_gram_batched(grams)

        # Verify each individually
        for t in range(T):
            for layer_idx in range(L):
                K = grams[t, layer_idx]
                row_mean = K.mean(dim=0, keepdim=True)
                col_mean = K.mean(dim=1, keepdim=True)
                grand_mean = K.mean()
                K_ref = K - row_mean - col_mean + grand_mean
                assert torch.allclose(K_ref, centered[t, layer_idx], atol=1e-6)


class TestSimilarityMatrixBatchedEquivalence:
    """Tests for batched similarity matrix computation equivalence."""

    def test_single_trial_matches_original(self):
        """Test that a single batched trial matches the original implementation."""
        torch.manual_seed(44)
        n, d, L = 32, 16, 4

        # Create single trial as list of layers
        repsA_list = [torch.randn(n, d) for _ in range(L)]
        repsB_list = [torch.randn(n, d) for _ in range(L)]

        # Stack into batched format (T=1)
        repsA_batched = torch.stack([torch.stack(repsA_list)], dim=0)  # (1, L, n, d)
        repsB_batched = torch.stack([torch.stack(repsB_list)], dim=0)  # (1, L, n, d)

        # Compute with original
        metric = _cka_metric()
        S_original = similarity_matrix_layerwise(
            repsA_list, repsB_list, metric, metric_name="cka_linear"
        )

        # Compute with batched
        S_batched = similarity_matrix_cka_batched(repsA_batched, repsB_batched)[0]

        max_diff = (S_original - S_batched).abs().max().item()
        assert max_diff < 1e-5, f"Max diff {max_diff} exceeds tolerance"

    def test_multiple_trials_match_original(self):
        """Test that multiple batched trials each match the original."""
        torch.manual_seed(45)
        T, n, d, L = 5, 24, 12, 3

        all_repsA = []
        all_repsB = []
        for _ in range(T):
            all_repsA.append([torch.randn(n, d) for _ in range(L)])
            all_repsB.append([torch.randn(n, d) for _ in range(L)])

        # Stack into batched format
        repsA_batched = torch.stack([torch.stack(layers) for layers in all_repsA])
        repsB_batched = torch.stack([torch.stack(layers) for layers in all_repsB])

        # Compute batched
        S_batched = similarity_matrix_cka_batched(repsA_batched, repsB_batched)

        # Compare each trial
        metric = _cka_metric()
        for t in range(T):
            S_original = similarity_matrix_layerwise(
                all_repsA[t], all_repsB[t], metric, metric_name="cka_linear"
            )
            max_diff = (S_original - S_batched[t]).abs().max().item()
            assert max_diff < 1e-5, f"Trial {t}: max diff {max_diff} exceeds tolerance"

    def test_asymmetric_layers(self):
        """Test with different number of layers in A and B."""
        torch.manual_seed(46)
        T, n, d = 3, 20, 10
        L_a, L_b = 4, 6

        repsA_batched = torch.randn(T, L_a, n, d)
        repsB_batched = torch.randn(T, L_b, n, d)

        S = similarity_matrix_cka_batched(repsA_batched, repsB_batched)
        assert S.shape == (T, L_a, L_b)

        # Verify values are valid CKA scores
        assert (S >= -0.1).all()  # CKA can be slightly negative due to centering
        assert (S <= 1.1).all()  # Allow small numerical tolerance


class TestPermutationNullBatchedEquivalence:
    """Tests for batched permutation null computation equivalence."""

    def test_null_distribution_statistics_match(self):
        """Test that null distributions have similar statistics."""
        torch.manual_seed(47)
        T, n, d, L = 10, 24, 12, 3
        num_perms = 50

        # Generate random data (null condition)
        repsA_batched = torch.randn(T, L, n, d)
        repsB_batched = torch.randn(T, L, n, d)

        # Compute batched null
        null_batched = permutation_null_batched(
            repsA_batched, repsB_batched, num_perms, seed=123
        )

        # Verify shape
        assert null_batched.shape == (T, num_perms)

        # Null max values should be bounded
        assert (null_batched >= -0.1).all()
        assert (null_batched <= 1.1).all()

        # Check null distribution is centered (random data should have low alignment)
        mean_null = null_batched.mean().item()
        assert mean_null < 0.6, f"Null mean {mean_null} unexpectedly high"

    def test_null_matrices_shape_correct(self):
        """Test that permutation_null_matrices_batched returns correct shape."""
        torch.manual_seed(48)
        T, n, d, L = 4, 16, 8, 3
        num_perms = 20

        repsA_batched = torch.randn(T, L, n, d)
        repsB_batched = torch.randn(T, L, n, d)

        null_matrices = permutation_null_matrices_batched(
            repsA_batched, repsB_batched, num_perms, seed=456
        )

        assert null_matrices.shape == (T, num_perms, L, L)

    def test_deterministic_with_seed(self):
        """Test that setting seed produces reproducible results."""
        torch.manual_seed(49)
        T, n, d, L = 3, 20, 10, 2
        num_perms = 30

        repsA = torch.randn(T, L, n, d)
        repsB = torch.randn(T, L, n, d)

        null1 = permutation_null_batched(repsA, repsB, num_perms, seed=789)
        null2 = permutation_null_batched(repsA, repsB, num_perms, seed=789)

        assert torch.allclose(null1, null2, atol=1e-6)


class TestGatingExperimentBatched:
    """Tests for batched gating experiment."""

    def test_output_keys_present(self):
        """Test that all expected output keys are present."""
        torch.manual_seed(50)
        T, n, d, L = 5, 16, 8, 3

        repsA = torch.randn(T, L, n, d)
        repsB = torch.randn(T, L, n, d)

        result = run_batched_gating_experiment(
            repsA, repsB, num_permutations=30, alpha=0.05, seed=100
        )

        expected_keys = [
            "raw",
            "gated",
            "p_value",
            "tau_alpha",
            "tail_strength",
            "mu0",
            "sd0",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
            assert result[key].shape == (T,), f"Wrong shape for {key}"

    def test_gated_scores_bounded(self):
        """Test that gated scores are in [0, 1]."""
        torch.manual_seed(51)
        T, n, d, L = 10, 20, 10, 3

        repsA = torch.randn(T, L, n, d)
        repsB = torch.randn(T, L, n, d)

        result = run_batched_gating_experiment(
            repsA, repsB, num_permutations=50, alpha=0.05, seed=200
        )

        assert (result["gated"] >= 0.0).all()
        assert (result["gated"] <= 1.0).all()

    def test_null_condition_low_gated_scores(self):
        """Test that random data (null) produces low gated scores."""
        torch.manual_seed(52)
        T, n, d, L = 20, 32, 16, 4

        # Random data - no alignment
        repsA = torch.randn(T, L, n, d)
        repsB = torch.randn(T, L, n, d)

        result = run_batched_gating_experiment(
            repsA, repsB, num_permutations=100, alpha=0.05, seed=300
        )

        # Most gated scores should be zero for null condition
        mean_gated = result["gated"].mean().item()
        assert mean_gated < 0.2, f"Null gated mean {mean_gated} too high"

    def test_p_values_bounded(self):
        """Test that p-values are in [0, 1]."""
        torch.manual_seed(53)
        T, n, d, L = 8, 16, 8, 3

        repsA = torch.randn(T, L, n, d)
        repsB = torch.randn(T, L, n, d)

        result = run_batched_gating_experiment(
            repsA, repsB, num_permutations=50, alpha=0.05, seed=400
        )

        assert (result["p_value"] >= 0.0).all()
        assert (result["p_value"] <= 1.0).all()


class TestMultiAggregatorExperimentBatched:
    """Tests for batched multi-aggregator experiment."""

    def test_all_aggregators_present(self):
        """Test that all requested aggregators are in output."""
        torch.manual_seed(54)
        T, n, d, L = 5, 16, 8, 3

        repsA = torch.randn(T, L, n, d)
        repsB = torch.randn(T, L, n, d)

        aggregator_names = ["max", "rowmax_mean", "colmax_mean", "topk_5", "topk_10"]
        result = run_batched_multi_aggregator_experiment(
            repsA, repsB, aggregator_names, num_permutations=30, seed=500
        )

        for name in aggregator_names:
            assert name in result, f"Missing aggregator: {name}"
            assert "raw" in result[name]
            assert "gated" in result[name]
            assert result[name]["raw"].shape == (T,)
            assert result[name]["gated"].shape == (T,)

    def test_aggregator_ordering(self):
        """Test that max >= other aggregators (as expected by definition)."""
        torch.manual_seed(55)
        T, n, d, L = 10, 20, 10, 4

        repsA = torch.randn(T, L, n, d)
        repsB = torch.randn(T, L, n, d)

        aggregator_names = ["max", "rowmax_mean", "topk_5"]
        result = run_batched_multi_aggregator_experiment(
            repsA, repsB, aggregator_names, num_permutations=50, seed=600
        )

        # Max should be >= rowmax_mean and topk_5
        assert (result["max"]["raw"] >= result["rowmax_mean"]["raw"] - 1e-5).all()
        assert (result["max"]["raw"] >= result["topk_5"]["raw"] - 1e-5).all()

    def test_gated_scores_bounded_all_aggregators(self):
        """Test that all aggregators produce bounded gated scores."""
        torch.manual_seed(56)
        T, n, d, L = 8, 16, 8, 3

        repsA = torch.randn(T, L, n, d)
        repsB = torch.randn(T, L, n, d)

        aggregator_names = ["max", "rowmax_mean", "colmax_mean", "topk_5", "topk_10"]
        result = run_batched_multi_aggregator_experiment(
            repsA, repsB, aggregator_names, num_permutations=30, seed=700
        )

        for name in aggregator_names:
            assert (result[name]["gated"] >= 0.0).all(), f"{name}: gated < 0"
            assert (result[name]["gated"] <= 1.0).all(), f"{name}: gated > 1"


class TestEndToEndEquivalence:
    """End-to-end tests comparing batched vs original implementations."""

    def test_single_trial_gating_behavior(self):
        """Test that a single trial produces expected gating behavior."""
        torch.manual_seed(57)
        n, d, L = 32, 16, 4

        # Create aligned data (signal condition)
        Z = torch.randn(n, 8)  # shared latent
        A = torch.randn(8, d)
        B = torch.randn(8, d)
        noise_A = 0.3 * torch.randn(n, d)
        noise_B = 0.3 * torch.randn(n, d)

        repsA_signal = [Z @ A + noise_A for _ in range(L)]
        repsB_signal = [Z @ B + noise_B for _ in range(L)]

        # Stack into batched format
        repsA_batched = torch.stack([torch.stack(repsA_signal)], dim=0)
        repsB_batched = torch.stack([torch.stack(repsB_signal)], dim=0)

        result = run_batched_gating_experiment(
            repsA_batched,
            repsB_batched,
            num_permutations=100,
            alpha=0.05,
            seed=800,
        )

        # With signal, raw score should be high
        assert result["raw"][0] > 0.3, f"Signal raw score {result['raw'][0]} too low"

    def test_null_vs_signal_discrimination(self):
        """Test that batched implementation discriminates null from signal."""
        torch.manual_seed(58)
        T_null, T_signal = 10, 10
        n, d, L = 24, 12, 3

        # Null condition (random)
        null_A = torch.randn(T_null, L, n, d)
        null_B = torch.randn(T_null, L, n, d)

        # Signal condition (shared structure per trial)
        signal_A = []
        signal_B = []
        for _ in range(T_signal):
            Z = torch.randn(n, 6)
            A = torch.randn(6, d)
            B = torch.randn(6, d)
            trial_A = torch.stack([Z @ A + 0.2 * torch.randn(n, d) for _ in range(L)])
            trial_B = torch.stack([Z @ B + 0.2 * torch.randn(n, d) for _ in range(L)])
            signal_A.append(trial_A)
            signal_B.append(trial_B)
        signal_A = torch.stack(signal_A)
        signal_B = torch.stack(signal_B)

        result_null = run_batched_gating_experiment(
            null_A, null_B, num_permutations=100, alpha=0.05, seed=900
        )
        result_signal = run_batched_gating_experiment(
            signal_A, signal_B, num_permutations=100, alpha=0.05, seed=901
        )

        # Signal should have higher raw scores on average
        null_mean = result_null["raw"].mean().item()
        signal_mean = result_signal["raw"].mean().item()
        assert (
            signal_mean > null_mean
        ), f"Signal mean {signal_mean} not > null mean {null_mean}"

        # Signal gated scores should be higher
        null_gated_mean = result_null["gated"].mean().item()
        signal_gated_mean = result_signal["gated"].mean().item()
        assert (
            signal_gated_mean > null_gated_mean
        ), f"Signal gated mean {signal_gated_mean} not > null gated mean {null_gated_mean}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDAEquivalence:
    """Tests specifically for CUDA execution."""

    def test_cuda_cpu_equivalence(self):
        """Test that CUDA and CPU produce same results."""
        torch.manual_seed(59)
        T, n, d, L = 5, 16, 8, 3

        repsA_cpu = torch.randn(T, L, n, d)
        repsB_cpu = torch.randn(T, L, n, d)
        repsA_cuda = repsA_cpu.cuda()
        repsB_cuda = repsB_cpu.cuda()

        S_cpu = similarity_matrix_cka_batched(repsA_cpu, repsB_cpu)
        S_cuda = similarity_matrix_cka_batched(repsA_cuda, repsB_cuda)

        assert torch.allclose(S_cpu, S_cuda.cpu(), atol=1e-5)

    def test_cuda_gating_experiment(self):
        """Test full gating experiment on CUDA."""
        torch.manual_seed(60)
        T, n, d, L = 8, 20, 10, 3

        repsA = torch.randn(T, L, n, d, device="cuda")
        repsB = torch.randn(T, L, n, d, device="cuda")

        result = run_batched_gating_experiment(
            repsA, repsB, num_permutations=50, alpha=0.05, seed=1000
        )

        # Check results are on correct device
        assert result["raw"].device.type == "cuda"
        assert result["gated"].device.type == "cuda"

        # Check values are valid
        assert (result["gated"] >= 0.0).all()
        assert (result["gated"] <= 1.0).all()
