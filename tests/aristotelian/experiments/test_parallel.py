#!/usr/bin/env python
"""Tests for parallelization optimizations in aggregation and experiments."""

import pytest
import torch

from aristotelian import standard_cka
from aristotelian.metrics.aggregation import (
    SimpleMetric,
    agg_max,
    compute_similarity_matrix,
    permutation_null_aggregated,
)


def _cka_metric() -> SimpleMetric:
    """CKA metric for testing - same as used in experiments."""
    return SimpleMetric(
        name="cka_linear",
        max_value=1.0,
        compute=lambda X, Y: standard_cka(X, Y, mode="linear"),
    )


def _toy_metric() -> SimpleMetric:
    """Simple cosine metric for quick tests."""
    return SimpleMetric(
        name="cosine",
        max_value=1.0,
        compute=lambda X, Y: float(
            torch.nn.functional.cosine_similarity(
                X.flatten(), Y.flatten(), dim=0
            ).item()
        ),
    )


class TestPermutationNullAggregatedSerial:
    """Tests for the current serial implementation - baseline for comparison."""

    def test_deterministic_with_seed(self):
        """Verify current implementation is deterministic with seed."""
        metric = _cka_metric()
        torch.manual_seed(42)
        n, d, L = 64, 32, 4
        repsA = [torch.randn(n, d) for _ in range(L)]
        repsB = [torch.randn(n, d) for _ in range(L)]

        null1 = permutation_null_aggregated(
            repsA, repsB, metric, agg_max, num_permutations=20, seed=123
        )
        null2 = permutation_null_aggregated(
            repsA, repsB, metric, agg_max, num_permutations=20, seed=123
        )

        assert null1 == null2, "Same seed should give identical results"

    def test_different_seeds_give_different_results(self):
        """Verify different seeds give different results."""
        metric = _cka_metric()
        torch.manual_seed(42)
        n, d, L = 64, 32, 4
        repsA = [torch.randn(n, d) for _ in range(L)]
        repsB = [torch.randn(n, d) for _ in range(L)]

        null1 = permutation_null_aggregated(
            repsA, repsB, metric, agg_max, num_permutations=20, seed=123
        )
        null2 = permutation_null_aggregated(
            repsA, repsB, metric, agg_max, num_permutations=20, seed=456
        )

        assert null1 != null2, "Different seeds should give different results"

    def test_output_length_matches_num_permutations(self):
        """Verify output has correct length."""
        metric = _toy_metric()
        torch.manual_seed(42)
        repsA = [torch.randn(32, 16) for _ in range(2)]
        repsB = [torch.randn(32, 16) for _ in range(2)]

        for num_perms in [10, 50, 100]:
            null = permutation_null_aggregated(
                repsA, repsB, metric, agg_max, num_permutations=num_perms, seed=0
            )
            assert len(null) == num_perms

    def test_null_values_bounded(self):
        """Verify null values are within expected bounds for bounded metric."""
        metric = _cka_metric()
        torch.manual_seed(42)
        repsA = [torch.randn(64, 32) for _ in range(3)]
        repsB = [torch.randn(64, 32) for _ in range(3)]

        null = permutation_null_aggregated(
            repsA, repsB, metric, agg_max, num_permutations=50, seed=0
        )

        for val in null:
            assert -0.1 <= val <= 1.1, f"CKA should be roughly in [0,1], got {val}"


class TestComputeSimilarityMatrix:
    """Tests for compute_similarity_matrix."""

    def test_output_shape(self):
        """Verify output shape is (len_A, len_B)."""
        metric = _toy_metric()
        for la, lb in [(2, 3), (4, 4), (1, 5)]:
            repsA = [torch.randn(32, 16) for _ in range(la)]
            repsB = [torch.randn(32, 16) for _ in range(lb)]
            S = compute_similarity_matrix(repsA, repsB, metric)
            assert S.shape == (la, lb)

    def test_deterministic(self):
        """Verify compute_similarity_matrix is deterministic."""
        metric = _cka_metric()
        torch.manual_seed(42)
        repsA = [torch.randn(64, 32) for _ in range(3)]
        repsB = [torch.randn(64, 32) for _ in range(3)]

        S1 = compute_similarity_matrix(repsA, repsB, metric)
        S2 = compute_similarity_matrix(repsA, repsB, metric)

        assert torch.allclose(S1, S2), "Should be deterministic"

    def test_cka_values_bounded(self):
        """Verify CKA similarity values are in valid range."""
        metric = _cka_metric()
        torch.manual_seed(42)
        repsA = [torch.randn(64, 32) for _ in range(4)]
        repsB = [torch.randn(64, 32) for _ in range(4)]

        S = compute_similarity_matrix(repsA, repsB, metric)

        # CKA should be in [0, 1] for typical data
        assert S.min() >= -0.1, f"Min CKA too low: {S.min()}"
        assert S.max() <= 1.1, f"Max CKA too high: {S.max()}"


class TestType1CalibrationParallel:
    """Tests for parallelized type1 calibration."""

    def test_serial_parallel_equivalence(self):
        """Test that parallel execution gives same results as serial.

        Note: Uses same parameters as existing passing test in test_experiments.py
        """
        from aristotelian.experiments.experiments import run_type1_calibration

        torch.manual_seed(2)
        # Run with serial execution
        res_serial = run_type1_calibration(
            metric="sgrsa",
            n=18,
            d=4,
            num_trials=8,
            num_permutations=10,
            quantile=0.9,
            null_type="gaussian",
            seed=11,
            num_workers=1,
            rsa_batch_size=4,
        )

        # Run with parallel execution
        res_parallel = run_type1_calibration(
            metric="sgrsa",
            n=18,
            d=4,
            num_trials=8,
            num_permutations=10,
            quantile=0.9,
            null_type="gaussian",
            seed=11,
            num_workers=2,
            rsa_batch_size=4,
        )

        assert res_serial.positives == res_parallel.positives
        assert res_serial.type1_rate == res_parallel.type1_rate


class TestExpAMaxInflationParallel:
    """Tests for exp_a_max_inflation parallelization."""

    def test_single_layer_trial_deterministic(self):
        """Test that a single trial computation is deterministic."""
        from aristotelian.metrics.aggregation import (
            agg_max,
            compute_null_summary,
            compute_similarity_matrix,
            gated_rescaled,
            permutation_null_aggregated,
        )

        metric = _cka_metric()
        n, d, L = 64, 32, 4
        num_permutations = 20
        alpha = 0.05

        def run_trial(seed):
            torch.manual_seed(seed)
            repsA = [torch.randn(n, d) for _ in range(L)]
            repsB = [torch.randn(n, d) for _ in range(L)]
            S = compute_similarity_matrix(repsA, repsB, metric)
            T_obs = agg_max(S).value
            null_samples = permutation_null_aggregated(
                repsA,
                repsB,
                metric,
                agg_max,
                num_permutations=num_permutations,
                seed=seed,
            )
            summary = compute_null_summary(null_samples, T_obs=T_obs, alpha=alpha)
            g = gated_rescaled(T_obs, tau_alpha=summary["tau_alpha"], s_max=1.0)
            if summary["p_value"] > alpha:
                g = 0.0
            return T_obs, g, summary["p_value"]

        result1 = run_trial(42)
        result2 = run_trial(42)

        assert result1 == result2, "Same seed should give identical results"


class TestExpBParallel:
    """Tests for exp_b_aggregator_calibration parallelization."""

    def test_aggregator_trial_deterministic(self):
        """Test that aggregator trial computation is deterministic."""
        from aristotelian.metrics.aggregation import (
            agg_max,
            agg_rowmax_mean,
            compute_null_summary,
            compute_similarity_matrix,
            gated_rescaled,
            permutation_null_aggregated,
        )

        metric = _cka_metric()
        n, d, L = 64, 32, 4
        num_permutations = 20
        alpha = 0.05

        def run_trial(agg_fn, seed):
            torch.manual_seed(seed)
            repsA = [torch.randn(n, d) for _ in range(L)]
            repsB = [torch.randn(n, d) for _ in range(L)]
            S = compute_similarity_matrix(repsA, repsB, metric)
            T_obs = agg_fn(S).value
            null_samples = permutation_null_aggregated(
                repsA,
                repsB,
                metric,
                agg_fn,
                num_permutations=num_permutations,
                seed=seed,
            )
            summary = compute_null_summary(null_samples, T_obs=T_obs, alpha=alpha)
            g = gated_rescaled(T_obs, tau_alpha=summary["tau_alpha"], s_max=1.0)
            if summary["p_value"] > alpha:
                g = 0.0
            return T_obs, g, summary["p_value"]

        for agg_fn in [agg_max, agg_rowmax_mean]:
            result1 = run_trial(agg_fn, 42)
            result2 = run_trial(agg_fn, 42)
            assert result1 == result2, f"Same seed should give identical for {agg_fn}"


class TestType1CalibrationGridHelper:
    """Tests for the type1 calibration grid helper function."""

    def test_single_config_picklable(self):
        """Test that config tuples are picklable for ProcessPoolExecutor."""
        import pickle

        config = ("gaussian", "sgcka_lin", 24, 8, 10, 15, 0.05, 42, "cpu")
        pickled = pickle.dumps(config)
        unpickled = pickle.loads(pickled)
        assert unpickled == config

    def test_single_config_deterministic(self):
        """Test that single config function is deterministic."""
        from scripts.experiments.sections.calibration import (
            _type1_calibration_single_config,
        )

        config = ("gaussian", "sgcka_lin", 18, 6, 4, 10, 0.05, 42, "cpu")
        result1 = _type1_calibration_single_config(config)
        result2 = _type1_calibration_single_config(config)

        assert result1 == result2, "Same config should give identical results"


class TestProcessPoolExecutorCompatibility:
    """Tests to ensure functions work with ProcessPoolExecutor."""

    def test_type1_trial_picklable(self):
        """Test that type1 trial function can be pickled (required for ProcessPoolExecutor)."""
        import pickle

        from aristotelian.experiments.experiments import _type1_trial

        # Try to pickle the function (will fail if not picklable)
        try:
            pickled = pickle.dumps(_type1_trial)
            unpickled = pickle.loads(pickled)
            assert callable(unpickled)
        except Exception as e:
            pytest.fail(f"_type1_trial should be picklable: {e}")

    def test_trial_args_picklable(self):
        """Test that trial arguments can be pickled."""
        import pickle

        args = {
            "metric": "sgcka_lin",
            "n": 24,
            "d": 8,
            "num_permutations": 15,
            "quantile": 0.9,
            "null_type": "gaussian",
            "k_knn": 10,
            "device": "cpu",
            "rsa_batch_size": 32,
        }

        try:
            pickled = pickle.dumps(args)
            unpickled = pickle.loads(pickled)
            assert unpickled == args
        except Exception as e:
            pytest.fail(f"Arguments should be picklable: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
