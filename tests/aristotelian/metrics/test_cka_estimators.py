"""Tests for CKA estimator variants.

Tests the ported CKA estimators from:
- Re-Align (ICLR 2024): Debiased CKA
- arxiv 2502.15104: Dependent-columns CKA
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from aristotelian.metrics.estimators import (
    cka_biased,
    cka_debiased,
    cka_depcols,
    cka_estimators_all,
    compare_cka_estimators,
)


class TestCKAEstimatorsBasic:
    """Basic functionality tests."""

    def test_identical_matrices_high_score(self):
        """Identical matrices should have high CKA."""
        torch.manual_seed(42)
        X = torch.randn(100, 50)

        assert cka_biased(X, X) > 0.99
        assert cka_debiased(X, X) > 0.95

    def test_random_independent_matrices(self):
        """Random independent matrices should have relatively low CKA."""
        torch.manual_seed(42)
        n, d = 100, 50
        X = torch.randn(n, d)
        Y = torch.randn(n, d)

        biased = cka_biased(X, Y)
        debiased = cka_debiased(X, Y)

        assert biased < 0.6
        assert abs(debiased) < 0.5
        assert biased >= debiased - 0.1

    def test_random_matrices_debiased_near_zero(self):
        """Random independent matrices should have debiased CKA near 0."""
        torch.manual_seed(42)
        X = torch.randn(100, 50)
        Y = torch.randn(100, 50)

        biased = cka_biased(X, Y)
        debiased = cka_debiased(X, Y)

        assert abs(debiased) < abs(biased) + 0.1

    def test_estimators_all_returns_three_values(self):
        """cka_estimators_all should return three estimates."""
        torch.manual_seed(42)
        X = torch.randn(50, 30)
        Y = torch.randn(50, 30)

        naive, song, depcols = cka_estimators_all(X, Y)

        assert isinstance(naive, float)
        assert isinstance(song, float)
        assert isinstance(depcols, float)

    def test_compare_cka_estimators_dict_keys(self):
        """compare_cka_estimators should return dict with correct keys."""
        torch.manual_seed(42)
        X = torch.randn(50, 30)
        Y = torch.randn(50, 30)

        result = compare_cka_estimators(X, Y)

        assert "biased" in result
        assert "debiased" in result
        assert "depcols" in result


class TestCKAEstimatorsHighDNRegime:
    """Tests for the high d/n regime where biased CKA fails."""

    def test_high_dn_biased_inflates(self):
        """In high d/n regime, biased CKA should be inflated."""
        torch.manual_seed(42)
        n = 50
        d = 500

        X = torch.randn(n, d)
        Y = torch.randn(n, d)

        biased = cka_biased(X, Y)
        debiased = cka_debiased(X, Y)

        assert biased > 0.3
        assert abs(debiased) < 0.3

    def test_increasing_dn_ratio(self):
        """Biased CKA should increase with d/n ratio, debiased should not."""
        torch.manual_seed(42)
        n = 50

        biased_values = []
        debiased_values = []

        for ratio in [0.5, 1.0, 2.0, 5.0, 10.0]:
            d = int(n * ratio)
            X = torch.randn(n, d)
            Y = torch.randn(n, d)

            biased_values.append(cka_biased(X, Y))
            debiased_values.append(cka_debiased(X, Y))

        assert biased_values[-1] > biased_values[0]
        assert all(abs(v) < 0.5 for v in debiased_values)


class TestCKAEstimatorsWithSignal:
    """Tests that estimators can detect true signal."""

    def test_shared_signal_detected(self):
        """All estimators should detect shared low-rank signal."""
        torch.manual_seed(42)
        n, d = 100, 50
        rank = 5

        U = torch.randn(n, rank)
        V_x = torch.randn(rank, d)
        V_y = torch.randn(rank, d)

        signal_x = U @ V_x
        signal_y = U @ V_y

        X = signal_x + 0.1 * torch.randn(n, d)
        Y = signal_y + 0.1 * torch.randn(n, d)

        biased = cka_biased(X, Y)
        debiased = cka_debiased(X, Y)
        depcols = cka_depcols(X, Y)

        assert biased > 0.3
        assert debiased > 0.2
        assert depcols > 0.2


class TestCKAEstimatorsConsistency:
    """Tests for consistency between different implementations."""

    def test_moment_estimators_internally_consistent(self):
        """Moment-based estimators should be internally consistent."""
        torch.manual_seed(42)
        X = torch.randn(50, 30)
        Y = torch.randn(50, 30)

        naive, song, depcols = cka_estimators_all(X, Y, indep_cols=True)

        assert isinstance(naive, float)
        assert isinstance(song, float)
        assert isinstance(depcols, float)

        assert abs(song - depcols) < 1e-6

    def test_gram_based_biased_vs_debiased(self):
        """Biased CKA should generally be higher than debiased for random data."""
        torch.manual_seed(42)
        X = torch.randn(50, 200)
        Y = torch.randn(50, 200)

        biased = cka_biased(X, Y)
        debiased = cka_debiased(X, Y)

        assert biased > debiased

    def test_depcols_indep_equals_song_moment(self):
        """With indep_cols=True, depcols should equal song in moment formulation."""
        torch.manual_seed(42)
        X = torch.randn(50, 30)
        Y = torch.randn(50, 30)

        _, song, depcols = cka_estimators_all(X, Y, indep_cols=True)

        assert abs(song - depcols) < 1e-6


class TestCKAEstimatorsEdgeCases:
    """Edge case tests."""

    def test_different_dimensions(self):
        """Should work with different feature dimensions."""
        torch.manual_seed(42)
        X = torch.randn(100, 50)
        Y = torch.randn(100, 30)

        biased = cka_biased(X, Y)
        debiased = cka_debiased(X, Y)

        assert isinstance(biased, float)
        assert isinstance(debiased, float)

    def test_mismatched_samples_raises(self):
        """Should raise error for mismatched sample counts."""
        X = torch.randn(100, 50)
        Y = torch.randn(80, 50)

        with pytest.raises(ValueError, match="same number of samples"):
            cka_biased(X, Y)

        with pytest.raises(ValueError, match="same number of samples"):
            cka_debiased(X, Y)

    def test_small_sample_size(self):
        """Should work with small sample sizes (n > 4 for unbiased)."""
        torch.manual_seed(42)
        X = torch.randn(10, 5)
        Y = torch.randn(10, 5)

        biased = cka_biased(X, Y)
        debiased = cka_debiased(X, Y)

        assert isinstance(biased, float)
        assert isinstance(debiased, float)
        assert not np.isnan(biased)
        assert not np.isnan(debiased)

    def test_gpu_if_available(self):
        """Should work on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        X = torch.randn(100, 50, device="cuda")
        Y = torch.randn(100, 50, device="cuda")

        biased = cka_biased(X, Y)
        debiased = cka_debiased(X, Y)

        assert isinstance(biased, float)
        assert isinstance(debiased, float)
