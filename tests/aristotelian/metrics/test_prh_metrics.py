"""Tests to verify prh_metrics.py matches the original Platonic-Rep implementation."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add external_code to path to import original implementation
sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "external_code" / "platonic-rep")
)

from metrics import AlignmentMetrics as OriginalMetrics
from metrics import compute_knn_accuracy as original_compute_knn_accuracy
from metrics import compute_nearest_neighbors as original_compute_nearest_neighbors
from metrics import hsic_biased as original_hsic_biased
from metrics import hsic_unbiased as original_hsic_unbiased

from aristotelian import (
    cka,
    cknna,
    compute_knn_accuracy,
    compute_nearest_neighbors,
    cycle_knn,
    hsic_biased,
    hsic_unbiased,
    mutual_knn,
    svcca,
    unbiased_cka,
)


@pytest.fixture
def sample_features():
    """Generate deterministic sample features for testing."""
    torch.manual_seed(42)
    np.random.seed(42)
    feats_A = torch.randn(32, 64)
    feats_B = torch.randn(32, 64)
    return feats_A, feats_B


@pytest.fixture
def normalized_features(sample_features):
    """Normalized features (as used in original PRH examples)."""
    feats_A, feats_B = sample_features
    feats_A = torch.nn.functional.normalize(feats_A, dim=-1)
    feats_B = torch.nn.functional.normalize(feats_B, dim=-1)
    return feats_A, feats_B


class TestComputeNearestNeighbors:
    def test_matches_original(self, sample_features):
        feats_A, _ = sample_features
        topk = 5
        ours = compute_nearest_neighbors(feats_A, topk)
        original = original_compute_nearest_neighbors(feats_A, topk)
        assert torch.equal(ours, original)

    def test_different_topk_values(self, sample_features):
        feats_A, _ = sample_features
        for topk in [1, 3, 10, 20]:
            ours = compute_nearest_neighbors(feats_A, topk)
            original = original_compute_nearest_neighbors(feats_A, topk)
            assert torch.equal(ours, original), f"Mismatch at topk={topk}"


class TestComputeKnnAccuracy:
    def test_matches_original(self, sample_features):
        feats_A, _ = sample_features
        knn = compute_nearest_neighbors(feats_A, 5)
        ours = compute_knn_accuracy(knn)
        original = original_compute_knn_accuracy(knn)
        assert torch.isclose(ours, original)


class TestHSIC:
    def test_hsic_unbiased_matches_original(self, sample_features):
        feats_A, feats_B = sample_features
        K = feats_A @ feats_A.T
        L = feats_B @ feats_B.T
        ours = hsic_unbiased(K, L)
        original = original_hsic_unbiased(K, L)
        assert torch.isclose(ours, original, rtol=1e-5)

    def test_hsic_biased_matches_original(self, sample_features):
        feats_A, feats_B = sample_features
        K = feats_A @ feats_A.T
        L = feats_B @ feats_B.T
        ours = hsic_biased(K, L)
        original = original_hsic_biased(K, L)
        assert torch.isclose(ours, original, rtol=1e-5)


class TestCycleKnn:
    def test_matches_original(self, normalized_features):
        feats_A, feats_B = normalized_features
        topk = 10
        ours = cycle_knn(feats_A, feats_B, topk=topk)
        original = OriginalMetrics.cycle_knn(feats_A, feats_B, topk)
        assert np.isclose(ours, original, rtol=1e-5)


class TestMutualKnn:
    def test_matches_original(self, normalized_features):
        feats_A, feats_B = normalized_features
        topk = 10
        ours = mutual_knn(feats_A, feats_B, topk=topk)
        original = OriginalMetrics.mutual_knn(feats_A, feats_B, topk)
        assert np.isclose(ours, original, rtol=1e-5)

    def test_different_topk_values(self, normalized_features):
        feats_A, feats_B = normalized_features
        for topk in [1, 5, 10, 20]:
            ours = mutual_knn(feats_A, feats_B, topk=topk)
            original = OriginalMetrics.mutual_knn(feats_A, feats_B, topk)
            assert np.isclose(ours, original, rtol=1e-5), f"Mismatch at topk={topk}"


class TestCKA:
    def test_linear_matches_original(self, normalized_features):
        feats_A, feats_B = normalized_features
        ours = cka(feats_A, feats_B, kernel_metric="ip")
        original = OriginalMetrics.cka(feats_A, feats_B, kernel_metric="ip")
        assert np.isclose(ours, original, rtol=1e-5)

    def test_rbf_matches_original(self, normalized_features):
        feats_A, feats_B = normalized_features
        ours = cka(feats_A, feats_B, kernel_metric="rbf", rbf_sigma=1.0)
        original = OriginalMetrics.cka(
            feats_A, feats_B, kernel_metric="rbf", rbf_sigma=1.0
        )
        assert np.isclose(ours, original, rtol=1e-5)

    def test_unbiased_matches_original(self, normalized_features):
        feats_A, feats_B = normalized_features
        ours = unbiased_cka(feats_A, feats_B)
        original = OriginalMetrics.unbiased_cka(feats_A, feats_B)
        # Relaxed tolerance due to minor numerical differences in HSIC computation
        assert np.isclose(ours, original, rtol=1e-4)


class TestSVCCA:
    def test_score_range(self, normalized_features):
        feats_A, feats_B = normalized_features
        score = svcca(feats_A, feats_B, cca_dim=10)
        assert 0.0 <= score <= 1.0, f"SVCCA score out of range: {score}"

    def test_identical_high(self, normalized_features):
        feats_A, _ = normalized_features
        score = svcca(feats_A, feats_A.clone(), cca_dim=10)
        assert score > 0.99, f"SVCCA of identical inputs should be ~1, got {score}"


class TestCKNNA:
    def test_matches_original(self, normalized_features):
        feats_A, feats_B = normalized_features
        topk = 10
        ours = cknna(feats_A, feats_B, topk=topk)
        original = OriginalMetrics.cknna(feats_A, feats_B, topk=topk)
        assert np.isclose(ours, original, rtol=1e-4)

    def test_topk_none_uses_all_neighbors(self, normalized_features):
        feats_A, feats_B = normalized_features
        n = feats_A.shape[0]
        # Our implementation with topk=None should equal topk=n-1
        ours_none = cknna(feats_A, feats_B, topk=None)
        ours_explicit = cknna(feats_A, feats_B, topk=n - 1)
        assert np.isclose(ours_none, ours_explicit, rtol=1e-5)

    def test_distance_agnostic_returns_valid_score(self, normalized_features):
        # Note: Original PRH code has a bug with distance_agnostic=True
        # (returns 2D tensor instead of scalar). We fixed this by summing
        # the overlap mask. This test verifies our fix produces valid scores.
        feats_A, feats_B = normalized_features
        topk = 10
        ours = cknna(feats_A, feats_B, topk=topk, distance_agnostic=True)
        # Should return a valid float between 0 and some reasonable upper bound
        assert isinstance(ours, float)
        assert ours >= 0.0

    def test_biased_matches_original(self, normalized_features):
        feats_A, feats_B = normalized_features
        topk = 10
        ours = cknna(feats_A, feats_B, topk=topk, unbiased=False)
        original = OriginalMetrics.cknna(feats_A, feats_B, topk=topk, unbiased=False)
        assert np.isclose(ours, original, rtol=1e-5)


class TestConsistencyWithMainMetrics:
    def test_mutual_knn_matches_sg_knn_raw(self, normalized_features):
        """Verify mutual_knn and mutual_knn_overlap use the same algorithm."""
        from aristotelian import mutual_knn_overlap

        feats_A, feats_B = normalized_features
        topk = 10
        prh_result = mutual_knn(feats_A, feats_B, topk=topk)
        main_result = mutual_knn_overlap(feats_A, feats_B, k=topk)
        assert np.isclose(prh_result, main_result, rtol=1e-5)


class TestEdgeCases:
    def test_identical_features_high_similarity(self):
        torch.manual_seed(0)
        feats = torch.randn(20, 32)
        feats = torch.nn.functional.normalize(feats, dim=-1)

        assert mutual_knn(feats, feats, topk=5) > 0.99
        assert cka(feats, feats, kernel_metric="ip") > 0.99
        assert unbiased_cka(feats, feats) > 0.99

    def test_cknna_requires_topk_ge_2(self):
        torch.manual_seed(0)
        feats_A = torch.randn(10, 16)
        feats_B = torch.randn(10, 16)
        with pytest.raises(ValueError, match="topk >= 2"):
            cknna(feats_A, feats_B, topk=1)

    def test_compute_nearest_neighbors_requires_2d(self):
        feats_3d = torch.randn(10, 16, 8)
        with pytest.raises(ValueError, match="2D"):
            compute_nearest_neighbors(feats_3d, topk=5)
