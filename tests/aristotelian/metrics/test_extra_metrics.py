import numpy as np

from aristotelian.metrics.cca import (
    cca_mean,
    cca_mean_approx,
    pwcca_mean,
    rv_coefficient,
    sg_cca_mean,
    sg_svcca_mean,
    svcca_mean,
    svcca_mean_k,
)
from aristotelian.metrics.other_metrics import procrustes_score


def test_extra_metrics_basic_ranges():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 10))
    Y = rng.standard_normal((40, 10))
    for fn in [cca_mean, svcca_mean, pwcca_mean, rv_coefficient]:
        val = fn(X, Y)
        assert 0.0 <= val <= 1.0


def test_extra_metrics_identical_near_one():
    rng = np.random.default_rng(10)
    X = rng.standard_normal((32, 12))
    for fn in [cca_mean, svcca_mean, pwcca_mean, rv_coefficient]:
        val = fn(X, X.copy())
        assert val > 0.99


def test_procrustes_score_identical():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 8))
    val = procrustes_score(X, X.copy())
    assert 0.9 <= val <= 1.0


def test_svcca_mean_k_basic():
    """Test svcca_mean_k with fixed k components."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((40, 20))
    Y = rng.standard_normal((40, 20))
    val = svcca_mean_k(X, Y, k=5)
    assert 0.0 <= val <= 1.0


def test_cca_mean_approx_basic():
    """Test cca_mean_approx with PCA projection."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, 32))
    Y = rng.standard_normal((40, 32))
    val = cca_mean_approx(X, Y, proj_dim=16)
    assert 0.0 <= val <= 1.0


def test_sg_svcca_mean_with_k():
    """Test sg_svcca_mean with k parameter for speedup."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((32, 16))
    Y = rng.standard_normal((32, 16))
    result = sg_svcca_mean(X, Y, num_permutations=10, k=5)
    assert 0.0 <= result.raw <= 1.0
    assert result.pvalue > 0


def test_sg_cca_mean_with_proj_dim():
    """Test sg_cca_mean with proj_dim parameter for speedup."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((32, 16))
    Y = rng.standard_normal((32, 16))
    result = sg_cca_mean(X, Y, num_permutations=10, proj_dim=8)
    assert 0.0 <= result.raw <= 1.0
    assert result.pvalue > 0
