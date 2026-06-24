import itertools

import numpy as np
import pytest
import torch

from aristotelian import mutual_knn_overlap, rsa_vector, sg_knn, sg_rsa, spearman_corr
from aristotelian.metrics.rsa import _rsa_auto_pair_samples


def test_mutual_knn_overlap_identical():
    torch.manual_seed(0)
    X = torch.randn(16, 4)
    raw = mutual_knn_overlap(X, X, k=5)
    assert np.isclose(raw, 1.0)  # identical data should have perfect overlap


def test_sg_knn_null_returns_nonnegative_and_p_in_range():
    torch.manual_seed(1)
    X = torch.randn(20, 6)
    Y = torch.randn(20, 6)
    res = sg_knn(X, Y, k=5, num_permutations=20, quantile=0.9)
    assert res.gated >= 0.0
    assert 0.0 <= res.pvalue <= 1.0
    assert len(res.null_samples) == 20


def test_sg_knn_identical_has_positive_gated_and_low_p():
    torch.manual_seed(2)
    X = torch.randn(24, 5)
    res = sg_knn(X, X, k=5, num_permutations=30, quantile=0.9)
    assert np.isclose(res.raw, 1.0)
    assert res.gated > 0.0
    assert res.pvalue <= 1.0 / (30 + 1.0)


def test_sg_rsa_null_behaves():
    torch.manual_seed(3)
    X = torch.randn(18, 5)
    Y = torch.randn(18, 5)
    res = sg_rsa(X, Y, num_permutations=20, quantile=0.9)
    assert res.gated >= 0.0
    assert 0.0 <= res.pvalue <= 1.0
    assert len(res.null_samples) == 20


def test_sg_rsa_identical_is_detected():
    torch.manual_seed(4)
    X = torch.randn(16, 4)
    res = sg_rsa(X, X, num_permutations=30, quantile=0.9)
    assert res.raw > 0.99
    assert res.gated > 0.0
    assert res.pvalue <= 1.0 / (30 + 1.0)


def test_sg_rsa_batch_matches_default():
    torch.manual_seed(5)
    X = torch.randn(20, 6)
    Y = torch.randn(20, 6)
    rng_state = torch.random.get_rng_state()
    res_default = sg_rsa(X, Y, num_permutations=12, quantile=0.9)
    torch.random.set_rng_state(rng_state)
    res_batch = sg_rsa(X, Y, num_permutations=12, quantile=0.9, batch_size=4)
    assert np.isclose(res_default.raw, res_batch.raw)
    assert np.isclose(res_default.tau, res_batch.tau)
    assert np.isclose(res_default.gated, res_batch.gated)
    assert np.isclose(res_default.pvalue, res_batch.pvalue)


def test_rsa_vector_matches_cdist_upper_triangle():
    torch.manual_seed(7)
    X = torch.randn(12, 4)
    dist = torch.cdist(X, X, p=2)
    idx = torch.triu_indices(dist.shape[0], dist.shape[0], offset=1)
    expected = dist[idx[0], idx[1]].flatten()
    got = rsa_vector(X)
    assert torch.allclose(expected, got)


def test_sg_rsa_auto_pair_samples_matches_explicit():
    torch.manual_seed(8)
    X = torch.randn(256, 16)
    Y = torch.randn(256, 16)
    total_pairs = (X.shape[0] * (X.shape[0] - 1)) // 2
    auto_samples = _rsa_auto_pair_samples(total_pairs)
    assert auto_samples is not None
    assert auto_samples < total_pairs

    perms = torch.stack([torch.randperm(X.shape[0]) for _ in range(12)])
    rng_state = torch.random.get_rng_state()
    res_auto = sg_rsa(
        X,
        Y,
        num_permutations=12,
        quantile=0.9,
        perms=perms,
    )
    torch.random.set_rng_state(rng_state)
    res_explicit = sg_rsa(
        X,
        Y,
        num_permutations=12,
        quantile=0.9,
        perms=perms,
        pair_samples=auto_samples,
    )
    assert np.isclose(res_auto.raw, res_explicit.raw)
    assert np.isclose(res_auto.tau, res_explicit.tau)
    assert np.isclose(res_auto.gated, res_explicit.gated)


def test_sg_knn_respects_perms_and_null_matches_manual():
    torch.manual_seed(9)
    X = torch.randn(6, 4)
    Y = torch.randn(6, 4)
    perms = torch.stack(
        [
            torch.arange(6),
            torch.tensor([1, 2, 3, 4, 5, 0]),
            torch.tensor([2, 3, 4, 5, 0, 1]),
            torch.tensor([5, 4, 3, 2, 1, 0]),
        ]
    )
    k = 2
    raw = mutual_knn_overlap(X, Y, k=k)
    null_scores = [mutual_knn_overlap(X, Y[p], k=k) for p in perms]
    res = sg_knn(
        X,
        Y,
        k=k,
        num_permutations=perms.shape[0],
        quantile=0.75,
        perms=perms,
    )
    assert np.isclose(res.raw, raw)
    assert np.allclose(np.asarray(res.null_samples), np.asarray(null_scores))
    # tau is computed from null_scores + observed value
    expected_tau = float(np.quantile(null_scores + [raw], 0.75))
    assert np.isclose(res.tau, expected_tau)
    expected_p = (sum(s >= raw for s in null_scores) + 1.0) / (len(null_scores) + 1.0)
    assert np.isclose(res.pvalue, expected_p)


def test_sg_rsa_respects_perms_and_null_matches_manual():
    torch.manual_seed(10)
    X = torch.randn(6, 3)
    Y = torch.randn(6, 3)
    perms = torch.stack(
        [
            torch.arange(6),
            torch.tensor([1, 2, 3, 4, 5, 0]),
            torch.tensor([2, 3, 4, 5, 0, 1]),
        ]
    )
    raw = spearman_corr(rsa_vector(X).cpu().numpy(), rsa_vector(Y).cpu().numpy())
    null_scores = []
    for p in perms:
        y_perm = Y[p]
        null_scores.append(
            spearman_corr(rsa_vector(X).cpu().numpy(), rsa_vector(y_perm).cpu().numpy())
        )
    res = sg_rsa(
        X,
        Y,
        num_permutations=perms.shape[0],
        quantile=0.67,
        batch_size=None,
        perms=perms,
    )
    assert np.isclose(res.raw, raw)
    assert np.allclose(np.asarray(res.null_samples), np.asarray(null_scores))
    # tau is the exact permutation cutoff (order statistic) including the observed value
    from aristotelian.metrics.aggregation import tau_order_statistic

    expected_tau = tau_order_statistic(null_scores, 0.67, obs=raw)
    assert np.isclose(res.tau, expected_tau)
    expected_p = (sum(s >= raw for s in null_scores) + 1.0) / (len(null_scores) + 1.0)
    assert np.isclose(res.pvalue, expected_p)


def test_mutual_knn_overlap_validates_inputs():
    torch.manual_seed(11)
    X = torch.randn(5, 3)
    Y = torch.randn(6, 3)
    with pytest.raises(ValueError, match="same number of samples"):
        mutual_knn_overlap(X, Y, k=2)
    with pytest.raises(ValueError, match="0 < k < n"):
        mutual_knn_overlap(X, X, k=0)
    with pytest.raises(ValueError, match="0 < k < n"):
        mutual_knn_overlap(X, X, k=5)


def test_sg_knn_validates_inputs():
    torch.manual_seed(12)
    X = torch.randn(5, 3)
    with pytest.raises(ValueError, match="0 < k < n"):
        sg_knn(X, X, k=0, num_permutations=5)
    with pytest.raises(ValueError, match="0 < k < n"):
        sg_knn(X, X, k=5, num_permutations=5)
    bad_perms = torch.arange(5)
    with pytest.raises(ValueError, match="perms must have shape"):
        sg_knn(X, X, k=2, num_permutations=5, perms=bad_perms)


def test_sg_rsa_validates_inputs():
    torch.manual_seed(13)
    X = torch.randn(5, 3)
    with pytest.raises(ValueError, match="pair_samples must be a positive"):
        sg_rsa(X, X, num_permutations=5, pair_samples=0)
    with pytest.raises(ValueError, match="batch_size must be a positive"):
        sg_rsa(X, X, num_permutations=5, batch_size=0)
    bad_perms = torch.arange(5)
    with pytest.raises(ValueError, match="perms must have shape"):
        sg_rsa(X, X, num_permutations=5, perms=bad_perms)


def test_sg_knn_all_perms_exact_pvalue():
    torch.manual_seed(20)
    n = 5
    X = torch.randn(n, 3)
    Y = torch.randn(n, 3)
    perms = torch.tensor(list(itertools.permutations(range(n))))
    res = sg_knn(
        X,
        Y,
        k=2,
        num_permutations=perms.shape[0],
        quantile=0.8,
        perms=perms,
    )
    null_scores = [mutual_knn_overlap(X, Y[p], k=2) for p in perms]
    expected_p = (sum(s >= res.raw for s in null_scores) + 1.0) / (
        len(null_scores) + 1.0
    )
    assert np.isclose(res.pvalue, expected_p)


def test_sg_rsa_pair_samples_matches_manual():
    torch.manual_seed(21)
    n = 6
    X = torch.randn(n, 4)
    Y = torch.randn(n, 4)
    perms = torch.stack(
        [
            torch.arange(n),
            torch.tensor([1, 2, 3, 4, 5, 0]),
            torch.tensor([2, 3, 4, 5, 0, 1]),
        ]
    )
    pair_samples = 10
    rng_state = torch.random.get_rng_state()
    res = sg_rsa(
        X,
        Y,
        num_permutations=perms.shape[0],
        quantile=0.7,
        batch_size=None,
        pair_samples=pair_samples,
        perms=perms,
    )
    torch.random.set_rng_state(rng_state)
    total_pairs = (n * (n - 1)) // 2
    sample_idx = torch.randperm(total_pairs)[:pair_samples]
    idx0, idx1 = torch.triu_indices(n, n, offset=1)
    idx0 = idx0[sample_idx]
    idx1 = idx1[sample_idx]
    vx = torch.norm(X[idx0] - X[idx1], dim=1).cpu().numpy()
    vy = torch.norm(Y[idx0] - Y[idx1], dim=1).cpu().numpy()
    raw = spearman_corr(vx, vy)
    null_scores = []
    for p in perms:
        y0 = Y[p][idx0]
        y1 = Y[p][idx1]
        vy_perm = torch.norm(y0 - y1, dim=1).cpu().numpy()
        null_scores.append(spearman_corr(vx, vy_perm))
    assert np.isclose(res.raw, raw)
    assert np.allclose(np.asarray(res.null_samples), np.asarray(null_scores))


def test_sg_knn_quantile_one_clamps_tail_strength():
    torch.manual_seed(22)
    X = torch.randn(8, 3)
    Y = torch.randn(8, 3)
    res = sg_knn(X, Y, k=3, num_permutations=10, quantile=1.0)
    assert res.tail_strength == 0.0


def test_sg_rsa_quantile_one_clamps_tail_strength():
    torch.manual_seed(23)
    X = torch.randn(8, 3)
    Y = torch.randn(8, 3)
    res = sg_rsa(X, Y, num_permutations=10, quantile=1.0)
    assert res.tail_strength == 0.0


def test_sg_knn_rejects_invalid_quantile():
    torch.manual_seed(24)
    X = torch.randn(6, 3)
    with pytest.raises(ValueError, match="quantile must be in"):
        sg_knn(X, X, k=2, num_permutations=5, quantile=1.1)


def test_sg_rsa_rejects_invalid_quantile():
    torch.manual_seed(25)
    X = torch.randn(6, 3)
    with pytest.raises(ValueError, match="quantile must be in"):
        sg_rsa(X, X, num_permutations=5, quantile=-0.1)


def test_sg_rsa_isometry_invariance():
    torch.manual_seed(26)
    X = torch.randn(10, 4)
    q, _ = torch.linalg.qr(torch.randn(4, 4))
    Y = X @ q
    res = sg_rsa(X, Y, num_permutations=10, quantile=0.9)
    assert res.raw > 0.99


def test_sg_knn_superuniform_under_null_all_perms():
    torch.manual_seed(27)
    n = 5
    X = torch.randn(n, 3)
    Y = torch.randn(n, 3)
    perms = torch.tensor(list(itertools.permutations(range(n))))
    alpha = 0.2
    pvals = []
    for p in perms:
        res = sg_knn(
            X,
            Y[p],
            k=2,
            num_permutations=perms.shape[0],
            quantile=1.0 - alpha,
            perms=perms,
        )
        pvals.append(res.pvalue)
    rate = float(np.mean([p <= alpha for p in pvals]))
    assert rate <= alpha


def test_sg_rsa_superuniform_under_null_all_perms():
    torch.manual_seed(28)
    n = 5
    X = torch.randn(n, 3)
    Y = torch.randn(n, 3)
    perms = torch.tensor(list(itertools.permutations(range(n))))
    alpha = 0.2
    pvals = []
    for p in perms:
        res = sg_rsa(
            X,
            Y[p],
            num_permutations=perms.shape[0],
            quantile=1.0 - alpha,
            batch_size=None,
            perms=perms,
        )
        pvals.append(res.pvalue)
    rate = float(np.mean([p <= alpha for p in pvals]))
    assert rate <= alpha


# =============================================================================
# Regression tests for fixed bugs
# =============================================================================


def test_rv_coefficient_self_similarity_is_one():
    """RV(X, X) must equal 1.0 (regression: denominator used trace of
    element-wise square instead of Frobenius norm squared)."""
    from aristotelian.metrics import MetricRegistry

    torch.manual_seed(42)
    X = torch.randn(50, 20)
    rv = MetricRegistry.compute_raw("rv_coefficient", X, X)
    assert np.isclose(rv, 1.0, atol=1e-5), f"RV(X, X) = {rv}, expected 1.0"


def test_rv_coefficient_bounded():
    """RV coefficient must be in [0, 1] for independent data."""
    from aristotelian.metrics import MetricRegistry

    torch.manual_seed(43)
    X = torch.randn(50, 20)
    Y = torch.randn(50, 20)
    rv = MetricRegistry.compute_raw("rv_coefficient", X, Y)
    assert 0.0 <= rv <= 1.0 + 1e-6, f"RV(X, Y) = {rv}, expected in [0, 1]"


def test_rv_coefficient_null_uses_correct_denominator():
    """RV null distribution scores must also be in [0, 1] (regression:
    same denominator bug affected null computation)."""
    from aristotelian.metrics import MetricConfig, MetricRegistry

    torch.manual_seed(44)
    X = torch.randn(30, 10)
    Y = torch.randn(30, 10)
    config = MetricConfig(calibrate=True, num_permutations=20, quantile=0.95)
    result = MetricRegistry.compute("rv_coefficient", X, Y, config)
    null_arr = np.asarray(result.null_samples, dtype=float)
    assert np.all(null_arr >= -1e-6), f"Null scores below 0: {null_arr.min()}"
    assert np.all(null_arr <= 1.0 + 1e-6), f"Null scores above 1: {null_arr.max()}"
