import math

import numpy as np
import torch

from aristotelian import sg_cka_kernel, sg_cka_linear, sg_knn, sg_rsa
from aristotelian.metrics.aggregation import gated_rescaled


def _pvalue_from_null(null_samples, obs):
    arr = np.asarray(null_samples, dtype=float)
    return (float(np.sum(arr >= obs)) + 1.0) / (len(arr) + 1.0)


def test_sgknn_pvalue_matches_gate_statistic():
    torch.manual_seed(0)
    X = torch.randn(20, 6)
    Y = torch.randn(20, 6)
    res = sg_knn(X, Y, k=4, num_permutations=20, quantile=0.9)

    # tau is computed from null_samples + observed value
    expected_tau = float(np.quantile(list(res.null_samples) + [res.raw], 0.9))
    assert math.isclose(res.tau, expected_tau, rel_tol=1e-6)
    expected_gated = gated_rescaled(res.raw, tau_alpha=res.tau, s_max=1.0)
    assert math.isclose(res.gated, expected_gated, rel_tol=1e-6)

    expected_p = _pvalue_from_null(res.null_samples, res.raw)
    assert math.isclose(res.pvalue, expected_p, rel_tol=1e-6)


def test_sgrsa_pvalue_matches_gate_statistic():
    torch.manual_seed(1)
    X = torch.randn(18, 5)
    Y = torch.randn(18, 5)
    res = sg_rsa(X, Y, num_permutations=20, quantile=0.9)

    # tau is computed from null_samples + observed value
    expected_tau = float(np.quantile(list(res.null_samples) + [res.raw], 0.9))
    assert math.isclose(res.tau, expected_tau, rel_tol=1e-6)
    expected_gated = gated_rescaled(res.raw, tau_alpha=res.tau, s_max=1.0)
    assert math.isclose(res.gated, expected_gated, rel_tol=1e-6)

    expected_p = _pvalue_from_null(res.null_samples, res.raw)
    assert math.isclose(res.pvalue, expected_p, rel_tol=1e-6)


def test_sgcka_linear_gate_stat_and_pvalues():
    seed = 2
    torch.manual_seed(seed)
    X = torch.randn(16, 5)
    Y = torch.randn(16, 5)
    res = sg_cka_linear(X, Y, num_permutations=20, quantile=0.9)

    # tau is computed from null_samples + observed value
    expected_tau = float(np.quantile(list(res.null_samples) + [res.raw], 0.9))
    assert math.isclose(res.tau, expected_tau, rel_tol=1e-6)
    expected_gated = gated_rescaled(res.raw, tau_alpha=res.tau, s_max=1.0)
    assert math.isclose(res.gated, expected_gated, rel_tol=1e-6)

    expected_p = _pvalue_from_null(res.null_samples, res.raw)
    assert math.isclose(res.pvalue, expected_p, rel_tol=1e-6)


def test_sgcka_kernel_gate_stat_and_pvalues():
    seed = 3
    torch.manual_seed(seed)
    X = torch.randn(14, 4)
    Y = torch.randn(14, 4)
    res = sg_cka_kernel(X, Y, num_permutations=20, quantile=0.9)

    # tau is computed from null_samples + observed value
    expected_tau = float(np.quantile(list(res.null_samples) + [res.raw], 0.9))
    assert math.isclose(res.tau, expected_tau, rel_tol=1e-6)
    expected_gated = gated_rescaled(res.raw, tau_alpha=res.tau, s_max=1.0)
    assert math.isclose(res.gated, expected_gated, rel_tol=1e-6)

    expected_p = _pvalue_from_null(res.null_samples, res.raw)
    assert math.isclose(res.pvalue, expected_p, rel_tol=1e-6)


def test_gate_coherence():
    torch.manual_seed(5)
    X = torch.randn(18, 6)
    Y = torch.randn(18, 6)
    quantile = 0.9
    alpha = 1.0 - quantile
    res = sg_knn(X, Y, k=4, num_permutations=30, quantile=quantile)
    if res.pvalue > alpha:
        assert res.gated == 0.0
    if res.gated > 0.0:
        assert res.pvalue <= alpha + 1.0 / (30 + 1.0)
