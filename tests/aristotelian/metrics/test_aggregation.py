import numpy as np
import pytest
import torch

from aristotelian.metrics.aggregation import (
    SimpleMetric,
    agg_colmax_mean,
    agg_max,
    agg_rowmax_mean,
    agg_topk_mean,
    bootstrap_statistic,
    compute_null_summary,
    compute_similarity_matrix,
    gated_rescaled,
    permutation_null_aggregated,
)


def _toy_metric() -> SimpleMetric:
    return SimpleMetric(
        name="cosine",
        max_value=1.0,
        compute=lambda X, Y: float(
            torch.nn.functional.cosine_similarity(
                X.flatten(), Y.flatten(), dim=0
            ).item()
        ),
    )


def test_compute_similarity_matrix_shapes_and_values():
    metric = _toy_metric()
    X1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    X2 = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    Y1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    Y2 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    S = compute_similarity_matrix([X1, X2], [Y1, Y2], metric)
    assert S.shape == (2, 2)
    assert S[0, 0] > S[0, 1]


def test_aggregators_match_expected_values():
    S = torch.tensor([[0.1, 0.9], [0.4, 0.2]])
    assert np.isclose(agg_max(S).value, 0.9)
    assert np.isclose(agg_rowmax_mean(S).value, (0.9 + 0.4) / 2.0)
    assert np.isclose(agg_colmax_mean(S).value, (0.4 + 0.9) / 2.0)
    assert np.isclose(agg_topk_mean(S, k=2).value, (0.9 + 0.4) / 2.0)


def test_permutation_null_and_summary():
    metric = _toy_metric()
    X = torch.randn(12, 4)
    Y = torch.randn(12, 4)
    repsA = [X]
    repsB = [Y]
    null_samples = permutation_null_aggregated(
        repsA,
        repsB,
        metric,
        agg_max,
        num_permutations=15,
        seed=0,
    )
    assert len(null_samples) == 15
    summary = compute_null_summary(
        null_samples, T_obs=float(null_samples[0]), alpha=0.1
    )
    assert 0.0 <= summary["p_value"] <= 1.0


def test_gated_rescaled_behaves_at_bounds():
    assert gated_rescaled(0.2, tau_alpha=0.3, s_max=1.0) == 0.0
    assert np.isclose(gated_rescaled(1.0, tau_alpha=0.3, s_max=1.0), 1.0)
    assert np.isclose(gated_rescaled(0.5, tau_alpha=0.3, s_max=None), 0.2)
    assert np.isclose(gated_rescaled(1.1, tau_alpha=1.0, s_max=1.0), 1.0)
    assert np.isclose(gated_rescaled(1.1, tau_alpha=1.0, s_max=0.5), 1.0)


def test_bootstrap_statistic_has_ci_bounds():
    metric = _toy_metric()
    X = torch.randn(10, 3)
    Y = torch.randn(10, 3)
    repsA = [X]
    repsB = [Y]
    out = bootstrap_statistic(
        repsA,
        repsB,
        metric,
        agg_max,
        num_bootstrap=20,
        seed=123,
    )
    assert len(out.samples) == 20
    assert out.ci_low <= out.ci_high


def test_compute_null_summary_alpha_zero_clamps_tail_strength():
    null_samples = [0.1, 0.2, 0.3]
    summary = compute_null_summary(null_samples, T_obs=0.2, alpha=0.0)
    assert summary["tail_strength"] == 0.0


def test_compute_null_summary_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="alpha must be in"):
        compute_null_summary([0.1, 0.2], T_obs=0.2, alpha=-0.1)
