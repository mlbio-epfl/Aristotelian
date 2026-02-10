import itertools

import numpy as np
import pytest

from aristotelian.metrics.aggregation import gated_rescaled
from aristotelian.metrics.calibration import compute_null_variants
from aristotelian.metrics.cca import (
    cca_mean,
    pwcca_mean,
    rv_coefficient,
    sg_cca_mean,
    sg_cca_multiq,
    sg_pwcca_mean,
    sg_pwcca_multiq,
    sg_rv_coefficient,
    sg_svcca_mean,
    sg_svcca_multiq,
    svcca_mean,
)
from aristotelian.metrics.other_metrics import procrustes_score, sg_procrustes_score


def _make_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((8, 5))
    Y = rng.standard_normal((8, 5))
    return X, Y


def _make_perms(n: int) -> np.ndarray:
    base = np.arange(n)
    return np.stack(
        [
            base,
            np.roll(base, 1),
            np.roll(base, 2),
            np.roll(base, 3),
        ],
        axis=0,
    )


def _assert_sg_matches_expected(
    *,
    metric_fn,
    sg_fn,
    min_score: float,
    max_score: float,
):
    X, Y = _make_data()
    perms = _make_perms(X.shape[0])
    quantile = 0.5
    raw = metric_fn(X, Y)
    null_samples = [metric_fn(X, Y[p]) for p in perms]
    # tau is computed from null_samples + observed value
    tau = float(np.quantile(null_samples + [raw], quantile))
    gated = gated_rescaled(raw, tau_alpha=tau, s_max=1.0)
    pvalue = (sum(s >= raw for s in null_samples) + 1.0) / (len(null_samples) + 1.0)
    alpha = 1.0 - quantile
    if alpha <= 0:
        tail_strength = 0.0
    else:
        tail_strength = float(max(0.0, min(1.0, (alpha - pvalue) / alpha)))
    variants = compute_null_variants(
        raw, null_samples, min_score=min_score, max_score=max_score
    )

    res = sg_fn(X, Y, num_permutations=perms.shape[0], quantile=quantile, perms=perms)
    assert np.isclose(res.raw, raw)
    assert np.isclose(res.tau, tau)
    assert np.isclose(res.gated, gated)
    assert np.isclose(res.pvalue, pvalue)
    assert np.isclose(res.tail_strength, tail_strength)
    assert np.allclose(np.asarray(res.null_samples, dtype=float), null_samples)
    assert np.isclose(res.mean_null, variants.mean_null)
    assert np.isclose(res.median_null, variants.median_null)
    assert np.isclose(res.std_null, variants.std_null)
    assert np.isclose(res.null_centered, variants.null_centered)
    assert np.isclose(res.z, variants.z)
    assert np.isclose(res.ari, variants.ari)


def _assert_multiq_matches_single(*, sg_fn, multiq_fn):
    X, Y = _make_data()
    perms = _make_perms(X.shape[0])
    quantiles = [0.5, 0.8]
    res_multi = multiq_fn(
        X,
        Y,
        num_permutations=perms.shape[0],
        quantiles=quantiles,
        perms=perms,
    )
    res_single = sg_fn(
        X,
        Y,
        num_permutations=perms.shape[0],
        quantile=quantiles[0],
        perms=perms,
    )
    assert np.isclose(res_multi["raw"], res_single.raw)
    assert np.isclose(res_multi["p_value"], res_single.pvalue)
    assert np.allclose(
        np.asarray(res_multi["null_samples"], dtype=float),
        np.asarray(res_single.null_samples, dtype=float),
    )
    assert np.isclose(res_multi["variants"].mean_null, res_single.mean_null)
    assert np.isclose(res_multi["variants"].median_null, res_single.median_null)
    assert np.isclose(res_multi["variants"].std_null, res_single.std_null)
    assert np.isclose(res_multi["variants"].null_centered, res_single.null_centered)
    assert np.isclose(res_multi["variants"].z, res_single.z)
    assert np.isclose(res_multi["variants"].ari, res_single.ari)

    for q in quantiles:
        res_q = sg_fn(
            X,
            Y,
            num_permutations=perms.shape[0],
            quantile=q,
            perms=perms,
        )
        assert np.isclose(res_multi["tau"][q], res_q.tau)
        assert np.isclose(res_multi["gated"][q], res_q.gated)
        assert np.isclose(res_multi["tail_strength"][q], res_q.tail_strength)


def test_sg_cca_matches_null_summary():
    _assert_sg_matches_expected(
        metric_fn=cca_mean,
        sg_fn=sg_cca_mean,
        min_score=0.0,
        max_score=1.0,
    )


def test_sg_svcca_matches_null_summary():
    _assert_sg_matches_expected(
        metric_fn=svcca_mean,
        sg_fn=sg_svcca_mean,
        min_score=0.0,
        max_score=1.0,
    )


def test_sg_pwcca_matches_null_summary():
    _assert_sg_matches_expected(
        metric_fn=pwcca_mean,
        sg_fn=sg_pwcca_mean,
        min_score=0.0,
        max_score=1.0,
    )


def test_sg_rv_matches_null_summary():
    _assert_sg_matches_expected(
        metric_fn=rv_coefficient,
        sg_fn=sg_rv_coefficient,
        min_score=0.0,
        max_score=1.0,
    )


def test_sg_procrustes_matches_null_summary():
    _assert_sg_matches_expected(
        metric_fn=procrustes_score,
        sg_fn=sg_procrustes_score,
        min_score=-1.0,
        max_score=1.0,
    )


def test_sg_cca_multiq_matches_single():
    _assert_multiq_matches_single(sg_fn=sg_cca_mean, multiq_fn=sg_cca_multiq)


def test_sg_svcca_multiq_matches_single():
    _assert_multiq_matches_single(sg_fn=sg_svcca_mean, multiq_fn=sg_svcca_multiq)


def test_sg_pwcca_multiq_matches_single():
    _assert_multiq_matches_single(sg_fn=sg_pwcca_mean, multiq_fn=sg_pwcca_multiq)


def test_sg_extra_metrics_reject_invalid_quantile():
    X, Y = _make_data()
    with pytest.raises(ValueError, match="quantile must be in"):
        sg_cca_mean(X, Y, num_permutations=5, quantile=1.2)
    with pytest.raises(ValueError, match="quantiles must be in"):
        sg_cca_multiq(X, Y, num_permutations=5, quantiles=[0.5, -0.1])


def test_sg_cca_superuniform_under_null_all_perms():
    rng = np.random.default_rng(123)
    n = 5
    X = rng.standard_normal((n, 3))
    Y = rng.standard_normal((n, 3))
    perms = np.array(list(itertools.permutations(range(n))))
    alpha = 0.2
    pvals = []
    for p in perms:
        res = sg_cca_mean(
            X,
            Y[p],
            num_permutations=perms.shape[0],
            quantile=1.0 - alpha,
            perms=perms,
        )
        pvals.append(res.pvalue)
    rate = float(np.mean([p <= alpha for p in pvals]))
    assert rate <= alpha


def test_sg_extra_metrics_superuniform_under_null_all_perms():
    rng = np.random.default_rng(456)
    n = 5
    X = rng.standard_normal((n, 4))
    Y = rng.standard_normal((n, 4))
    perms = np.array(list(itertools.permutations(range(n))))
    alpha = 0.2
    metrics = [
        sg_svcca_mean,
        sg_pwcca_mean,
        sg_rv_coefficient,
        sg_procrustes_score,
    ]
    for metric in metrics:
        pvals = []
        for p in perms:
            res = metric(
                X,
                Y[p],
                num_permutations=perms.shape[0],
                quantile=1.0 - alpha,
                perms=perms,
            )
            pvals.append(res.pvalue)
        rate = float(np.mean([p <= alpha for p in pvals]))
        assert rate <= alpha
