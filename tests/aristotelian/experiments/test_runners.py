import numpy as np
import pytest
import torch

from aristotelian.experiments.experiments import (
    run_permutation_budget,
    run_type1_calibration,
)
from scripts.experiments.sections.aggregation import (
    _aggregate_nulls_from_matrices,
    _cka_metric,
)


def test_run_permutation_budget_shapes():
    torch.manual_seed(0)
    out = run_permutation_budget(
        metric="sgcka_lin",
        n=20,
        d=6,
        budgets=(5, 10),
        num_trials=3,
        quantile=0.9,
        null_type="gaussian",
        seed=123,
    )
    assert set(out.keys()) == {5, 10}
    for summary in out.values():
        assert summary.mean_null is not None
        assert summary.std_null is not None
        assert 0.0 <= summary.zeros_fraction <= 1.0


def test_run_permutation_budget_rsa_batch():
    torch.manual_seed(3)
    out = run_permutation_budget(
        metric="sgrsa",
        n=16,
        d=5,
        budgets=(5,),
        num_trials=2,
        quantile=0.9,
        null_type="gaussian",
        seed=5,
        rsa_batch_size=4,
    )
    assert set(out.keys()) == {5}


def test_aggregate_nulls_from_matrices_matches_direct():
    torch.manual_seed(4)
    n, d, num_permutations = 12, 5, 6
    repsA = [torch.randn(n, d), torch.randn(n, d)]
    repsB = [torch.randn(n, d), torch.randn(n, d)]
    metric = _cka_metric()
    S_obs = None
    null_mats = []
    for _ in range(num_permutations):
        perm = torch.randperm(n)
        repsB_perm = [Y[perm] for Y in repsB]
        S = torch.empty((len(repsA), len(repsB)))
        for i, Xa in enumerate(repsA):
            for j, Xb in enumerate(repsB_perm):
                S[i, j] = metric.compute(Xa, Xb)
        null_mats.append(S)
        if S_obs is None:
            S_obs = S

    def agg_fn(S):
        return float(S.max().item())

    direct = [agg_fn(S) for S in null_mats]
    from_mats = _aggregate_nulls_from_matrices(null_mats, agg_fn)
    assert np.allclose(direct, from_mats)


def test_run_type1_calibration_shapes():
    torch.manual_seed(1)
    res = run_type1_calibration(
        metric="sgknn",
        n=24,
        d=5,
        num_trials=10,
        num_permutations=15,
        quantile=0.9,
        null_type="gaussian",
        seed=7,
    )
    assert 0.0 <= res.type1_rate <= 1.0
    assert len(res.positives) == 10


def test_run_type1_calibration_parallel_matches_serial():
    torch.manual_seed(2)
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


@pytest.mark.parametrize("null_type", ["heavy", "shuffled"])
def test_run_permutation_budget_null_types_smoke(null_type):
    torch.manual_seed(0)
    out = run_permutation_budget(
        metric="sgcka_lin",
        n=12,
        d=4,
        budgets=(3,),
        num_trials=2,
        quantile=0.9,
        null_type=null_type,
        seed=19,
    )
    assert set(out.keys()) == {3}


@pytest.mark.parametrize("null_type", ["heavy", "shuffled"])
def test_run_type1_calibration_null_types_smoke(null_type):
    torch.manual_seed(0)
    res = run_type1_calibration(
        metric="sgknn",
        n=12,
        d=4,
        num_trials=4,
        num_permutations=6,
        quantile=0.9,
        null_type=null_type,
        seed=23,
    )
    assert len(res.positives) == 4
