import pytest
import torch

from aristotelian.metrics.baselines import run_null_baselines


def test_run_null_baselines_shapes():
    torch.manual_seed(0)
    out = run_null_baselines(
        n=20,
        d=6,
        num_trials=3,
        num_permutations=10,
        quantiles=(0.9, 0.95),
        null_type="gaussian",
        k_knn=5,
        device="cpu",
        seed=123,
    )
    assert set(out.keys()) == {"sgcka_lin", "sgcka_rbf", "sgknn", "sgrsa"}
    for metric_dict in out.values():
        assert set(metric_dict.keys()) == {0.9, 0.95}
        for v in metric_dict.values():
            assert v.mean >= 0.0
            assert 0.0 <= v.zeros_fraction <= 1.0


def test_run_null_baselines_feature_permute():
    torch.manual_seed(0)
    out = run_null_baselines(
        n=16,
        d=6,
        num_trials=2,
        num_permutations=5,
        quantiles=(0.9,),
        null_type="feature_permute",
        device="cpu",
    )
    assert "sgcka_lin" in out


def test_run_null_baselines_within_class():
    torch.manual_seed(0)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    out = run_null_baselines(
        n=8,
        d=4,
        num_trials=2,
        num_permutations=5,
        quantiles=(0.9,),
        null_type="within_class",
        k_knn=3,
        labels=labels,
        device="cpu",
    )
    assert "sgcka_lin" in out


@pytest.mark.parametrize("null_type", ["heavy", "shuffled"])
def test_run_null_baselines_other_nulls_smoke(null_type):
    torch.manual_seed(0)
    out = run_null_baselines(
        n=12,
        d=4,
        num_trials=2,
        num_permutations=5,
        quantiles=(0.9,),
        null_type=null_type,
        k_knn=3,
        device="cpu",
        seed=11,
    )
    assert "sgcka_lin" in out
