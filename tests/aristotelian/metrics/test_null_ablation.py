import torch

from aristotelian.metrics.baselines import run_null_type_ablation


def test_null_type_ablation_shapes():
    torch.manual_seed(0)
    X = torch.randn(20, 6)
    Y = torch.randn(20, 6)
    labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    out = run_null_type_ablation(
        X,
        Y,
        metric="sgcka_lin",
        null_types=("sample", "feature", "within_class"),
        num_permutations=5,
        quantile=0.9,
        labels=labels,
        seed=123,
    )
    assert set(out.keys()) == {"sample", "feature", "within_class"}
    for entry in out.values():
        assert hasattr(entry, "tau")
