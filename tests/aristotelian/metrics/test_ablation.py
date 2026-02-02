import torch

from aristotelian.metrics.ablation import run_preprocess_ablation


def test_preprocess_ablation_shapes():
    torch.manual_seed(0)
    X = torch.randn(30, 8)
    Y = torch.randn(30, 8)
    res = run_preprocess_ablation(
        X,
        Y,
        metric="sgcka_lin",
        preprocess_options=("none", "center", "whiten"),
        num_permutations=10,
        quantile=0.9,
    )
    assert set(res.keys()) == {"none", "center", "whiten"}
    for entry in res.values():
        assert "raw" in entry
        assert "gated" in entry
