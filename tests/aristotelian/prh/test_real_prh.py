import torch

from aristotelian.models.real_models import (
    run_seed_reproducibility,
    run_width_depth_scaling,
)
from aristotelian.prh.prh import run_prh_alignment


def test_run_seed_reproducibility_shapes():
    torch.manual_seed(0)
    acts = {
        "seed1": [torch.randn(40, 16), torch.randn(40, 16)],
        "seed2": [torch.randn(40, 16), torch.randn(40, 16)],
    }
    res = run_seed_reproducibility(
        activations=acts,
        metric="sgknn",
        num_permutations=10,
        quantile=0.9,
    )
    assert any(k.startswith("seed1__") for k in res.keys())
    assert any(k.endswith("__seed2") or k.endswith("__seed1") for k in res.keys())


def test_run_width_depth_scaling_shapes():
    torch.manual_seed(1)
    acts = {
        "small": [torch.randn(30, 12), torch.randn(30, 12)],
        "large": [torch.randn(30, 12), torch.randn(30, 12)],
    }
    res = run_width_depth_scaling(
        activations=acts,
        metric="sgcka_lin",
        num_permutations=10,
        quantile=0.9,
    )
    assert any(k.startswith("small__") for k in res.keys())
    assert any(k.endswith("__large") or k.endswith("__small") for k in res.keys())


def test_run_prh_alignment_shapes():
    torch.manual_seed(2)
    acts = {
        "modelA": [torch.randn(50, 10), torch.randn(50, 10)],
        "modelB": [torch.randn(50, 10), torch.randn(50, 10)],
    }
    res = run_prh_alignment(
        activations=acts,
        metric="sgknn",
        num_permutations=10,
        quantile=0.9,
        alpha=0.1,
    )
    assert "pvalues" in res
    assert "fdr_mask" in res
