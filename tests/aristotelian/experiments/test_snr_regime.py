import numpy as np
import torch

from scripts.experiments.generators.low_rank import make_low_rank_signal_unitvar


def _flat_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x_np = x.detach().cpu().numpy().reshape(-1)
    y_np = y.detach().cpu().numpy().reshape(-1)
    return float(np.corrcoef(x_np, y_np)[0, 1])


def test_low_rank_unitvar_variance():
    torch.manual_seed(0)
    X, Y = make_low_rank_signal_unitvar(
        n=256, d=128, rank=4, signal_strength=1.0, noise_std=0.0, device="cpu"
    )
    assert torch.allclose(X, Y)
    var = float(X.var().item())
    assert 0.7 < var < 1.3


def test_low_rank_unitvar_noise_monotonic():
    corrs = []
    for sigma in [0.0, 0.5, 1.0]:
        torch.manual_seed(123)
        X, Y = make_low_rank_signal_unitvar(
            n=256, d=128, rank=4, signal_strength=1.0, noise_std=sigma, device="cpu"
        )
        corrs.append(_flat_corr(X, Y))
    assert corrs[0] > corrs[1] > corrs[2]
