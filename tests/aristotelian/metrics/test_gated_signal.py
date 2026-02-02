import torch

from aristotelian import sg_cka_linear, sg_knn


def _make_signal_pair(
    seed: int, *, n: int = 128, d: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    x = torch.randn(n, d, generator=rng)
    y = x + 0.01 * torch.randn(n, d, generator=rng)
    return x, y


def _gated_positive_rate(fn, *, trials: int = 5, **kwargs) -> float:
    positives = 0
    for t in range(trials):
        x, y = _make_signal_pair(1000 + t)
        res = fn(x, y, **kwargs)
        positives += int(res.gated > 0.0)
    return positives / float(trials)


def test_gated_preserves_strong_signal_cka() -> None:
    rate = _gated_positive_rate(
        sg_cka_linear,
        trials=5,
        num_permutations=50,
        quantile=0.95,
        device="cpu",
    )
    assert rate >= 0.8


def test_gated_preserves_strong_signal_knn() -> None:
    rate = _gated_positive_rate(
        sg_knn,
        trials=5,
        num_permutations=50,
        quantile=0.95,
        device="cpu",
        k=10,
    )
    assert rate >= 0.8
