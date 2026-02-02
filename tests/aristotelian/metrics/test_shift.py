import torch

from aristotelian.metrics.shift import (
    generate_shifted_pair,
    run_shift_sensitivity,
    run_stimulus_sensitivity,
)


def test_generate_shifted_pair_shapes():
    torch.manual_seed(0)
    X, Y = generate_shifted_pair(
        n=30,
        d=8,
        shift_strength=1.0,
        shift_type="mean",
        snr=2.0,
        seed=123,
    )
    assert X.shape == (30, 8)
    assert Y.shape == (30, 8)


def test_shift_sensitivity_outputs():
    torch.manual_seed(1)
    res = run_shift_sensitivity(
        metric="sgcka_lin",
        n=40,
        d=10,
        shifts=(0.0, 1.0),
        shift_type="cov",
        snr=1.0,
        num_trials=3,
        num_permutations=10,
        quantile=0.9,
        seed=7,
    )
    assert set(res.keys()) == {0.0, 1.0}
    for entry in res.values():
        assert "gated_mean" in entry
        assert "z_mean" in entry


def test_stimulus_sensitivity_outputs():
    torch.manual_seed(2)
    res = run_stimulus_sensitivity(
        metric="sgknn",
        n_values=(20, 30),
        d=6,
        shift_strength=0.5,
        shift_type="mean",
        snr=1.5,
        num_trials=3,
        num_permutations=10,
        quantile=0.9,
        seed=11,
    )
    assert set(res.keys()) == {20, 30}
    for entry in res.values():
        assert "gated_mean" in entry
        assert "ari_mean" in entry
