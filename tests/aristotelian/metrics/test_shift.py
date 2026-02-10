"""Tests for dataset shift and stimulus-count sensitivity utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from aristotelian.metrics.utils import EPS
from aristotelian.metrics.utils import resolve_sg_metric as _resolve_metric


# ============================================================================
# Inlined from aristotelian/metrics/shift.py (test-only utilities)
# ============================================================================


def generate_shifted_pair(
    *,
    n: int,
    d: int,
    shift_strength: float,
    shift_type: str,
    snr: float,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate paired representations with controlled dataset shift.

    shift_type: "mean" (mean shift), "cov" (covariance shift via rotation+scaling)
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    Z = torch.randn(n, d, generator=rng)
    signal = Z
    noise = 1.0 / max(snr, EPS)
    X = signal + noise * torch.randn(n, d, generator=rng)

    if shift_type == "mean":
        shift = shift_strength * torch.randn(d, generator=rng)
        Y = signal + shift + noise * torch.randn(n, d, generator=rng)
    elif shift_type == "cov":
        Q, _ = torch.linalg.qr(torch.randn(d, d, generator=rng))
        scale = torch.diag(1.0 + shift_strength * torch.linspace(0.0, 1.0, d))
        Y = signal @ (Q @ scale @ Q.T) + noise * torch.randn(n, d, generator=rng)
    else:
        raise ValueError("shift_type must be one of {'mean','cov'}")
    return X, Y


def run_shift_sensitivity(
    *,
    metric: str,
    n: int,
    d: int,
    shifts: Iterable[float],
    shift_type: str,
    snr: float,
    num_trials: int = 50,
    num_permutations: int = 200,
    quantile: float = 0.95,
    k_knn: int = 10,
    seed: int | None = None,
) -> dict[float, dict[str, float]]:
    """Sweep shift strength and report effect-size stability."""
    metric_fn = _resolve_metric(metric)
    results: dict[float, dict[str, float]] = {}
    for shift_strength in shifts:
        gated = []
        z = []
        ari = []
        for t in range(num_trials):
            X, Y = generate_shifted_pair(
                n=n,
                d=d,
                shift_strength=shift_strength,
                shift_type=shift_type,
                snr=snr,
                seed=None if seed is None else seed + t,
            )
            kwargs = dict(num_permutations=num_permutations, quantile=quantile)
            if metric == "sgknn":
                kwargs["k"] = k_knn
            res = metric_fn(X, Y, **kwargs)
            gated.append(res.gated)
            z.append(res.z)
            ari.append(res.ari)
        results[float(shift_strength)] = {
            "gated_mean": float(np.mean(gated)),
            "gated_std": float(np.std(gated)),
            "z_mean": float(np.mean(z)),
            "ari_mean": float(np.mean(ari)),
        }
    return results


def run_stimulus_sensitivity(
    *,
    metric: str,
    n_values: Iterable[int],
    d: int,
    shift_strength: float,
    shift_type: str,
    snr: float,
    num_trials: int = 50,
    num_permutations: int = 200,
    quantile: float = 0.95,
    k_knn: int = 10,
    seed: int | None = None,
) -> dict[int, dict[str, float]]:
    """Sweep sample size n and report stability of calibrated metrics."""
    metric_fn = _resolve_metric(metric)
    results: dict[int, dict[str, float]] = {}
    for n in n_values:
        gated = []
        z = []
        ari = []
        for t in range(num_trials):
            X, Y = generate_shifted_pair(
                n=n,
                d=d,
                shift_strength=shift_strength,
                shift_type=shift_type,
                snr=snr,
                seed=None if seed is None else seed + t,
            )
            kwargs = dict(num_permutations=num_permutations, quantile=quantile)
            if metric == "sgknn":
                kwargs["k"] = k_knn
            res = metric_fn(X, Y, **kwargs)
            gated.append(res.gated)
            z.append(res.z)
            ari.append(res.ari)
        results[int(n)] = {
            "gated_mean": float(np.mean(gated)),
            "gated_std": float(np.std(gated)),
            "z_mean": float(np.mean(z)),
            "ari_mean": float(np.mean(ari)),
        }
    return results


# ============================================================================
# Tests
# ============================================================================


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
