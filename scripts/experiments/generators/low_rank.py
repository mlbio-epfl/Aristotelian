"""Low-rank signal generation utilities."""

from __future__ import annotations

import math

import torch


def make_pure_noise(
    n: int, d: int, *, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate pair of independent Gaussian noise matrices."""
    return torch.randn(n, d, device=device), torch.randn(n, d, device=device)


def make_low_rank_signal(
    n: int,
    d: int,
    rank: int,
    *,
    signal_strength: float = 5.0,
    noise_std: float = 1.0,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate pair with shared low-rank signal plus noise.

    Creates X = signal_strength * (U @ V.T) + noise_std * noise_X
    and Y = signal_strength * (U @ V.T) + noise_std * noise_Y.
    """
    U, _, _ = torch.linalg.svd(torch.randn(n, rank, device=device), full_matrices=False)
    V, _, _ = torch.linalg.svd(torch.randn(d, rank, device=device), full_matrices=False)
    signal = signal_strength * (U @ V.T)
    X = signal + noise_std * torch.randn(n, d, device=device)
    Y = signal + noise_std * torch.randn(n, d, device=device)
    return X, Y


def make_low_rank_signal_unitvar(
    n: int,
    d: int,
    rank: int,
    *,
    signal_strength: float,
    noise_std: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate pair with unit-variance normalized low-rank signal plus noise."""
    U, _, _ = torch.linalg.svd(torch.randn(n, rank, device=device), full_matrices=False)
    V, _, _ = torch.linalg.svd(torch.randn(d, rank, device=device), full_matrices=False)
    signal = U @ V.T
    scale = math.sqrt(float(n * d) / float(max(rank, 1)))
    signal = signal * scale
    X = signal_strength * signal + noise_std * torch.randn(n, d, device=device)
    Y = signal_strength * signal + noise_std * torch.randn(n, d, device=device)
    return X, Y
