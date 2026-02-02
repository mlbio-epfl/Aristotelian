"""Layerwise representation generators for aggregation experiments."""

from __future__ import annotations

import torch

from .common import projection_matrix_no_rng
from .gen2 import gen2_linear_from_state
from .noise import strength_from_snr


def make_random_layers(
    n: int,
    d: int,
    num_layers: int,
    *,
    device: str,
) -> list[torch.Tensor]:
    """Generate random layers without explicit RNG."""
    return [torch.randn(n, d, device=device) for _ in range(num_layers)]


def make_random_layers_batched(
    num_trials: int,
    n: int,
    d: int,
    num_layers: int,
    *,
    device: str,
) -> torch.Tensor:
    """Generate batched random layers.

    Returns:
        Tensor of shape (num_trials, num_layers, n, d).
    """
    return torch.randn(num_trials, num_layers, n, d, device=device)


def make_random_layers_with_rng(
    n: int,
    d: int,
    num_layers: int,
    *,
    rng: torch.Generator,
    device: str,
) -> list[torch.Tensor]:
    """Generate random layers with explicit RNG for reproducibility."""
    return [torch.randn(n, d, generator=rng, device=device) for _ in range(num_layers)]


def make_signal_layers(
    n: int,
    d: int,
    num_layers: int,
    *,
    rank: int,
    signal_strength: float,
    noise_std: float,
    noise_type: str,
    device: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Generate layerwise representations with diagonal alignment only.

    Each layer pair (i, i) shares a latent Z, creating alignment on the diagonal.
    Different layers use independent Z's, so off-diagonal pairs (i, j) where i != j
    have no shared structure and behave as nulls.

    This is critical for selection bias experiments: max aggregation over L×L
    matrices should show inflation under null (off-diagonal) but preserve signal
    on diagonal entries.
    """
    strength = strength_from_snr(signal_strength, noise_std)
    repsA = []
    repsB = []
    for _ in range(num_layers):
        # Each layer gets its OWN latent Z - shared between A and B for this layer
        # but independent across layers. This ensures:
        # - Diagonal (i,i): aligned (same Z)
        # - Off-diagonal (i,j): null-like (independent Z's)
        Z_layer = torch.randn(n, rank, device=device)
        state = {
            "Z": Z_layer,
            "A": projection_matrix_no_rng(rank, d, device),
            "B": projection_matrix_no_rng(rank, d, device),
        }
        X, Y = gen2_linear_from_state(state, strength, noise_type=noise_type)
        repsA.append(X)
        repsB.append(Y)
    return repsA, repsB


def make_signal_layers_batched(
    num_trials: int,
    n: int,
    d: int,
    num_layers: int,
    *,
    rank: int,
    signal_strength: float,
    noise_std: float,
    noise_type: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate batched layerwise representations with diagonal alignment.

    Returns:
        Tuple of (repsA, repsB), each of shape (num_trials, num_layers, n, d).
    """
    import math

    from .noise import sample_noise

    strength = strength_from_snr(signal_strength, noise_std)
    noise_scale = math.sqrt(max(1.0 - strength**2, 0.0))

    repsA = torch.empty(num_trials, num_layers, n, d, device=device)
    repsB = torch.empty(num_trials, num_layers, n, d, device=device)

    for t in range(num_trials):
        for layer in range(num_layers):
            Z_layer = torch.randn(n, rank, device=device)
            A = projection_matrix_no_rng(rank, d, device)
            B = projection_matrix_no_rng(rank, d, device)
            Xs = Z_layer @ A
            Ys = Z_layer @ B
            Nx = sample_noise(n, d, 1.0, noise_type, device=device)
            Ny = sample_noise(n, d, 1.0, noise_type, device=device)
            repsA[t, layer] = strength * Xs + noise_scale * Nx
            repsB[t, layer] = strength * Ys + noise_scale * Ny

    return repsA, repsB
