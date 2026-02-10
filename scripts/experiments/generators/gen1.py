"""Gen1 mixture invariance data generator."""

from __future__ import annotations

import math

import torch

from .common import projection_matrix


def gen1_mixture_invariance(
    n: int,
    d: int,
    eta: float,
    rng: torch.Generator,
    intrinsic_dim: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate X, Y pair with controllable mixture of shared and independent components.

    X = sqrt(1-eta) * Z + sqrt(eta) * N1
    Y = sqrt(1-eta) * Z + sqrt(eta) * N2

    When intrinsic_dim < d, the data lives in a lower-dimensional subspace.
    """
    eta = float(max(0.0, min(1.0, eta)))
    if intrinsic_dim is None or intrinsic_dim >= d:
        Z = torch.randn(n, d, generator=rng, device=rng.device)
        N1 = torch.randn(n, d, generator=rng, device=rng.device)
        N2 = torch.randn(n, d, generator=rng, device=rng.device)
    else:
        m = int(intrinsic_dim)
        Z0 = torch.randn(n, m, generator=rng, device=rng.device)
        N10 = torch.randn(n, m, generator=rng, device=rng.device)
        N20 = torch.randn(n, m, generator=rng, device=rng.device)
        P = projection_matrix(rng, m, d)
        Z = Z0 @ P
        N1 = N10 @ P
        N2 = N20 @ P
    X = math.sqrt(1.0 - eta) * Z + math.sqrt(eta) * N1
    Y = math.sqrt(1.0 - eta) * Z + math.sqrt(eta) * N2
    return X, Y
