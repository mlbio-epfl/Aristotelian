"""Gen2 data generators (linear, geometry, local)."""

from __future__ import annotations

import math
from typing import Dict

import torch

from .common import projection_matrix, projection_matrix_no_rng
from .noise import sample_noise, strength_from_snr


def gen2_linear_shared_state(
    n: int, d: int, r: int, *, device: str
) -> Dict[str, torch.Tensor]:
    """Create shared state for gen2_linear without explicit RNG.

    The state contains latent Z and projection matrices A, B.
    """
    Z = torch.randn(n, r, device=device)
    A = projection_matrix_no_rng(r, d, device)
    B = projection_matrix_no_rng(r, d, device)
    return {"Z": Z, "A": A, "B": B}


def gen2_linear_from_state(
    state: Dict[str, torch.Tensor],
    strength: float,
    *,
    noise_type: str,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate X, Y pair from shared state with given strength.

    X = strength * (Z @ A) + sqrt(1 - strength^2) * Nx
    Y = strength * (Z @ B) + sqrt(1 - strength^2) * Ny
    """
    strength = float(max(0.0, min(1.0, strength)))
    Z = state["Z"]
    A = state["A"]
    B = state["B"]
    device = Z.device
    n = Z.shape[0]
    d = A.shape[1]
    Nx = sample_noise(n, d, 1.0, noise_type, device=str(device), rng=rng)
    Ny = sample_noise(n, d, 1.0, noise_type, device=str(device), rng=rng)
    Xs = Z @ A
    Ys = Z @ B
    noise_scale = math.sqrt(max(1.0 - strength**2, 0.0))
    X = strength * Xs + noise_scale * Nx
    Y = strength * Ys + noise_scale * Ny
    return X, Y


def make_gen2_linear_signal(
    n: int,
    d: int,
    rank: int,
    *,
    signal_strength: float,
    noise_std: float,
    noise_type: str,
    device: str,
    shared_state: Dict[str, torch.Tensor] | None = None,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Generate signal pair with optional shared state, converting SNR to strength."""
    strength = strength_from_snr(signal_strength, noise_std)
    if shared_state is None:
        shared_state = gen2_linear_shared_state(n, d, rank, device=device)
    X, Y = gen2_linear_from_state(
        shared_state, strength, noise_type=noise_type, rng=rng
    )
    return X, Y, shared_state


def gen2_linear_state(
    n: int, d: int, r: int, rng: torch.Generator
) -> Dict[str, torch.Tensor]:
    """Create full state for gen2_linear with pre-sampled noise."""
    Z = torch.randn(n, r, generator=rng, device=rng.device)
    A = projection_matrix(rng, r, d)
    B = projection_matrix(rng, r, d)
    Nx = torch.randn(n, d, generator=rng, device=rng.device)
    Ny = torch.randn(n, d, generator=rng, device=rng.device)
    return {"Z": Z, "A": A, "B": B, "Nx": Nx, "Ny": Ny}


def gen2_linear(
    n: int,
    d: int,
    r: int,
    strength: float,
    rng: torch.Generator,
    *,
    state: Dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate X, Y pair with linear relationship and controllable strength."""
    strength = float(max(0.0, min(1.0, strength)))
    if state is None:
        state = gen2_linear_state(n, d, r, rng)
    Z = state["Z"]
    A = state["A"]
    B = state["B"]
    Nx = state["Nx"]
    Ny = state["Ny"]
    Xs = Z @ A
    Ys = Z @ B
    X = strength * Xs + math.sqrt(max(1.0 - strength**2, 0.0)) * Nx
    Y = strength * Ys + math.sqrt(max(1.0 - strength**2, 0.0)) * Ny
    return X, Y


def gen2_geometry_state(
    n: int, d: int, m: int, rng: torch.Generator, *, noise_type: str
) -> Dict[str, torch.Tensor]:
    """Create state for gen2_geometry generator."""
    U = torch.randn(n, m, generator=rng, device=rng.device)
    P = projection_matrix(rng, m, d)
    E1 = sample_noise(n, d, 1.0, noise_type, device=str(rng.device), rng=rng)
    E2 = sample_noise(n, d, 1.0, noise_type, device=str(rng.device), rng=rng)
    return {"U": U, "P": P, "E1": E1, "E2": E2}


def gen2_geometry(
    n: int,
    d: int,
    m: int,
    sigma: float,
    rng: torch.Generator,
    noise_type: str,
    *,
    state: Dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate X, Y pair sharing geometric structure in subspace.

    X = U @ P + sigma * E1
    Y = U @ P + sigma * E2
    """
    if state is None:
        state = gen2_geometry_state(n, d, m, rng, noise_type=noise_type)
    U = state["U"]
    P = state["P"]
    E1 = state["E1"]
    E2 = state["E2"]
    X0 = U @ P
    X = X0 + sigma * E1
    Y = X0 + sigma * E2
    return X, Y


def gen2_local_state(
    n: int,
    d: int,
    m: int,
    clusters: int,
    noise: float,
    rng: torch.Generator,
    *,
    noise_type: str,
) -> Dict[str, torch.Tensor]:
    """Create state for gen2_local generator with cluster structure."""
    centers_dir = torch.randn(clusters, m, generator=rng, device=rng.device)
    centers_dir = centers_dir / (centers_dir.norm(dim=1, keepdim=True) + 1e-8)
    labels = torch.randint(0, clusters, (n,), generator=rng, device=rng.device)
    eps = sample_noise(n, m, noise, noise_type, device=str(rng.device), rng=rng)
    P = projection_matrix(rng, m, d)
    E1 = sample_noise(n, d, noise, noise_type, device=str(rng.device), rng=rng)
    E2 = sample_noise(n, d, noise, noise_type, device=str(rng.device), rng=rng)
    return {
        "centers_dir": centers_dir,
        "labels": labels,
        "eps": eps,
        "P": P,
        "E1": E1,
        "E2": E2,
    }


def gen2_local(
    n: int,
    d: int,
    m: int,
    sep: float,
    clusters: int,
    noise: float,
    rng: torch.Generator,
    noise_type: str,
    *,
    state: Dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate X, Y pair with local cluster structure.

    Points are assigned to clusters with shared centers, then projected
    to high-dimensional space with additive noise.
    """
    if state is None:
        state = gen2_local_state(n, d, m, clusters, noise, rng, noise_type=noise_type)
    centers = state["centers_dir"] * sep
    labels = state["labels"]
    eps = state["eps"]
    P = state["P"]
    E1 = state["E1"]
    E2 = state["E2"]
    U = centers[labels] + eps
    U_emb = U @ P
    X = U_emb + E1
    Y = U_emb + E2
    return X, Y
