"""Noise generation utilities."""

from __future__ import annotations

import math

import torch

NOISE_TYPES = ("gaussian", "student_t", "laplace", "mixture")
DEFAULT_NOISE_TYPE = "gaussian"
STUDENT_T_DF = 3
MIXTURE_PROB = 0.1


def strength_from_snr(signal_strength: float, noise_std: float) -> float:
    """Convert signal strength and noise to normalized strength in [0, 1]."""
    denom = math.sqrt(signal_strength**2 + noise_std**2)
    if denom <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, signal_strength / denom)))


def sample_noise(
    n: int,
    d: int,
    noise_std: float,
    noise_type: str,
    *,
    device: str,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample noise tensor with specified distribution.

    Args:
        n: Number of samples.
        d: Dimensionality.
        noise_std: Standard deviation of noise.
        noise_type: One of 'gaussian', 'student_t', 'laplace', 'mixture'.
        device: Torch device.
        rng: Optional random generator.

    Returns:
        Noise tensor of shape (n, d).
    """
    if noise_std <= 0.0:
        return torch.zeros((n, d), device=device)
    if noise_type == "gaussian":
        return noise_std * torch.randn(n, d, device=device, generator=rng)
    if noise_type == "laplace":
        u = torch.rand(n, d, device=device, generator=rng) - 0.5
        # Avoid log1p(-1) -> -inf when u is extremely close to +/-0.5.
        u = u.clamp(min=-0.5 + 1e-6, max=0.5 - 1e-6)
        b = noise_std / math.sqrt(2.0)
        return -b * torch.sign(u) * torch.log1p(-2.0 * u.abs())
    if noise_type == "student_t":
        df = STUDENT_T_DF
        z = torch.randn(n, d, device=device, generator=rng)
        chi = torch.zeros(n, d, device=device)
        for _ in range(df):
            chi = chi + torch.randn(n, d, device=device, generator=rng) ** 2
        t = z / torch.sqrt(chi / float(df))
        scale = noise_std / math.sqrt(float(df) / float(df - 2))
        return scale * t
    if noise_type == "mixture":
        gauss = noise_std * torch.randn(n, d, device=device, generator=rng)
        df = STUDENT_T_DF
        z = torch.randn(n, d, device=device, generator=rng)
        chi = torch.zeros(n, d, device=device)
        for _ in range(df):
            chi = chi + torch.randn(n, d, device=device, generator=rng) ** 2
        t = z / torch.sqrt(chi / float(df))
        scale = noise_std / math.sqrt(float(df) / float(df - 2))
        heavy = scale * t
        mask = torch.rand(n, d, device=device, generator=rng) < MIXTURE_PROB
        return torch.where(mask, heavy, gauss)
    raise ValueError(f"Unknown noise type: {noise_type}")
