"""Null-model utilities used for calibrated representation similarity experiments."""

from __future__ import annotations

import torch


def spectrum_matched_view(
    X: torch.Tensor,
    *,
    seed: int | None = None,
    device: str | None = None,
) -> torch.Tensor:
    """
    Return X' with the same singular values as X but randomized singular vectors.

    This is useful for spectrum-matched nulls that preserve anisotropy while
    destroying alignment structure.
    """
    if device is None:
        device = X.device.type
    X = X.to(device)

    n, d = X.shape
    r = min(n, d)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    # Random orthonormal bases for left/right spaces.
    A = torch.randn(n, r, generator=generator, device=device, dtype=X.dtype)
    B = torch.randn(d, r, generator=generator, device=device, dtype=X.dtype)
    Qa, _ = torch.linalg.qr(A, mode="reduced")
    Qb, _ = torch.linalg.qr(B, mode="reduced")

    s = torch.linalg.svdvals(X)
    return Qa @ torch.diag(s) @ Qb.T
