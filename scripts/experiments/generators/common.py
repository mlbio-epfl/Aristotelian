"""Common generator utilities."""

from __future__ import annotations

import torch


def projection_matrix_no_rng(m: int, d: int, device: str) -> torch.Tensor:
    """Create random orthonormal projection matrix (m x d) without explicit RNG."""
    mat = torch.randn(d, m, device=device)
    q, _ = torch.linalg.qr(mat)
    return q.T


def projection_matrix(rng: torch.Generator, m: int, d: int) -> torch.Tensor:
    """Create random orthonormal projection matrix (m x d) with explicit RNG."""
    mat = torch.randn(d, m, generator=rng, device=rng.device)
    q, _ = torch.linalg.qr(mat)
    return q.T


def center_norm_features(X: torch.Tensor) -> torch.Tensor:
    """Center and normalize features to unit Frobenius norm."""
    Xc = X - X.mean(0, keepdim=True)
    return Xc / (torch.norm(Xc, p="fro") + 1e-8)


def knn_indicator(knn_idx: torch.Tensor, n: int) -> torch.Tensor:
    """Convert kNN indices to indicator matrix."""
    mask = torch.zeros((n, n), device=knn_idx.device, dtype=torch.bool)
    rows = torch.arange(n, device=knn_idx.device).unsqueeze(1).expand_as(knn_idx)
    mask[rows, knn_idx] = True
    mask.fill_diagonal_(False)
    return mask
