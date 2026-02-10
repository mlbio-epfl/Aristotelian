"""Common utility functions shared across the aristotelian package.

This module consolidates duplicated helper functions to avoid code duplication
and ensure consistent behavior across all metric implementations.
"""

from __future__ import annotations

from typing import Generator

import numpy as np
import torch

# =============================================================================
# Numerical Constants
# =============================================================================

# Standard epsilon for numerical stability in denominator operations.
# Used across all metrics for consistent behavior.
EPS = 1e-8

# =============================================================================
# kNN Indicator / Mask Functions
# =============================================================================


def knn_indicator(knn_idx: torch.Tensor, n: int) -> torch.Tensor:
    """Convert kNN indices (n, k) to boolean adjacency matrix (n, n)."""
    mask = torch.zeros((n, n), device=knn_idx.device, dtype=torch.bool)
    rows = torch.arange(n, device=knn_idx.device).unsqueeze(1).expand_as(knn_idx)
    mask[rows, knn_idx] = True
    mask.fill_diagonal_(False)
    return mask


# =============================================================================
# HSIC (Hilbert-Schmidt Independence Criterion) Functions
# =============================================================================


def hsic_biased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Biased HSIC estimator (O(1/n) bias, lower variance)."""
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype, device=K.device) - 1.0 / n
    return torch.trace(K @ H @ L @ H)


def hsic_unbiased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Unbiased HSIC estimator (Song et al. 2012)."""
    m = K.shape[0]
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)
    hsic_value = (
        (torch.sum(K_tilde * L_tilde.T))
        + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
    )
    hsic_value /= m * (m - 3)
    return hsic_value


# =============================================================================
# Centering Functions
# =============================================================================


def center_np(X: np.ndarray) -> np.ndarray:
    """Center columns to zero mean."""
    return X - X.mean(axis=0, keepdims=True)


def center_gram(K: torch.Tensor) -> torch.Tensor:
    """Double-center a Gram matrix: K - row_mean - col_mean + grand_mean."""
    row_mean = K.mean(1, keepdim=True)
    col_mean = K.mean(0, keepdim=True)
    grand_mean = K.mean()
    return K - row_mean - col_mean + grand_mean


# =============================================================================
# Batched Permutation Generator
# =============================================================================


def batched_perms(
    n: int,
    num_permutations: int,
    *,
    device: torch.device | str = "cpu",
    seed: int | None = None,
    chunk_size: int = 16,
) -> Generator[torch.Tensor, None, None]:
    """Generate random permutations in batched chunks.

    Yields batches of random permutations, which is useful for memory-efficient
    null distribution computation.

    Args:
        n: The length of each permutation.
        num_permutations: Total number of permutations to generate.
        device: Device to create tensors on.
        seed: Optional random seed for reproducibility.
        chunk_size: Number of permutations per batch.

    Yields:
        Tensors of shape (batch_size, n) where batch_size <= chunk_size.
    """
    if isinstance(device, str):
        device = torch.device(device)
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
    remaining = num_permutations
    while remaining > 0:
        size = min(chunk_size, remaining)
        perms = torch.stack(
            [torch.randperm(n, generator=rng, device=device) for _ in range(size)]
        )
        yield perms
        remaining -= size


def batched_perms_simple(
    n: int, num_permutations: int, device: str | torch.device
) -> torch.Tensor:
    """Generate all permutations at once (non-generator version)."""
    if isinstance(device, str):
        device = torch.device(device)
    return torch.stack(
        [torch.randperm(n, device=device) for _ in range(num_permutations)]
    )


# =============================================================================
# Memory-Aware Batch Sizing for Permutation Loops
# =============================================================================

# Target peak working memory for batched eigvalsh / eigh operations.
_MAX_PERM_BATCH_BYTES = 512 * 1024 * 1024  # 512 MB


def perm_batch_size(num_perms: int, d: int, dtype: torch.dtype = torch.float64) -> int:
    """Compute batch size for vectorised permutation-test loops.

    The main memory consumer is the (B, d, d) tensor fed to eigvalsh/eigh,
    plus the (B, d, d) intermediate T @ T.T.  We budget for ~3 such tensors.
    """
    elem = (
        dtype.itemsize if hasattr(dtype, "itemsize") else (torch.finfo(dtype).bits // 8)
    )
    per_perm = 3 * d * d * elem
    chunk = max(1, _MAX_PERM_BATCH_BYTES // max(per_perm, 1))
    return min(num_perms, chunk)


# =============================================================================
# SVCCA Preprocessing
# =============================================================================


def svcca_preprocess(feats: torch.Tensor) -> torch.Tensor:
    """Center and standardize features (mean 0, unit variance per feature)."""
    feats = feats - torch.mean(feats, axis=0)
    feats = feats / (torch.std(feats, axis=0) + EPS)
    return feats


# =============================================================================
# Metric Resolution
# =============================================================================


def resolve_sg_metric(metric: str):
    """Resolve a metric string name to its significance-gated function.

    Args:
        metric: One of 'sgcka_lin', 'sgcka_rbf', 'sgknn', 'sgrsa'.

    Returns:
        The corresponding metric function.

    Raises:
        ValueError: If metric is not recognized.
    """
    # Import here to avoid circular imports
    from .. import sg_cka_kernel, sg_cka_linear, sg_knn, sg_rsa

    if metric == "sgcka_lin":
        return sg_cka_linear
    if metric == "sgcka_rbf":
        return sg_cka_kernel
    if metric == "sgknn":
        return sg_knn
    if metric == "sgrsa":
        return sg_rsa
    raise ValueError("metric must be one of {'sgcka_lin','sgcka_rbf','sgknn','sgrsa'}")


# =============================================================================
# SVD-based PCA Utilities
# =============================================================================


def svd_pca(X: np.ndarray, var_threshold: float = 0.99) -> np.ndarray:
    """PCA using SVD with variance threshold.

    Computes principal components using SVD and retains enough components
    to explain the specified fraction of total variance.

    Args:
        X: Input array of shape (n, d).
        var_threshold: Fraction of variance to retain (default 0.99).

    Returns:
        Projected data of shape (n, k) where k is the number of components
        needed to explain var_threshold of the variance.
    """
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    var = (S**2) / np.sum(S**2)
    k = int(np.searchsorted(np.cumsum(var), var_threshold) + 1)
    return U[:, :k] * S[:k]


def svd_pca_k(X: np.ndarray, k: int) -> np.ndarray:
    """PCA using SVD with fixed k components.

    Computes principal components using SVD and retains exactly k components.

    Args:
        X: Input array of shape (n, d).
        k: Number of principal components to retain. Must be <= min(n, d).

    Returns:
        Projected data of shape (n, k).

    Raises:
        ValueError: If k exceeds the maximum possible components.
    """
    n, d = X.shape
    max_k = min(n, d)
    if k > max_k:
        raise ValueError(
            f"k={k} exceeds maximum possible components min(n={n}, d={d})={max_k}"
        )
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :k] * S[:k]


# =============================================================================
# Data Sampling Utilities
# =============================================================================


def sample_student_t(
    n: int,
    d: int,
    *,
    df: int,
    device: str,
    rng: torch.Generator,
) -> torch.Tensor:
    """Sample from a multivariate Student-t distribution.

    Args:
        n: Number of samples.
        d: Dimensionality.
        df: Degrees of freedom.
        device: Device to create tensor on.
        rng: Random number generator for reproducibility.

    Returns:
        Tensor of shape (n, d) with samples from Student-t distribution.
    """
    dist = torch.distributions.StudentT(df=df)
    try:
        return dist.sample((n, d), generator=rng).to(device)
    except TypeError:
        # Older torch versions don't accept a generator for distribution sampling.
        seed_tensor = torch.randint(0, 2**31 - 1, (1,), generator=rng, device=device)
        seed = int(seed_tensor.item())
        if device.startswith("cuda"):
            device_index = torch.device(device).index
            if device_index is None:
                device_index = torch.cuda.current_device()
            devices = [device_index]
        else:
            devices = []
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(seed)
            if device.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)
            return dist.sample((n, d)).to(device)
