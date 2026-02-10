"""CKA (Centered Kernel Alignment) metrics.

This module provides CKA variants:
- cka_linear: Linear kernel CKA
- cka_rbf: RBF kernel CKA
- cka_unbiased: Unbiased CKA estimator

All variants support null calibration via permutation testing.
"""

from __future__ import annotations

import torch

from .base import BaseMetric, MetricConfig
from .registry import register_metric
from .utils import EPS, batched_perms_simple, center_gram, hsic_biased, hsic_unbiased


def _get_centered(
    X: torch.Tensor, *, cache: dict[str, torch.Tensor], key: str
) -> torch.Tensor:
    """Center X and memoize by key in the metric cache."""
    if key in cache:
        return cache[key]
    Xc = X - X.mean(0, keepdim=True)
    cache[key] = Xc
    return Xc


def _rbf_kernel(X: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """Compute RBF kernel matrix with median heuristic for bandwidth."""
    dist_sq = torch.cdist(X, X, p=2) ** 2
    if sigma is None:
        med = torch.median(dist_sq)
        sigma = float(torch.sqrt(med).item()) if med > 0 else 1.0
    gamma = 1.0 / (2 * sigma**2)
    return torch.exp(-gamma * dist_sq)


def _center_norm_gram(K: torch.Tensor) -> torch.Tensor:
    """Center and normalize a Gram matrix."""
    Kc = center_gram(K)
    return Kc / (torch.norm(Kc, p="fro") + EPS)


@register_metric
class CKALinear(BaseMetric):
    """Linear kernel CKA metric.

    Measures representation similarity using linear kernel alignment.
    This is the most common CKA variant.

    Score range: [0, 1] where 1 means perfect alignment.
    """

    name = "cka_linear"
    min_score = 0.0
    max_score = 1.0
    supports_caching = True
    cache_keys = ("gram_X", "gram_Y", "Xc", "Yc")

    def _compute_raw(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> float:
        cache = config.cache

        Xc = _get_centered(X, cache=cache, key="Xc")
        Yc = _get_centered(Y, cache=cache, key="Yc")

        # Compute CKA via feature space (more efficient)
        x_xt = Xc.T @ Xc
        y_yt = Yc.T @ Yc
        denom = torch.norm(x_xt) * torch.norm(y_yt) + EPS
        num = torch.norm(Yc.T @ Xc) ** 2
        return float((num / denom).item())

    def _compute_null_distribution(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> list[float]:
        """Optimized null distribution using vectorized cross-product."""
        device = config.device
        num_perms = config.num_permutations
        cache = config.cache

        Xc = _get_centered(X, cache=cache, key="Xc")
        Yc = _get_centered(Y, cache=cache, key="Yc")

        Xc = Xc.to(device)
        Yc = Yc.to(device)

        x_xt = Xc.T @ Xc
        y_yt = Yc.T @ Yc
        denom = torch.norm(x_xt) * torch.norm(y_yt) + EPS

        if config.perms is not None:
            perms = config.perms.to(device)
        else:
            perms = batched_perms_simple(Xc.size(0), num_perms, device)

        # Vectorized computation
        Y_perm = Yc[perms]  # (B, n, d)
        cross = torch.einsum("bni,nj->bij", Y_perm, Xc)  # (B, d, d)
        nums = torch.linalg.matrix_norm(cross, ord="fro") ** 2
        null_samples = nums / denom

        return null_samples.detach().cpu().tolist()


@register_metric
class CKARBF(BaseMetric):
    """RBF kernel CKA metric.

    Uses RBF (Gaussian) kernel for CKA computation.
    Sigma (bandwidth) can be specified or computed via median heuristic.

    Score range: [0, 1] where 1 means perfect alignment.
    """

    name = "cka_rbf"
    min_score = 0.0
    max_score = 1.0
    supports_caching = True
    cache_keys = ("Kx_norm", "Ky_norm")

    def _compute_raw(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> float:
        cache = config.cache
        sigma = config.sigma

        if "Kx_norm" in cache:
            Kx = cache["Kx_norm"]
        else:
            Kx = _center_norm_gram(_rbf_kernel(X, sigma))
            cache["Kx_norm"] = Kx

        if "Ky_norm" in cache:
            Ky = cache["Ky_norm"]
        else:
            Ky = _center_norm_gram(_rbf_kernel(Y, sigma))
            cache["Ky_norm"] = Ky

        return float(torch.sum(Kx * Ky).item())

    def _compute_null_distribution(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> list[float]:
        """Optimized null distribution using Gram matrix permutation."""
        device = config.device
        num_perms = config.num_permutations
        cache = config.cache
        sigma = config.sigma

        if "Kx_norm" in cache:
            Kx = cache["Kx_norm"].to(device)
        else:
            Kx = _center_norm_gram(_rbf_kernel(X.to(device), sigma))
            cache["Kx_norm"] = Kx

        if "Ky_norm" in cache:
            Ky = cache["Ky_norm"].to(device)
        else:
            Ky = _center_norm_gram(_rbf_kernel(Y.to(device), sigma))
            cache["Ky_norm"] = Ky

        if config.perms is not None:
            perms = config.perms.to(device)
        else:
            perms = batched_perms_simple(Kx.size(0), num_perms, device)

        # Permute both rows and columns of Ky
        temp = Ky[perms]  # (B, n, n)
        Ky_perm = torch.gather(temp, 2, perms.unsqueeze(1).expand(-1, Ky.size(0), -1))
        null_samples = torch.sum(Kx.unsqueeze(0) * Ky_perm, dim=(1, 2))

        return null_samples.detach().cpu().tolist()


@register_metric
class CKAUnbiased(BaseMetric):
    """Unbiased CKA estimator.

    Uses unbiased HSIC estimator for more accurate small-sample results.

    Score range: Typically [0, 1] but can be slightly negative due to bias correction.
    """

    name = "cka_unbiased"
    min_score = -0.1  # Can go slightly negative due to bias correction
    max_score = 1.0
    supports_caching = True
    cache_keys = ("gram_X_ip", "gram_Y_ip")

    def _compute_raw(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> float:
        cache = config.cache

        if "gram_X_ip" in cache:
            K = cache["gram_X_ip"]
        else:
            K = X @ X.T
            cache["gram_X_ip"] = K

        if "gram_Y_ip" in cache:
            L = cache["gram_Y_ip"]
        else:
            L = Y @ Y.T
            cache["gram_Y_ip"] = L

        hsic_kk = hsic_unbiased(K, K)
        hsic_ll = hsic_unbiased(L, L)
        hsic_kl = hsic_unbiased(K, L)
        cka = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + EPS)
        return float(cka.item())

    def _compute_null_distribution(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> list[float]:
        """Batched null distribution for unbiased CKA.

        Permutation only affects L (Gram of Y). The denominator
        sqrt(HSIC_u(K,K) * HSIC_u(L,L)) is invariant under row/col
        permutation of L, so we precompute it and batch only the
        HSIC_u(K, L_perm) numerator.
        """
        n = X.shape[0]
        device = config.device
        num_perms = config.num_permutations

        K = (X @ X.T).to(device)
        L = (Y @ Y.T).to(device)

        # Denominator is invariant under permutation of L
        denom = torch.sqrt(hsic_unbiased(K, K) * hsic_unbiased(L, L)) + EPS

        # Precompute K_tilde (diagonal zeroed) constants
        K_tilde = K.clone().fill_diagonal_(0)
        sum_K = torch.sum(K_tilde)
        sum_L = torch.sum(L) - torch.trace(L)  # = sum(L_tilde), invariant under perm
        krow = K_tilde.sum(dim=0)  # (n,) — for term3

        if config.perms is not None:
            perms = config.perms.to(device)
        else:
            perms = batched_perms_simple(n, num_perms, device)

        m = n
        scale = 1.0 / (m * (m - 3))
        term2_const = float(sum_K.item()) * float(sum_L.item()) / ((m - 1) * (m - 2))

        # Process in chunks to limit memory: (chunk, n, n)
        max_chunk = max(1, (512 * 1024 * 1024) // (n * n * K.element_size()))
        null_scores: list[float] = []
        for start in range(0, num_perms, max_chunk):
            end = min(start + max_chunk, num_perms)
            batch_perms = perms[start:end]  # (B, n)

            # Permute rows+cols of L: L_perm[b,i,j] = L[perm[b,i], perm[b,j]]
            temp = L[batch_perms]  # (B, n, n) — row permutation
            L_perm = torch.gather(
                temp, 2, batch_perms.unsqueeze(1).expand(-1, n, -1)
            )  # (B, n, n)

            # term1: sum(K_tilde * L_perm) per batch (K_tilde diagonal is 0,
            # so L_perm diagonal doesn't contribute)
            t1 = (K_tilde.unsqueeze(0) * L_perm).sum(dim=(1, 2))  # (B,)

            # term3: 2 * ones^T (K_tilde @ L_tilde_perm) ones / (m-2)
            # L_tilde_perm = L_perm with diagonal zeroed
            L_tilde_perm = L_perm.clone()
            diag_idx = torch.arange(m, device=device)
            L_tilde_perm[:, diag_idx, diag_idx] = 0
            # (K_tilde @ L_tilde_perm).sum() = krow @ L_tilde_perm.sum(dim=-1)
            lcol = L_tilde_perm.sum(dim=-1)  # (B, n)
            t3 = 2.0 * torch.einsum("n,bn->b", krow, lcol) / (m - 2)

            hsic_batch = scale * (t1 + term2_const - t3)
            scores = hsic_batch / denom
            null_scores.extend(scores.cpu().tolist())

        return null_scores


@register_metric
class CKA(BaseMetric):
    """General CKA metric with kernel selection.

    Supports both linear (ip) and RBF kernels via config.kernel.
    This is for backward compatibility with PRH metric interface.

    Score range: [0, 1] where 1 means perfect alignment.
    """

    name = "cka"
    min_score = 0.0
    max_score = 1.0
    supports_caching = True

    def _compute_raw(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> float:
        kernel = config.kernel

        if kernel in ("linear", "ip"):
            # Linear/inner-product kernel
            K = X @ X.T
            L = Y @ Y.T
        elif kernel == "rbf":
            sigma = config.rbf_sigma
            K = torch.exp(-torch.cdist(X, X) ** 2 / (2 * sigma**2))
            L = torch.exp(-torch.cdist(Y, Y) ** 2 / (2 * sigma**2))
        else:
            raise ValueError(f"Invalid kernel: {kernel}")

        hsic_fn = hsic_unbiased if config.unbiased else hsic_biased
        hsic_kk = hsic_fn(K, K)
        hsic_ll = hsic_fn(L, L)
        hsic_kl = hsic_fn(K, L)
        cka = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + EPS)
        return float(cka.item())
