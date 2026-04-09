"""Other representation similarity metrics.

This module provides:
- procrustes: Procrustes distance-based similarity
- cknna: Centered Kernel Neighborhood Nearest Neighbor Alignment
"""

from __future__ import annotations

import numpy as np
import torch

from .base import BaseMetric, MetricConfig, MetricResult
from .extra_base import _sg_metric
from .registry import register_metric
from .utils import EPS, center_np, hsic_biased, hsic_unbiased


@register_metric
class Procrustes(BaseMetric):
    """Procrustes distance-based similarity.

    Finds optimal orthogonal alignment and measures residual distance.
    Score = 1 - sqrt(||X||² + ||Y||² - 2*sum(S)) / ||Y||
    where S = singular_values(Xc.T @ Yc).

    Score range: [-inf, 1] where 1 means perfect alignment.
    Typically in [-1, 1] for normalized data.
    """

    name = "procrustes"
    min_score = -1.0
    max_score = 1.0
    supports_calibration = True

    def _compute_raw(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> float:
        Xc = X - X.mean(dim=0, keepdim=True)
        Yc = Y - Y.mean(dim=0, keepdim=True)
        return _procrustes_from_centered(Xc, Yc)

    def _compute_null_distribution(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> list[float]:
        """Optimized null: GPU eigvalsh, precomputed norms."""
        n = X.shape[0]
        device = config.device
        num_perms = config.num_permutations

        if config.perms is not None:
            perms = config.perms.to(device)
            if perms.dim() != 2 or perms.size(1) != n:
                raise ValueError("perms must have shape (B, n)")
        else:
            perms = torch.stack(
                [torch.randperm(n, device=device) for _ in range(num_perms)]
            )

        Xc = X - X.mean(dim=0, keepdim=True)
        Yc = Y - Y.mean(dim=0, keepdim=True)
        norm_Xc_sq = torch.sum(Xc**2)
        norm_Yc_sq = torch.sum(Yc**2)  # invariant under row permutation
        norm_Yc = torch.sqrt(norm_Yc_sq)
        XcT = Xc.T  # (d, n) — precompute

        from .utils import perm_batch_size

        d_x = Xc.shape[1]
        d_y = Yc.shape[1]
        d_eig = min(d_x, d_y)
        chunk = perm_batch_size(num_perms, d_eig, Xc.dtype)
        null_scores: list[float] = []
        for start in range(0, num_perms, chunk):
            end = min(start + chunk, num_perms)
            Yc_perm = Yc[perms[start:end]]  # (B, n, d_y)
            # M_batch[b] = Xc.T @ Yc_perm[b]  →  (B, d_x, d_y)
            M_batch = torch.einsum("dn,bnk->bdk", XcT, Yc_perm)
            # Use the smaller Gram matrix for eigvalsh: O(min(d_x,d_y)^3)
            if d_x <= d_y:
                G = M_batch @ M_batch.transpose(-1, -2)  # (B, d_x, d_x)
            else:
                G = M_batch.transpose(-1, -2) @ M_batch  # (B, d_y, d_y)
            try:
                eigvals = torch.linalg.eigvalsh(G)  # (B, d_eig)
                sum_svals = torch.sqrt(torch.clamp(eigvals, min=0.0)).sum(dim=-1)  # (B,)
            except torch._C._LinAlgError:
                svals = torch.linalg.svdvals(M_batch)
                sum_svals = svals.sum(dim=-1)
            residual_sq = norm_Xc_sq + norm_Yc_sq - 2.0 * sum_svals
            residual = torch.sqrt(torch.clamp(residual_sq, min=0.0))
            scores = 1.0 - residual / (norm_Yc + EPS)
            null_scores.extend(scores.cpu().tolist())

        return null_scores


def _procrustes_from_centered(Xc: torch.Tensor, Yc: torch.Tensor) -> float:
    """Procrustes score from centered tensors, using eigvalsh for robustness."""
    M = Xc.T @ Yc  # (d_x, d_y)
    # Use the smaller Gram matrix: M@M.T is (d_x,d_x), M.T@M is (d_y,d_y).
    # Both share the same non-zero eigenvalues (squared singular values of M).
    if M.shape[0] <= M.shape[1]:
        G = M @ M.T
    else:
        G = M.T @ M
    try:
        eigvals = torch.linalg.eigvalsh(G)
        sum_svals = torch.sqrt(torch.clamp(eigvals, min=0.0)).sum()
    except torch._C._LinAlgError:
        sum_svals = torch.linalg.svdvals(M).sum()
    norm_Xc_sq = torch.sum(Xc**2)
    norm_Yc = torch.norm(Yc, p="fro")
    residual_sq = norm_Xc_sq + norm_Yc**2 - 2.0 * sum_svals
    residual = torch.sqrt(torch.clamp(residual_sq, min=0.0))
    return float((1.0 - residual / (norm_Yc + EPS)).item())


@register_metric
class CKNNA(BaseMetric):
    """Centered Kernel Neighborhood Nearest Neighbor Alignment.

    HSIC-based alignment restricted to k-nearest neighbors.
    Can be distance-aware or distance-agnostic.

    Score range: [0, 1] where 1 means perfect neighborhood alignment.
    """

    name = "cknna"
    min_score = 0.0
    max_score = 1.0
    supports_calibration = True
    supports_caching = True
    cache_keys = ("gram_X", "gram_Y")

    def _compute_raw(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> float:
        n = X.shape[0]
        # Default to all neighbors (matching original PRH)
        k = config.topk if config.topk is not None else n - 1
        if k < 2:
            raise ValueError("CKNNA requires topk >= 2")

        cache = config.cache
        unbiased = config.unbiased
        distance_agnostic = config.distance_agnostic

        # Compute Gram matrices
        if "gram_X" in cache:
            K = cache["gram_X"]
        else:
            K = X @ X.T
            cache["gram_X"] = K

        if "gram_Y" in cache:
            L = cache["gram_Y"]
        else:
            L = Y @ Y.T
            cache["gram_Y"] = L

        def similarity(
            Km: torch.Tensor, Lm: torch.Tensor, topk_inner: int
        ) -> torch.Tensor:
            if unbiased:
                K_hat = Km.clone().fill_diagonal_(float("-inf"))
                L_hat = Lm.clone().fill_diagonal_(float("-inf"))
            else:
                K_hat, L_hat = Km, Lm

            _, topk_K_indices = torch.topk(K_hat, topk_inner, dim=1)
            _, topk_L_indices = torch.topk(L_hat, topk_inner, dim=1)

            mask_K = torch.zeros(n, n, device=Km.device).scatter_(1, topk_K_indices, 1)
            mask_L = torch.zeros(n, n, device=Km.device).scatter_(1, topk_L_indices, 1)
            mask = mask_K * mask_L

            if distance_agnostic:
                # Just count overlapping neighbors
                sim = mask.sum()
            else:
                hsic_fn = hsic_unbiased if unbiased else hsic_biased
                sim = hsic_fn(mask * Km, mask * Lm)
            return sim

        sim_kl = similarity(K, L, k)
        sim_kk = similarity(K, K, k)
        sim_ll = similarity(L, L, k)
        # Use 1e-6 to match original PRH implementation
        return float(sim_kl.item() / (torch.sqrt(sim_kk * sim_ll) + 1e-6).item())

    def _compute_null_distribution(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> list[float]:
        """Null distribution for CKNNA using permuted Gram matrices."""
        n = X.shape[0]
        k = config.topk if config.topk is not None else n - 1
        device = config.device
        num_perms = config.num_permutations
        unbiased = config.unbiased
        distance_agnostic = config.distance_agnostic

        K = (X @ X.T).to(device)
        L = (Y @ Y.T).to(device)

        if config.perms is not None:
            perms = config.perms.to(device)
            if perms.dim() != 2 or perms.size(1) != n:
                raise ValueError("perms must have shape (B, n)")
        else:
            perms = torch.stack(
                [torch.randperm(n, device=device) for _ in range(num_perms)]
            )

        def _similarity(
            Km: torch.Tensor, Lm: torch.Tensor, topk_inner: int
        ) -> torch.Tensor:
            if unbiased:
                K_hat = Km.clone().fill_diagonal_(float("-inf"))
                L_hat = Lm.clone().fill_diagonal_(float("-inf"))
            else:
                K_hat, L_hat = Km, Lm
            _, topk_K_idx = torch.topk(K_hat, topk_inner, dim=1)
            _, topk_L_idx = torch.topk(L_hat, topk_inner, dim=1)
            mask_K = torch.zeros(n, n, device=device).scatter_(1, topk_K_idx, 1)
            mask_L = torch.zeros(n, n, device=device).scatter_(1, topk_L_idx, 1)
            mask = mask_K * mask_L
            if distance_agnostic:
                return mask.sum()
            hsic_fn = hsic_unbiased if unbiased else hsic_biased
            return hsic_fn(mask * Km, mask * Lm)

        # sim_kk is invariant (K doesn't change)
        sim_kk = _similarity(K, K, k)

        null_scores: list[float] = []
        for perm in perms:
            L_perm = L[perm][:, perm]
            sim_kl = _similarity(K, L_perm, k)
            sim_ll = _similarity(L_perm, L_perm, k)
            score = float(sim_kl.item() / (torch.sqrt(sim_kk * sim_ll) + 1e-6).item())
            null_scores.append(score)

        return null_scores


def procrustes_score(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = center_np(X)
    Yc = center_np(Y)
    U, _, Vt = np.linalg.svd(Xc.T @ Yc, full_matrices=False)
    R = U @ Vt
    aligned = Xc @ R
    num = np.linalg.norm(aligned - Yc, ord="fro")
    denom = np.linalg.norm(Yc, ord="fro") + EPS
    return float(1.0 - num / denom)


def sg_procrustes_score(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    num_permutations: int = 200,
    quantile: float = 0.95,
    perms: np.ndarray | None = None,
) -> MetricResult:
    return _sg_metric(
        X,
        Y,
        metric_fn=procrustes_score,
        num_permutations=num_permutations,
        quantile=quantile,
        perms=perms,
        min_score=-1.0,
        max_score=1.0,
    )
