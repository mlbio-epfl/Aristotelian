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


def _cknna_centered_bounded(K: torch.Tensor, L: torch.Tensor, k: int) -> torch.Tensor:
    """Centre-first CKNNA that is bounded above by 1.

    Order follows Huh et al. (2024), Appendix A, Eq. 16: double-centre the Gram
    matrices first, then keep only the mutual k-nearest neighbours. Neighbours
    are taken from the raw Gram (self excluded); the values summed are the
    centred Grams. The numerator sums over a subset of each model's own
    neighbours, so Cauchy-Schwarz bounds the ratio by 1.

    This differs from the default CKNNA path, which masks the raw Gram first and
    centres afterwards inside HSIC. With anisotropic features (high mean cosine)
    that default can exceed 1 at large k. See the CKNNA class docstring.
    """
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype, device=K.device) - 1.0 / n
    Kc = H @ K @ H
    Lc = H @ L @ H

    def knn_mask(G: torch.Tensor) -> torch.Tensor:
        G = G.clone().fill_diagonal_(float("-inf"))
        _, idx = torch.topk(G, k, dim=1)
        return torch.zeros(n, n, dtype=K.dtype, device=K.device).scatter_(1, idx, 1.0)

    mask_K = knn_mask(K)
    mask_L = knn_mask(L)
    num = (mask_K * mask_L * Kc * Lc).sum()
    den = torch.sqrt((mask_K * Kc**2).sum() * (mask_L * Lc**2).sum())
    return num / (den + EPS)


@register_metric
class CKNNA(BaseMetric):
    """Centered Kernel Neighborhood Nearest Neighbor Alignment.

    HSIC-based alignment restricted to k-nearest neighbors.
    Can be distance-aware or distance-agnostic.

    Two variants, chosen by ``config.cknna_bounded``:

    - ``cknna_bounded=False`` (default). Reproduces the platonic-rep
      implementation (Huh et al. 2024): mask the raw Gram matrix to the mutual
      k-nearest neighbours, then centre inside HSIC. This is the metric the
      paper uses and the one comparable to platonic-rep and work built on it.
      With anisotropic features (mean cosine well above 0) it is NOT bounded by
      1 at large k: the numerator uses the shared neighbour mask while the two
      denominators use each model's own mask, so Cauchy-Schwarz does not bound
      the ratio, and the score can climb above 1 for k beyond about n/2. It
      stays well inside [0, 1] at the small k the paper reports (k up to 100
      with n=1024). See issue #2.

    - ``cknna_bounded=True``. Centre first, then keep the mutual neighbours
      (Huh et al. 2024, Appendix A, Eq. 16). Bounded above by 1 at every k by
      Cauchy-Schwarz. Like CKA it can sit marginally below 0 for near-orthogonal
      data. It agrees with the default at small k and ranks model pairs almost
      identically, so it is a drop-in for analyses that go to larger k or need
      the 1 upper bound. See ``_cknna_centered_bounded``.

    The distance-agnostic path (neighbour overlap count) is already in [0, 1]
    for both variants, so ``cknna_bounded`` only changes the distance-aware
    (HSIC) path.

    Score range: bounded above by 1 with ``cknna_bounded=True`` (and near 0 at
    the low end, as for CKA); with the default, within 1 only at small k.
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

        # Bounded variant: centre first, then mask (see class docstring). Only
        # the distance-aware path can exceed 1, so the flag only changes it.
        if config.cknna_bounded and not distance_agnostic:
            return float(_cknna_centered_bounded(K, L, k).item())

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

        bounded = config.cknna_bounded and not distance_agnostic
        if bounded:
            null_scores: list[float] = []
            for perm in perms:
                L_perm = L[perm][:, perm]
                null_scores.append(float(_cknna_centered_bounded(K, L_perm, k).item()))
            return null_scores

        # sim_kk is invariant (K doesn't change)
        sim_kk = _similarity(K, K, k)

        null_scores = []
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
