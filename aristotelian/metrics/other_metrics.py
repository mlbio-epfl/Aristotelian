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
    Score = 1 - (||aligned_X - Y|| / ||Y||)

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
        X_np = X.detach().cpu().numpy()
        Y_np = Y.detach().cpu().numpy()
        return procrustes_score(X_np, Y_np)


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
