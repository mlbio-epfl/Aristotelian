"""kNN-based representation similarity metrics.

This module provides kNN-based metrics:
- mutual_knn: Mutual k-nearest neighbor overlap (PRH metric)
- cycle_knn: Cycle-consistent kNN accuracy
- lcs_knn: Longest common subsequence of kNN
- edit_distance_knn: Edit distance normalized kNN similarity

All metrics use cosine similarity (dot product) for neighbor selection,
matching the original Platonic Representation Hypothesis implementation.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import BaseMetric, MetricConfig
from .registry import register_metric
from .utils import knn_indicator


def _compute_knn_indices(feats: torch.Tensor, k: int) -> torch.Tensor:
    """Get k-nearest neighbor indices using cosine similarity."""
    sim = feats @ feats.T
    sim.fill_diagonal_(-1e8)
    return torch.topk(sim, k=k, largest=True).indices


def _get_knn_from_cache(
    cache: dict, *, cache_key: str, feats: torch.Tensor, k: int
) -> torch.Tensor:
    k_key = f"{cache_key}_k"
    if cache_key in cache and cache.get(k_key) == k:
        return cache[cache_key]
    knn = _compute_knn_indices(feats, k)
    cache[cache_key] = knn
    cache[k_key] = k
    return knn


def _get_knn_mask_from_cache(
    cache: dict, *, cache_key: str, knn: torch.Tensor, n: int, k: int
) -> torch.Tensor:
    k_key = f"{cache_key}_k"
    if cache_key in cache and cache.get(k_key) == k:
        return cache[cache_key]
    mask = knn_indicator(knn, n)
    cache[cache_key] = mask
    cache[k_key] = k
    return mask


@register_metric
class MutualKNN(BaseMetric):
    """Mutual k-nearest neighbor overlap metric.

    Measures the average overlap between k-nearest neighbors computed
    in each representation space. This is the core PRH metric.

    Score range: [0, 1] where 1 means perfect neighbor agreement.
    """

    name = "mutual_knn"
    min_score = 0.0
    max_score = 1.0
    supports_caching = True
    cache_keys = ("knn_X", "knn_Y", "mask_X")

    def _compute_raw(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> float:
        n = X.shape[0]
        k = config.topk if config.topk is not None else 10
        if k <= 0 or k >= n:
            raise ValueError("k must satisfy 0 < k < n")

        cache = config.cache
        knn_X = _get_knn_from_cache(cache, cache_key="knn_X", feats=X, k=k)
        knn_Y = _get_knn_from_cache(cache, cache_key="knn_Y", feats=Y, k=k)
        mask_X = _get_knn_mask_from_cache(
            cache, cache_key="mask_X", knn=knn_X, n=n, k=k
        )

        mask_Y = knn_indicator(knn_Y, n)
        overlap = (mask_X & mask_Y).sum(dim=1).float() / float(k)
        return float(overlap.mean().item())

    def _compute_null_distribution(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> list[float]:
        """Optimized null distribution using kNN cache."""
        n = X.shape[0]
        k = config.topk if config.topk is not None else 10
        device = config.device
        num_perms = config.num_permutations

        cache = config.cache
        knn_X = _get_knn_from_cache(cache, cache_key="knn_X", feats=X, k=k)
        knn_Y = _get_knn_from_cache(cache, cache_key="knn_Y", feats=Y, k=k)
        mask_X = _get_knn_mask_from_cache(
            cache, cache_key="mask_X", knn=knn_X, n=n, k=k
        )

        # Generate or use provided permutations
        if config.perms is not None:
            perms = config.perms.to(device)
            if perms.dim() != 2 or perms.size(1) != n:
                raise ValueError("perms must have shape (B, n)")
        else:
            perms = torch.stack(
                [torch.randperm(n, device=device) for _ in range(num_perms)]
            )

        # Efficient null computation using kNN index permutation
        knn_Y = knn_Y.to(device)
        inv_base = torch.arange(n, device=device)
        null_scores = []

        for perm in perms:
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = inv_base
            knn_Y_perm = inv_perm[knn_Y[perm]]
            mask_Y_perm = knn_indicator(knn_Y_perm, n)
            overlap = (mask_X & mask_Y_perm).sum(dim=1).float() / float(k)
            null_scores.append(float(overlap.mean().item()))

        return null_scores


@register_metric
class CycleKNN(BaseMetric):
    """Cycle-consistent kNN accuracy metric.

    Measures how often a point's neighbors in X map back to itself
    when going through Y's neighbor structure.

    Score range: [0, 1] where 1 means perfect cycle consistency.
    """

    name = "cycle_knn"
    min_score = 0.0
    max_score = 1.0
    supports_caching = True
    cache_keys = ("knn_X", "knn_Y")

    def _compute_raw(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> float:
        n = X.shape[0]
        k = config.topk if config.topk is not None else 10
        if k <= 0 or k >= n:
            raise ValueError("k must satisfy 0 < k < n")

        cache = config.cache
        knn_X = _get_knn_from_cache(cache, cache_key="knn_X", feats=X, k=k)
        knn_Y = _get_knn_from_cache(cache, cache_key="knn_Y", feats=Y, k=k)

        # Cycle through: X's neighbors -> Y's neighbors -> check if original
        cycle_idx = knn_X[knn_Y]
        acc = (cycle_idx == torch.arange(n, device=X.device).view(-1, 1, 1)).float()
        acc = acc.view(n, -1).max(dim=1).values.mean()
        return float(acc.item())


@register_metric
class LCSKNN(BaseMetric):
    """Longest common subsequence of kNN metric.

    Measures the average LCS length between kNN neighbor lists.

    Score range: [0, k] where k is the number of neighbors.
    """

    name = "lcs_knn"
    min_score = 0.0
    max_score = 1.0  # Normalized
    supports_calibration = False  # LCS is expensive, calibration typically not used

    def _compute_raw(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> float:
        n = X.shape[0]
        k = config.topk if config.topk is not None else 10
        if k <= 0 or k >= n:
            raise ValueError("k must satisfy 0 < k < n")

        knn_X = _compute_knn_indices(X, k)
        knn_Y = _compute_knn_indices(Y, k)

        def lcs_length(x: np.ndarray, y: np.ndarray) -> int:
            m, ln = len(x), len(y)
            dp = [[0] * (ln + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, ln + 1):
                    if x[i - 1] == y[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[m][ln]

        knn_X_np = knn_X.cpu().numpy()
        knn_Y_np = knn_Y.cpu().numpy()
        lcs_scores = [lcs_length(knn_X_np[i], knn_Y_np[i]) for i in range(n)]
        return float(np.mean(lcs_scores))


@register_metric
class EditDistanceKNN(BaseMetric):
    """Edit distance normalized kNN similarity.

    Measures similarity as 1 - (edit_distance / k).

    Score range: [0, 1] where 1 means identical neighbor lists.

    Note: Requires torchaudio to be installed.
    """

    name = "edit_distance_knn"
    min_score = 0.0
    max_score = 1.0
    supports_calibration = False  # Edit distance is expensive

    def _compute_raw(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> float:
        try:
            import torchaudio.functional as TAF
        except ImportError as e:
            raise ImportError(
                "EditDistanceKNN requires torchaudio. "
                "Install it with: pip install torchaudio"
            ) from e

        n = X.shape[0]
        k = config.topk if config.topk is not None else 10
        if k <= 0 or k >= n:
            raise ValueError("k must satisfy 0 < k < n")

        knn_X = _compute_knn_indices(X, k)
        knn_Y = _compute_knn_indices(Y, k)

        distances = []
        for i in range(n):
            ed = TAF.edit_distance(knn_X[i].cpu(), knn_Y[i].cpu())
            distances.append(float(ed))

        edit_dist = torch.tensor(distances)
        return float(1.0 - torch.mean(edit_dist) / k)


# Register "knn" as an alias for "mutual_knn" for backward compatibility.
# This is done after import to ensure MutualKNN is registered first.
def _register_knn_alias() -> None:
    """Register knn as alias for mutual_knn after module load."""
    from .registry import MetricRegistry

    if MetricRegistry.has("mutual_knn") and not MetricRegistry.has("knn"):
        MetricRegistry.register_alias("mutual_knn", "knn")


_register_knn_alias()
