"""RSA (Representational Similarity Analysis) metrics.

This module provides RSA metrics:
- rsa: Spearman correlation of RDM vectors

RSA uses Representational Dissimilarity Matrices (RDMs) to compare
how different representations structure data.
"""

from __future__ import annotations

import torch

from .base import BaseMetric, MetricConfig
from .registry import register_metric
from .utils import EPS


def _rankdata_torch(a: torch.Tensor) -> torch.Tensor:
    """Torch-based rankdata with average ranks for ties."""
    a_flat = a.flatten()
    order = torch.argsort(a_flat)
    ranks = torch.empty_like(a_flat, dtype=torch.float)
    ranks[order] = torch.arange(len(a_flat), device=a.device).float()

    sorted_vals = a_flat[order]
    diffs = torch.diff(sorted_vals)
    boundaries = torch.nonzero(diffs != 0, as_tuple=False).flatten() + 1
    boundaries = torch.cat(
        [
            torch.tensor([0], device=a.device),
            boundaries,
            torch.tensor([len(a_flat)], device=a.device),
        ]
    )
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i].item(), boundaries[i + 1].item()
        if e - s > 1:
            avg = (s + e - 1) / 2.0
            ranks[order[s:e]] = avg
    return ranks.reshape(a.shape).float()


def _rankdata_torch_batch(a: torch.Tensor) -> torch.Tensor:
    """Batch rankdata with average ranks for ties (shape: B x M)."""
    order = torch.argsort(a, dim=1)
    sorted_vals = torch.gather(a, 1, order)
    diffs = sorted_vals[:, 1:] != sorted_vals[:, :-1]
    first = torch.ones((a.size(0), 1), device=a.device, dtype=torch.bool)
    seg_starts = torch.cat([first, diffs], dim=1)
    seg_ids = torch.cumsum(seg_starts, dim=1).to(torch.long) - 1

    positions = torch.arange(a.size(1), device=a.device, dtype=torch.float).expand_as(
        sorted_vals
    )
    seg_sums = torch.zeros_like(sorted_vals, dtype=torch.float)
    seg_counts = torch.zeros_like(sorted_vals, dtype=torch.float)
    seg_sums.scatter_add_(1, seg_ids, positions)
    seg_counts.scatter_add_(1, seg_ids, torch.ones_like(positions))
    avg = seg_sums / seg_counts.clamp_min(1.0)
    ranks_sorted = avg.gather(1, seg_ids)
    ranks = torch.zeros_like(sorted_vals, dtype=torch.float)
    ranks.scatter_(1, order, ranks_sorted)
    return ranks


def _spearman_corr_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Spearman correlation using torch."""
    rx = _rankdata_torch(x)
    ry = _rankdata_torch(y)
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = torch.norm(rx_c) * torch.norm(ry_c) + EPS
    return torch.sum(rx_c * ry_c) / denom


def _rsa_vector(X: torch.Tensor) -> torch.Tensor:
    """Return the upper-triangular (vectorized) RDM using Euclidean distances."""
    return torch.pdist(X, p=2)


def _rsa_auto_pair_samples(total_pairs: int) -> int | None:
    """Auto-determine pair sampling for large RDMs."""
    auto_samples = 8192
    min_pairs = 2 * auto_samples
    if total_pairs >= min_pairs:
        return min(auto_samples, total_pairs)
    return None


@register_metric
class RSA(BaseMetric):
    """Representational Similarity Analysis metric.

    Computes Spearman correlation between RDM vectors.
    Uses Euclidean distance for RDM construction.

    Score range: [-1, 1] where 1 means perfect correlation.
    """

    name = "rsa"
    min_score = -1.0
    max_score = 1.0
    supports_calibration = True
    supports_caching = True
    cache_keys = ("rdm_X", "rdm_Y", "dist_Y")

    def _compute_raw(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> float:
        cache = config.cache
        n = X.shape[0]
        total_pairs = (n * (n - 1)) // 2
        pair_samples = config.pair_samples

        # Validate pair_samples
        if pair_samples is not None and pair_samples <= 0:
            raise ValueError("pair_samples must be a positive integer or None")
        if pair_samples is not None and pair_samples >= total_pairs:
            pair_samples = None  # Use full RDM

        # Auto-enable pair sampling for large RDMs
        if pair_samples is None:
            pair_samples = _rsa_auto_pair_samples(total_pairs)

        if pair_samples is None:
            # Full RDM computation
            if "rdm_X" in cache:
                vx = cache["rdm_X"]
            else:
                vx = _rsa_vector(X)
                cache["rdm_X"] = vx

            if "rdm_Y" in cache:
                vy = cache["rdm_Y"]
            else:
                vy = _rsa_vector(Y)
                cache["rdm_Y"] = vy
        else:
            # Sampled pairs for efficiency - cache indices for null computation
            idx0_full, idx1_full = torch.triu_indices(n, n, offset=1, device=X.device)
            if "sample_idx" in cache and cache.get("sample_size") == pair_samples:
                sample_idx = cache["sample_idx"]
            else:
                sample_idx = torch.randperm(total_pairs, device=X.device)[:pair_samples]
                cache["sample_idx"] = sample_idx
                cache["sample_size"] = pair_samples
            idx0 = idx0_full[sample_idx]
            idx1 = idx1_full[sample_idx]
            vx = torch.norm(X[idx0] - X[idx1], dim=1)
            vy = torch.norm(Y[idx0] - Y[idx1], dim=1)
            # Cache vx for null computation
            cache["vx_sampled"] = vx

        return float(_spearman_corr_torch(vx, vy).item())

    def _compute_null_distribution(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> list[float]:
        """Optimized null distribution with batched Spearman computation."""
        n = X.shape[0]
        device = config.device
        num_perms = config.num_permutations
        batch_size = config.batch_size if config.batch_size else 32
        pair_samples = config.pair_samples
        total_pairs = (n * (n - 1)) // 2

        # Validate batch_size
        if config.batch_size is not None and config.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer or None")

        # Validate pair_samples
        if pair_samples is not None and pair_samples <= 0:
            raise ValueError("pair_samples must be a positive integer or None")
        if pair_samples is not None and pair_samples >= total_pairs:
            pair_samples = None  # Use full RDM

        # Auto-enable pair sampling for large RDMs
        if pair_samples is None:
            pair_samples = _rsa_auto_pair_samples(total_pairs)

        # Get or generate permutations
        if config.perms is not None:
            perms = config.perms.to(device)
            if perms.dim() != 2 or perms.size(1) != n:
                raise ValueError("perms must have shape (B, n)")
        else:
            perms = torch.stack(
                [torch.randperm(n, device=device) for _ in range(num_perms)]
            )

        cache = config.cache
        X = X.to(device)
        Y = Y.to(device)

        if pair_samples is None:
            # Full RDM computation
            if "rdm_X" in cache:
                vx = cache["rdm_X"].to(device)
            else:
                vx = _rsa_vector(X)
                cache["rdm_X"] = vx

            if "dist_Y" in cache:
                dist_y = cache["dist_Y"].to(device)
            else:
                dist_y = torch.cdist(Y, Y, p=2)
                cache["dist_Y"] = dist_y

            idx0, idx1 = torch.triu_indices(n, n, offset=1)

            # Batched computation
            rx = _rankdata_torch(vx)
            rx_c = rx - rx.mean()
            rx_norm = torch.norm(rx_c) + EPS
            null_scores = []

            for start in range(0, num_perms, batch_size):
                batch = perms[start : start + batch_size]
                vy_perm = dist_y[batch[:, idx0], batch[:, idx1]]
                ry = _rankdata_torch_batch(vy_perm)
                ry_c = ry - ry.mean(dim=1, keepdim=True)
                denom = (ry_c.norm(dim=1) * rx_norm) + EPS
                corr = (ry_c * rx_c).sum(dim=1) / denom
                null_scores.extend(corr.detach().cpu().tolist())
        else:
            # Sampled pairs - use cached indices from _compute_raw if available
            idx0_full, idx1_full = torch.triu_indices(n, n, offset=1, device=device)
            if "sample_idx" in cache and cache.get("sample_size") == pair_samples:
                sample_idx = cache["sample_idx"].to(device)
            else:
                sample_idx = torch.randperm(total_pairs, device=device)[:pair_samples]
                cache["sample_idx"] = sample_idx
                cache["sample_size"] = pair_samples
            idx0 = idx0_full[sample_idx]
            idx1 = idx1_full[sample_idx]

            # Use cached vx if available
            if "vx_sampled" in cache:
                vx = cache["vx_sampled"].to(device)
            else:
                vx = torch.norm(X[idx0] - X[idx1], dim=1)

            # Precompute full distance matrix for efficient permutation indexing
            # (avoids materializing huge (batch, pairs, d) tensors per batch)
            if "dist_Y" in cache:
                dist_y = cache["dist_Y"].to(device)
            else:
                dist_y = torch.cdist(Y, Y, p=2)
                cache["dist_Y"] = dist_y

            rx = _rankdata_torch(vx)
            rx_c = rx - rx.mean()
            rx_norm = torch.norm(rx_c) + EPS
            null_scores = []

            for start in range(0, num_perms, batch_size):
                batch = perms[start : start + batch_size]
                vy_perm = dist_y[batch[:, idx0], batch[:, idx1]]
                ry = _rankdata_torch_batch(vy_perm)
                ry_c = ry - ry.mean(dim=1, keepdim=True)
                denom = (ry_c.norm(dim=1) * rx_norm) + EPS
                corr = (ry_c * rx_c).sum(dim=1) / denom
                null_scores.extend(corr.detach().cpu().tolist())

        return null_scores
