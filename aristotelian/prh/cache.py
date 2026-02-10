"""PRH cache helpers for kNN and Gram matrices."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from ..experiments import layerwise_engine as lwe


def _cached_feats_match(payload: dict, expected_len: int) -> bool:
    cached = payload.get("feats")
    return cached is not None and cached.shape[0] == expected_len


def build_knn_cache(
    feats: torch.Tensor | Sequence[torch.Tensor],
    *,
    topk: int,
    normalize: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    return lwe.build_knn_cache(feats, topk=topk, normalize=normalize)


def build_knn_cache_with_indices(
    feats: torch.Tensor | Sequence[torch.Tensor],
    *,
    topk: int,
    normalize: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    return lwe.build_knn_cache_with_indices(feats, topk=topk, normalize=normalize)


def build_gram_cache(
    feats: torch.Tensor | Sequence[torch.Tensor],
    *,
    normalize: bool = True,
    kernel: str = "linear",
    rbf_sigma: float = 1.0,
) -> List[torch.Tensor]:
    return lwe.build_gram_cache(
        feats, normalize=normalize, kernel=kernel, rbf_sigma=rbf_sigma
    )
