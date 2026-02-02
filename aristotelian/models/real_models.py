"""Real-model reproducibility utilities (seed consistency, width/depth scaling)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from ..metrics.utils import resolve_sg_metric as _resolve_metric


def _pairwise_layer_scores(
    acts_a: List[torch.Tensor],
    acts_b: List[torch.Tensor],
    *,
    metric: str,
    num_permutations: int,
    quantile: float,
    k_knn: int,
) -> Dict[str, np.ndarray]:
    metric_fn = _resolve_metric(metric)
    L1, L2 = len(acts_a), len(acts_b)
    raw = np.zeros((L1, L2))
    gated = np.zeros((L1, L2))
    z = np.zeros((L1, L2))
    ari = np.zeros((L1, L2))
    for i, Xa in enumerate(acts_a):
        for j, Xb in enumerate(acts_b):
            kwargs = dict(num_permutations=num_permutations, quantile=quantile)
            if metric == "sgknn":
                kwargs["k"] = k_knn
            res = metric_fn(Xa, Xb, **kwargs)
            raw[i, j] = res.raw
            gated[i, j] = res.gated
            z[i, j] = res.z
            ari[i, j] = res.ari
    return {"raw": raw, "gated": gated, "z": z, "ari": ari}


def run_seed_reproducibility(
    *,
    activations: Dict[str, List[torch.Tensor]],
    metric: str,
    num_permutations: int = 200,
    quantile: float = 0.95,
    k_knn: int = 10,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compare same-architecture seeds by layer-pair similarity matrices."""
    keys = list(activations.keys())
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for i, key_a in enumerate(keys):
        for key_b in keys[i:]:
            scores = _pairwise_layer_scores(
                activations[key_a],
                activations[key_b],
                metric=metric,
                num_permutations=num_permutations,
                quantile=quantile,
                k_knn=k_knn,
            )
            out[f"{key_a}__{key_b}"] = scores
    return out


def run_width_depth_scaling(
    *,
    activations: Dict[str, List[torch.Tensor]],
    metric: str,
    num_permutations: int = 200,
    quantile: float = 0.95,
    k_knn: int = 10,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compare width/depth variants by layer-pair similarity matrices."""
    return run_seed_reproducibility(
        activations=activations,
        metric=metric,
        num_permutations=num_permutations,
        quantile=quantile,
        k_knn=k_knn,
    )
