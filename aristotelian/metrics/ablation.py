"""Ablation utilities for preprocessing choices."""

from __future__ import annotations

from typing import Iterable

import torch

from .utils import resolve_sg_metric as _resolve_metric


def _preprocess(X: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return X
    if mode == "center":
        return X - X.mean(0, keepdim=True)
    if mode == "whiten":
        Xc = X - X.mean(0, keepdim=True)
        cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
        evals, evecs = torch.linalg.eigh(cov)
        eps = 1e-6
        W = evecs @ torch.diag(1.0 / torch.sqrt(evals + eps)) @ evecs.T
        return Xc @ W
    raise ValueError("mode must be one of {'none','center','whiten'}")


def run_preprocess_ablation(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    metric: str,
    preprocess_options: Iterable[str] = ("none", "center", "whiten"),
    num_permutations: int = 200,
    quantile: float = 0.95,
    k_knn: int = 10,
) -> dict[str, dict[str, float]]:
    """Run metric across preprocessing modes for ablation tables."""
    metric_fn = _resolve_metric(metric)
    out: dict[str, dict[str, float]] = {}
    for mode in preprocess_options:
        Xp = _preprocess(X, mode)
        Yp = _preprocess(Y, mode)
        kwargs = dict(num_permutations=num_permutations, quantile=quantile)
        if metric == "sgknn":
            kwargs["k"] = k_knn
        res = metric_fn(Xp, Yp, **kwargs)
        out[mode] = {
            "raw": float(res.raw),
            "gated": float(res.gated),
            "z": float(res.z),
            "ari": float(res.ari),
        }
    return out
