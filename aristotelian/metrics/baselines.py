"""Utilities to run null baselines for multiple similarity metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .. import (
    mutual_knn_overlap,
    rsa_vector,
    sg_cka_kernel,
    sg_cka_linear,
    sg_knn,
    sg_rsa,
    standard_cka,
)
from .aggregation import tau_order_statistic
from .calibration import compute_null_variants
from .rsa import _spearman_corr_torch
from .utils import sample_student_t


@dataclass
class BaselineSummary:
    mean: float
    std: float
    zeros_fraction: float
    tau: float
    pvalues: Sequence[float]
    raw_mean: float
    raw_std: float
    mean_null: float
    median_null: float
    std_null: float
    null_centered_mean: float
    z_mean: float
    ari_mean: float


@dataclass
class NullTypeSummary:
    tau: float
    mean_null: float
    median_null: float
    std_null: float
    null_centered: float
    z: float
    ari: float


def _aggregate(results) -> BaselineSummary:
    gated = np.array([r.gated for r in results], dtype=float)
    raw = np.array([r.raw for r in results], dtype=float)
    taus = np.array([r.tau for r in results], dtype=float)
    pvals = np.array([r.pvalue for r in results], dtype=float)
    mean_null = np.array([r.mean_null for r in results], dtype=float)
    median_null = np.array([r.median_null for r in results], dtype=float)
    std_null = np.array([r.std_null for r in results], dtype=float)
    null_centered = np.array([r.null_centered for r in results], dtype=float)
    z = np.array([r.z for r in results], dtype=float)
    ari = np.array([r.ari for r in results], dtype=float)
    return BaselineSummary(
        mean=float(gated.mean()),
        std=float(gated.std()),
        zeros_fraction=float(np.mean(gated == 0.0)),
        tau=float(taus.mean()),
        pvalues=pvals,
        raw_mean=float(raw.mean()),
        raw_std=float(raw.std()),
        mean_null=float(mean_null.mean()),
        median_null=float(median_null.mean()),
        std_null=float(std_null.mean()),
        null_centered_mean=float(null_centered.mean()),
        z_mean=float(z.mean()),
        ari_mean=float(ari.mean()),
    )


def run_null_baselines(
    n: int,
    d: int,
    *,
    num_trials: int = 50,
    num_permutations: int = 200,
    quantiles: Sequence[float] = (0.95,),
    null_type: str = "gaussian",
    k_knn: int = 10,
    device: str = "cpu",
    seed: int | None = None,
    labels: torch.Tensor | None = None,
) -> dict[str, dict[float, BaselineSummary]]:
    """
    Run multiple null baselines for sgCKA/sg-kNN/sgRSA.

    null_type: "gaussian" (independent), "heavy" (Student-t df=3), "shuffled" (permute Y),
        "feature_permute" (permute Y features), "within_class" (permute within labels)
    """
    if null_type not in {
        "gaussian",
        "heavy",
        "shuffled",
        "feature_permute",
        "within_class",
    }:
        raise ValueError(
            "null_type must be one of {'gaussian','heavy','shuffled','feature_permute','within_class'}"
        )
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)

    def sample_pair():
        if null_type == "gaussian":
            X = torch.randn(n, d, generator=rng, device=device)
            Y = torch.randn(n, d, generator=rng, device=device)
        elif null_type == "heavy":
            X = sample_student_t(n, d, df=3, device=device, rng=rng)
            Y = sample_student_t(n, d, df=3, device=device, rng=rng)
        elif null_type == "shuffled":
            X = torch.randn(n, d, generator=rng, device=device)
            idx = torch.randperm(n, generator=rng, device=device)
            Y = X[idx]
        elif null_type == "feature_permute":
            X = torch.randn(n, d, generator=rng, device=device)
            Y = torch.randn(n, d, generator=rng, device=device)
            perm = torch.randperm(d, generator=rng, device=device)
            Y = Y[:, perm]
        else:  # within_class
            if labels is None:
                raise ValueError("labels are required for within_class nulls")
            X = torch.randn(n, d, generator=rng, device=device)
            Y = X + 0.5 * torch.randn(n, d, generator=rng, device=device)
            Y = _permute_within_class(Y, labels, rng)
        return X, Y

    metrics = ["sgcka_lin", "sgcka_rbf", "sgknn", "sgrsa"]
    # nested dict: metric -> quantile -> list of results
    accum: dict[str, dict[float, list]] = {
        m: {q: [] for q in quantiles} for m in metrics
    }

    for _ in range(num_trials):
        X, Y = sample_pair()
        for q in quantiles:
            accum["sgcka_lin"][q].append(
                sg_cka_linear(
                    X, Y, num_permutations=num_permutations, quantile=q, device=device
                )
            )
            accum["sgcka_rbf"][q].append(
                sg_cka_kernel(
                    X, Y, num_permutations=num_permutations, quantile=q, device=device
                )
            )
            accum["sgknn"][q].append(
                sg_knn(
                    X,
                    Y,
                    k=k_knn,
                    num_permutations=num_permutations,
                    quantile=q,
                    device=device,
                )
            )
            accum["sgrsa"][q].append(
                sg_rsa(
                    X, Y, num_permutations=num_permutations, quantile=q, device=device
                )
            )

    return {
        m: {q: _aggregate(v) for q, v in qdict.items()} for m, qdict in accum.items()
    }


def _permute_within_class(
    Y: torch.Tensor, labels: torch.Tensor, rng: torch.Generator
) -> torch.Tensor:
    labels_t = labels.to(Y.device)
    Yp = Y.clone()
    for label in torch.unique(labels_t):
        idx = torch.nonzero(labels_t == label, as_tuple=False).flatten()
        if idx.numel() <= 1:
            continue
        perm = idx[torch.randperm(idx.numel(), generator=rng, device=Y.device)]
        Yp[idx] = Y[perm]
    return Yp


def run_null_type_ablation(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    metric: str,
    null_types: Sequence[str],
    num_permutations: int = 200,
    quantile: float = 0.95,
    k_knn: int = 10,
    labels: torch.Tensor | None = None,
    device: str = "cpu",
    seed: int | None = None,
) -> dict[str, NullTypeSummary]:
    """Compare null types (sample/feature/within-class) for a given metric."""
    X = X.to(device)
    Y = Y.to(device)
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)

    if metric == "sgcka_lin":

        def raw_fn(a, b):
            return standard_cka(a, b, mode="linear")

        min_score, max_score = 0.0, 1.0
    elif metric == "sgcka_rbf":

        def raw_fn(a, b):
            return standard_cka(a, b, mode="kernel")

        min_score, max_score = 0.0, 1.0
    elif metric == "sgknn":

        def raw_fn(a, b):
            return mutual_knn_overlap(a, b, k=k_knn)

        min_score, max_score = 0.0, 1.0
    elif metric == "sgrsa":

        def raw_fn(a, b):
            return float(_spearman_corr_torch(rsa_vector(a), rsa_vector(b)).item())

        min_score, max_score = -1.0, 1.0
    else:
        raise ValueError(
            "metric must be one of {'sgcka_lin','sgcka_rbf','sgknn','sgrsa'}"
        )

    def permute(kind: str) -> torch.Tensor:
        if kind == "sample":
            idx = torch.randperm(Y.shape[0], generator=rng, device=Y.device)
            return Y[idx]
        if kind == "feature":
            idx = torch.randperm(Y.shape[1], generator=rng, device=Y.device)
            return Y[:, idx]
        if kind == "within_class":
            if labels is None:
                raise ValueError("labels are required for within_class nulls")
            return _permute_within_class(Y, labels, rng)
        raise ValueError("null type must be one of {'sample','feature','within_class'}")

    out: dict[str, NullTypeSummary] = {}
    raw = float(raw_fn(X, Y))
    for kind in null_types:
        null_scores = []
        for _ in range(num_permutations):
            Yp = permute(kind)
            null_scores.append(float(raw_fn(X, Yp)))
        tau = tau_order_statistic(null_scores, quantile, obs=raw)
        variants = compute_null_variants(
            raw, null_scores, min_score=min_score, max_score=max_score
        )
        out[kind] = NullTypeSummary(
            tau=tau,
            mean_null=variants.mean_null,
            median_null=variants.median_null,
            std_null=variants.std_null,
            null_centered=variants.null_centered,
            z=variants.z,
            ari=variants.ari,
        )
    return out
