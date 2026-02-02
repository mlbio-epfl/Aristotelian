"""PRH alignment helpers."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..experiments import layerwise_engine as lwe
from ..metrics.aggregation import (
    SimpleMetric,
    agg_max,
    compute_null_summary,
    gated_rescaled,
    permutation_null_aggregated,
)
from ..metrics.api import prh_metric_spec
from .layers import _as_layers, _normalize_layers
from .preprocess import prepare_features


def compute_alignment_gated_cached(
    x_layers: Sequence[torch.Tensor],
    y_layers: Sequence[torch.Tensor],
    x_masks: Sequence[torch.Tensor],
    y_masks: Sequence[torch.Tensor],
    *,
    topk: int = 10,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    return lwe.compute_alignment_gated_cached(
        x_layers,
        y_layers,
        x_masks,
        y_masks,
        topk=topk,
        num_permutations=num_permutations,
        alpha=alpha,
        seed=seed,
    )


def compute_alignment_gated_knn_cached(
    x_knn: Sequence[torch.Tensor],
    y_knn: Sequence[torch.Tensor],
    *,
    topk: int = 10,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    return lwe.compute_alignment_gated_knn_cached(
        x_knn,
        y_knn,
        topk=topk,
        num_permutations=num_permutations,
        alpha=alpha,
        seed=seed,
    )


def compute_alignment_gated_cknna_cached(
    x_grams: Sequence[torch.Tensor],
    y_grams: Sequence[torch.Tensor],
    *,
    topk: int = 10,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
    unbiased: bool = True,
) -> Dict[str, float | Tuple[int, int]]:
    return lwe.compute_alignment_gated_cknna_cached(
        x_grams,
        y_grams,
        topk=topk,
        num_permutations=num_permutations,
        alpha=alpha,
        seed=seed,
        unbiased=unbiased,
    )


def compute_alignment_gated_cycle_knn_cached(
    x_knn: Sequence[torch.Tensor],
    y_knn: Sequence[torch.Tensor],
    *,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    return lwe.compute_alignment_gated_cycle_knn_cached(
        x_knn,
        y_knn,
        num_permutations=num_permutations,
        alpha=alpha,
        seed=seed,
    )


def compute_alignment_gated_svcca_cached(
    x_layers: Sequence[torch.Tensor],
    y_layers: Sequence[torch.Tensor],
    *,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    return lwe.compute_alignment_gated_svcca_cached(
        x_layers,
        y_layers,
        num_permutations=num_permutations,
        alpha=alpha,
        seed=seed,
    )


def compute_alignment_gated_pwcca_cached(
    x_layers: Sequence[torch.Tensor],
    y_layers: Sequence[torch.Tensor],
    *,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    return lwe.compute_alignment_gated_pwcca_cached(
        x_layers,
        y_layers,
        num_permutations=num_permutations,
        alpha=alpha,
        seed=seed,
    )


def compute_alignment_gated_procrustes_cached(
    x_layers: Sequence[torch.Tensor],
    y_layers: Sequence[torch.Tensor],
    *,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    return lwe.compute_alignment_gated_procrustes_cached(
        x_layers,
        y_layers,
        num_permutations=num_permutations,
        alpha=alpha,
        seed=seed,
    )


def compute_alignment_gated_cka_cached(
    x_grams: Sequence[torch.Tensor],
    y_grams: Sequence[torch.Tensor],
    *,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
    unbiased: bool = False,
) -> Dict[str, float | Tuple[int, int]]:
    return lwe.compute_alignment_gated_cka_cached(
        x_grams,
        y_grams,
        num_permutations=num_permutations,
        alpha=alpha,
        seed=seed,
        unbiased=unbiased,
    )


def compute_alignment_gated(
    x_feats: torch.Tensor | Sequence[torch.Tensor],
    y_feats: torch.Tensor | Sequence[torch.Tensor],
    *,
    metric: str = "mutual_knn",
    topk: int = 10,
    normalize: bool = True,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    if metric not in {
        "cycle_knn",
        "knn",
        "mutual_knn",
        "lcs_knn",
        "cka",
        "cka_lin",
        "cka_rbf",
        "unbiased_cka",
        "cknna",
        "svcca",
        "pwcca",
        "procrustes",
        "edit_distance_knn",
    }:
        raise ValueError(f"Unsupported metric {metric}")
    x_layers = _as_layers(x_feats)
    y_layers = _as_layers(y_feats)
    if normalize:
        x_layers = _normalize_layers(x_layers)
        y_layers = _normalize_layers(y_layers)

    metric_compute, max_value = prh_metric_spec(metric, topk=topk)

    metric_fn = SimpleMetric(
        name=metric,
        max_value=max_value,
        compute=metric_compute,
    )

    S = torch.empty((len(x_layers), len(y_layers)), device=x_layers[0].device)
    for i, x in enumerate(x_layers):
        for j, y in enumerate(y_layers):
            S[i, j] = metric_fn.compute(x, y)

    agg = agg_max(S, return_indices=True)
    T_obs = float(agg.value)
    best_indices = (
        (int(agg.indices["i"]), int(agg.indices["j"])) if agg.indices else (0, 0)
    )

    null_samples = permutation_null_aggregated(
        x_layers,
        y_layers,
        metric_fn,
        agg_max,
        num_permutations=num_permutations,
        seed=seed,
    )
    summary = compute_null_summary(null_samples, T_obs=T_obs, alpha=alpha)
    g_score = gated_rescaled(
        T_obs, tau_alpha=summary["tau_alpha"], s_max=metric_fn.max_value
    )

    return {
        "raw_score": T_obs,
        "best_indices": best_indices,
        "p_value": summary["p_value"],
        "tau_alpha": summary["tau_alpha"],
        "tail_strength": summary["tail_strength"],
        "g_score": g_score,
        "mu0": summary["mu0"],
        "sd0": summary["sd0"],
    }


def compute_outlier_ablation(
    feats_x: Sequence[torch.Tensor],
    feats_y: Sequence[torch.Tensor],
    *,
    q_values: Sequence[float],
    metric: str = "mutual_knn",
    topk: int = 10,
    normalize: bool = True,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
    device: str = "cpu",
    show_progress: bool = True,
) -> Dict[str, Sequence[float]]:
    mean_raw = []
    mean_gated = []
    mean_pvalue = []
    mean_tau = []
    total = len(q_values) * len(feats_x) * len(feats_y)
    pbar = tqdm(total=total, desc="PRH outlier ablation", disable=not show_progress)
    for q in q_values:
        raw_scores = []
        gated_scores = []
        pvalues = []
        taus = []
        prepped_x = [
            prepare_features(x, q=q, exact=False, device=device) for x in feats_x
        ]
        prepped_y = [
            prepare_features(y, q=q, exact=False, device=device) for y in feats_y
        ]
        for x_prepped in prepped_x:
            for y_prepped in prepped_y:
                res = compute_alignment_gated(
                    x_prepped,
                    y_prepped,
                    metric=metric,
                    topk=topk,
                    normalize=normalize,
                    num_permutations=num_permutations,
                    alpha=alpha,
                    seed=seed,
                )
                raw_scores.append(res["raw_score"])
                gated_scores.append(res["g_score"])
                pvalues.append(res["p_value"])
                taus.append(res["tau_alpha"])
                pbar.update(1)
        mean_raw.append(float(np.mean(raw_scores)) if raw_scores else 0.0)
        mean_gated.append(float(np.mean(gated_scores)) if gated_scores else 0.0)
        mean_pvalue.append(float(np.mean(pvalues)) if pvalues else 1.0)
        mean_tau.append(float(np.mean(taus)) if taus else 0.0)
    pbar.close()
    return {
        "q_values": list(q_values),
        "mean_raw": mean_raw,
        "mean_gated": mean_gated,
        "mean_pvalue": mean_pvalue,
        "mean_tau": mean_tau,
    }
