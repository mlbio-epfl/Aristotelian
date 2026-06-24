"""Aggregation-aware null calibration and bootstrap utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

from .utils import EPS


@dataclass(frozen=True)
class SimpleMetric:
    name: str
    max_value: float | None
    compute: Callable[[torch.Tensor, torch.Tensor], float]


@dataclass(frozen=True)
class AggregationResult:
    value: float
    indices: dict[str, np.ndarray | torch.Tensor | int] | None = None


@dataclass(frozen=True)
class BootstrapSummary:
    samples: Sequence[float]
    ci_low: float
    ci_high: float
    mean: float
    std: float


def _as_value(result: float | AggregationResult) -> float:
    if isinstance(result, AggregationResult):
        return float(result.value)
    return float(result)


def compute_similarity_matrix(
    repsA: Sequence[torch.Tensor],
    repsB: Sequence[torch.Tensor],
    metric: SimpleMetric,
) -> torch.Tensor:
    if not repsA or not repsB:
        raise ValueError("repsA and repsB must be non-empty")
    n = repsA[0].shape[0]
    for X in list(repsA) + list(repsB):
        if X.shape[0] != n:
            raise ValueError("All layers must share the same number of samples")
    S = torch.empty((len(repsA), len(repsB)), device=repsA[0].device)
    for i, Xa in enumerate(repsA):
        for j, Xb in enumerate(repsB):
            S[i, j] = metric.compute(Xa, Xb)
    return S


def agg_max(S: torch.Tensor, *, return_indices: bool = False) -> AggregationResult:
    flat_idx = int(torch.argmax(S).item())
    value = float(S.flatten()[flat_idx].item())
    if not return_indices:
        return AggregationResult(value=value, indices=None)
    i = flat_idx // S.shape[1]
    j = flat_idx % S.shape[1]
    return AggregationResult(value=value, indices={"i": i, "j": j})


def agg_rowmax_mean(
    S: torch.Tensor, *, return_indices: bool = False
) -> AggregationResult:
    row_max, row_idx = torch.max(S, dim=1)
    value = float(row_max.mean().item())
    if not return_indices:
        return AggregationResult(value=value, indices=None)
    return AggregationResult(
        value=value, indices={"row": torch.arange(S.size(0)), "col": row_idx}
    )


def agg_colmax_mean(
    S: torch.Tensor, *, return_indices: bool = False
) -> AggregationResult:
    col_max, col_idx = torch.max(S, dim=0)
    value = float(col_max.mean().item())
    if not return_indices:
        return AggregationResult(value=value, indices=None)
    return AggregationResult(
        value=value, indices={"row": col_idx, "col": torch.arange(S.size(1))}
    )


def agg_topk_mean(
    S: torch.Tensor, *, k: int, return_indices: bool = False
) -> AggregationResult:
    if k <= 0:
        raise ValueError("k must be positive")
    flat = S.flatten()
    k = min(int(k), flat.numel())
    vals, idx = torch.topk(flat, k)
    value = float(vals.mean().item())
    if not return_indices:
        return AggregationResult(value=value, indices=None)
    return AggregationResult(value=value, indices={"flat": idx})


def agg_hungarian_match_mean(S: torch.Tensor) -> AggregationResult:
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
    except ImportError as exc:
        raise ImportError("scipy is required for agg_hungarian_match_mean") from exc

    cost = (-S).detach().cpu().numpy()
    rows, cols = linear_sum_assignment(cost)
    vals = S[rows, cols]
    value = float(vals.mean().item())
    return AggregationResult(value=value, indices={"row": rows, "col": cols})


def permutation_null_aggregated(
    repsA_layers: Sequence[torch.Tensor],
    repsB_layers: Sequence[torch.Tensor],
    metric: SimpleMetric,
    aggregator: Callable[[torch.Tensor], float | AggregationResult],
    *,
    num_permutations: int,
    seed: int | None = None,
    return_matrices: bool = False,
) -> Sequence[float] | tuple[Sequence[float], Sequence[torch.Tensor]]:
    if num_permutations <= 0:
        raise ValueError("num_permutations must be positive")
    n = repsA_layers[0].shape[0]
    device = repsA_layers[0].device
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    null_samples: list[float] = []
    matrices: list[torch.Tensor] = []
    for _ in range(num_permutations):
        perm = torch.randperm(n, generator=rng, device=device)
        repsB_perm = [Y[perm, :] for Y in repsB_layers]
        S_b = compute_similarity_matrix(repsA_layers, repsB_perm, metric)
        if return_matrices:
            matrices.append(S_b)
        null_samples.append(_as_value(aggregator(S_b)))

    if return_matrices:
        return null_samples, matrices
    return null_samples


def tau_order_statistic(
    null_samples: "Sequence[float] | np.ndarray",
    quantile: float,
    *,
    obs: float | None = None,
) -> float:
    """Exact permutation threshold: the ceiling order statistic at ``quantile``.

    ``tau = s_(ceil(quantile * m))`` of the sorted sample (the null, with the observed
    value appended when ``obs`` is given, as in a permutation test). This is the cutoff
    that controls Type-I at <= ``1 - quantile``. ``np.quantile`` interpolates and sits
    below this, under-controlling Type-I at small K, so every gating threshold routes
    through this single helper for a consistent definition across all metrics/paths.
    """
    arr = np.asarray(null_samples, dtype=float)
    if obs is not None:
        arr = np.append(arr, float(obs))
    arr = np.sort(arr)
    m = arr.size
    if m == 0:
        raise ValueError("null_samples must be non-empty")
    rank = min(max(int(np.ceil(float(quantile) * m)), 1), m)
    return float(arr[rank - 1])


def compute_null_summary(
    null_samples_T: Sequence[float],
    *,
    T_obs: float,
    alpha: float,
) -> dict[str, float]:
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    null_arr = np.asarray(null_samples_T, dtype=float)
    # Exact permutation cutoff (ceiling order statistic of null + observed).
    tau_alpha = tau_order_statistic(null_arr, 1.0 - alpha, obs=T_obs)
    p_value = (1.0 + float(np.sum(null_arr >= float(T_obs)))) / (len(null_arr) + 1.0)
    if alpha <= 0:
        tail_strength = 0.0
    else:
        tail_strength = float(max(0.0, min(1.0, (alpha - p_value) / alpha)))
    mu0 = float(null_arr.mean())
    sd0 = float(null_arr.std())
    return {
        "tau_alpha": tau_alpha,
        "p_value": p_value,
        "tail_strength": tail_strength,
        "mu0": mu0,
        "sd0": sd0,
    }


def gated_rescaled(T_obs: float, *, tau_alpha: float, s_max: float | None) -> float:
    if s_max is None:
        return float(max(T_obs - tau_alpha, 0.0))
    if T_obs <= tau_alpha:
        return 0.0
    if s_max <= tau_alpha:
        return 1.0
    denom = s_max - tau_alpha
    if abs(denom) <= EPS:
        return 1.0
    return float(min((T_obs - tau_alpha) / denom, 1.0))


def bootstrap_statistic(
    repsA_layers: Sequence[torch.Tensor],
    repsB_layers: Sequence[torch.Tensor],
    metric: SimpleMetric,
    aggregator: Callable[[torch.Tensor], float | AggregationResult],
    *,
    num_bootstrap: int,
    seed: int | None = None,
    ci: tuple[float, float] = (0.025, 0.975),
) -> BootstrapSummary:
    if num_bootstrap <= 0:
        raise ValueError("num_bootstrap must be positive")
    n = repsA_layers[0].shape[0]
    device = repsA_layers[0].device
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    samples: list[float] = []
    for _ in range(num_bootstrap):
        idx = torch.randint(0, n, (n,), generator=rng, device=device)
        repsA_bs = [X[idx, :] for X in repsA_layers]
        repsB_bs = [Y[idx, :] for Y in repsB_layers]
        S_k = compute_similarity_matrix(repsA_bs, repsB_bs, metric)
        samples.append(_as_value(aggregator(S_k)))

    samples_arr = np.asarray(samples, dtype=float)
    ci_low, ci_high = np.quantile(samples_arr, ci)
    return BootstrapSummary(
        samples=samples,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        mean=float(samples_arr.mean()),
        std=float(samples_arr.std()),
    )
