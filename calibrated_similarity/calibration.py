"""Null-calibration algorithms for representation similarity metrics.

This module implements two core algorithms:

1. `calibrate()` - Scalar null-calibration (Algorithm 1)
   Calibrates a single similarity score against a permutation null distribution.

2. `calibrate_layers()` - Aggregation-aware calibration (Algorithm 2)
   Calibrates layer-wise similarity matrices with proper multiple comparison handling.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

# Numerical stability constant for division
_EPSILON = 1e-12


def _validate_inputs(X: torch.Tensor, Y: torch.Tensor) -> None:
    """Validate input tensors for calibration functions."""
    if X.dim() != 2:
        raise ValueError(f"X must be 2-dimensional, got {X.dim()}D")
    if Y.dim() != 2:
        raise ValueError(f"Y must be 2-dimensional, got {Y.dim()}D")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of samples, "
            f"got {X.shape[0]} and {Y.shape[0]}"
        )
    if X.shape[0] == 0:
        raise ValueError("X and Y must have at least one sample")
    if X.device != Y.device:
        raise ValueError(
            f"X and Y must be on the same device, got {X.device} and {Y.device}"
        )


def _validate_params(K: int, alpha: float, smax: float | None) -> None:
    """Validate calibration parameters."""
    if not isinstance(K, int) or K < 1:
        raise ValueError(f"K must be a positive integer, got {K}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if smax is not None and smax <= 0:
        raise ValueError(f"smax must be positive, got {smax}")


def _validate_layer_inputs(
    X_layers: list[torch.Tensor], Y_layers: list[torch.Tensor]
) -> tuple[int, torch.device]:
    """Validate layer inputs and return (n_samples, device)."""
    if len(X_layers) == 0:
        raise ValueError("X_layers must not be empty")
    if len(Y_layers) == 0:
        raise ValueError("Y_layers must not be empty")

    n = X_layers[0].shape[0]
    device = X_layers[0].device

    for i, layer in enumerate(X_layers):
        if layer.dim() != 2:
            raise ValueError(f"X_layers[{i}] must be 2-dimensional, got {layer.dim()}D")
        if layer.shape[0] != n:
            raise ValueError(
                f"All X_layers must have the same number of samples, "
                f"X_layers[0] has {n} but X_layers[{i}] has {layer.shape[0]}"
            )
        if layer.device != device:
            raise ValueError(
                f"All layers must be on the same device, "
                f"X_layers[0] is on {device} but X_layers[{i}] is on {layer.device}"
            )

    for i, layer in enumerate(Y_layers):
        if layer.dim() != 2:
            raise ValueError(f"Y_layers[{i}] must be 2-dimensional, got {layer.dim()}D")
        if layer.shape[0] != n:
            raise ValueError(
                f"All Y_layers must have the same number of samples as X_layers, "
                f"expected {n} but Y_layers[{i}] has {layer.shape[0]}"
            )
        if layer.device != device:
            raise ValueError(
                f"All layers must be on the same device, "
                f"X_layers[0] is on {device} but Y_layers[{i}] is on {layer.device}"
            )

    return n, device


@torch.no_grad()
def calibrate(
    X: torch.Tensor,
    Y: torch.Tensor,
    sim: Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor],
    *,
    K: int = 200,
    alpha: float = 0.05,
    smax: float | None = 1.0,
    generator: torch.Generator | None = None,
    perm_fn: Callable[[int, torch.device], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scalar null-calibration via permutation testing (Algorithm 1).

    Breaks row correspondences between X and Y by permuting Y's rows K times,
    estimates the (1-alpha) critical threshold tau from the combined distribution
    (observed + null), and returns a calibrated score that is 0 under the null
    hypothesis at the specified alpha level.

    Args:
        X: Feature matrix of shape [n, d1] for the first representation.
        Y: Feature matrix of shape [n, d2] for the second representation.
            Row i of X and Y should correspond to the same sample.
        sim: Similarity function with signature sim(X, Y) -> scalar.
            Should return values bounded in [0, smax] for proper normalization.
        K: Number of permutations for null distribution estimation.
            Higher values give more precise p-values but increase computation.
            Must be >= 1. Default: 200.
        alpha: Significance level for the critical threshold (must be in (0, 1)).
            tau is set to the (1-alpha) quantile of the null distribution.
            Default: 0.05.
        smax: Maximum attainable similarity value (used for normalization).
            If None, returns unnormalized effect size: relu(obs - tau).
            This is useful when the similarity function has no natural upper bound.
            Must be positive if provided. Default: 1.0.
        generator: Optional torch.Generator for reproducible permutations.
        perm_fn: Optional custom permutation function for restricted permutations.
            Signature: perm_fn(n, device) -> LongTensor of shape [n].
            Useful for block permutations or other constrained designs.

    Returns:
        A tuple (calibrated_score, p_value, tau) where:
        - calibrated_score: Normalized score in [0, 1] if smax is set,
          or relu(obs - tau) if smax is None. Zero under null at level alpha.
        - p_value: Add-one permutation p-value in (0, 1].
          Formula: (1 + #{null >= obs}) / (K + 1).
        - tau: The (1-alpha) critical threshold from the combined distribution.

    Raises:
        ValueError: If inputs have invalid shapes, devices, or parameters are
            out of valid ranges.

    Example:
        >>> import torch
        >>> from calibrated_similarity import calibrate
        >>>
        >>> def cka(X, Y):
        ...     X, Y = X - X.mean(0), Y - Y.mean(0)
        ...     hsic = (X @ X.T * (Y @ Y.T)).sum()
        ...     return hsic / (X @ X.T).norm() / (Y @ Y.T).norm()
        >>>
        >>> X = torch.randn(100, 64)
        >>> Y = torch.randn(100, 64)
        >>> score, pval, tau = calibrate(X, Y, cka, K=100)
        >>> print(f"Calibrated: {score:.3f}, p={pval:.3f}")
    """
    _validate_inputs(X, Y)
    _validate_params(K, alpha, smax)

    n = X.shape[0]
    device = X.device

    if perm_fn is None:

        def perm_fn(n_: int, device_: torch.device) -> torch.Tensor:
            return torch.randperm(n_, device=device_, generator=generator)

    # Observed similarity
    sobs = sim(X, Y)
    sobs = torch.as_tensor(sobs, device=device, dtype=torch.float32)

    # Generate null distribution
    null = torch.empty((K,), device=device, dtype=torch.float32)
    for k in range(K):
        pi = perm_fn(n, device)
        null[k] = torch.as_tensor(sim(X, Y[pi]), device=device, dtype=torch.float32)

    # Compute (1-alpha) critical value from combined distribution
    combined = torch.cat([sobs.view(1), null], dim=0)  # [K+1]
    combined_sorted, _ = torch.sort(combined)
    # Order statistic index: ceil((1-alpha)(K+1)) in {1..K+1}, converted to 0-based
    idx = int(math.ceil((1.0 - alpha) * (K + 1))) - 1
    idx = max(0, min(idx, K))  # Clamp to valid range [0, K]
    tau = combined_sorted[idx]

    # Add-one right-tail p-value: (1 + #{null >= obs}) / (K + 1)
    p = (1.0 + (null >= sobs).to(torch.float32).sum()) / float(K + 1)

    # Compute calibrated score
    if smax is None:
        scal = torch.relu(sobs - tau)
    else:
        smax_t = torch.tensor(float(smax), device=device, dtype=torch.float32)
        denom = (smax_t - tau).clamp_min(_EPSILON)
        scal = torch.clamp((sobs - tau) / denom, min=0.0, max=1.0)

    return scal, p, tau


@torch.no_grad()
def calibrate_layers(
    X_layers: list[torch.Tensor],
    Y_layers: list[torch.Tensor],
    sim: Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor],
    *,
    agg: str | Callable[[torch.Tensor], torch.Tensor] = "max",
    K: int = 200,
    alpha: float = 0.05,
    smax: float | None = 1.0,
    generator: torch.Generator | None = None,
    perm_fn: Callable[[int, torch.device], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregation-aware null-calibration for layer-wise comparisons (Algorithm 2).

    Computes a similarity matrix S[i, j] = sim(X_layers[i], Y_layers[j]) and
    aggregates it (e.g., max or mean). The aggregate is then calibrated against
    a permutation null where the SAME row permutation is applied across all
    layers of Y, properly accounting for the multiple comparison structure.

    This is essential when searching for the best-matching layer pair, as naive
    per-pair calibration would inflate false positives.

    Args:
        X_layers: List of LA feature matrices, each of shape [n, d_l].
        Y_layers: List of LB feature matrices, each of shape [n, d_l'].
            All matrices must have the same number of samples n and be on the
            same device.
        sim: Similarity function with signature sim(X, Y) -> scalar.
        agg: Aggregation strategy for the similarity matrix.
            - "max": Maximum similarity (best layer pair) - default
            - "mean": Mean similarity (average alignment)
            - callable: Custom function taking a [LA, LB] tensor and returning
              a scalar tensor.
        K: Number of permutations. Must be >= 1. Default: 200.
        alpha: Significance level (must be in (0, 1)). Default: 0.05.
        smax: Maximum attainable similarity. Must be positive if provided.
            Default: 1.0.
        generator: Optional torch.Generator for reproducibility.
        perm_fn: Optional custom permutation function.

    Returns:
        A tuple (calibrated_agg, p_value, tau_agg) where:
        - calibrated_agg: Calibrated aggregate score.
        - p_value: Add-one permutation p-value for the aggregate.
        - tau_agg: Critical threshold for the aggregate.

    Raises:
        ValueError: If layer lists are empty, have inconsistent shapes/devices,
            or parameters are out of valid ranges.

    Example:
        >>> import torch
        >>> from calibrated_similarity import calibrate_layers
        >>>
        >>> def cosine_sim(X, Y):
        ...     X = X / X.norm(dim=1, keepdim=True)
        ...     Y = Y / Y.norm(dim=1, keepdim=True)
        ...     return (X * Y).sum(dim=1).mean()
        >>>
        >>> # Compare 5 layers from model A to 3 layers from model B
        >>> X_layers = [torch.randn(100, 64) for _ in range(5)]
        >>> Y_layers = [torch.randn(100, 64) for _ in range(3)]
        >>> score, pval, tau = calibrate_layers(
        ...     X_layers, Y_layers, cosine_sim, agg="max"
        ... )
    """
    n, device = _validate_layer_inputs(X_layers, Y_layers)
    _validate_params(K, alpha, smax)

    LA, LB = len(X_layers), len(Y_layers)

    if perm_fn is None:

        def perm_fn(n_: int, device_: torch.device) -> torch.Tensor:
            return torch.randperm(n_, device=device_, generator=generator)

    # Set up aggregation function
    def _agg_max(S: torch.Tensor) -> torch.Tensor:
        return S.max()

    def _agg_mean(S: torch.Tensor) -> torch.Tensor:
        return S.mean()

    if isinstance(agg, str):
        if agg == "max":
            agg_fn: Callable[[torch.Tensor], torch.Tensor] = _agg_max
        elif agg == "mean":
            agg_fn = _agg_mean
        else:
            raise ValueError("agg must be 'max', 'mean', or a callable.")
    else:
        if not callable(agg):
            raise ValueError("agg must be 'max', 'mean', or a callable.")
        agg_fn = agg

    # Compute observed similarity matrix and aggregate
    S = torch.empty((LA, LB), device=device, dtype=torch.float32)
    for i in range(LA):
        for j in range(LB):
            S[i, j] = torch.as_tensor(
                sim(X_layers[i], Y_layers[j]), device=device, dtype=torch.float32
            )
    Tobs = torch.as_tensor(agg_fn(S), device=device, dtype=torch.float32)

    # Generate null distribution for the aggregate
    # Key: same permutation is applied across ALL Y layers
    null_T = torch.empty((K,), device=device, dtype=torch.float32)
    for k in range(K):
        pi = perm_fn(n, device)
        S_k = torch.empty((LA, LB), device=device, dtype=torch.float32)
        for i in range(LA):
            for j in range(LB):
                S_k[i, j] = torch.as_tensor(
                    sim(X_layers[i], Y_layers[j][pi]),
                    device=device,
                    dtype=torch.float32,
                )
        null_T[k] = torch.as_tensor(agg_fn(S_k), device=device, dtype=torch.float32)

    # Compute critical value from combined distribution
    combined = torch.cat([Tobs.view(1), null_T], dim=0)
    combined_sorted, _ = torch.sort(combined)
    idx = int(math.ceil((1.0 - alpha) * (K + 1))) - 1
    idx = max(0, min(idx, K))
    tau_agg = combined_sorted[idx]

    # Add-one p-value
    pagg = (1.0 + (null_T >= Tobs).to(torch.float32).sum()) / float(K + 1)

    # Compute calibrated aggregate
    if smax is None:
        Tcal = torch.relu(Tobs - tau_agg)
    else:
        smax_t = torch.tensor(float(smax), device=device, dtype=torch.float32)
        denom = (smax_t - tau_agg).clamp_min(_EPSILON)
        Tcal = torch.clamp((Tobs - tau_agg) / denom, min=0.0, max=1.0)

    return Tcal, pagg, tau_agg
