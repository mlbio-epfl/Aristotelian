"""Base classes for the metrics system.

This module provides the foundational abstractions for all metrics:
- MetricConfig: Configuration dataclass for metric computation
- MetricResult: Unified result structure for all metrics
- BaseMetric: Abstract base class that all metrics inherit from
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from .calibration import compute_calibration_stats


@dataclass
class MetricConfig:
    """Configuration for metric computation.

    This dataclass captures all parameters needed to compute any metric,
    including null calibration settings. Unused parameters for a given
    metric are simply ignored.
    """

    # kNN parameters
    topk: int | None = None

    # Kernel parameters
    kernel: str = "linear"  # "linear" or "rbf"
    sigma: float | None = None  # RBF bandwidth (None = median heuristic)
    rbf_sigma: float = 1.0  # For PRH-style RBF CKA

    # CCA parameters (0 = no projection, >0 = PCA to this dim)
    cca_dim: int = 0

    # HSIC/CKA parameters
    unbiased: bool = False

    # Distance-agnostic flag (for CKNNA)
    distance_agnostic: bool = False

    # Null calibration settings
    num_permutations: int = 200
    quantile: float = 0.95
    calibrate: bool = False  # Whether to perform null calibration

    # RSA-specific parameters
    batch_size: int | None = 32
    pair_samples: int | None = None

    # Device settings
    device: str = "cpu"

    # Pre-computed permutations (for efficiency in layer-wise computation)
    perms: torch.Tensor | None = None

    # Cache for intermediate computations (Gram matrices, kNN indices, etc.)
    cache: dict[str, Any] = field(default_factory=dict)

    def with_updates(self, **kwargs: Any) -> MetricConfig:
        """Return a new config with specified fields updated."""
        from dataclasses import replace

        return replace(self, **kwargs)


@dataclass
class MetricResult:
    """Unified result structure for all metrics.

    All metrics return this structure. For non-calibrated metrics,
    only `raw` is populated. For significance-gated (SG) metrics,
    additional fields contain null distribution statistics.
    """

    # Core result
    raw: float

    # Significance-gated (SG) results (None if not calibrated)
    gated: float | None = None
    tau: float | None = None
    pvalue: float | None = None
    tail_strength: float | None = None

    # Null distribution statistics (None if not calibrated)
    null_samples: Sequence[float] | torch.Tensor | None = None
    mean_null: float | None = None
    median_null: float | None = None
    std_null: float | None = None
    null_centered: float | None = None
    z: float | None = None
    ari: float | None = None

    # Additional metadata
    recovered_rank: int | None = None
    obs_max: float | None = None

    @classmethod
    def from_raw(cls, raw: float) -> MetricResult:
        """Create a result with only the raw score (no calibration)."""
        return cls(raw=raw)

    @classmethod
    def from_calibrated(
        cls,
        raw: float,
        null_samples: Sequence[float] | torch.Tensor,
        *,
        quantile: float = 0.95,
        min_score: float = 0.0,
        max_score: float = 1.0,
    ) -> MetricResult:
        """Create a result with full null calibration statistics."""
        stats = compute_calibration_stats(
            raw,
            null_samples,
            quantile=quantile,
            min_score=min_score,
            max_score=max_score,
        )

        return cls(
            raw=raw,
            gated=stats.gated,
            tau=stats.tau,
            pvalue=stats.pvalue,
            tail_strength=stats.tail_strength,
            null_samples=null_samples,
            mean_null=stats.variants.mean_null,
            median_null=stats.variants.median_null,
            std_null=stats.variants.std_null,
            null_centered=stats.variants.null_centered,
            z=stats.variants.z,
            ari=stats.variants.ari,
        )


class BaseMetric(ABC):
    """Abstract base class for all metrics.

    Subclasses must implement:
    - name: The metric's canonical name for registry lookup
    - _compute_raw: The core metric computation (no calibration)
    - min_score / max_score: The metric's theoretical bounds

    Optional overrides:
    - _compute_null_distribution: Custom null distribution computation
    - supports_caching: Whether the metric can use cached intermediates
    - cache_keys: What cache entries this metric uses/produces
    """

    # Must be set by subclasses
    name: str = ""
    min_score: float = 0.0
    max_score: float = 1.0

    # Whether null calibration is supported/meaningful
    supports_calibration: bool = True

    # Whether the metric can use cached intermediates
    supports_caching: bool = False
    cache_keys: tuple[str, ...] = ()

    def _validate_inputs(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """Validate input tensors have compatible shapes.

        Args:
            X: Feature matrix of shape (n, d1).
            Y: Feature matrix of shape (n, d2).

        Raises:
            ValueError: If X and Y have different number of samples.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of samples, "
                f"got {X.shape[0]} and {Y.shape[0]}"
            )

    @abstractmethod
    def _compute_raw(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> float:
        """Compute the raw metric value.

        Args:
            X: Feature matrix of shape (n, d1) for first representation.
            Y: Feature matrix of shape (n, d2) for second representation.
            config: Metric configuration.

        Returns:
            The raw (uncalibrated) metric value.
        """
        ...

    def _compute_null_distribution(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> list[float]:
        """Compute null distribution via permutation testing.

        Default implementation permutes Y and recomputes the metric.
        Subclasses can override for optimized implementations.

        Args:
            X: Feature matrix of shape (n, d1).
            Y: Feature matrix of shape (n, d2).
            config: Metric configuration (uses num_permutations, device, perms).

        Returns:
            List of null scores from permutation testing.
        """
        n = X.shape[0]
        device = config.device
        num_perms = config.num_permutations

        if config.perms is not None:
            perms = config.perms.to(device)
            if perms.dim() != 2 or perms.size(1) != n:
                raise ValueError("perms must have shape (B, n)")
        else:
            perms = torch.stack(
                [torch.randperm(n, device=device) for _ in range(num_perms)]
            )

        # Create a config without calibration for null computation
        null_config = config.with_updates(calibrate=False)

        null_scores = []
        for perm in perms:
            Y_perm = Y[perm]
            score = self._compute_raw(X, Y_perm, null_config)
            null_scores.append(score)

        return null_scores

    def compute(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig | None = None
    ) -> MetricResult:
        """Compute the metric with optional null calibration.

        Args:
            X: Feature matrix of shape (n, d1).
            Y: Feature matrix of shape (n, d2).
            config: Metric configuration. If None, uses defaults.

        Returns:
            MetricResult with raw score and optional calibration statistics.
        """
        if config is None:
            config = MetricConfig()

        X = X.to(config.device)
        Y = Y.to(config.device)
        self._validate_inputs(X, Y)

        raw = self._compute_raw(X, Y, config)

        if not config.calibrate or not self.supports_calibration:
            return MetricResult.from_raw(raw)

        # Validate quantile for calibration
        if not 0.0 <= float(config.quantile) <= 1.0:
            raise ValueError("quantile must be in [0, 1]")

        null_samples = self._compute_null_distribution(X, Y, config)
        return MetricResult.from_calibrated(
            raw,
            null_samples,
            quantile=config.quantile,
            min_score=self.min_score,
            max_score=self.max_score,
        )

    def compute_raw(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig | None = None
    ) -> float:
        """Compute only the raw metric value (faster, no calibration).

        Args:
            X: Feature matrix of shape (n, d1).
            Y: Feature matrix of shape (n, d2).
            config: Metric configuration. If None, uses defaults.

        Returns:
            The raw metric value.
        """
        if config is None:
            config = MetricConfig()

        X = X.to(config.device)
        Y = Y.to(config.device)
        self._validate_inputs(X, Y)

        return self._compute_raw(X, Y, config)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
