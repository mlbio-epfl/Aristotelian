"""MetricRegistry: Singleton for managing all available metrics.

This module provides a central registry for metric lookup and computation.
Metrics are registered automatically when their modules are imported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .base import BaseMetric, MetricConfig, MetricResult


class MetricRegistry:
    """Singleton registry for all metrics.

    Provides a central place to register, look up, and compute metrics.
    Metrics register themselves at import time.

    Usage:
        # Get a metric by name
        metric = MetricRegistry.get("mutual_knn")

        # Compute a metric directly
        result = MetricRegistry.compute("cka_linear", X, Y)

        # List all available metrics
        names = MetricRegistry.list_metrics()
    """

    _metrics: dict[str, BaseMetric] = {}

    @classmethod
    def register(cls, metric: BaseMetric) -> None:
        """Register a metric instance.

        Args:
            metric: The metric instance to register.

        Raises:
            ValueError: If a metric with the same name is already registered.
        """
        if metric.name in cls._metrics:
            raise ValueError(
                f"Metric '{metric.name}' is already registered. "
                f"Existing: {cls._metrics[metric.name]}, New: {metric}"
            )
        cls._metrics[metric.name] = metric

    @classmethod
    def register_alias(cls, name: str, alias: str) -> None:
        """Register an alias for an existing metric.

        Args:
            name: The canonical metric name.
            alias: The alias to register.

        Raises:
            KeyError: If the original metric is not registered.
            ValueError: If the alias is already registered.
        """
        if name not in cls._metrics:
            raise KeyError(f"Metric '{name}' is not registered")
        if alias in cls._metrics:
            raise ValueError(f"Alias '{alias}' is already registered")
        cls._metrics[alias] = cls._metrics[name]

    @classmethod
    def get(cls, name: str) -> BaseMetric:
        """Get a metric by name.

        Args:
            name: The metric name (case-sensitive).

        Returns:
            The metric instance.

        Raises:
            KeyError: If no metric with the given name is registered.
        """
        if name not in cls._metrics:
            available = ", ".join(sorted(cls._metrics.keys()))
            raise KeyError(f"Unknown metric '{name}'. Available metrics: {available}")
        return cls._metrics[name]

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if a metric is registered.

        Args:
            name: The metric name to check.

        Returns:
            True if the metric is registered.
        """
        return name in cls._metrics

    @classmethod
    def list_metrics(cls) -> list[str]:
        """List all registered metric names.

        Returns:
            Sorted list of metric names.
        """
        return sorted(cls._metrics.keys())

    @classmethod
    def compute(
        cls,
        name: str,
        X: torch.Tensor,
        Y: torch.Tensor,
        config: MetricConfig | None = None,
    ) -> MetricResult:
        """Compute a metric by name.

        Convenience method that combines get() and compute().

        Args:
            name: The metric name.
            X: Feature matrix of shape (n, d1).
            Y: Feature matrix of shape (n, d2).
            config: Metric configuration. If None, uses defaults.

        Returns:
            MetricResult with the computed score(s).
        """
        metric = cls.get(name)
        return metric.compute(X, Y, config)

    @classmethod
    def compute_raw(
        cls,
        name: str,
        X: torch.Tensor,
        Y: torch.Tensor,
        config: MetricConfig | None = None,
    ) -> float:
        """Compute only the raw metric value by name.

        Convenience method for when you don't need calibration.

        Args:
            name: The metric name.
            X: Feature matrix of shape (n, d1).
            Y: Feature matrix of shape (n, d2).
            config: Metric configuration. If None, uses defaults.

        Returns:
            The raw metric value.
        """
        metric = cls.get(name)
        return metric.compute_raw(X, Y, config)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered metrics. Primarily for testing."""
        cls._metrics.clear()


def register_metric(metric_class: type[BaseMetric]) -> type[BaseMetric]:
    """Decorator to register a metric class.

    Usage:
        @register_metric
        class MyMetric(BaseMetric):
            name = "my_metric"
            ...

    The metric instance is created and registered when the class is defined.
    """
    instance = metric_class()
    MetricRegistry.register(instance)
    return metric_class
