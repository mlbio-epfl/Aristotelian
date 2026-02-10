"""Metric definition helpers for tests and experiments."""

from __future__ import annotations

from typing import Callable, Dict, Sequence

from aristotelian.metrics.api import metric_definitions as registry_metric_definitions


def _metric_definitions(
    *, num_permutations: int, device: str
) -> tuple[Sequence[tuple[str, Callable, Callable | None]], Dict[str, Callable]]:
    """Return metric definitions and multi-quantile helpers."""
    return registry_metric_definitions(num_permutations=num_permutations, device=device)
