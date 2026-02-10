"""Shared helpers for experiment sweeps."""

from __future__ import annotations

import torch

from ..metrics.utils import sample_student_t


def _sample_pair(
    n: int,
    d: int,
    *,
    null_type: str,
    device: str,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    else:
        raise ValueError("null_type must be one of {'gaussian','heavy','shuffled'}")
    return X, Y
