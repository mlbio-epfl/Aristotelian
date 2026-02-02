"""PRH preprocessing helpers."""

from __future__ import annotations

from typing import Sequence

import torch


def remove_outliers(
    feats: torch.Tensor,
    q: float,
    *,
    exact: bool = False,
    max_threshold: float | None = None,
) -> torch.Tensor:
    if q >= 1.0:
        return feats
    if exact:
        q_val = feats.view(-1).abs().sort().values[int(q * feats.numel())]
    else:
        q_val = torch.quantile(feats.abs().flatten(start_dim=1), q, dim=1).mean()
    if max_threshold is not None:
        q_val = max(q_val, max_threshold)
    return feats.clamp(-q_val, q_val)


def prepare_features(
    feats: torch.Tensor | Sequence[torch.Tensor],
    *,
    q: float = 0.95,
    exact: bool = False,
    device: str = "cpu",
) -> torch.Tensor | list[torch.Tensor]:
    if isinstance(feats, torch.Tensor):
        return remove_outliers(feats.float(), q=q, exact=exact).to(device)
    return [remove_outliers(f.float(), q=q, exact=exact).to(device) for f in feats]
