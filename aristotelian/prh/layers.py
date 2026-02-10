"""PRH layer helpers."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def _as_layers(feats: torch.Tensor | Sequence[torch.Tensor]) -> list[torch.Tensor]:
    if isinstance(feats, torch.Tensor):
        return [feats[:, i, :] for i in range(feats.shape[1])]
    return list(feats)


def _normalize_layers(layers: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    return [F.normalize(x, p=2, dim=-1) for x in layers]
