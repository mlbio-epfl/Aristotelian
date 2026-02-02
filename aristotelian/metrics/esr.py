"""Effective shared rank (ESR) utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from .. import sg_cka_linear
from .utils import EPS


@dataclass
class ESRResult:
    recovered_rank: int
    effective_rank: float


def compute_esr(svals: torch.Tensor, *, tau: float) -> ESRResult:
    """Compute effective shared rank from singular values with a null threshold."""
    svals = svals.detach().cpu()
    recovered_rank = int((svals > tau).sum().item())
    if recovered_rank == 0:
        return ESRResult(recovered_rank=0, effective_rank=0.0)
    top = svals[svals > tau]
    # Effective rank via participation ratio on gated spectrum.
    effective_rank = float((top.sum() ** 2 / (top.pow(2).sum() + EPS)).item())
    return ESRResult(recovered_rank=recovered_rank, effective_rank=effective_rank)


def _sample_low_rank(
    n: int,
    d: int,
    r: int,
    snr: float,
    *,
    device: str,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    Z = torch.randn(n, r, generator=rng, device=device)
    A = torch.randn(r, d, generator=rng, device=device)
    B = torch.randn(r, d, generator=rng, device=device)
    Sx = Z @ A
    Sy = Z @ B
    noise = 1.0 / max(snr, EPS)
    X = Sx + noise * torch.randn(n, d, generator=rng, device=device)
    Y = Sy + noise * torch.randn(n, d, generator=rng, device=device)
    return X, Y


def run_esr_validation(
    *,
    n: int,
    d: int,
    ranks: Iterable[int],
    snr_values: Iterable[float],
    num_trials: int = 50,
    num_permutations: int = 200,
    quantile: float = 0.95,
    device: str = "cpu",
    seed: int | None = None,
) -> dict[int, dict[float, dict[str, float]]]:
    """Validate ESR by comparing recovered ranks against ground truth."""
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)

    results: dict[int, dict[float, dict[str, float]]] = {}
    for r in ranks:
        results[r] = {}
        for snr in snr_values:
            recovered = []
            eff = []
            for _ in range(num_trials):
                X, Y = _sample_low_rank(n, d, r, snr, device=device, rng=rng)
                res = sg_cka_linear(
                    X,
                    Y,
                    num_permutations=num_permutations,
                    quantile=quantile,
                    device=device,
                )
                # Approximate ESR from the same threshold on the real spectrum.
                svals = torch.linalg.svdvals((Y - Y.mean(0)).T @ (X - X.mean(0)))
                esr = compute_esr(svals, tau=res.tau)
                recovered.append(float(esr.recovered_rank))
                eff.append(esr.effective_rank)
            results[r][float(snr)] = {
                "recovered_rank_mean": float(np.mean(recovered)),
                "recovered_rank_std": float(np.std(recovered)),
                "effective_rank_mean": float(np.mean(eff)),
                "effective_rank_std": float(np.std(eff)),
            }
    return results
