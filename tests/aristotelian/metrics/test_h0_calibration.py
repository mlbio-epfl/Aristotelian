"""Under H0, calibration must control Type-I and keep the observed score exchangeable
with its permutation null -- for EVERY calibrated metric, across regimes (incl. d>=n).

This guards a class of bug that is invisible on CPU but appears on GPU: for d>=n the
whitened cross-covariance operator is rank-deficient with a large degenerate near-zero
eigenspace whose eigenvectors are arbitrary. If a metric's weighting uses that subspace
(e.g. PWCCA before the rank-truncation fix), `_compute_raw` (single eigh) and
`_compute_null_distribution` (batched eigh) disagree, the observed raw is pinned above
the whole null, Type-I ~ 1, and the calibrated score cannot gate to 0.

The test runs on CUDA when available (the regime where such bugs surface) and on CPU
otherwise. It checks the permutation test directly (add-one p-value), so it is metric-
and device-agnostic.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import aristotelian.metrics  # noqa: F401  (registers metrics)
from aristotelian.metrics.base import MetricConfig
from aristotelian.metrics.registry import MetricRegistry

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA = 0.05
TRIALS = 15
PERMS = 150

CALIBRATED = [
    ("cca", {}),
    ("svcca", {"cca_dim": 10}),
    ("pwcca", {}),
    ("rv_coefficient", {}),
    ("cka_linear", {}),
    ("cka_rbf", {}),
    ("cka_unbiased", {}),
    ("procrustes", {}),
    ("rsa", {"batch_size": 32}),
    ("mutual_knn", {"topk": 5}),
    ("cycle_knn", {"topk": 5}),
    ("cknna", {"topk": 5}),
]
# (n, d): d<n, d=n, d>n, d>>n. The d>=n cells are the degenerate regime.
REGIMES = [(96, 48), (96, 96), (96, 192), (128, 384)]


@pytest.mark.parametrize("n,d", REGIMES, ids=[f"n{n}d{d}" for n, d in REGIMES])
@pytest.mark.parametrize("name,extra", CALIBRATED, ids=[m[0] for m in CALIBRATED])
def test_h0_calibration_typeI_and_exchangeability(name, extra, n, d):
    metric = MetricRegistry.get(name)
    if not metric.supports_calibration:
        pytest.skip("metric does not support calibration")

    n_reject = 0
    pctiles = []
    for t in range(TRIALS):
        torch.manual_seed(20_240_000 + 1000 * n + d + t)
        X = torch.randn(n, d, device=DEVICE)
        Y = torch.randn(n, d, device=DEVICE)
        perms = torch.stack([torch.randperm(n, device=DEVICE) for _ in range(PERMS)])
        raw = metric._compute_raw(X, Y, MetricConfig(device=DEVICE, **extra))
        null = np.asarray(
            metric._compute_null_distribution(
                X, Y,
                MetricConfig(device=DEVICE, num_permutations=PERMS, perms=perms, **extra),
            ),
            dtype=float,
        )
        # Valid (add-one) one-sided permutation p-value.
        pval = (1 + int((null >= raw).sum())) / (PERMS + 1)
        n_reject += int(pval <= ALPHA)
        pctiles.append(float((null < raw).mean()))

    type_i = n_reject / TRIALS
    mean_pctile = float(np.mean(pctiles))

    # Type-I must be controlled near alpha. The bug pinned raw above the whole null,
    # giving Type-I ~ 1. Bound is loose enough for TRIALS-sample noise around alpha.
    assert type_i <= 0.25, (
        f"{name} n={n} d={d} dev={DEVICE}: Type-I={type_i:.2f} >> alpha={ALPHA} "
        f"(observed raw not exchangeable with its null; mean raw-percentile={mean_pctile:.2f})"
    )
    # Observed raw must sit inside the null, not be pinned at the top.
    assert mean_pctile <= 0.85, (
        f"{name} n={n} d={d} dev={DEVICE}: mean raw-percentile={mean_pctile:.2f} "
        f"(raw sits systematically above the null -> calibration cannot gate to 0)"
    )
