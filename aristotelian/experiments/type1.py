"""Type-I calibration utilities."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from loguru import logger

from ..metrics.utils import resolve_sg_metric
from .common import _sample_pair


@dataclass
class Type1Summary:
    positives: Sequence[bool]
    type1_rate: float


def _type1_trial(
    *,
    metric: str,
    n: int,
    d: int,
    num_permutations: int,
    quantile: float,
    null_type: str,
    k_knn: int,
    device: str,
    seed: int | None,
    rsa_batch_size: int | None,
) -> bool:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
    X, Y = _sample_pair(n, d, null_type=null_type, device=device, rng=rng)
    metric_fn = resolve_sg_metric(metric)
    kwargs = dict(num_permutations=num_permutations, quantile=quantile, device=device)
    if metric == "sgknn":
        kwargs["k"] = k_knn
    if metric == "sgrsa":
        kwargs["batch_size"] = rsa_batch_size
    res = metric_fn(X, Y, **kwargs)
    return float(res.gated) > 0.0


def _type1_trial_from_args(args: tuple[int | None, dict[str, object]]) -> bool:
    seed, kwargs = args
    return _type1_trial(seed=seed, **kwargs)


def run_type1_calibration(
    *,
    metric: str,
    n: int,
    d: int,
    num_trials: int = 50,
    num_permutations: int = 200,
    quantile: float = 0.95,
    alpha: float = 0.05,
    null_type: str = "gaussian",
    k_knn: int = 10,
    device: str = "cpu",
    seed: int | None = None,
    rsa_batch_size: int | None = 32,
    num_workers: int = 1,
) -> Type1Summary:
    """Estimate Type I error under the null using the gating event."""
    logger.info(f"Running Type-I calibration: metric={metric}, n={n}, d={d}")
    logger.info(
        f"Trials: {num_trials}, permutations: {num_permutations}, alpha: {alpha}"
    )

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        logger.debug(f"Random seed set: {seed}")

    if seed is None:
        trial_seeds = [None] * num_trials
    else:
        rng = np.random.default_rng(seed)
        trial_seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=num_trials)]

    task_kwargs: dict[str, object] = dict(
        metric=metric,
        n=n,
        d=d,
        num_permutations=num_permutations,
        quantile=quantile,
        null_type=null_type,
        k_knn=k_knn,
        device=device,
        rsa_batch_size=rsa_batch_size,
    )
    if num_workers <= 1:
        positives = [
            _type1_trial(seed=trial_seed, **task_kwargs) for trial_seed in trial_seeds
        ]
    else:
        executor_class = ProcessPoolExecutor if device == "cpu" else ThreadPoolExecutor
        with executor_class(max_workers=num_workers) as executor:
            positives = list(
                executor.map(
                    _type1_trial_from_args,
                    [(trial_seed, task_kwargs) for trial_seed in trial_seeds],
                )
            )
    type1_rate = float(np.mean(np.array(positives)))

    logger.info(f"Type-I (gate): {type1_rate:.4f} (expected: {alpha})")
    if abs(type1_rate - alpha) > 0.02:
        logger.warning(
            f"Type-I error deviates from nominal by {abs(type1_rate - alpha):.4f}"
        )
    return Type1Summary(positives=positives, type1_rate=type1_rate)
