"""PRH-style alignment utilities: mutual-kNN + null-calibration + FDR/FWER."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from .. import sg_knn
from ..experiments.multiple import bh_fdr, max_stat_threshold


def _pairwise_knn(
    acts: Dict[str, List[torch.Tensor]],
    *,
    k: int,
    num_permutations: int,
    quantile: float,
) -> Tuple[np.ndarray, np.ndarray]:
    keys = list(acts.keys())
    L = len(acts[keys[0]])
    num_pairs = len(keys) * len(keys)
    pvals = np.zeros((num_pairs, L, L))
    gated = np.zeros((num_pairs, L, L))
    idx = 0
    for a in keys:
        for b in keys:
            for i, Xa in enumerate(acts[a]):
                for j, Xb in enumerate(acts[b]):
                    res = sg_knn(
                        Xa,
                        Xb,
                        k=k,
                        num_permutations=num_permutations,
                        quantile=quantile,
                    )
                    pvals[idx, i, j] = res.pvalue
                    gated[idx, i, j] = res.gated
            idx += 1
    return pvals, gated


def run_prh_alignment(
    *,
    activations: Dict[str, List[torch.Tensor]],
    metric: str = "sgknn",
    k: int = 10,
    num_permutations: int = 200,
    quantile: float = 0.95,
    alpha: float = 0.05,
    fwer: bool = False,
) -> Dict[str, np.ndarray]:
    """Compute PRH-style mutual-kNN alignment with significance control."""
    if metric != "sgknn":
        raise ValueError("PRH alignment currently supports metric='sgknn' only")

    pvals, gated = _pairwise_knn(
        activations, k=k, num_permutations=num_permutations, quantile=quantile
    )
    flat_p = pvals.reshape(-1)
    if fwer:
        # Use permutation max-stat as a placeholder threshold from p-values.
        thr = max_stat_threshold(flat_p, alpha=alpha)
        mask = pvals <= thr
    else:
        thr, mask_flat = bh_fdr(flat_p, alpha=alpha)
        mask = mask_flat.reshape(pvals.shape)
    return {
        "pvalues": pvals,
        "gated": gated,
        "fdr_threshold": np.array(thr),
        "fdr_mask": mask,
    }
