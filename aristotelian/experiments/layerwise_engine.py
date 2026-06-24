"""Layerwise similarity helpers with optional cached backends.

This module provides optimized implementations for computing similarity
matrices across many layer pairs. It uses pre-built caches (Gram matrices,
kNN indices) for efficiency.

NOTE: The canonical metric implementations are in aristotelian.metrics.
This module mirrors those implementations but is optimized for batch
layer-wise computation. For single-pair computation, prefer using:

    from aristotelian.metrics import MetricRegistry, MetricConfig
    result = MetricRegistry.compute("cka_linear", X, Y)
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from sklearn.cross_decomposition import CCA
from tqdm import tqdm

from .. import compute_nearest_neighbors
from ..metrics.aggregation import (
    SimpleMetric,
    agg_max,
    compute_null_summary,
    compute_similarity_matrix,
    gated_rescaled,
)
from ..metrics.cca import _pwcca_weighted_mean
from ..metrics.utils import (
    batched_perms,
    center_gram,
    center_np,
    hsic_biased,
    hsic_unbiased,
    knn_indicator,
    svcca_preprocess,
)
from ..prh.layers import _as_layers, _normalize_layers

# Note: Metric implementations here are optimized for batch computation.
# For single-pair computation, use aristotelian.metrics.MetricRegistry.


def _knn_overlap_from_indices(
    knn_a: torch.Tensor, knn_b: torch.Tensor, *, topk: int
) -> torch.Tensor:
    matches = (knn_a.unsqueeze(2) == knn_b.unsqueeze(1)).sum(dim=(1, 2)).float()
    return matches.mean() / float(topk)


def _build_gated_summary(
    null_samples: Sequence[float],
    *,
    T_obs: float,
    best_indices: Tuple[int, int],
    alpha: float,
) -> Dict[str, float | Tuple[int, int]]:
    summary = compute_null_summary(null_samples, T_obs=T_obs, alpha=alpha)
    g_score = gated_rescaled(T_obs, tau_alpha=summary["tau_alpha"], s_max=1.0)
    return {
        "raw_score": T_obs,
        "best_indices": best_indices,
        "p_value": summary["p_value"],
        "tau_alpha": summary["tau_alpha"],
        "tail_strength": summary["tail_strength"],
        "g_score": g_score,
        "mu0": summary["mu0"],
        "sd0": summary["sd0"],
    }


def _build_knn_cache(
    feats: torch.Tensor | Sequence[torch.Tensor],
    *,
    topk: int,
    normalize: bool,
    return_indices: bool,
) -> Tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor] | None]:
    layers = _as_layers(feats)
    if normalize:
        layers = _normalize_layers(layers)
    indices: list[torch.Tensor] = []
    masks = []
    for layer in layers:
        knn_idx = compute_nearest_neighbors(layer, topk=topk)
        if return_indices:
            indices.append(knn_idx)
        masks.append(knn_indicator(knn_idx, layer.shape[0]))
    return layers, masks, indices if return_indices else None


def build_knn_cache(
    feats: torch.Tensor | Sequence[torch.Tensor],
    *,
    topk: int,
    normalize: bool = True,
) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
    layers, masks, _ = _build_knn_cache(
        feats, topk=topk, normalize=normalize, return_indices=False
    )
    return layers, masks


def build_knn_cache_with_indices(
    feats: torch.Tensor | Sequence[torch.Tensor],
    *,
    topk: int,
    normalize: bool = True,
) -> Tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    layers, masks, indices = _build_knn_cache(
        feats, topk=topk, normalize=normalize, return_indices=True
    )
    return layers, indices or [], masks


def _cka_from_grams(K: torch.Tensor, L: torch.Tensor, unbiased: bool = False) -> float:
    hsic_fn = hsic_unbiased if unbiased else hsic_biased
    hsic_kl = hsic_fn(K, L)
    hsic_kk = hsic_fn(K, K)
    hsic_ll = hsic_fn(L, L)
    cka_val = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)
    return float(cka_val.item())


def _cknna_mask_from_gram(
    K: torch.Tensor, topk: int, *, unbiased: bool = True
) -> torch.Tensor:
    n = K.shape[0]
    if unbiased:
        K_hat = K.clone().fill_diagonal_(float("-inf"))
    else:
        K_hat = K
    _, topk_idx = torch.topk(K_hat, topk, dim=1)
    mask = torch.zeros(n, n, device=K.device).scatter_(1, topk_idx, 1)
    return mask


def build_gram_cache(
    feats: torch.Tensor | Sequence[torch.Tensor],
    *,
    normalize: bool = True,
    kernel: str = "linear",
    rbf_sigma: float = 1.0,
) -> list[torch.Tensor]:
    layers = _as_layers(feats)
    if normalize:
        layers = _normalize_layers(layers)
    grams = []
    for layer in layers:
        if kernel == "linear":
            K = layer @ layer.T
        elif kernel == "rbf":
            dists = torch.cdist(layer, layer, p=2)
            K = torch.exp(-(dists**2) / (2 * rbf_sigma**2))
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")
        grams.append(K)
    return grams


def compute_alignment_gated_cached(
    x_layers: Sequence[torch.Tensor],
    y_layers: Sequence[torch.Tensor],
    x_masks: Sequence[torch.Tensor],
    y_masks: Sequence[torch.Tensor],
    *,
    topk: int = 10,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    if len(x_layers) != len(x_masks) or len(y_layers) != len(y_masks):
        raise ValueError("Layer and mask lists must have matching lengths")
    device = x_masks[0].device
    n = x_masks[0].shape[0]
    S = torch.empty((len(x_masks), len(y_masks)), device=device)
    for i, mx in enumerate(x_masks):
        for j, my in enumerate(y_masks):
            overlap = (mx & my).sum(dim=1).float() / float(topk)
            S[i, j] = overlap.mean()

    agg = agg_max(S, return_indices=True)
    T_obs = float(agg.value)
    best_indices = (
        (int(agg.indices["i"]), int(agg.indices["j"])) if agg.indices else (0, 0)
    )

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    null_samples = []
    for _ in range(num_permutations):
        perm = torch.randperm(n, generator=rng, device=device)
        S_perm = torch.empty_like(S)
        for j, my in enumerate(y_masks):
            my_perm = my[perm][:, perm]
            for i, mx in enumerate(x_masks):
                overlap = (mx & my_perm).sum(dim=1).float() / float(topk)
                S_perm[i, j] = overlap.mean()
        null_samples.append(float(agg_max(S_perm).value))

    return _build_gated_summary(
        null_samples, T_obs=T_obs, best_indices=best_indices, alpha=alpha
    )


def compute_alignment_gated_knn_cached(
    x_knn: Sequence[torch.Tensor],
    y_knn: Sequence[torch.Tensor],
    *,
    topk: int = 10,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    if not x_knn or not y_knn:
        raise ValueError("x_knn and y_knn must be non-empty")
    device = x_knn[0].device
    n = x_knn[0].shape[0]

    S = torch.empty((len(x_knn), len(y_knn)), device=device)
    for i, knn_A in enumerate(x_knn):
        for j, knn_B in enumerate(y_knn):
            S[i, j] = _knn_overlap_from_indices(knn_A, knn_B, topk=topk)

    agg = agg_max(S, return_indices=True)
    T_obs = float(agg.value)
    best_indices = (
        (int(agg.indices["i"]), int(agg.indices["j"])) if agg.indices else (0, 0)
    )

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    null_samples = []
    for _ in range(num_permutations):
        perm = torch.randperm(n, generator=rng, device=device)
        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = torch.arange(n, device=device)
        S_perm = torch.empty_like(S)
        for j, knn_B in enumerate(y_knn):
            knn_B_perm = perm_inv[knn_B[perm]]
            for i, knn_A in enumerate(x_knn):
                S_perm[i, j] = _knn_overlap_from_indices(knn_A, knn_B_perm, topk=topk)
        null_samples.append(float(agg_max(S_perm).value))

    return _build_gated_summary(
        null_samples, T_obs=T_obs, best_indices=best_indices, alpha=alpha
    )


def compute_alignment_gated_cycle_knn_cached(
    x_knn: Sequence[torch.Tensor],
    y_knn: Sequence[torch.Tensor],
    *,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    if not x_knn or not y_knn:
        raise ValueError("x_knn and y_knn must be non-empty")
    device = x_knn[0].device
    n = x_knn[0].shape[0]

    def cycle_knn_score(knn_A: torch.Tensor, knn_B: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(n, device=knn_A.device).view(-1, 1, 1)
        return (knn_A[knn_B] == idx).float().view(n, -1).max(dim=1).values.mean()

    S = torch.empty((len(x_knn), len(y_knn)), device=device)
    for i, knn_A in enumerate(x_knn):
        for j, knn_B in enumerate(y_knn):
            S[i, j] = cycle_knn_score(knn_A, knn_B)

    agg = agg_max(S, return_indices=True)
    T_obs = float(agg.value)
    best_indices = (
        (int(agg.indices["i"]), int(agg.indices["j"])) if agg.indices else (0, 0)
    )

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    null_samples = []
    for _ in range(num_permutations):
        perm = torch.randperm(n, generator=rng, device=device)
        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = torch.arange(n, device=device)
        S_perm = torch.empty_like(S)
        for j, knn_B in enumerate(y_knn):
            knn_B_perm = perm_inv[knn_B[perm]]
            for i, knn_A in enumerate(x_knn):
                S_perm[i, j] = cycle_knn_score(knn_A, knn_B_perm)
        null_samples.append(float(agg_max(S_perm).value))

    return _build_gated_summary(
        null_samples, T_obs=T_obs, best_indices=best_indices, alpha=alpha
    )


def _precompute_unbiased_terms(K: torch.Tensor) -> dict:
    K_tilde = K.clone().fill_diagonal_(0)
    return {
        "K_tilde": K_tilde,
        "sum": torch.sum(K_tilde),
        "hsic": hsic_unbiased(K, K),
    }


def _unbiased_hsic_from_terms(
    K_terms: dict,
    L_terms: dict,
    *,
    n: int,
) -> torch.Tensor:
    K_tilde = K_terms["K_tilde"]
    L_tilde = L_terms["K_tilde"]
    term1 = torch.sum(K_tilde * L_tilde)
    term2 = (K_terms["sum"] * L_terms["sum"]) / ((n - 1) * (n - 2))
    term3 = (2.0 * torch.sum(torch.mm(K_tilde, L_tilde))) / (n - 2)
    hsic = (term1 + term2 - term3) / (n * (n - 3))
    return hsic


def compute_alignment_gated_cka_cached(
    x_grams: Sequence[torch.Tensor],
    y_grams: Sequence[torch.Tensor],
    *,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
    unbiased: bool = False,
) -> Dict[str, float | Tuple[int, int]]:
    if not x_grams or not y_grams:
        raise ValueError("x_grams and y_grams must be non-empty")
    device = x_grams[0].device
    n = x_grams[0].shape[0]

    S = torch.empty((len(x_grams), len(y_grams)), device=device)
    if unbiased:
        x_terms = [_precompute_unbiased_terms(K) for K in x_grams]
        y_terms = [_precompute_unbiased_terms(L) for L in y_grams]
        for i, K_terms in enumerate(x_terms):
            for j, L_terms in enumerate(y_terms):
                hsic_kl = _unbiased_hsic_from_terms(K_terms, L_terms, n=n)
                denom = torch.sqrt(K_terms["hsic"] * L_terms["hsic"]) + 1e-6
                S[i, j] = hsic_kl / denom
    else:
        x_centered = [center_gram(K) for K in x_grams]
        y_centered = [center_gram(L) for L in y_grams]
        x_norms = [torch.sum(Kc * Kc) for Kc in x_centered]
        y_norms = [torch.sum(Lc * Lc) for Lc in y_centered]
        for i, Kc in enumerate(x_centered):
            for j, Lc in enumerate(y_centered):
                denom = torch.sqrt(x_norms[i] * y_norms[j]) + 1e-6
                S[i, j] = torch.sum(Kc * Lc) / denom

    agg = agg_max(S, return_indices=True)
    T_obs = float(agg.value)
    best_indices = (
        (int(agg.indices["i"]), int(agg.indices["j"])) if agg.indices else (0, 0)
    )

    null_samples = []
    if unbiased:
        chunk_size = 8
        denom_const = (n - 1) * (n - 2)
        hsic_div = n * (n - 3)
        for perms in batched_perms(
            n,
            num_permutations,
            device=device,
            seed=seed,
            chunk_size=chunk_size,
        ):
            batch = perms.size(0)
            S_perm = torch.empty((batch, len(x_grams), len(y_grams)), device=device)
            for j, L_terms in enumerate(y_terms):
                L_tilde = L_terms["K_tilde"]
                temp = L_tilde[perms]
                L_perm = torch.gather(temp, 2, perms.unsqueeze(1).expand(-1, n, -1))
                sumL = L_terms["sum"]
                hsic_ll = L_terms["hsic"]
                for i, K_terms in enumerate(x_terms):
                    K_tilde = K_terms["K_tilde"]
                    sumK = K_terms["sum"]
                    hsic_kk = K_terms["hsic"]
                    term1 = (L_perm * K_tilde).sum(dim=(1, 2))
                    prod = torch.einsum("ij,bjk->bik", K_tilde, L_perm)
                    term3 = (2.0 * prod.sum(dim=(1, 2))) / (n - 2)
                    hsic_kl = (term1 + (sumK * sumL) / denom_const - term3) / hsic_div
                    denom = torch.sqrt(hsic_kk * hsic_ll) + 1e-6
                    S_perm[:, i, j] = hsic_kl / denom
            null_samples.extend(S_perm.view(batch, -1).max(dim=1).values.tolist())
    else:
        chunk_size = 16
        for perms in batched_perms(
            n,
            num_permutations,
            device=device,
            seed=seed,
            chunk_size=chunk_size,
        ):
            batch = perms.size(0)
            S_perm = torch.empty((batch, len(x_grams), len(y_grams)), device=device)
            for j, Lc in enumerate(y_centered):
                temp = Lc[perms]
                Lc_perm = torch.gather(temp, 2, perms.unsqueeze(1).expand(-1, n, -1))
                for i, Kc in enumerate(x_centered):
                    denom = torch.sqrt(x_norms[i] * y_norms[j]) + 1e-6
                    S_perm[:, i, j] = (Lc_perm * Kc).sum(dim=(1, 2)) / denom
            null_samples.extend(S_perm.view(batch, -1).max(dim=1).values.tolist())

    return _build_gated_summary(
        null_samples, T_obs=T_obs, best_indices=best_indices, alpha=alpha
    )


def compute_alignment_gated_cknna_cached(
    x_grams: Sequence[torch.Tensor],
    y_grams: Sequence[torch.Tensor],
    *,
    topk: int = 10,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
    unbiased: bool = True,
) -> Dict[str, float | Tuple[int, int]]:
    if not x_grams or not y_grams:
        raise ValueError("x_grams and y_grams must be non-empty")
    device = x_grams[0].device
    n = x_grams[0].shape[0]
    hsic_fn = hsic_unbiased if unbiased else hsic_biased

    x_masks = [_cknna_mask_from_gram(K, topk=topk, unbiased=unbiased) for K in x_grams]
    y_masks = [_cknna_mask_from_gram(L, topk=topk, unbiased=unbiased) for L in y_grams]
    x_sim_kk = [hsic_fn(mask * K, mask * K) for mask, K in zip(x_masks, x_grams)]
    y_sim_ll = [hsic_fn(mask * L, mask * L) for mask, L in zip(y_masks, y_grams)]

    S = torch.empty((len(x_grams), len(y_grams)), device=device)
    for i, K in enumerate(x_grams):
        mask_K = x_masks[i]
        sim_kk = x_sim_kk[i]
        for j, L in enumerate(y_grams):
            mask = mask_K * y_masks[j]
            sim_kl = hsic_fn(mask * K, mask * L)
            denom = torch.sqrt(sim_kk * y_sim_ll[j]) + 1e-6
            S[i, j] = sim_kl / denom

    agg = agg_max(S, return_indices=True)
    T_obs = float(agg.value)
    best_indices = (
        (int(agg.indices["i"]), int(agg.indices["j"])) if agg.indices else (0, 0)
    )

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    null_samples = []
    if unbiased:
        chunk_size = 8
        denom_const = (n - 1) * (n - 2)
        hsic_div = n * (n - 3)
        for perms in batched_perms(
            n,
            num_permutations,
            device=device,
            seed=seed,
            chunk_size=chunk_size,
        ):
            batch = perms.size(0)
            S_perm = torch.empty((batch, len(x_grams), len(y_grams)), device=device)
            for j, L in enumerate(y_grams):
                L_perm = L[perms]
                L_perm = torch.gather(L_perm, 2, perms.unsqueeze(1).expand(-1, n, -1))
                mask_L_perm = y_masks[j][perms]
                mask_L_perm = torch.gather(
                    mask_L_perm, 2, perms.unsqueeze(1).expand(-1, n, -1)
                )
                sim_ll = y_sim_ll[j]
                for i, K in enumerate(x_grams):
                    mask = x_masks[i] * mask_L_perm
                    K_mask = mask * K
                    L_mask = mask * L_perm
                    term1 = (K_mask * L_mask.transpose(1, 2)).sum(dim=(1, 2))
                    sumK = K_mask.sum(dim=(1, 2))
                    sumL = L_mask.sum(dim=(1, 2))
                    term3 = 2.0 * torch.bmm(K_mask, L_mask).sum(dim=(1, 2)) / (n - 2)
                    hsic_kl = (term1 + (sumK * sumL) / denom_const - term3) / hsic_div
                    denom = torch.sqrt(x_sim_kk[i] * sim_ll) + 1e-6
                    S_perm[:, i, j] = hsic_kl / denom
            null_samples.extend(S_perm.view(batch, -1).max(dim=1).values.tolist())
    else:
        for _ in range(num_permutations):
            perm = torch.randperm(n, generator=rng, device=device)
            S_perm = torch.empty_like(S)
            for j, L in enumerate(y_grams):
                L_perm = L[perm][:, perm]
                mask_L_perm = y_masks[j][perm][:, perm]
                sim_ll = y_sim_ll[j]
                for i, K in enumerate(x_grams):
                    mask = x_masks[i] * mask_L_perm
                    sim_kl = hsic_fn(mask * K, mask * L_perm)
                    denom = torch.sqrt(x_sim_kk[i] * sim_ll) + 1e-6
                    S_perm[i, j] = sim_kl / denom
            null_samples.append(float(agg_max(S_perm).value))

    return _build_gated_summary(
        null_samples, T_obs=T_obs, best_indices=best_indices, alpha=alpha
    )


def _svcca_from_preprocessed(
    feats_A: torch.Tensor, feats_B: torch.Tensor, *, cca_dim: int
) -> float:
    U1, _, _ = torch.svd_lowrank(feats_A, q=cca_dim)
    U2, _, _ = torch.svd_lowrank(feats_B, q=cca_dim)
    U1 = U1.cpu().detach().numpy()
    U2 = U2.cpu().detach().numpy()
    cca = CCA(n_components=cca_dim)
    cca.fit(U1, U2)
    U1_c, U2_c = cca.transform(U1, U2)
    U1_c += 1e-10 * np.random.randn(*U1_c.shape)
    U2_c += 1e-10 * np.random.randn(*U2_c.shape)
    svcca_similarity = np.mean(
        [np.corrcoef(U1_c[:, i], U2_c[:, i])[0, 1] for i in range(cca_dim)]
    )
    return float(svcca_similarity)


def compute_alignment_gated_svcca_cached(
    x_layers: Sequence[torch.Tensor],
    y_layers: Sequence[torch.Tensor],
    *,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    if not x_layers or not y_layers:
        raise ValueError("x_layers and y_layers must be non-empty")
    device = x_layers[0].device
    n = x_layers[0].shape[0]

    x_pre = [svcca_preprocess(x) for x in x_layers]
    y_pre = [svcca_preprocess(y) for y in y_layers]

    S = torch.empty((len(x_pre), len(y_pre)), device=device)
    for i, x in enumerate(x_pre):
        for j, y in enumerate(y_pre):
            cca_dim = min(10, x.shape[1], y.shape[1])
            S[i, j] = _svcca_from_preprocessed(x, y, cca_dim=cca_dim)

    agg = agg_max(S, return_indices=True)
    T_obs = float(agg.value)
    best_indices = (
        (int(agg.indices["i"]), int(agg.indices["j"])) if agg.indices else (0, 0)
    )

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    null_samples = []
    for _ in range(num_permutations):
        perm = torch.randperm(n, generator=rng, device=device)
        y_perm = [y[perm] for y in y_pre]
        S_perm = torch.empty_like(S)
        for i, x in enumerate(x_pre):
            for j, y in enumerate(y_perm):
                cca_dim = min(10, x.shape[1], y.shape[1])
                S_perm[i, j] = _svcca_from_preprocessed(x, y, cca_dim=cca_dim)
        null_samples.append(float(agg_max(S_perm).value))

    return _build_gated_summary(
        null_samples, T_obs=T_obs, best_indices=best_indices, alpha=alpha
    )


def _cca_whiten(
    layer: torch.Tensor, reg: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Precompute whitening for one layer. Returns (Xc, C_inv_sqrt, dim)."""
    device = layer.device
    Xc = (layer - layer.mean(0, keepdim=True)).to(dtype=torch.float32)
    d = Xc.shape[1]
    n = Xc.shape[0]
    eye = torch.eye(d, device=device, dtype=torch.float32)
    C = (Xc.T @ Xc) / (n - 1) + reg * eye
    S, U = torch.linalg.eigh(C)
    S = torch.clamp(S, min=reg)
    C_inv_sqrt = U @ torch.diag(1.0 / torch.sqrt(S)) @ U.T
    return Xc, C_inv_sqrt, d


def compute_alignment_gated_pwcca_cached(
    x_layers: Sequence[torch.Tensor],
    y_layers: Sequence[torch.Tensor],
    *,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
    reg: float = 1e-6,
) -> Dict[str, float | Tuple[int, int]]:
    """PWCCA with GPU-accelerated batched null distribution.

    Uses batched eigh (not eigvalsh) because PWCCA needs eigenvectors
    for projection weighting. Whitening precomputed per layer.
    """
    if not x_layers or not y_layers:
        raise ValueError("x_layers and y_layers must be non-empty")
    device = x_layers[0].device
    n = x_layers[0].shape[0]
    n_x = len(x_layers)
    n_y = len(y_layers)

    # Precompute whitening per X layer (need Cxx_inv_sqrt for weighting)
    x_cache = []
    for x_layer in x_layers:
        Xc, Cxx_inv_sqrt, dx = _cca_whiten(x_layer, reg)
        L = Cxx_inv_sqrt @ Xc.T  # (dx, n)
        x_cache.append((L, Cxx_inv_sqrt, Xc, dx))

    # Precompute whitening per Y layer
    y_cache = []
    for y_layer in y_layers:
        Yc, Cyy_inv_sqrt, dy = _cca_whiten(y_layer, reg)
        R = Yc @ Cyy_inv_sqrt  # (n, dy)
        y_cache.append((R, dy))

    # Raw similarity matrix with PWCCA scoring
    S = torch.empty((n_x, n_y), device=device)
    for i, (L_i, Cxx_inv_sqrt_i, Xc_i, dx_i) in enumerate(x_cache):
        for j, (R_j, dy_j) in enumerate(y_cache):
            T = (L_i @ R_j) / (n - 1)
            TTt = T @ T.T
            eigvals, U = torch.linalg.eigh(TTt)
            # eigh ascending → reverse for SVD descending convention
            svals = torch.sqrt(torch.clamp(eigvals.flip(-1), min=0.0))
            U = U.flip(-1)
            S[i, j] = _pwcca_weighted_mean(
                svals, U, Cxx_inv_sqrt_i, Xc_i, min(dx_i, dy_j, n - 1)
            )

    agg = agg_max(S, return_indices=True)
    T_obs = float(agg.value)
    best_indices = (
        (int(agg.indices["i"]), int(agg.indices["j"])) if agg.indices else (0, 0)
    )

    # Generate permutations
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    perms = torch.stack(
        [
            torch.randperm(n, generator=rng, device=device)
            for _ in range(num_permutations)
        ]
    )

    # Memory-aware batch size
    max_dx = max(dx for _, _, _, dx in x_cache)
    max_dy = max(dy for _, dy in y_cache)
    elem = 4  # float32
    per_perm_bytes = (n * max_dy + max_dx * max_dy + max_dx * max_dx * 2) * elem
    max_batch_bytes = 512 * 1024 * 1024
    chunk = max(1, max_batch_bytes // max(per_perm_bytes, 1))
    chunk = min(num_permutations, chunk)

    # Batched null distribution
    null_samples: list[float] = []
    for start in range(0, num_permutations, chunk):
        end = min(start + chunk, num_permutations)
        batch_perms = perms[start:end]
        B = batch_perms.shape[0]

        S_perm = torch.empty(B, n_x, n_y, device=device)
        for j, (R_j, dy_j) in enumerate(y_cache):
            R_perm = R_j[batch_perms]  # (B, n, dy)
            for i, (L_i, Cxx_inv_sqrt_i, Xc_i, dx_i) in enumerate(x_cache):
                T_batch = torch.einsum("xn,bny->bxy", L_i, R_perm) / (n - 1)
                TTt = T_batch @ T_batch.transpose(-1, -2)
                eigvals, U_batch = torch.linalg.eigh(TTt)  # (B, dx), (B, dx, dx)
                svals = torch.sqrt(torch.clamp(eigvals.flip(-1), min=0.0))
                U_batch = U_batch.flip(-1)
                S_perm[:, i, j] = _pwcca_weighted_mean(
                    svals, U_batch, Cxx_inv_sqrt_i, Xc_i, min(dx_i, dy_j, n - 1)
                )

        null_samples.extend(S_perm.view(B, -1).max(dim=1).values.tolist())

    return _build_gated_summary(
        null_samples, T_obs=T_obs, best_indices=best_indices, alpha=alpha
    )


def _procrustes_score_cached(
    Xc: np.ndarray,
    Yc_perm: np.ndarray,
) -> float:
    U, _, Vt = np.linalg.svd(Xc.T @ Yc_perm, full_matrices=False)
    R = U @ Vt
    aligned = Xc @ R
    num = np.linalg.norm(aligned - Yc_perm, ord="fro")
    denom = np.linalg.norm(Yc_perm, ord="fro") + 1e-8
    return float(1.0 - num / denom)


def compute_alignment_gated_procrustes_cached(
    x_layers: Sequence[torch.Tensor],
    y_layers: Sequence[torch.Tensor],
    *,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    if not x_layers or not y_layers:
        raise ValueError("x_layers and y_layers must be non-empty")
    device = x_layers[0].device
    n = x_layers[0].shape[0]

    x_cache = [center_np(x.detach().cpu().numpy()) for x in x_layers]
    y_cache = [center_np(y.detach().cpu().numpy()) for y in y_layers]

    S = torch.empty((len(x_cache), len(y_cache)), device=device)
    for i, Xc in enumerate(x_cache):
        for j, Yc in enumerate(y_cache):
            S[i, j] = _procrustes_score_cached(Xc, Yc)

    agg = agg_max(S, return_indices=True)
    T_obs = float(agg.value)
    best_indices = (
        (int(agg.indices["i"]), int(agg.indices["j"])) if agg.indices else (0, 0)
    )

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    null_samples = []
    for _ in range(num_permutations):
        perm = torch.randperm(n, generator=rng, device=device)
        perm_np = perm.cpu().numpy()
        S_perm = torch.empty_like(S)
        y_perm = [Yc[perm_np] for Yc in y_cache]
        for i, Xc in enumerate(x_cache):
            for j, Yc_perm in enumerate(y_perm):
                S_perm[i, j] = _procrustes_score_cached(Xc, Yc_perm)
        null_samples.append(float(agg_max(S_perm).value))

    return _build_gated_summary(
        null_samples, T_obs=T_obs, best_indices=best_indices, alpha=alpha
    )


def _cka_linear_grams(layers: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    grams = []
    for layer in layers:
        Xc = layer - layer.mean(0, keepdim=True)
        grams.append(Xc @ Xc.T)
    return grams


def _similarity_matrix_cka_linear_cached(
    repsA_layers: Sequence[torch.Tensor],
    repsB_layers: Sequence[torch.Tensor],
) -> torch.Tensor:
    x_grams = _cka_linear_grams(repsA_layers)
    y_grams = _cka_linear_grams(repsB_layers)
    S = torch.empty((len(x_grams), len(y_grams)), device=x_grams[0].device)
    for i, K in enumerate(x_grams):
        for j, L in enumerate(y_grams):
            S[i, j] = _cka_from_grams(K, L, unbiased=False)
    return S


# =============================================================================
# Batched operations for GPU-accelerated experiments
# =============================================================================


def _center_gram_batched(K: torch.Tensor) -> torch.Tensor:
    """Center Gram matrices in batch. K shape: (..., n, n)."""
    row_mean = K.mean(dim=-1, keepdim=True)
    col_mean = K.mean(dim=-2, keepdim=True)
    grand_mean = K.mean(dim=(-2, -1), keepdim=True)
    return K - row_mean - col_mean + grand_mean


def similarity_matrix_cka_batched(
    repsA: torch.Tensor,
    repsB: torch.Tensor,
) -> torch.Tensor:
    """Compute CKA similarity matrices for batched layer representations.

    Args:
        repsA: Shape (T, L_a, n, d) - T trials, L_a layers.
        repsB: Shape (T, L_b, n, d) - T trials, L_b layers.

    Returns:
        Similarity matrices of shape (T, L_a, L_b).
    """
    T, L_a, n, d = repsA.shape
    L_b = repsB.shape[1]
    device = repsA.device

    # Center representations
    repsA_c = repsA - repsA.mean(dim=2, keepdim=True)  # (T, L_a, n, d)
    repsB_c = repsB - repsB.mean(dim=2, keepdim=True)  # (T, L_b, n, d)

    # Compute Gram matrices: gramsA[t,l] = repsA_c[t,l] @ repsA_c[t,l].T -> (T, L, n, n)
    gramsA = torch.bmm(
        repsA_c.view(T * L_a, n, d),
        repsA_c.view(T * L_a, n, d).transpose(-1, -2),
    ).view(T, L_a, n, n)
    gramsB = torch.bmm(
        repsB_c.view(T * L_b, n, d),
        repsB_c.view(T * L_b, n, d).transpose(-1, -2),
    ).view(T, L_b, n, n)

    # Center Gram matrices
    gramsA_c = _center_gram_batched(gramsA)  # (T, L_a, n, n)
    gramsB_c = _center_gram_batched(gramsB)  # (T, L_b, n, n)

    # Precompute HSIC(K, K) for all layers
    hsic_A = (gramsA_c * gramsA_c).sum(dim=(-2, -1))  # (T, L_a)
    hsic_B = (gramsB_c * gramsB_c).sum(dim=(-2, -1))  # (T, L_b)

    # Compute all pairwise CKA values
    S = torch.empty(T, L_a, L_b, device=device)
    for i in range(L_a):
        for j in range(L_b):
            Ka = gramsA_c[:, i]  # (T, n, n)
            Lb = gramsB_c[:, j]  # (T, n, n)
            hsic_kl = (Ka * Lb).sum(dim=(-2, -1))  # (T,)
            denom = torch.sqrt(hsic_A[:, i] * hsic_B[:, j]) + 1e-8
            S[:, i, j] = hsic_kl / denom

    return S


def permutation_null_batched(
    repsA: torch.Tensor,
    repsB: torch.Tensor,
    num_permutations: int,
    *,
    seed: int | None = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """Compute permutation null distributions for batched trials.

    Args:
        repsA: Shape (T, L_a, n, d).
        repsB: Shape (T, L_b, n, d).
        num_permutations: Number of permutations.
        seed: Random seed.
        show_progress: Whether to show tqdm progress bar.

    Returns:
        Null max values of shape (T, num_permutations).
    """
    T, L_a, n, d = repsA.shape
    L_b = repsB.shape[1]
    device = repsA.device

    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    # Center representations once
    repsA_c = repsA - repsA.mean(dim=2, keepdim=True)
    repsB_c = repsB - repsB.mean(dim=2, keepdim=True)

    # Compute Gram matrices for A (fixed)
    gramsA = torch.bmm(
        repsA_c.view(T * L_a, n, d),
        repsA_c.view(T * L_a, n, d).transpose(-1, -2),
    ).view(T, L_a, n, n)
    gramsA_c = _center_gram_batched(gramsA)
    hsic_A = (gramsA_c * gramsA_c).sum(dim=(-2, -1))  # (T, L_a)

    # Compute B Gram matrices (will be permuted)
    gramsB = torch.bmm(
        repsB_c.view(T * L_b, n, d),
        repsB_c.view(T * L_b, n, d).transpose(-1, -2),
    ).view(T, L_b, n, n)
    gramsB_c = _center_gram_batched(gramsB)
    hsic_B = (gramsB_c * gramsB_c).sum(dim=(-2, -1))  # (T, L_b)

    null_maxes = torch.empty(T, num_permutations, device=device)

    perm_iter = range(num_permutations)
    if show_progress:
        perm_iter = tqdm(perm_iter, desc="Permutations", unit="perm", leave=False)

    for p in perm_iter:
        perm = torch.randperm(n, device=device)
        # Permute B Gram matrices: L[perm][:, perm]
        gramsB_perm = gramsB_c[:, :, perm][:, :, :, perm]  # (T, L_b, n, n)

        # Compute similarity matrix for this permutation
        S_perm = torch.empty(T, L_a, L_b, device=device)
        for i in range(L_a):
            for j in range(L_b):
                Ka = gramsA_c[:, i]  # (T, n, n)
                Lb_perm = gramsB_perm[:, j]  # (T, n, n)
                hsic_kl = (Ka * Lb_perm).sum(dim=(-2, -1))
                denom = torch.sqrt(hsic_A[:, i] * hsic_B[:, j]) + 1e-8
                S_perm[:, i, j] = hsic_kl / denom

        # Get max per trial
        null_maxes[:, p] = S_perm.view(T, -1).max(dim=1).values

    return null_maxes


def run_batched_gating_experiment(
    repsA: torch.Tensor,
    repsB: torch.Tensor,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
    show_progress: bool = False,
) -> Dict[str, torch.Tensor]:
    """Run gating experiment on batched trials.

    Args:
        repsA: Shape (T, L, n, d).
        repsB: Shape (T, L, n, d).
        num_permutations: Number of permutations for null.
        alpha: Significance level.
        seed: Random seed.
        show_progress: Whether to show tqdm progress bar.

    Returns:
        Dictionary with:
        - raw: (T,) raw max scores
        - gated: (T,) gated scores
        - p_value: (T,) p-values
        - tau_alpha: (T,) thresholds
    """
    T = repsA.shape[0]
    _device = repsA.device  # noqa: F841 - kept for reference

    # Compute observed similarity matrices
    S_obs = similarity_matrix_cka_batched(repsA, repsB)  # (T, L, L)
    T_obs = S_obs.view(T, -1).max(dim=1).values  # (T,)

    # Compute permutation nulls
    null_maxes = permutation_null_batched(
        repsA, repsB, num_permutations, seed=seed, show_progress=show_progress
    )  # (T, num_permutations)

    # Compute p-values
    p_values = (null_maxes >= T_obs.unsqueeze(1)).float().mean(dim=1)

    # Compute tau_alpha (1-alpha quantile of null)
    sorted_nulls, _ = null_maxes.sort(dim=1)
    tau_idx = int((1 - alpha) * num_permutations)
    tau_alpha = sorted_nulls[:, min(tau_idx, num_permutations - 1)]

    # Compute gated scores
    s_max = 1.0  # CKA max value
    gated = (T_obs - tau_alpha) / (s_max - tau_alpha + 1e-8)
    gated = gated.clamp(min=0.0, max=1.0)
    gated = torch.where(p_values > alpha, torch.zeros_like(gated), gated)

    # Compute null statistics
    mu0 = null_maxes.mean(dim=1)
    sd0 = null_maxes.std(dim=1)
    tail_strength = (T_obs - mu0) / (sd0 + 1e-8)

    return {
        "raw": T_obs,
        "gated": gated,
        "p_value": p_values,
        "tau_alpha": tau_alpha,
        "tail_strength": tail_strength,
        "mu0": mu0,
        "sd0": sd0,
    }


def permutation_null_matrices_batched(
    repsA: torch.Tensor,
    repsB: torch.Tensor,
    num_permutations: int,
    *,
    seed: int | None = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """Compute batched permutation null similarity matrices.

    Args:
        repsA: Shape (T, L_a, n, d).
        repsB: Shape (T, L_b, n, d).
        num_permutations: Number of permutations.
        seed: Random seed.
        show_progress: Whether to show tqdm progress bar.

    Returns:
        Null similarity matrices of shape (T, num_permutations, L_a, L_b).
    """
    T, L_a, n, d = repsA.shape
    L_b = repsB.shape[1]
    device = repsA.device

    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    # Center representations once
    repsA_c = repsA - repsA.mean(dim=2, keepdim=True)
    repsB_c = repsB - repsB.mean(dim=2, keepdim=True)

    # Compute Gram matrices for A (fixed)
    gramsA = torch.bmm(
        repsA_c.view(T * L_a, n, d),
        repsA_c.view(T * L_a, n, d).transpose(-1, -2),
    ).view(T, L_a, n, n)
    gramsA_c = _center_gram_batched(gramsA)
    hsic_A = (gramsA_c * gramsA_c).sum(dim=(-2, -1))  # (T, L_a)

    # Compute B Gram matrices
    gramsB = torch.bmm(
        repsB_c.view(T * L_b, n, d),
        repsB_c.view(T * L_b, n, d).transpose(-1, -2),
    ).view(T, L_b, n, n)
    gramsB_c = _center_gram_batched(gramsB)
    hsic_B = (gramsB_c * gramsB_c).sum(dim=(-2, -1))  # (T, L_b)

    null_matrices = torch.empty(T, num_permutations, L_a, L_b, device=device)

    perm_iter = range(num_permutations)
    if show_progress:
        perm_iter = tqdm(perm_iter, desc="Permutations", unit="perm", leave=False)

    for p in perm_iter:
        perm = torch.randperm(n, device=device)
        # Permute B Gram matrices
        gramsB_perm = gramsB_c[:, :, perm][:, :, :, perm]  # (T, L_b, n, n)

        # Compute similarity matrix for this permutation
        for i in range(L_a):
            for j in range(L_b):
                Ka = gramsA_c[:, i]  # (T, n, n)
                Lb_perm = gramsB_perm[:, j]  # (T, n, n)
                hsic_kl = (Ka * Lb_perm).sum(dim=(-2, -1))
                denom = torch.sqrt(hsic_A[:, i] * hsic_B[:, j]) + 1e-8
                null_matrices[:, p, i, j] = hsic_kl / denom

    return null_matrices


def run_batched_multi_aggregator_experiment(
    repsA: torch.Tensor,
    repsB: torch.Tensor,
    aggregator_names: Sequence[str],
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
    show_progress: bool = False,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Run gating experiment with multiple aggregators on batched trials.

    Computes similarity matrices and permutation nulls once, then applies
    multiple aggregators efficiently.

    Args:
        repsA: Shape (T, L, n, d).
        repsB: Shape (T, L, n, d).
        aggregator_names: List of aggregator names ('max', 'rowmax_mean', etc.).
        num_permutations: Number of permutations for null.
        alpha: Significance level.
        seed: Random seed.
        show_progress: Whether to show tqdm progress bar.

    Returns:
        Dictionary mapping aggregator names to result dicts.
    """
    _T = repsA.shape[0]  # noqa: F841 - kept for reference
    _device = repsA.device  # noqa: F841 - kept for reference
    s_max = 1.0  # CKA max value

    # Compute observed similarity matrices once
    S_obs = similarity_matrix_cka_batched(repsA, repsB)  # (T, L, L)

    # Compute permutation null matrices once
    null_matrices = permutation_null_matrices_batched(
        repsA, repsB, num_permutations, seed=seed, show_progress=show_progress
    )  # (T, num_permutations, L, L)

    # Define aggregator functions
    def agg_max_batch(S: torch.Tensor) -> torch.Tensor:
        """S shape: (T, L, L) or (T, P, L, L)"""
        return S.view(*S.shape[:-2], -1).max(dim=-1).values

    def agg_rowmax_mean_batch(S: torch.Tensor) -> torch.Tensor:
        """S shape: (T, L, L) or (T, P, L, L)"""
        return S.max(dim=-1).values.mean(dim=-1)

    def agg_colmax_mean_batch(S: torch.Tensor) -> torch.Tensor:
        """S shape: (T, L, L) or (T, P, L, L)"""
        return S.max(dim=-2).values.mean(dim=-1)

    def agg_topk_mean_batch(S: torch.Tensor, k: int) -> torch.Tensor:
        """S shape: (T, L, L) or (T, P, L, L)"""
        flat = S.view(*S.shape[:-2], -1)
        topk_vals, _ = flat.topk(min(k, flat.shape[-1]), dim=-1)
        return topk_vals.mean(dim=-1)

    agg_fns = {
        "max": agg_max_batch,
        "rowmax_mean": agg_rowmax_mean_batch,
        "colmax_mean": agg_colmax_mean_batch,
        "topk_5": lambda S: agg_topk_mean_batch(S, 5),
        "topk_10": lambda S: agg_topk_mean_batch(S, 10),
    }

    results = {}
    for agg_name in aggregator_names:
        agg_fn = agg_fns[agg_name]

        # Apply aggregator to observed and null matrices
        T_obs = agg_fn(S_obs)  # (T,)
        null_agg = agg_fn(null_matrices)  # (T, num_permutations)

        # Compute p-values
        p_values = (null_agg >= T_obs.unsqueeze(1)).float().mean(dim=1)

        # Compute tau_alpha
        sorted_nulls, _ = null_agg.sort(dim=1)
        tau_idx = int((1 - alpha) * num_permutations)
        tau_alpha = sorted_nulls[:, min(tau_idx, num_permutations - 1)]

        # Compute gated scores
        gated = (T_obs - tau_alpha) / (s_max - tau_alpha + 1e-8)
        gated = gated.clamp(min=0.0, max=1.0)
        gated = torch.where(p_values > alpha, torch.zeros_like(gated), gated)

        # Compute null statistics
        mu0 = null_agg.mean(dim=1)
        sd0 = null_agg.std(dim=1)
        tail_strength = (T_obs - mu0) / (sd0 + 1e-8)

        results[agg_name] = {
            "raw": T_obs,
            "gated": gated,
            "p_value": p_values,
            "tau_alpha": tau_alpha,
            "tail_strength": tail_strength,
        }

    return results


def similarity_matrix_layerwise(
    repsA_layers: Sequence[torch.Tensor],
    repsB_layers: Sequence[torch.Tensor],
    metric: SimpleMetric,
    *,
    metric_name: str | None = None,
) -> torch.Tensor:
    if metric_name in {"cka_linear", "cka_lin"}:
        return _similarity_matrix_cka_linear_cached(repsA_layers, repsB_layers)
    return compute_similarity_matrix(repsA_layers, repsB_layers, metric)


def permutation_null_matrices_layerwise(
    repsA_layers: Sequence[torch.Tensor],
    repsB_layers: Sequence[torch.Tensor],
    metric: SimpleMetric,
    *,
    metric_name: str | None = None,
    num_permutations: int,
    seed: int | None,
) -> list[torch.Tensor]:
    if num_permutations <= 0:
        raise ValueError("num_permutations must be positive")
    n = repsA_layers[0].shape[0]
    device = repsA_layers[0].device
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    if metric_name in {"cka_linear", "cka_lin"}:
        x_grams = _cka_linear_grams(repsA_layers)
        y_grams = _cka_linear_grams(repsB_layers)
        matrices = []
        for _ in range(num_permutations):
            perm = torch.randperm(n, generator=rng, device=device)
            S_perm = torch.empty((len(x_grams), len(y_grams)), device=device)
            for j, L in enumerate(y_grams):
                L_perm = L[perm][:, perm]
                for i, K in enumerate(x_grams):
                    S_perm[i, j] = _cka_from_grams(K, L_perm, unbiased=False)
            matrices.append(S_perm)
        return matrices

    matrices = []
    for _ in range(num_permutations):
        perm = torch.randperm(n, generator=rng, device=device)
        repsB_perm = [Y[perm, :] for Y in repsB_layers]
        matrices.append(compute_similarity_matrix(repsA_layers, repsB_perm, metric))
    return matrices
