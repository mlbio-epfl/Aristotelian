import torch

from aristotelian.experiments.layerwise_engine import (
    build_gram_cache,
    compute_alignment_gated_cka_cached,
    compute_alignment_gated_cknna_cached,
)
from aristotelian.metrics.aggregation import agg_max


def _hsic_biased_gram(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype, device=K.device) - 1.0 / n
    return torch.trace(K @ H @ L @ H)


def _cka_from_grams(K: torch.Tensor, L: torch.Tensor) -> float:
    hsic_kl = _hsic_biased_gram(K, L)
    hsic_kk = _hsic_biased_gram(K, K)
    hsic_ll = _hsic_biased_gram(L, L)
    cka_val = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)
    return float(cka_val.item())


def _cknna_from_grams_reference(K: torch.Tensor, L: torch.Tensor, topk: int) -> float:
    n = K.shape[0]

    def similarity(Km: torch.Tensor, Lm: torch.Tensor) -> torch.Tensor:
        K_hat = Km.clone().fill_diagonal_(float("-inf"))
        L_hat = Lm.clone().fill_diagonal_(float("-inf"))
        _, topk_K_indices = torch.topk(K_hat, topk, dim=1)
        _, topk_L_indices = torch.topk(L_hat, topk, dim=1)
        mask_K = torch.zeros(n, n, device=Km.device).scatter_(1, topk_K_indices, 1)
        mask_L = torch.zeros(n, n, device=Km.device).scatter_(1, topk_L_indices, 1)
        mask = mask_K * mask_L
        return _hsic_unbiased_gram_reference(mask * Km, mask * Lm)

    def _hsic_unbiased_gram_reference(
        Km: torch.Tensor, Lm: torch.Tensor
    ) -> torch.Tensor:
        m = Km.shape[0]
        K_tilde = Km.clone().fill_diagonal_(0)
        L_tilde = Lm.clone().fill_diagonal_(0)
        hsic_value = (
            (torch.sum(K_tilde * L_tilde.T))
            + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
            - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
        )
        hsic_value /= m * (m - 3)
        return hsic_value

    sim_kl = similarity(K, L)
    sim_kk = similarity(K, K)
    sim_ll = similarity(L, L)
    return float(sim_kl.item() / (torch.sqrt(sim_kk * sim_ll) + 1e-6).item())


def test_layerwise_cka_cached_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(8, 2, 4)
    y = torch.randn(8, 2, 4)
    x_grams = build_gram_cache(x, normalize=True, kernel="linear")
    y_grams = build_gram_cache(y, normalize=True, kernel="linear")
    res = compute_alignment_gated_cka_cached(
        x_grams,
        y_grams,
        num_permutations=4,
        alpha=0.05,
        seed=123,
        unbiased=False,
    )

    # Reference: compute S + permutation null using direct CKA-from-grams
    device = x_grams[0].device
    S = torch.empty((len(x_grams), len(y_grams)), device=device)
    for i, K in enumerate(x_grams):
        for j, L in enumerate(y_grams):
            S[i, j] = _cka_from_grams(K, L)
    T_obs = float(agg_max(S).value)
    assert abs(res["raw_score"] - T_obs) < 1e-6


def test_layerwise_cknna_cached_matches_reference():
    torch.manual_seed(1)
    x = torch.randn(10, 2, 4)
    y = torch.randn(10, 2, 4)
    x_grams = build_gram_cache(x, normalize=True, kernel="linear")
    y_grams = build_gram_cache(y, normalize=True, kernel="linear")
    res = compute_alignment_gated_cknna_cached(
        x_grams,
        y_grams,
        topk=3,
        num_permutations=3,
        alpha=0.05,
        seed=5,
        unbiased=True,
    )

    device = x_grams[0].device
    S = torch.empty((len(x_grams), len(y_grams)), device=device)
    for i, K in enumerate(x_grams):
        for j, L in enumerate(y_grams):
            S[i, j] = _cknna_from_grams_reference(K, L, topk=3)
    T_obs = float(agg_max(S).value)
    assert abs(res["raw_score"] - T_obs) < 1e-6
