import torch

from aristotelian.metrics.aggregation import (
    agg_max,
    compute_null_summary,
    gated_rescaled,
)
from aristotelian.prh.prh_experiment import (
    build_gram_cache,
    compute_alignment_gated_cka_cached,
)


def _hsic_biased_gram(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype, device=K.device) - 1.0 / n
    return torch.trace(K @ H @ L @ H)


def _hsic_unbiased_gram(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    m = K.shape[0]
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)
    hsic_value = (
        (torch.sum(K_tilde * L_tilde.T))
        + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
    )
    hsic_value /= m * (m - 3)
    return hsic_value


def _cka_from_grams(K: torch.Tensor, L: torch.Tensor, *, unbiased: bool) -> float:
    hsic_fn = _hsic_unbiased_gram if unbiased else _hsic_biased_gram
    hsic_kl = hsic_fn(K, L)
    hsic_kk = hsic_fn(K, K)
    hsic_ll = hsic_fn(L, L)
    cka_val = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)
    return float(cka_val.item())


def _reference_alignment(
    x_grams,
    y_grams,
    *,
    num_permutations: int,
    alpha: float,
    seed: int,
    unbiased: bool,
):
    device = x_grams[0].device
    n = x_grams[0].shape[0]
    S = torch.empty((len(x_grams), len(y_grams)), device=device)
    for i, K in enumerate(x_grams):
        for j, L in enumerate(y_grams):
            S[i, j] = _cka_from_grams(K, L, unbiased=unbiased)

    agg = agg_max(S, return_indices=True)
    T_obs = float(agg.value)
    best_indices = (
        (int(agg.indices["i"]), int(agg.indices["j"])) if agg.indices else (0, 0)
    )

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    torch.manual_seed(seed)

    null_samples = []
    for _ in range(num_permutations):
        perm = torch.randperm(n, generator=rng, device=device)
        S_perm = torch.empty_like(S)
        for j, L in enumerate(y_grams):
            L_perm = L[perm][:, perm]
            for i, K in enumerate(x_grams):
                S_perm[i, j] = _cka_from_grams(K, L_perm, unbiased=unbiased)
        null_samples.append(float(agg_max(S_perm).value))

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


def test_compute_alignment_gated_cka_cached_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(6, 2, 4)
    y = torch.randn(6, 2, 4)
    x_grams = build_gram_cache(x, normalize=True, kernel="linear")
    y_grams = build_gram_cache(y, normalize=True, kernel="linear")
    res = compute_alignment_gated_cka_cached(
        x_grams,
        y_grams,
        num_permutations=5,
        alpha=0.05,
        seed=123,
        unbiased=False,
    )
    ref = _reference_alignment(
        x_grams,
        y_grams,
        num_permutations=5,
        alpha=0.05,
        seed=123,
        unbiased=False,
    )
    assert res["best_indices"] == ref["best_indices"]
    for key in [
        "raw_score",
        "p_value",
        "tau_alpha",
        "tail_strength",
        "g_score",
        "mu0",
        "sd0",
    ]:
        torch.testing.assert_close(
            torch.tensor(res[key]), torch.tensor(ref[key]), rtol=1e-6, atol=1e-6
        )


def test_compute_alignment_gated_unbiased_cka_cached_matches_reference():
    torch.manual_seed(1)
    x = torch.randn(6, 2, 4)
    y = torch.randn(6, 2, 4)
    x_grams = build_gram_cache(x, normalize=True, kernel="linear")
    y_grams = build_gram_cache(y, normalize=True, kernel="linear")
    res = compute_alignment_gated_cka_cached(
        x_grams,
        y_grams,
        num_permutations=5,
        alpha=0.05,
        seed=321,
        unbiased=True,
    )
    ref = _reference_alignment(
        x_grams,
        y_grams,
        num_permutations=5,
        alpha=0.05,
        seed=321,
        unbiased=True,
    )
    assert res["best_indices"] == ref["best_indices"]
    for key in [
        "raw_score",
        "p_value",
        "tau_alpha",
        "tail_strength",
        "g_score",
        "mu0",
        "sd0",
    ]:
        torch.testing.assert_close(
            torch.tensor(res[key]), torch.tensor(ref[key]), rtol=1e-6, atol=1e-6
        )
