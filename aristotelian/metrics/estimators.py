"""CKA Estimator Variants for Comparison.

Provides multiple CKA estimators for the bias comparison experiment:

1. **Biased CKA** (cka_biased): Standard CKA with biased HSIC.
2. **Debiased CKA** (cka_debiased): Unbiased HSIC from Song et al. (2012).
3. **Dependent-columns CKA** (cka_depcols): From arxiv 2502.15104.
"""

from __future__ import annotations

import torch

from .utils import EPS


def center_gram_biased(gram: torch.Tensor) -> torch.Tensor:
    """Center a Gram matrix using standard double centering."""
    means = gram.mean(dim=0)
    means -= means.mean() / 2
    centered = gram - means.unsqueeze(0) - means.unsqueeze(1)
    return centered


def center_gram_unbiased(gram: torch.Tensor) -> torch.Tensor:
    """Center a Gram matrix using unbiased centering for U-statistic."""
    gram = gram.clone()
    n = gram.shape[0]
    gram.fill_diagonal_(0)
    means = gram.sum(dim=0, dtype=torch.float64) / (n - 2)
    means -= means.sum() / (2 * (n - 1))
    centered = gram - means.unsqueeze(0) - means.unsqueeze(1)
    centered.fill_diagonal_(0)
    return centered


def cka_biased(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute biased CKA (standard formulation)."""
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    gram_x = X @ X.T
    gram_y = Y @ Y.T
    gram_x_c = center_gram_biased(gram_x)
    gram_y_c = center_gram_biased(gram_y)
    hsic_xy = (gram_x_c * gram_y_c).sum()
    hsic_xx = (gram_x_c * gram_x_c).sum()
    hsic_yy = (gram_y_c * gram_y_c).sum()
    cka = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + EPS)
    return float(cka.item())


def cka_debiased(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute debiased CKA using Song/Kong-Valiant estimator."""
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    gram_x = X @ X.T
    gram_y = Y @ Y.T
    gram_x_c = center_gram_unbiased(gram_x)
    gram_y_c = center_gram_unbiased(gram_y)
    hsic_xy = (gram_x_c * gram_y_c).sum()
    hsic_xx = (gram_x_c * gram_x_c).sum()
    hsic_yy = (gram_y_c * gram_y_c).sum()
    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom < EPS:
        return 0.0
    cka = hsic_xy / (denom + EPS)
    return float(cka.item())


def _compute_moment_terms(
    A: torch.Tensor, B: torch.Tensor, indep_cols: bool = True
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Compute moment terms for CKA estimators."""
    patterns = ["ijji", "iiii", "ijjj", "iiij", "ijjl", "iijj", "iijl", "ijll", "ijlm"]
    results = {}
    for pattern in patterns:
        i, j, l, m = list(pattern)
        pexp = f"{i}a,{j}a,{l}b,{m}b->"
        pval = torch.einsum(pexp, A, A, B, B)
        if indep_cols or A.shape != B.shape:
            pqval = torch.tensor(0.0, device=A.device, dtype=A.dtype)
        else:
            qexp = f"{i}a,{j}a,{l}a,{m}a->"
            pqval = pval - torch.einsum(qexp, A, A, B, B)
        results[pattern] = (pval, pqval)
    return results


def _hsic_estimates(
    terms: dict[str, tuple[torch.Tensor, torch.Tensor]],
    P: int,
    Q: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract naive, Song, and dep-cols HSIC estimates from moment terms.

    If Q is None or Q <= 1, the dep-cols estimate falls back to Song.
    """
    t1, t1d = terms["ijji"]
    t2, t2d = terms["iiii"]
    t3, t3d = terms["ijjj"]
    t4, t4d = terms["iiij"]
    t5, t5d = terms["ijjl"]
    t6, t6d = terms["iijj"]
    t7, t7d = terms["iijl"]
    t8, t8d = terms["ijll"]
    t9, t9d = terms["ijlm"]

    f1 = P / (P - 2)
    f2 = 2 / (P - 2)
    f3 = (1 / (P - 1)) * (1 / (P - 2))

    naive = t1 - 2 / P * t5 + (1 / P) ** 2 * t9
    song = (P / (P - 3)) * (
        t1 - f1 * t2 + f2 * (t3 + t4 - t5) + f3 * (t6 - t7 - t8 + t9)
    )
    if Q is not None and Q > 1:
        depcols = (
            (P / (P - 3))
            * (Q / (Q - 1))
            * (t1d - f1 * t2d + f2 * (t3d + t4d - t5d) + f3 * (t6d - t7d - t8d + t9d))
        )
    else:
        depcols = song
    return naive, song, depcols


def cka_estimators_all(
    X: torch.Tensor, Y: torch.Tensor, indep_cols: bool = True
) -> tuple[float, float, float]:
    """Compute all three CKA estimators: naive, debiased (Song), and dependent-cols.

    CKA = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))

    The cross-HSIC numerator always uses Song — columns of X and Y are
    independently sampled, so the dep-cols correction is not needed
    (Chun et al. 2025). The dep-cols correction applies only to the
    self-HSICs in the denominator, where columns are inherently dependent.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")

    P = X.shape[0]
    Qa = X.shape[1]
    Qb = Y.shape[1]

    # Cast to float64: dep-cols correction has heavy cancellation in pval−qval
    # that requires double precision.
    A = X.to(dtype=torch.float64) / ((P**0.5) * (Qa**0.5))
    B = Y.to(dtype=torch.float64) / ((P**0.5) * (Qb**0.5))

    # Cross-HSIC (numerator): always Song — X and Y columns are independent.
    terms_cross = _compute_moment_terms(A, B, indep_cols=True)
    naive_cross, song_cross, _ = _hsic_estimates(terms_cross, P)

    # Self-HSICs (denominator): dep-cols correction applies here.
    terms_xx = _compute_moment_terms(A, A, indep_cols=indep_cols)
    terms_yy = _compute_moment_terms(B, B, indep_cols=indep_cols)
    naive_xx, song_xx, depcols_xx = _hsic_estimates(
        terms_xx, P, Q=Qa if not indep_cols else None,
    )
    naive_yy, song_yy, depcols_yy = _hsic_estimates(
        terms_yy, P, Q=Qb if not indep_cols else None,
    )

    def _cka(num: torch.Tensor, dxx: torch.Tensor, dyy: torch.Tensor) -> float:
        product = dxx * dyy
        if product <= 0:
            return 0.0
        denom = torch.sqrt(product) + EPS
        return float((num / denom).item())

    return (
        _cka(naive_cross, naive_xx, naive_yy),
        _cka(song_cross, song_xx, song_yy),
        _cka(song_cross, depcols_xx, depcols_yy),
    )


def cka_naive(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute naive (biased) CKA using moment-based formulation."""
    naive, _, _ = cka_estimators_all(X, Y, indep_cols=True)
    return naive


def cka_song(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute Song/Kong-Valiant debiased CKA."""
    _, debiased, _ = cka_estimators_all(X, Y, indep_cols=True)
    return debiased


def cka_depcols(X: torch.Tensor, Y: torch.Tensor, indep_cols: bool = False) -> float:
    """Compute dependent-columns CKA estimator."""
    _, _, depcols = cka_estimators_all(X, Y, indep_cols=indep_cols)
    return depcols


def compare_cka_estimators(
    X: torch.Tensor, Y: torch.Tensor, indep_cols: bool = True
) -> dict[str, float]:
    """Compare all CKA estimators on the same data."""
    naive, debiased, depcols = cka_estimators_all(X, Y, indep_cols=indep_cols)
    return {
        "biased": naive,
        "debiased": debiased,
        "depcols": depcols,
    }
