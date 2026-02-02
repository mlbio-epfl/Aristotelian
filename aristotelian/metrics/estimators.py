"""CKA Estimator Variants for Comparison.

This module provides multiple CKA estimators for empirical comparison:

**Gram-matrix based (standard CKA formulation):**

1. **Biased CKA** (cka_biased): Standard CKA with biased HSIC estimator.
   - Has O(1/n) bias that inflates scores, especially in high-d/low-n regime.

2. **Debiased CKA** (cka_debiased): Uses unbiased HSIC from Song et al. (2012).
   - E[estimator] = true_value (no systematic bias).
   - Used by Re-Align (ICLR 2024) paper.

**Moment-based estimators (from arxiv 2502.15104):**

3. **Moment estimators** (cka_estimators_all): Alternative formulation using moments.
   - naive: Biased moment estimator
   - song: Kong-Valiant moment estimator
   - depcols: Estimator correcting for dependent columns

Note: The moment-based estimators compute a differently-scaled quantity than
standard CKA. They are provided for research comparison but may not be
directly comparable to standard CKA values.

All estimators are implemented in PyTorch for GPU acceleration.

References:
- Song et al. (2012): "Feature Selection via Dependence Maximization"
- Kornblith et al. (2019): "Similarity of Neural Network Representations Revisited"
- Re-Align (ICLR 2024): "Correcting Biased CKA Measures in Biological and ANNs"
- arxiv 2502.15104: "CKA estimators"
"""

from __future__ import annotations

import torch

from .utils import EPS

# =============================================================================
# Gram Matrix Centering (Biased and Unbiased)
# =============================================================================


def center_gram_biased(gram: torch.Tensor) -> torch.Tensor:
    """Center a Gram matrix using standard double centering.

    This is equivalent to centering the features before computing the Gram matrix.
    Used by biased CKA.

    Args:
        gram: Symmetric Gram matrix of shape (n, n).

    Returns:
        Centered Gram matrix of shape (n, n).
    """
    means = gram.mean(dim=0)
    means -= means.mean() / 2
    centered = gram - means.unsqueeze(0) - means.unsqueeze(1)
    return centered


def center_gram_unbiased(gram: torch.Tensor) -> torch.Tensor:
    """Center a Gram matrix using unbiased centering for U-statistic.

    From Szekely & Rizzo (2014), "Partial distance correlation with methods
    for dissimilarities". This centering zeros out the diagonal and adjusts
    the means to produce an unbiased HSIC estimator.

    Args:
        gram: Symmetric Gram matrix of shape (n, n).

    Returns:
        Unbiased-centered Gram matrix of shape (n, n).
    """
    gram = gram.clone()
    n = gram.shape[0]
    gram.fill_diagonal_(0)
    means = gram.sum(dim=0, dtype=torch.float64) / (n - 2)
    means -= means.sum() / (2 * (n - 1))
    centered = gram - means.unsqueeze(0) - means.unsqueeze(1)
    centered.fill_diagonal_(0)
    return centered


# =============================================================================
# Individual CKA Estimators
# =============================================================================


def cka_biased(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute biased CKA (standard formulation).

    Uses the standard biased HSIC estimator. This estimator has O(1/n) bias
    that causes inflated scores, particularly problematic in high-d/low-n regime.

    Args:
        X: Feature matrix of shape (n, d_x).
        Y: Feature matrix of shape (n, d_y).

    Returns:
        Biased CKA score in [0, 1].
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")

    # Compute Gram matrices
    gram_x = X @ X.T
    gram_y = Y @ Y.T

    # Center Gram matrices (biased centering)
    gram_x_c = center_gram_biased(gram_x)
    gram_y_c = center_gram_biased(gram_y)

    # CKA = <K_c, L_c>_F / (||K_c||_F * ||L_c||_F)
    # Using scaled HSIC that cancels in CKA computation
    hsic_xy = (gram_x_c * gram_y_c).sum()
    hsic_xx = (gram_x_c * gram_x_c).sum()
    hsic_yy = (gram_y_c * gram_y_c).sum()

    cka = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + EPS)
    return float(cka.item())


def cka_debiased(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute debiased CKA using Song/Kong-Valiant estimator.

    Uses the unbiased HSIC estimator from Song et al. (2012). This removes
    the O(1/n) bias, giving E[estimator] = true_value.

    This is the "debiased CKA" recommended by the Re-Align paper (ICLR 2024).

    Note: Can return slightly negative values due to variance in small samples.

    Args:
        X: Feature matrix of shape (n, d_x).
        Y: Feature matrix of shape (n, d_y).

    Returns:
        Debiased CKA score, typically in [-0.1, 1].
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")

    # Compute Gram matrices
    gram_x = X @ X.T
    gram_y = Y @ Y.T

    # Center Gram matrices (unbiased centering)
    gram_x_c = center_gram_unbiased(gram_x)
    gram_y_c = center_gram_unbiased(gram_y)

    # CKA with unbiased-centered Gram matrices
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
    """Compute moment terms for CKA estimators.

    Ported from arxiv 2502.15104. Computes various index-contracted terms
    needed for the moment-based CKA estimators.

    Args:
        A: Normalized feature matrix of shape (P, Qa).
        B: Normalized feature matrix of shape (P, Qb).
        indep_cols: Whether columns of A and B are independent.

    Returns:
        Dictionary mapping index patterns to (pval, pqval) tuples.
    """
    patterns = ["ijji", "iiii", "ijjj", "iiij", "ijjl", "iijj", "iijl", "ijll", "ijlm"]
    results = {}

    for pattern in patterns:
        i, j, l, m = list(pattern)
        # Build einsum expression for independent case
        pexp = f"{i}a,{j}a,{l}b,{m}b->"
        pval = torch.einsum(pexp, A, A, B, B)

        if indep_cols or A.shape != B.shape:
            pqval = torch.tensor(0.0, device=A.device, dtype=A.dtype)
        else:
            # Additional term for dependent columns
            qexp = f"{i}a,{j}a,{l}a,{m}a->"
            pqval = pval - torch.einsum(qexp, A, A, B, B)

        results[pattern] = (pval, pqval)

    return results


def cka_estimators_all(
    X: torch.Tensor, Y: torch.Tensor, indep_cols: bool = True
) -> tuple[float, float, float]:
    """Compute all three CKA estimators: naive, debiased (Song), and dependent-cols.

    Ported from arxiv 2502.15104. Returns all three estimates efficiently
    in a single pass.

    Args:
        X: Feature matrix of shape (n, d_x).
        Y: Feature matrix of shape (n, d_y).
        indep_cols: Whether columns of X and Y are independent (default True).
            Set to False when comparing representations that may share structure.

    Returns:
        Tuple of (naive_cka, debiased_cka, depcols_cka).
        - naive_cka: Standard biased CKA.
        - debiased_cka: Song/Kong-Valiant unbiased estimator.
        - depcols_cka: Estimator correcting for dependent columns.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")

    P = X.shape[0]
    Qa = X.shape[1]
    Qb = Y.shape[1]

    # Normalize features
    nf_a = (P**0.5) * (Qa**0.5)
    nf_b = (P**0.5) * (Qb**0.5)

    A = X / nf_a
    B = Y / nf_b

    # Compute all moment terms
    terms = _compute_moment_terms(A, B, indep_cols=indep_cols)

    t1, t1d = terms["ijji"]
    t2, t2d = terms["iiii"]
    t3, t3d = terms["ijjj"]
    t4, t4d = terms["iiij"]
    t5, t5d = terms["ijjl"]
    t6, t6d = terms["iijj"]
    t7, t7d = terms["iijl"]
    t8, t8d = terms["ijll"]
    t9, t9d = terms["ijlm"]

    # Factors for debiasing
    f1 = P / (P - 2)
    f2 = 2 / (P - 2)
    f3 = (1 / (P - 1)) * (1 / (P - 2))

    # Naive estimate (biased)
    sums_n = t1 - 2 / P * t5 + (1 / P) ** 2 * t9

    # Kong-Valiant / Song estimate (debiased)
    sums = (P / (P - 3)) * (
        t1 - f1 * t2 + f2 * (t3 + t4 - t5) + f3 * (t6 - t7 - t8 + t9)
    )

    # Dependent-columns estimate
    if indep_cols or X.shape[1] != Y.shape[1]:
        sums_d = sums
    else:
        Q = Qa  # = Qb when shapes match
        sums_d = (
            (P / (P - 3))
            * (Q / (Q - 1))
            * (t1d - f1 * t2d + f2 * (t3d + t4d - t5d) + f3 * (t6d - t7d - t8d + t9d))
        )

    return float(sums_n.item()), float(sums.item()), float(sums_d.item())


# =============================================================================
# Convenience Functions
# =============================================================================


def cka_naive(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute naive (biased) CKA using moment-based formulation.

    Equivalent to cka_biased but using the moment-based computation.

    Args:
        X: Feature matrix of shape (n, d_x).
        Y: Feature matrix of shape (n, d_y).

    Returns:
        Naive CKA score.
    """
    naive, _, _ = cka_estimators_all(X, Y, indep_cols=True)
    return naive


def cka_song(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute Song/Kong-Valiant debiased CKA.

    Equivalent to cka_debiased but using the moment-based computation.

    Args:
        X: Feature matrix of shape (n, d_x).
        Y: Feature matrix of shape (n, d_y).

    Returns:
        Debiased CKA score.
    """
    _, debiased, _ = cka_estimators_all(X, Y, indep_cols=True)
    return debiased


def cka_depcols(X: torch.Tensor, Y: torch.Tensor, indep_cols: bool = False) -> float:
    """Compute dependent-columns CKA estimator.

    From arxiv 2502.15104. Corrects for both bias and non-independent columns.

    Args:
        X: Feature matrix of shape (n, d).
        Y: Feature matrix of shape (n, d). Must have same d as X.
        indep_cols: Whether to assume independent columns (default False).

    Returns:
        Dependent-columns CKA score.
    """
    _, _, depcols = cka_estimators_all(X, Y, indep_cols=indep_cols)
    return depcols


def compare_cka_estimators(
    X: torch.Tensor, Y: torch.Tensor, indep_cols: bool = True
) -> dict[str, float]:
    """Compare all CKA estimators on the same data.

    Convenience function that returns all estimators in a dictionary.

    Args:
        X: Feature matrix of shape (n, d_x).
        Y: Feature matrix of shape (n, d_y).
        indep_cols: Whether columns are independent.

    Returns:
        Dictionary with keys: 'biased', 'debiased', 'depcols'.
    """
    naive, debiased, depcols = cka_estimators_all(X, Y, indep_cols=indep_cols)
    return {
        "biased": naive,
        "debiased": debiased,
        "depcols": depcols,
    }
