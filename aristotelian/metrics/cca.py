"""CCA (Canonical Correlation Analysis) family metrics.

This module provides CCA-based metrics:
- cca: Mean canonical correlation
- svcca: Singular Vector CCA (with PCA preprocessing)
- pwcca: Projection Weighted CCA
- rv_coefficient: RV coefficient (multivariate generalization of R²)
"""

from __future__ import annotations

import numpy as np
import torch
from .base import BaseMetric, MetricConfig, MetricResult
from .extra_base import _sg_metric, _sg_metric_multiq
from .registry import register_metric
from .utils import EPS, center_np, perm_batch_size, svd_pca, svd_pca_k


def _svd_via_eigh(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute (U, singular_values) of T via eigh(T @ T.T).

    More robust than np.linalg.svd (no convergence failures) and
    comparable speed for square matrices.  Returns only the left
    singular vectors U and the singular values (no Vh).
    """
    # T @ T.T is symmetric PSD → eigh is stable
    TTt = T @ T.T
    eigvals, U = np.linalg.eigh(TTt)
    # Clamp numerical noise, then sqrt to get singular values
    svals = np.sqrt(np.maximum(eigvals, 0.0))
    # eigh returns ascending order; reverse for descending (SVD convention)
    return U[:, ::-1].copy(), svals[::-1].copy()


def _mean_canonical_corr(T: torch.Tensor) -> torch.Tensor:
    # symmetric mean over min(dx,dy) canonical corrs; decompose the smaller Gram
    gram = T @ T.mT if T.shape[-2] <= T.shape[-1] else T.mT @ T
    svals = torch.sqrt(torch.clamp(torch.linalg.eigvalsh(gram), min=0.0))
    return svals.mean(dim=-1)


def _pwcca_weighted_mean(
    svals: torch.Tensor,
    U: torch.Tensor,
    cxx_inv_sqrt: torch.Tensor,
    Xc: torch.Tensor,
    r: int,
) -> torch.Tensor:
    # Morcos projection-weighted mean over the top-r canonical components.
    # weight_i = sum over datapoints of |canonical variable i| = ||Xc h_i||_1;
    # this gives rank-deficient (null-space) directions zero weight, unlike the
    # direction-L1 ||h_i|| which Cxx^{-1/2} blows up by ~1/sqrt(reg) when d>=n.
    # r MUST be the rank min(dx,dy,n-1): including the degenerate near-zero
    # eigenspace makes single vs batched eigh disagree (arbitrary eigenvectors),
    # breaking raw/null exchangeability and inflating the calibrated score for d>=n.
    svals = svals[..., :r]
    A = cxx_inv_sqrt @ U[..., :, :r]
    Z = Xc @ A
    w = Z.abs().sum(dim=-2)
    w = w / (w.sum(dim=-1, keepdim=True) + EPS)
    return (w * svals).sum(dim=-1)


def _cca_project_pca(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    proj_dim: int | None = None,
    var_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project features using joint PCA basis."""
    Xc = center_np(X)
    Yc = center_np(Y)

    if proj_dim is None and var_threshold is None:
        raise ValueError("proj_dim or var_threshold must be set")
    if proj_dim is not None and var_threshold is not None:
        raise ValueError("set only one of proj_dim or var_threshold")

    if proj_dim is not None:
        if proj_dim <= 0:
            raise ValueError("proj_dim must be positive")
        if proj_dim > Xc.shape[1] or proj_dim > Yc.shape[1]:
            raise ValueError("proj_dim must be <= number of features")
    if var_threshold is not None and not 0.0 < var_threshold <= 1.0:
        raise ValueError("var_threshold must be in (0, 1]")

    Z = np.concatenate([Xc, Yc], axis=0)
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)

    if var_threshold is not None:
        var = (S**2) / np.sum(S**2)
        proj_dim = int(np.searchsorted(np.cumsum(var), var_threshold) + 1)

    basis = Vt[:proj_dim].T
    return Xc @ basis, Yc @ basis


def _cca_mean(X: np.ndarray, Y: np.ndarray, reg: float = 1e-6) -> float:
    """Compute mean canonical correlation."""
    # L/R factorisation; Cxx^{-1/2} @ Cxy @ Cyy^{-1/2} is float-unstable when d > n
    # (Cxy round-off in low-precision input is amplified by Cxx^{-1/2} ≈ 1/sqrt(reg)).
    Xc = center_np(X)
    Yc = center_np(Y)
    n = Xc.shape[0]

    Cxx = (Xc.T @ Xc) / (n - 1) + reg * np.eye(Xc.shape[1])
    Cyy = (Yc.T @ Yc) / (n - 1) + reg * np.eye(Yc.shape[1])

    # Whitening via eigh (faster & more stable for symmetric PD matrices)
    Sx, Ux = np.linalg.eigh(Cxx)
    Sy, Uy = np.linalg.eigh(Cyy)
    Sx = np.maximum(Sx, reg)
    Sy = np.maximum(Sy, reg)
    Cxx_inv_sqrt = Ux @ np.diag(1.0 / np.sqrt(Sx)) @ Ux.T
    Cyy_inv_sqrt = Uy @ np.diag(1.0 / np.sqrt(Sy)) @ Uy.T

    L = Cxx_inv_sqrt @ Xc.T
    R = Yc @ Cyy_inv_sqrt
    T = (L @ R) / (n - 1)
    # symmetric: mean over min(dx,dy) canonical corrs (decompose smaller Gram)
    gram = T @ T.T if T.shape[0] <= T.shape[1] else T.T @ T
    svals = np.sqrt(np.maximum(np.linalg.eigvalsh(gram), 0.0))
    return float(np.mean(svals))


def _pwcca_mean(X: np.ndarray, Y: np.ndarray, reg: float = 1e-6) -> float:
    """Compute projection weighted CCA (Morcos sample-L1, rank-truncated)."""
    # See _cca_mean for the L/R rationale.
    Xc = center_np(X)
    Yc = center_np(Y)
    n = Xc.shape[0]

    Cxx = (Xc.T @ Xc) / (n - 1) + reg * np.eye(Xc.shape[1])
    Cyy = (Yc.T @ Yc) / (n - 1) + reg * np.eye(Yc.shape[1])

    Sx, Ux = np.linalg.eigh(Cxx)
    Sy, Uy = np.linalg.eigh(Cyy)
    Sx = np.maximum(Sx, reg)
    Sy = np.maximum(Sy, reg)
    Cxx_inv_sqrt = Ux @ np.diag(1.0 / np.sqrt(Sx)) @ Ux.T
    Cyy_inv_sqrt = Uy @ np.diag(1.0 / np.sqrt(Sy)) @ Uy.T

    L = Cxx_inv_sqrt @ Xc.T
    R = Yc @ Cyy_inv_sqrt
    T = (L @ R) / (n - 1)
    U, svals = _svd_via_eigh(T)

    # Truncate to the canonical rank min(dx,dy,n-1), then Morcos projection
    # weights: sample-L1 of each canonical variable Z = Xc @ (Cxx^{-1/2} @ U)
    # over datapoints. The rank truncation drops the degenerate near-zero
    # eigenspace that exists when d >= n.
    r = min(Xc.shape[1], Yc.shape[1], Xc.shape[0] - 1)
    U = U[:, :r]
    svals = svals[:r]
    A = Cxx_inv_sqrt @ U
    Z = Xc @ A
    weights = np.sum(np.abs(Z), axis=0)
    weights = weights / (np.sum(weights) + EPS)
    return float(np.sum(weights * svals))


@register_metric
class CCAMean(BaseMetric):
    """Mean canonical correlation metric.

    Computes CCA and returns mean of canonical correlations.
    Operates on full feature dimensions to preserve O(d/n) scaling.

    Score range: [0, 1] where 1 means perfect correlation.
    """

    name = "cca"
    min_score = 0.0
    max_score = 1.0
    supports_calibration = True

    def _compute_raw(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> float:
        # L/R factorisation; the algebraically-equivalent Cxx^{-1/2} @ Cxy @ Cyy^{-1/2}
        # is float32-unstable when d > n.
        device = config.device
        reg = 1e-6
        Xc = (X - X.mean(0, keepdim=True)).to(device=device, dtype=torch.float32)
        Yc = (Y - Y.mean(0, keepdim=True)).to(device=device, dtype=torch.float32)
        n, dx, dy = Xc.shape[0], Xc.shape[1], Yc.shape[1]
        eye_x = torch.eye(dx, device=device, dtype=torch.float32)
        eye_y = torch.eye(dy, device=device, dtype=torch.float32)
        Cxx = (Xc.T @ Xc) / (n - 1) + reg * eye_x
        Cyy = (Yc.T @ Yc) / (n - 1) + reg * eye_y
        Sx, Ux = torch.linalg.eigh(Cxx)
        Sy, Uy = torch.linalg.eigh(Cyy)
        Sx = torch.clamp(Sx, min=reg)
        Sy = torch.clamp(Sy, min=reg)
        Cxx_inv_sqrt = Ux @ torch.diag(1.0 / torch.sqrt(Sx)) @ Ux.T
        Cyy_inv_sqrt = Uy @ torch.diag(1.0 / torch.sqrt(Sy)) @ Uy.T
        L = Cxx_inv_sqrt @ Xc.T
        R = Yc @ Cyy_inv_sqrt
        T = (L @ R) / (n - 1)
        try:
            score = _mean_canonical_corr(T)
        except torch._C._LinAlgError:
            score = torch.linalg.svdvals(T).mean()
        return float(score.item())

    def _compute_null_distribution(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> list[float]:
        """Batched null distribution: vectorised eigvalsh over all permutations.

        Cxx and Cyy are invariant under row permutation (same set of rows),
        so we precompute both whitening matrices and L = Cxx^{-1/2} @ Xc.T,
        R = Yc @ Cyy^{-1/2} once, then batch-compute
        T[b] = L @ R[perm_b] / (n-1) and eigvalsh(T @ T.T) for all perms
        in one call per chunk.
        """
        n = X.shape[0]
        device = config.device
        num_perms = config.num_permutations
        reg = 1e-6

        if config.perms is not None:
            perms = config.perms.to(device)
            if perms.dim() != 2 or perms.size(1) != n:
                raise ValueError("perms must have shape (B, n)")
        else:
            perms = torch.stack(
                [torch.randperm(n, device=device) for _ in range(num_perms)]
            )

        Xc = X.to(device=device, dtype=torch.float32)
        Yc = Y.to(device=device, dtype=torch.float32)
        Xc = Xc - Xc.mean(dim=0, keepdim=True)
        Yc = Yc - Yc.mean(dim=0, keepdim=True)
        dx, dy = Xc.shape[1], Yc.shape[1]

        # Precompute whitening (both are invariant under row permutation)
        eye_dx = torch.eye(dx, device=device, dtype=torch.float32)
        eye_dy = torch.eye(dy, device=device, dtype=torch.float32)
        Cxx = (Xc.T @ Xc) / (n - 1) + reg * eye_dx
        Cyy = (Yc.T @ Yc) / (n - 1) + reg * eye_dy

        Sx, Ux = torch.linalg.eigh(Cxx)
        Sy, Uy = torch.linalg.eigh(Cyy)
        Sx = torch.clamp(Sx, min=reg)
        Sy = torch.clamp(Sy, min=reg)
        Cxx_inv_sqrt = Ux @ torch.diag(1.0 / torch.sqrt(Sx)) @ Ux.T
        Cyy_inv_sqrt = Uy @ torch.diag(1.0 / torch.sqrt(Sy)) @ Uy.T

        L = Cxx_inv_sqrt @ Xc.T  # (dx, n)
        R = Yc @ Cyy_inv_sqrt  # (n, dy)

        # Batched: process permutations in memory-aware chunks
        chunk = perm_batch_size(num_perms, max(dx, dy), torch.float32)
        null_scores: list[float] = []
        for start in range(0, num_perms, chunk):
            end = min(start + chunk, num_perms)
            R_perm = R[perms[start:end]]  # (B, n, dy)
            # T_batch[b] = L @ R_perm[b] / (n-1)
            T_batch = torch.einsum("xn,bny->bxy", L, R_perm) / (n - 1)
            try:
                scores = _mean_canonical_corr(T_batch)  # (B,)
            except torch._C._LinAlgError:
                scores = torch.linalg.svdvals(T_batch).mean(dim=-1)
            null_scores.extend(scores.cpu().tolist())

        return null_scores


@register_metric
class SVCCA(BaseMetric):
    """Singular Vector CCA metric.

    Applies PCA/SVD preprocessing before CCA for efficiency
    and robustness to noise.

    Score range: [0, 1] where 1 means perfect correlation.
    """

    name = "svcca"
    min_score = 0.0
    max_score = 1.0
    supports_calibration = True

    def _compute_raw(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> float:
        X_np = X.detach().cpu().numpy()
        Y_np = Y.detach().cpu().numpy()

        # Validate and clamp cca_dim to valid range
        max_dim = min(X.shape[0], X.shape[1], Y.shape[1])
        cca_dim = config.cca_dim if config.cca_dim > 0 else 10
        cca_dim = min(cca_dim, max_dim)

        # SVD-based PCA with fixed k (faster)
        Xp = svd_pca_k(center_np(X_np), k=cca_dim)
        Yp = svd_pca_k(center_np(Y_np), k=cca_dim)

        return _cca_mean(Xp, Yp)

    def _compute_null_distribution(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> list[float]:
        """Batched null distribution for SVCCA.

        Key insight: PCA of row-permuted Y equals row-permuted PCA of Y,
        because centering and covariance are row-order invariant.
        So we precompute PCA projections once, then run CCA's batched
        eigvalsh approach on the tiny (cca_dim × cca_dim) matrices.
        """
        n = X.shape[0]
        device = config.device
        num_perms = config.num_permutations
        reg = 1e-6

        # Validate and clamp cca_dim
        max_dim = min(X.shape[0], X.shape[1], Y.shape[1])
        cca_dim = config.cca_dim if config.cca_dim > 0 else 10
        cca_dim = min(cca_dim, max_dim)

        # Precompute PCA projections (invariant under row permutation)
        X_np = X.detach().cpu().numpy()
        Y_np = Y.detach().cpu().numpy()
        Xp = torch.from_numpy(svd_pca_k(center_np(X_np), k=cca_dim)).to(
            device=device, dtype=torch.float32
        )
        Yp = torch.from_numpy(svd_pca_k(center_np(Y_np), k=cca_dim)).to(
            device=device, dtype=torch.float32
        )

        if config.perms is not None:
            perms = config.perms.to(device)
            if perms.dim() != 2 or perms.size(1) != n:
                raise ValueError("perms must have shape (B, n)")
        else:
            perms = torch.stack(
                [torch.randperm(n, device=device) for _ in range(num_perms)]
            )

        # Center PCA projections (they're already roughly centered, but be exact)
        Xp = Xp - Xp.mean(dim=0, keepdim=True)
        Yp = Yp - Yp.mean(dim=0, keepdim=True)
        dx, dy = Xp.shape[1], Yp.shape[1]

        # Precompute whitening (invariant under row permutation of Yp)
        eye_dx = torch.eye(dx, device=device, dtype=torch.float32)
        eye_dy = torch.eye(dy, device=device, dtype=torch.float32)
        Cxx = (Xp.T @ Xp) / (n - 1) + reg * eye_dx
        Cyy = (Yp.T @ Yp) / (n - 1) + reg * eye_dy

        Sx, Ux = torch.linalg.eigh(Cxx)
        Sy, Uy = torch.linalg.eigh(Cyy)
        Sx = torch.clamp(Sx, min=reg)
        Sy = torch.clamp(Sy, min=reg)
        Cxx_inv_sqrt = Ux @ torch.diag(1.0 / torch.sqrt(Sx)) @ Ux.T
        Cyy_inv_sqrt = Uy @ torch.diag(1.0 / torch.sqrt(Sy)) @ Uy.T

        L = Cxx_inv_sqrt @ Xp.T  # (cca_dim, n)
        R = Yp @ Cyy_inv_sqrt  # (n, cca_dim)

        # Batched eigvalsh on tiny (cca_dim × cca_dim) matrices
        chunk = perm_batch_size(num_perms, max(dx, dy), torch.float32)
        null_scores: list[float] = []
        for start in range(0, num_perms, chunk):
            end = min(start + chunk, num_perms)
            R_perm = R[perms[start:end]]  # (B, n, cca_dim)
            T_batch = torch.einsum("xn,bny->bxy", L, R_perm) / (n - 1)
            TTt = T_batch @ T_batch.transpose(-1, -2)  # (B, cca_dim, cca_dim)
            try:
                eigvals = torch.linalg.eigvalsh(TTt)
                svals = torch.sqrt(torch.clamp(eigvals, min=0.0))
            except torch._C._LinAlgError:
                svals = torch.linalg.svdvals(T_batch)
            null_scores.extend(svals.mean(dim=-1).cpu().tolist())

        return null_scores


@register_metric
class PWCCA(BaseMetric):
    """Projection Weighted CCA metric.

    Weights canonical correlations by projection importance.
    Operates on full feature dimensions to preserve O(d/n) scaling.

    Score range: [0, 1] where 1 means perfect weighted correlation.
    """

    name = "pwcca"
    min_score = 0.0
    max_score = 1.0
    supports_calibration = True

    def _compute_raw(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> float:
        # L/R factorisation; the algebraically-equivalent Cxx^{-1/2} @ Cxy @ Cyy^{-1/2}
        # is float32-unstable when d > n.
        device = config.device
        reg = 1e-6
        Xc = (X - X.mean(0, keepdim=True)).to(device=device, dtype=torch.float32)
        Yc = (Y - Y.mean(0, keepdim=True)).to(device=device, dtype=torch.float32)
        n, dx, dy = Xc.shape[0], Xc.shape[1], Yc.shape[1]
        eye_x = torch.eye(dx, device=device, dtype=torch.float32)
        eye_y = torch.eye(dy, device=device, dtype=torch.float32)
        Cxx = (Xc.T @ Xc) / (n - 1) + reg * eye_x
        Cyy = (Yc.T @ Yc) / (n - 1) + reg * eye_y
        Sx, Ux = torch.linalg.eigh(Cxx)
        Sy, Uy = torch.linalg.eigh(Cyy)
        Sx = torch.clamp(Sx, min=reg)
        Sy = torch.clamp(Sy, min=reg)
        Cxx_inv_sqrt = Ux @ torch.diag(1.0 / torch.sqrt(Sx)) @ Ux.T
        Cyy_inv_sqrt = Uy @ torch.diag(1.0 / torch.sqrt(Sy)) @ Uy.T
        L = Cxx_inv_sqrt @ Xc.T
        R = Yc @ Cyy_inv_sqrt
        T = (L @ R) / (n - 1)
        eigvals, U = torch.linalg.eigh(T @ T.T)
        # eigh ascending → reverse for SVD descending convention
        svals = torch.sqrt(torch.clamp(eigvals.flip(-1), min=0.0))
        U = U.flip(-1)
        score = _pwcca_weighted_mean(svals, U, Cxx_inv_sqrt, Xc, min(dx, dy, n - 1))
        return float(score.item())

    def _compute_null_distribution(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> list[float]:
        """Batched null distribution for PWCCA with vectorised eigh."""
        n = X.shape[0]
        device = config.device
        num_perms = config.num_permutations
        reg = 1e-6

        if config.perms is not None:
            perms = config.perms.to(device)
            if perms.dim() != 2 or perms.size(1) != n:
                raise ValueError("perms must have shape (B, n)")
        else:
            perms = torch.stack(
                [torch.randperm(n, device=device) for _ in range(num_perms)]
            )

        Xc = X.to(device=device, dtype=torch.float32)
        Yc = Y.to(device=device, dtype=torch.float32)
        Xc = Xc - Xc.mean(dim=0, keepdim=True)
        Yc = Yc - Yc.mean(dim=0, keepdim=True)
        dx, dy = Xc.shape[1], Yc.shape[1]

        eye_dx = torch.eye(dx, device=device, dtype=torch.float32)
        eye_dy = torch.eye(dy, device=device, dtype=torch.float32)
        Cxx = (Xc.T @ Xc) / (n - 1) + reg * eye_dx
        Cyy = (Yc.T @ Yc) / (n - 1) + reg * eye_dy

        Sx, Ux = torch.linalg.eigh(Cxx)
        Sy, Uy = torch.linalg.eigh(Cyy)
        Sx = torch.clamp(Sx, min=reg)
        Sy = torch.clamp(Sy, min=reg)
        Cxx_inv_sqrt = Ux @ torch.diag(1.0 / torch.sqrt(Sx)) @ Ux.T
        Cyy_inv_sqrt = Uy @ torch.diag(1.0 / torch.sqrt(Sy)) @ Uy.T

        L = Cxx_inv_sqrt @ Xc.T  # (dx, n)
        R = Yc @ Cyy_inv_sqrt  # (n, dy)

        chunk = perm_batch_size(num_perms, max(dx, dy), torch.float32)
        null_scores: list[float] = []
        for start in range(0, num_perms, chunk):
            end = min(start + chunk, num_perms)
            R_perm = R[perms[start:end]]  # (B, n, dy)
            T_batch = torch.einsum("xn,bny->bxy", L, R_perm) / (n - 1)
            TTt = T_batch @ T_batch.transpose(-1, -2)  # (B, dx, dx)
            eigvals, U_batch = torch.linalg.eigh(TTt)  # (B, dx), (B, dx, dx)
            # eigh returns ascending order; reverse for descending (SVD convention)
            svals = torch.sqrt(torch.clamp(eigvals.flip(-1), min=0.0))
            U_batch = U_batch.flip(-1)
            scores = _pwcca_weighted_mean(
                svals, U_batch, Cxx_inv_sqrt, Xc, min(dx, dy, n - 1)
            )
            null_scores.extend(scores.cpu().tolist())

        return null_scores


@register_metric
class RVCoefficient(BaseMetric):
    """RV coefficient metric.

    Multivariate generalization of the squared Pearson correlation.

    Score range: [0, 1] where 1 means perfect agreement.
    """

    name = "rv_coefficient"
    min_score = 0.0
    max_score = 1.0
    supports_calibration = True

    def _compute_raw(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> float:
        # Centered RV (== linear CKA, a known identity; documented in the appendix).
        Xc = X - X.mean(dim=0, keepdim=True)
        Yc = Y - Y.mean(dim=0, keepdim=True)
        Gx = Xc @ Xc.T
        Gy = Yc @ Yc.T
        num = torch.sum(Gx * Gy)  # = trace(Gx @ Gy)
        denom = torch.sqrt((Gx * Gx).sum() * (Gy * Gy).sum()) + EPS
        return float((num / denom).item())

    def _compute_null_distribution(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> list[float]:
        """Optimized null: Gram matrices precomputed, O(n²) per permutation.

        trace(Gx @ Gy_perm) = sum(Gx * Gy[perm][:,perm]) since row
        permutation of Y just permutes rows/cols of the Gram matrix.
        The denominator is invariant under permutation.
        """
        n = X.shape[0]
        device = config.device
        num_perms = config.num_permutations

        if config.perms is not None:
            perms = config.perms.to(device)
            if perms.dim() != 2 or perms.size(1) != n:
                raise ValueError("perms must have shape (B, n)")
        else:
            perms = torch.stack(
                [torch.randperm(n, device=device) for _ in range(num_perms)]
            )

        Xc = X - X.mean(dim=0, keepdim=True)
        Yc = Y - Y.mean(dim=0, keepdim=True)
        Gx = Xc @ Xc.T  # (n, n) — constant
        Gy = Yc @ Yc.T  # (n, n) — constant
        # Denominator is invariant: Frobenius norm is preserved under row/col permutation
        denom = torch.sqrt((Gx * Gx).sum() * (Gy * Gy).sum()) + EPS

        # Batched: permute rows+cols of Gy in chunks
        max_chunk = max(1, (512 * 1024 * 1024) // (n * n * Gx.element_size()))
        null_scores: list[float] = []
        for start in range(0, num_perms, max_chunk):
            end = min(start + max_chunk, num_perms)
            batch_perms = perms[start:end]  # (B, n)
            temp = Gy[batch_perms]  # (B, n, n) — row permutation
            Gy_perm = torch.gather(
                temp, 2, batch_perms.unsqueeze(1).expand(-1, n, -1)
            )  # (B, n, n)
            nums = (Gx.unsqueeze(0) * Gy_perm).sum(dim=(1, 2))  # (B,)
            null_scores.extend((nums / denom).cpu().tolist())

        return null_scores


# =============================================================================
# Numpy-first helpers and gated variants (used by tests/experiments)
# =============================================================================


def cca_mean(X: np.ndarray, Y: np.ndarray, reg: float = 1e-6) -> float:
    return _cca_mean(X, Y, reg=reg)


def cca_mean_approx(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    proj_dim: int | None = 128,
    var_threshold: float | None = None,
    reg: float = 1e-6,
) -> float:
    Xp, Yp = _cca_project_pca(X, Y, proj_dim=proj_dim, var_threshold=var_threshold)
    return _cca_mean(Xp, Yp, reg=reg)


def svcca_mean(X: np.ndarray, Y: np.ndarray, var_threshold: float = 0.99) -> float:
    Xp = svd_pca(center_np(X), var_threshold=var_threshold)
    Yp = svd_pca(center_np(Y), var_threshold=var_threshold)
    return _cca_mean(Xp, Yp)


def svcca_mean_k(X: np.ndarray, Y: np.ndarray, k: int = 10) -> float:
    """SVCCA with fixed k components (faster, similar to PRH implementation)."""
    Xp = svd_pca_k(center_np(X), k=k)
    Yp = svd_pca_k(center_np(Y), k=k)
    return _cca_mean(Xp, Yp)


def pwcca_mean(X: np.ndarray, Y: np.ndarray, reg: float = 1e-6) -> float:
    return _pwcca_mean(X, Y, reg=reg)


def pwcca_mean_approx(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    proj_dim: int = 32,
    reg: float = 1e-6,
) -> float:
    """PWCCA with PCA projection for speedup."""
    Xp, Yp = _cca_project_pca(X, Y, proj_dim=proj_dim)
    return _pwcca_mean(Xp, Yp, reg=reg)


def rv_coefficient(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = center_np(X)
    Yc = center_np(Y)
    Gx = Xc @ Xc.T
    Gy = Yc @ Yc.T
    num = np.trace(Gx @ Gy)
    denom = np.sqrt(np.sum(Gx * Gx) * np.sum(Gy * Gy)) + EPS
    return float(num / denom)


def sg_rv_coefficient(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    num_permutations: int = 200,
    quantile: float = 0.95,
    perms: np.ndarray | None = None,
) -> MetricResult:
    return _sg_metric(
        X,
        Y,
        metric_fn=rv_coefficient,
        num_permutations=num_permutations,
        quantile=quantile,
        perms=perms,
        min_score=0.0,
        max_score=1.0,
    )


def sg_cca_mean(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    num_permutations: int = 200,
    quantile: float = 0.95,
    perms: np.ndarray | None = None,
    reg: float = 1e-6,
    proj_dim: int | None = None,
) -> MetricResult:
    if proj_dim is not None:

        def metric_fn(a, b):
            return cca_mean_approx(a, b, proj_dim=proj_dim, reg=reg)

    else:

        def metric_fn(a, b):
            return cca_mean(a, b, reg=reg)

    return _sg_metric(
        X,
        Y,
        metric_fn=metric_fn,
        num_permutations=num_permutations,
        quantile=quantile,
        perms=perms,
        min_score=0.0,
        max_score=1.0,
    )


def sg_cca_multiq(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    num_permutations: int = 200,
    quantiles: list[float],
    perms: np.ndarray | None = None,
    reg: float = 1e-6,
    proj_dim: int | None = None,
) -> dict:
    if proj_dim is not None:

        def metric_fn(a, b):
            return cca_mean_approx(a, b, proj_dim=proj_dim, reg=reg)

    else:

        def metric_fn(a, b):
            return cca_mean(a, b, reg=reg)

    return _sg_metric_multiq(
        X,
        Y,
        metric_fn=metric_fn,
        num_permutations=num_permutations,
        quantiles=quantiles,
        perms=perms,
        min_score=0.0,
        max_score=1.0,
    )


def sg_svcca_mean(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    num_permutations: int = 200,
    quantile: float = 0.95,
    perms: np.ndarray | None = None,
    var_threshold: float = 0.99,
    k: int | None = None,
) -> MetricResult:
    if k is not None:

        def metric_fn(a, b):
            return svcca_mean_k(a, b, k=k)

    else:

        def metric_fn(a, b):
            return svcca_mean(a, b, var_threshold=var_threshold)

    return _sg_metric(
        X,
        Y,
        metric_fn=metric_fn,
        num_permutations=num_permutations,
        quantile=quantile,
        perms=perms,
        min_score=0.0,
        max_score=1.0,
    )


def sg_svcca_multiq(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    num_permutations: int = 200,
    quantiles: list[float],
    perms: np.ndarray | None = None,
    var_threshold: float = 0.99,
    k: int | None = None,
) -> dict:
    if k is not None:

        def metric_fn(a, b):
            return svcca_mean_k(a, b, k=k)

    else:

        def metric_fn(a, b):
            return svcca_mean(a, b, var_threshold=var_threshold)

    return _sg_metric_multiq(
        X,
        Y,
        metric_fn=metric_fn,
        num_permutations=num_permutations,
        quantiles=quantiles,
        perms=perms,
        min_score=0.0,
        max_score=1.0,
    )


def sg_pwcca_mean(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    num_permutations: int = 200,
    quantile: float = 0.95,
    perms: np.ndarray | None = None,
    proj_dim: int | None = None,
    reg: float = 1e-6,
) -> MetricResult:
    if proj_dim is not None:

        def metric_fn(a, b):
            return pwcca_mean_approx(a, b, proj_dim=proj_dim, reg=reg)

    else:

        def metric_fn(a, b):
            return pwcca_mean(a, b, reg=reg)

    return _sg_metric(
        X,
        Y,
        metric_fn=metric_fn,
        num_permutations=num_permutations,
        quantile=quantile,
        perms=perms,
        min_score=0.0,
        max_score=1.0,
    )


def sg_pwcca_multiq(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    num_permutations: int = 200,
    quantiles: list[float],
    perms: np.ndarray | None = None,
    proj_dim: int | None = None,
    reg: float = 1e-6,
) -> dict:
    if proj_dim is not None:

        def metric_fn(a, b):
            return pwcca_mean_approx(a, b, proj_dim=proj_dim, reg=reg)

    else:

        def metric_fn(a, b):
            return pwcca_mean(a, b, reg=reg)

    return _sg_metric_multiq(
        X,
        Y,
        metric_fn=metric_fn,
        num_permutations=num_permutations,
        quantiles=quantiles,
        perms=perms,
        min_score=0.0,
        max_score=1.0,
    )
