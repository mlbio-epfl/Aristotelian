"""CCA (Canonical Correlation Analysis) family metrics.

This module provides CCA-based metrics:
- cca: Mean canonical correlation
- svcca: Singular Vector CCA (with PCA preprocessing)
- pwcca: Projection Weighted CCA

All operate on numpy arrays (CPU) for compatibility with sklearn CCA.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cross_decomposition import CCA

from .base import BaseMetric, MetricConfig, MetricResult
from .extra_base import _sg_metric, _sg_metric_multiq
from .registry import register_metric
from .utils import EPS, center_np, svcca_preprocess, svd_pca, svd_pca_k


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
    Xc = center_np(X)
    Yc = center_np(Y)
    n = Xc.shape[0]

    Cxx = (Xc.T @ Xc) / (n - 1) + reg * np.eye(Xc.shape[1])
    Cyy = (Yc.T @ Yc) / (n - 1) + reg * np.eye(Yc.shape[1])
    Cxy = (Xc.T @ Yc) / (n - 1)

    # Whitening
    Ux, Sx, _ = np.linalg.svd(Cxx)
    Uy, Sy, _ = np.linalg.svd(Cyy)
    Cxx_inv_sqrt = Ux @ np.diag(1.0 / np.sqrt(Sx)) @ Ux.T
    Cyy_inv_sqrt = Uy @ np.diag(1.0 / np.sqrt(Sy)) @ Uy.T

    T = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt
    _, svals, _ = np.linalg.svd(T, full_matrices=False)
    return float(np.mean(svals))


def _pwcca_mean(X: np.ndarray, Y: np.ndarray, reg: float = 1e-6) -> float:
    """Compute projection weighted CCA."""
    Xc = center_np(X)
    Yc = center_np(Y)
    n = Xc.shape[0]

    Cxx = (Xc.T @ Xc) / (n - 1) + reg * np.eye(Xc.shape[1])
    Cyy = (Yc.T @ Yc) / (n - 1) + reg * np.eye(Yc.shape[1])
    Cxy = (Xc.T @ Yc) / (n - 1)

    Ux, Sx, _ = np.linalg.svd(Cxx)
    Uy, Sy, _ = np.linalg.svd(Cyy)
    Cxx_inv_sqrt = Ux @ np.diag(1.0 / np.sqrt(Sx)) @ Ux.T
    Cyy_inv_sqrt = Uy @ np.diag(1.0 / np.sqrt(Sy)) @ Uy.T

    T = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt
    U, svals, _ = np.linalg.svd(T, full_matrices=False)
    A = Cxx_inv_sqrt @ U

    # Projection weights
    weights = np.sum(np.abs(A), axis=0)
    weights = weights / (np.sum(weights) + EPS)
    return float(np.sum(weights * svals))


@register_metric
class CCAMean(BaseMetric):
    """Mean canonical correlation metric.

    Computes CCA and returns mean of canonical correlations.
    Does NOT apply PCA projection by default (matches legacy behavior).
    Use cca_approx for PCA-accelerated CCA.

    Score range: [0, 1] where 1 means perfect correlation.
    """

    name = "cca"
    min_score = 0.0
    max_score = 1.0
    supports_calibration = True

    def _compute_raw(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> float:
        X_np = X.detach().cpu().numpy()
        Y_np = Y.detach().cpu().numpy()

        return _cca_mean(X_np, Y_np)


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


@register_metric
class SVCCASklearn(BaseMetric):
    """SVCCA using sklearn CCA (PRH-compatible implementation).

    This matches the original Platonic Rep implementation using
    sklearn's CCA class.

    Score range: [0, 1] where 1 means perfect correlation.
    """

    name = "svcca_sklearn"
    min_score = 0.0
    max_score = 1.0
    supports_calibration = True

    def _compute_raw(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> float:
        # Preprocess (center + standardize)
        feats_A = svcca_preprocess(X)
        feats_B = svcca_preprocess(Y)

        # Validate and clamp cca_dim to valid range
        max_dim = min(X.shape[0], X.shape[1], Y.shape[1])
        cca_dim = config.cca_dim if config.cca_dim > 0 else 10
        cca_dim = min(cca_dim, max_dim)

        # SVD low-rank approximation
        U1, _, _ = torch.svd_lowrank(feats_A, q=cca_dim)
        U2, _, _ = torch.svd_lowrank(feats_B, q=cca_dim)
        U1 = U1.cpu().detach().numpy()
        U2 = U2.cpu().detach().numpy()

        # Sklearn CCA
        cca = CCA(n_components=cca_dim)
        cca.fit(U1, U2)
        U1_c, U2_c = cca.transform(U1, U2)

        # Add small noise for numerical stability
        U1_c += 1e-10 * np.random.randn(*U1_c.shape)
        U2_c += 1e-10 * np.random.randn(*U2_c.shape)

        # Compute correlations
        correlations = [
            np.corrcoef(U1_c[:, i], U2_c[:, i])[0, 1] for i in range(cca_dim)
        ]
        return float(np.mean(correlations))


@register_metric
class PWCCA(BaseMetric):
    """Projection Weighted CCA metric.

    Weights canonical correlations by projection importance.
    Does NOT apply PCA projection by default (matches legacy behavior).

    Score range: [0, 1] where 1 means perfect weighted correlation.
    """

    name = "pwcca"
    min_score = 0.0
    max_score = 1.0
    supports_calibration = True

    def _compute_raw(
        self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig
    ) -> float:
        X_np = X.detach().cpu().numpy()
        Y_np = Y.detach().cpu().numpy()
        return _pwcca_mean(X_np, Y_np)


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
        X_np = X.detach().cpu().numpy()
        Y_np = Y.detach().cpu().numpy()

        Xc = center_np(X_np)
        Yc = center_np(Y_np)

        num = np.trace(Xc @ Xc.T @ Yc @ Yc.T)
        denom = np.sqrt(np.trace((Xc @ Xc.T) ** 2) * np.trace((Yc @ Yc.T) ** 2)) + EPS
        return float(num / denom)


# =============================================================================
# Numpy-first helpers and gated variants (used by tests/experiments)
# =============================================================================


def cca_project_pca(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    proj_dim: int | None = None,
    var_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return _cca_project_pca(X, Y, proj_dim=proj_dim, var_threshold=var_threshold)


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
    num = np.trace(Xc @ Xc.T @ Yc @ Yc.T)
    denom = np.sqrt(np.trace((Xc @ Xc.T) ** 2) * np.trace((Yc @ Yc.T) ** 2)) + EPS
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
