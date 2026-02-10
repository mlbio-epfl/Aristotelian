"""Null-calibrated similarity metrics via permutation testing.

This package provides algorithms for null-calibrating similarity metrics,
enabling statistically valid comparisons between representations.

Example usage:

    import torch
    from calibrated_similarity import calibrate, calibrate_layers

    # Define a similarity function
    def cosine_sim(X, Y):
        X_norm = X / X.norm(dim=1, keepdim=True)
        Y_norm = Y / Y.norm(dim=1, keepdim=True)
        return (X_norm * Y_norm).sum(dim=1).mean()

    # Scalar calibration (Algorithm 1)
    X = torch.randn(100, 64)
    Y = torch.randn(100, 64)
    calibrated_score, p_value, threshold = calibrate(X, Y, cosine_sim)

    # Layer-wise calibration (Algorithm 2)
    X_layers = [torch.randn(100, 64) for _ in range(5)]
    Y_layers = [torch.randn(100, 64) for _ in range(5)]
    calibrated_agg, p_value, threshold = calibrate_layers(
        X_layers, Y_layers, cosine_sim, agg="max"
    )
"""

__version__ = "0.1.1"

from .calibration import calibrate, calibrate_layers

__all__ = ["calibrate", "calibrate_layers", "__version__"]
