# calibrated-similarity

[![PyPI](https://img.shields.io/pypi/v/calibrated-similarity)](https://pypi.org/project/calibrated-similarity/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Null-calibrated similarity metrics via permutation testing.

This package implements the calibration algorithms from:

> **Revisiting the Platonic Representation Hypothesis: An Aristotelian View**
> Gröger, Wen, Brbić.

## Installation

```bash
pip install calibrated-similarity
```

## The Problem

Representation similarity metrics (CKA, kNN overlap, RSA, etc.) have **non-zero random baselines** that depend on sample size, dimensionality, and the metric itself:

- A raw score of 0.3 might be highly significant in one setting but within random chance in another
- Comparing scores across studies with different configurations is misleading
- Searching for the best-matching layer pair inflates scores even under the null

## The Solution

**Null calibration** uses permutation testing to estimate what similarity scores look like under the null hypothesis (no true alignment). It returns a calibrated score that is **zero under the null**, enabling valid comparisons.

## Quick Start

```python
import torch
from calibrated_similarity import calibrate, calibrate_layers

# Define any similarity function
def cka(X, Y):
    X, Y = X - X.mean(0), Y - Y.mean(0)
    hsic_xy = (X @ X.T * (Y @ Y.T)).sum()
    hsic_xx = (X @ X.T * (X @ X.T)).sum()
    hsic_yy = (Y @ Y.T * (Y @ Y.T)).sum()
    return hsic_xy / torch.sqrt(hsic_xx * hsic_yy)

# Sample data
X = torch.randn(100, 64)
Y = torch.randn(100, 64)

# Calibrate the similarity
calibrated_score, p_value, threshold = calibrate(X, Y, cka)
print(f"Calibrated: {calibrated_score:.3f}, p={p_value:.3f}")
```

## API Reference

### `calibrate()` — Scalar Calibration

Calibrates a single similarity score against a permutation null distribution:

```python
calibrated_score, p_value, tau = calibrate(
    X, Y, sim_fn,
    K=200,        # Number of permutations
    alpha=0.05,   # Significance level
    smax=1.0,     # Maximum possible similarity
)
```

**Returns:**
- `calibrated_score`: Normalized score in [0, 1], zero under the null at level alpha
- `p_value`: Add-one permutation p-value
- `tau`: Critical threshold at the (1-alpha) quantile

### `calibrate_layers()` — Aggregation-Aware Calibration

For comparing multiple layers between two models, applies the **same permutation across all layers** to properly control for multiple comparisons:

```python
X_layers = [model_A.layer(i, data) for i in range(5)]
Y_layers = [model_B.layer(j, data) for j in range(5)]

calibrated_agg, p_value, tau = calibrate_layers(
    X_layers, Y_layers, sim_fn,
    agg="max",    # "max", "mean", or custom callable
)
```

## Features

- **Any similarity function**: Works with CKA, kNN, RSA, cosine similarity, or custom metrics
- **Valid p-values**: Uses the add-one formula for proper permutation p-values
- **GPU support**: Tensors stay on their original device
- **Reproducible**: Optional `generator` parameter for deterministic results

## Citation

```bibtex
@article{groger2026revisiting,
  title   = {Revisiting the Platonic Representation Hypothesis: An Aristotelian View},
  author  = {Gr{\"o}ger, Fabian and Wen, Shuo and Brbi{\'c}, Maria},
  journal = {arXiv preprint},
  year    = {2026},
}
```

## License

MIT
