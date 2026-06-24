# Revisiting the Platonic Representation Hypothesis: An Aristotelian View

[![ICML 2026](https://img.shields.io/badge/ICML-2026-8A2BE2.svg)](https://icml.cc/Conferences/2026)
[![Paper](https://img.shields.io/badge/arXiv-2602.14486-b31b1b.svg)](https://arxiv.org/abs/2602.14486)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://brbiclab.epfl.ch/projects/aristotelian/)

**[Fabian Gröger*](https://fabiangroeger96.github.io/)** · **[Shuo Wen*](https://wenshuo128.github.io/)** · **[Maria Brbić](https://brbiclab.epfl.ch/team/)**

---

## The Aristotelian Representation Hypothesis

> Neural networks, trained with different objectives on different data and modalities, are converging to shared local neighborhood relationships.

![The Aristotelian Representation Hypothesis](https://brbiclab.epfl.ch/wp-content/uploads/2026/02/Screenshot-2026-02-02-at-08.38.20.png)

The Platonic Representation Hypothesis suggests that representations from neural networks are converging to a common statistical model of reality. We show that the existing metrics used to measure representational similarity are **confounded by network scale**: increasing model depth or width can systematically inflate representational similarity scores. To correct these effects, we introduce a **permutation-based null-calibration framework** that transforms any representational similarity metric into a calibrated score with statistical guarantees. We revisit the Platonic Representation Hypothesis with our calibration framework, which reveals a nuanced picture: the apparent convergence reported by global spectral measures largely disappears after calibration, while local neighborhood similarity retains significant agreement across different modalities.

---

## Repository Structure

```
├── calibrated_similarity/    # Standalone Python package (pip installable)
│   ├── calibration.py        # Core algorithms (Algorithm 1 & 2 from paper)
│   └── __init__.py
├── aristotelian/             # Research code for paper experiments
│   ├── metrics/              # Similarity metrics (CKA, kNN, RSA, CCA, etc.)
│   ├── experiments/          # Experiment utilities
│   ├── prh/                  # PRH replication code
│   ├── utils/                # Utility functions
│   └── style/                # Plotting style
├── scripts/
│   ├── experiments/          # Paper experiment runners
│   └── plots/                # Figure generation
├── tests/                    # Test suite
```

---

## Using the calibration package only

```bash
pip install calibrated-similarity
```

See the [package documentation](README_PYPI.md) for usage examples.

---

## Reproducing Paper Experiments

```bash
# Step 1: Run experiments
python -m scripts.experiments.cli --device cuda

# Step 2: Generate all figures
python -m scripts.plots.experiments --sections all
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{groger2026revisiting,
  title     = {Revisiting the Platonic Representation Hypothesis: An Aristotelian View},
  author    = {Gr{\"o}ger, Fabian and Wen, Shuo and Brbi{\'c}, Maria},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
