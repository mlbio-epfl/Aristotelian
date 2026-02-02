import numpy as np

from aristotelian.experiments.multiple import bh_fdr, max_stat_threshold


def test_bh_fdr_basic():
    pvals = np.array([0.001, 0.02, 0.03, 0.2, 0.9])
    thr, mask = bh_fdr(pvals, alpha=0.05)
    assert 0.0 <= thr <= 0.05
    assert mask.sum() >= 1


def test_max_stat_threshold_quantile():
    null_max = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    thr = max_stat_threshold(null_max, alpha=0.1)
    assert 0.4 <= thr <= 0.5


def test_bh_fdr_all_ones():
    pvals = np.ones(5)
    thr, mask = bh_fdr(pvals, alpha=0.05)
    assert thr == 0.0
    assert not mask.any()


def test_bh_fdr_all_zeros():
    pvals = np.zeros(4)
    thr, mask = bh_fdr(pvals, alpha=0.1)
    assert thr == 0.0
    assert mask.all()


def test_bh_fdr_unsorted_input():
    pvals = np.array([0.2, 0.001, 0.05, 0.9])
    thr, mask = bh_fdr(pvals, alpha=0.05)
    assert np.all(mask == (pvals <= thr))
