import itertools

import pytest
import torch

from aristotelian import sg_cka_kernel, sg_cka_linear, standard_cka


def test_standard_cka_shapes():
    torch.manual_seed(0)
    X = torch.randn(12, 5)
    Y = torch.randn(12, 5)
    val = standard_cka(X, Y, mode="linear")
    assert isinstance(val, float)


def test_sg_cka_linear_null_nonnegative():
    torch.manual_seed(1)
    X = torch.randn(20, 6)
    Y = torch.randn(20, 6)
    res = sg_cka_linear(X, Y, num_permutations=20, quantile=0.9)
    assert res.gated >= 0.0
    assert 0.0 <= res.pvalue <= 1.0
    assert isinstance(res.pvalue, float)
    assert len(res.null_samples) == 20


def test_sg_cka_linear_identical_detects():
    torch.manual_seed(2)
    X = torch.randn(24, 5)
    res = sg_cka_linear(X, X, num_permutations=30, quantile=0.9)
    assert res.gated > 0.0
    assert res.raw > 0.99
    assert res.pvalue < 0.2
    assert isinstance(res.pvalue, float)


def test_sg_cka_kernel_paths():
    torch.manual_seed(3)
    X = torch.randn(16, 4)
    Y = torch.randn(16, 4)
    res = sg_cka_kernel(X, Y, num_permutations=15, quantile=0.9)
    assert res.gated >= 0.0
    assert 0.0 <= res.pvalue <= 1.0
    assert isinstance(res.pvalue, float)
    assert len(res.null_samples) == 15


def test_sg_cka_linear_respects_perms_and_pvalue():
    torch.manual_seed(5)
    X = torch.randn(6, 4)
    Y = torch.randn(6, 4)
    perms = torch.stack(
        [
            torch.arange(6),
            torch.tensor([1, 2, 3, 4, 5, 0]),
            torch.tensor([2, 3, 4, 5, 0, 1]),
        ]
    )
    res = sg_cka_linear(
        X, Y, num_permutations=perms.shape[0], quantile=0.7, perms=perms
    )
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    denom = torch.norm(Xc.T @ Xc) * torch.norm(Yc.T @ Yc) + 1e-8
    nums = []
    for p in perms:
        cross = Yc[p].T @ Xc
        nums.append(float((torch.norm(cross, p="fro") ** 2 / denom).item()))
    expected_p = (sum(s >= res.raw for s in nums) + 1.0) / (len(nums) + 1.0)
    assert torch.allclose(
        torch.tensor(nums), torch.tensor(res.null_samples, dtype=torch.float)
    )
    assert abs(res.pvalue - expected_p) <= 1e-8


def test_sg_cka_quantile_one_clamps_tail_strength():
    torch.manual_seed(6)
    X = torch.randn(8, 3)
    Y = torch.randn(8, 3)
    res = sg_cka_linear(X, Y, num_permutations=10, quantile=1.0)
    assert res.tail_strength == 0.0


def test_sg_cka_rejects_invalid_quantile():
    torch.manual_seed(7)
    X = torch.randn(8, 3)
    with pytest.raises(ValueError, match="quantile must be in"):
        sg_cka_linear(X, X, num_permutations=5, quantile=1.5)
    with pytest.raises(ValueError, match="quantile must be in"):
        sg_cka_kernel(X, X, num_permutations=5, quantile=-0.2)


def test_sg_cka_linear_isometry_invariance():
    torch.manual_seed(8)
    X = torch.randn(10, 4)
    q, _ = torch.linalg.qr(torch.randn(4, 4))
    Y = X @ q
    res = sg_cka_linear(X, Y, num_permutations=10, quantile=0.9)
    assert res.raw > 0.99


def test_sg_cka_linear_superuniform_under_null_all_perms():
    torch.manual_seed(9)
    n = 5
    X = torch.randn(n, 3)
    Y = torch.randn(n, 3)
    perms = torch.tensor(list(itertools.permutations(range(n))))
    alpha = 0.2
    pvals = []
    for p in perms:
        res = sg_cka_linear(
            X,
            Y[p],
            num_permutations=perms.shape[0],
            quantile=1.0 - alpha,
            perms=perms,
        )
        pvals.append(res.pvalue)
    rate = float(sum(p <= alpha for p in pvals)) / len(pvals)
    assert rate <= alpha


def test_standard_cka_invariance():
    torch.manual_seed(4)
    X = torch.randn(20, 6)
    q, _ = torch.linalg.qr(torch.randn(6, 6))
    Y = X @ q
    same = standard_cka(X, X, mode="linear")
    rotated = standard_cka(X, Y, mode="linear")
    assert same > 0.99
    assert rotated > 0.99
