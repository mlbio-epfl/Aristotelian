import torch

from aristotelian import sg_cka_kernel, sg_cka_linear, sg_knn, sg_rsa


def _assert_has_variants(res):
    assert isinstance(res.null_centered, float)
    assert isinstance(res.z, float)
    assert isinstance(res.ari, float)
    assert isinstance(res.mean_null, float)
    assert isinstance(res.median_null, float)
    assert isinstance(res.std_null, float)
    assert -1.0 <= res.ari <= 1.0


def test_variants_present_for_cka():
    torch.manual_seed(0)
    X = torch.randn(20, 6)
    Y = torch.randn(20, 6)
    res_lin = sg_cka_linear(X, Y, num_permutations=15, quantile=0.9)
    res_rbf = sg_cka_kernel(X, Y, num_permutations=15, quantile=0.9)
    _assert_has_variants(res_lin)
    _assert_has_variants(res_rbf)


def test_variants_present_for_knn_rsa():
    torch.manual_seed(1)
    X = torch.randn(22, 5)
    Y = torch.randn(22, 5)
    res_knn = sg_knn(X, Y, k=5, num_permutations=15, quantile=0.9)
    res_rsa = sg_rsa(X, Y, num_permutations=15, quantile=0.9)
    _assert_has_variants(res_knn)
    _assert_has_variants(res_rsa)


def test_compute_null_variants_accepts_tensor():
    from aristotelian.metrics.calibration import compute_null_variants

    null_samples = torch.tensor([0.1, 0.2, 0.3])
    out = compute_null_variants(0.25, null_samples, min_score=0.0, max_score=1.0)
    assert isinstance(out.mean_null, float)
    assert isinstance(out.null_centered, float)
