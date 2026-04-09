import numpy as np
import torch

from aristotelian import mutual_knn as prh_mutual_knn
from aristotelian import mutual_knn_overlap
from aristotelian.prh.cache import (
    build_gram_cache,
    build_knn_cache,
    build_knn_cache_with_indices,
)
from aristotelian.prh.preprocess import remove_outliers
from aristotelian.prh.alignment import compute_alignment_gated_cached
from aristotelian.prh.prh_experiment import (
    compute_alignment_gated,
    compute_alignment_gated_cka_cached,
    compute_alignment_gated_cknna_cached,
    compute_alignment_gated_cycle_knn_cached,
    compute_alignment_gated_knn_cached,
    compute_alignment_gated_procrustes_cached,
    compute_alignment_gated_pwcca_cached,
    compute_alignment_gated_svcca_cached,
    prepare_features,
    prh_alignment_filename,
    prh_feature_filename,
)


def test_prh_feature_filename_builds_path(tmp_path):
    path = prh_feature_filename(
        str(tmp_path),
        "minhuh/prh",
        "wit_1024",
        "model/name",
        pool="avg",
        prompt=False,
        caption_idx=0,
    )
    assert "minhuh/prh" in path
    assert "wit_1024" in path
    assert "model_name_pool-avg_prompt-False_cid-0.pt" in path


def test_prh_alignment_filename_builds_path(tmp_path):
    path = prh_alignment_filename(
        str(tmp_path),
        "minhuh/prh",
        "val",
        "language",
        "avg",
        False,
        "vision",
        "cls",
        False,
        "mutual_knn",
        10,
    )
    assert "minhuh/prh" in path
    assert "val" in path
    assert "mutual_knn_k10.npy" in path


def test_remove_outliers_clamps():
    feats = torch.tensor([[-10.0, 0.0, 10.0], [1.0, 2.0, 3.0]])
    out = remove_outliers(feats, q=0.5, exact=True)
    assert torch.max(out) <= torch.max(feats)
    assert torch.min(out) >= torch.min(feats)


def test_prepare_features_list():
    feats = [torch.randn(6, 4), torch.randn(6, 4)]
    prepared = prepare_features(feats, q=0.9, exact=True, device="cpu")
    assert isinstance(prepared, list)
    assert len(prepared) == 2
    assert prepared[0].shape == feats[0].shape


def test_compute_alignment_gated_returns_scores():
    torch.manual_seed(0)
    repsA = [torch.randn(12, 6), torch.randn(12, 6)]
    repsB = [repsA[0].clone(), repsA[1].clone()]
    out = compute_alignment_gated(
        repsA,
        repsB,
        metric="mutual_knn",
        topk=4,
        normalize=True,
        num_permutations=10,
        alpha=0.1,
        seed=0,
    )
    assert 0.0 <= out["p_value"] <= 1.0
    assert out["best_indices"][0] in [0, 1]
    assert out["best_indices"][1] in [0, 1]
    assert out["g_score"] >= 0.0
    assert out["raw_score"] >= 0.0


def test_compute_alignment_gated_skip_normalize_matches_normalize():
    torch.manual_seed(1)
    repsA = [torch.randn(10, 5), torch.randn(10, 5)]
    repsB = [torch.randn(10, 5), torch.randn(10, 5)]
    out_norm = compute_alignment_gated(
        repsA,
        repsB,
        metric="mutual_knn",
        topk=3,
        normalize=True,
        num_permutations=8,
        alpha=0.1,
        seed=1,
    )
    repsA_norm = [torch.nn.functional.normalize(x, p=2, dim=-1) for x in repsA]
    repsB_norm = [torch.nn.functional.normalize(y, p=2, dim=-1) for y in repsB]
    out_skip = compute_alignment_gated(
        repsA_norm,
        repsB_norm,
        metric="mutual_knn",
        topk=3,
        normalize=False,
        num_permutations=8,
        alpha=0.1,
        seed=1,
    )
    assert np.isclose(out_norm["raw_score"], out_skip["raw_score"])
    assert np.isclose(out_norm["g_score"], out_skip["g_score"])


def test_compute_alignment_gated_cached_matches_default():
    torch.manual_seed(3)
    repsA = [torch.randn(14, 6), torch.randn(14, 6)]
    repsB = [torch.randn(14, 6), torch.randn(14, 6)]
    out_default = compute_alignment_gated(
        repsA,
        repsB,
        metric="mutual_knn",
        topk=4,
        normalize=True,
        num_permutations=10,
        alpha=0.1,
        seed=2,
    )
    layers_a, masks_a = build_knn_cache(repsA, topk=4, normalize=True)
    layers_b, masks_b = build_knn_cache(repsB, topk=4, normalize=True)
    out_cached = compute_alignment_gated_cached(
        layers_a,
        layers_b,
        masks_a,
        masks_b,
        topk=4,
        num_permutations=10,
        alpha=0.1,
        seed=2,
    )
    assert np.isclose(out_default["raw_score"], out_cached["raw_score"])
    assert np.isclose(out_default["g_score"], out_cached["g_score"])


def test_prh_mutual_knn_matches_overlap_for_normalized():
    torch.manual_seed(2)
    X = torch.randn(20, 6)
    Y = torch.randn(20, 6)
    Xn = torch.nn.functional.normalize(X, p=2, dim=-1)
    Yn = torch.nn.functional.normalize(Y, p=2, dim=-1)
    prh_score = prh_mutual_knn(Xn, Yn, topk=5)
    overlap_score = mutual_knn_overlap(Xn, Yn, k=5)
    assert np.isclose(prh_score, overlap_score, atol=1e-5)


def test_compute_alignment_gated_supports_additional_metrics():
    torch.manual_seed(4)
    repsA = [torch.randn(10, 5), torch.randn(10, 5)]
    repsB = [torch.randn(10, 5), torch.randn(10, 5)]
    metrics = ["cka", "knn", "svcca", "pwcca", "procrustes"]
    for metric in metrics:
        out = compute_alignment_gated(
            repsA,
            repsB,
            metric=metric,
            topk=3,
            normalize=True,
            num_permutations=6,
            alpha=0.2,
            seed=0,
        )
        assert 0.0 <= out["p_value"] <= 1.0
        assert out["g_score"] >= 0.0


def test_compute_alignment_gated_cka_cached_matches_default():
    """Test that CKA cached (linear) matches the non-cached implementation."""
    torch.manual_seed(5)
    repsA = [torch.randn(14, 6), torch.randn(14, 6)]
    repsB = [torch.randn(14, 6), torch.randn(14, 6)]
    out_default = compute_alignment_gated(
        repsA,
        repsB,
        metric="cka_lin",
        topk=4,
        normalize=True,
        num_permutations=10,
        alpha=0.1,
        seed=2,
    )
    x_grams = build_gram_cache(repsA, normalize=True, kernel="linear")
    y_grams = build_gram_cache(repsB, normalize=True, kernel="linear")
    out_cached = compute_alignment_gated_cka_cached(
        x_grams,
        y_grams,
        num_permutations=10,
        alpha=0.1,
        seed=2,
        unbiased=False,
    )
    assert np.isclose(out_default["raw_score"], out_cached["raw_score"], atol=1e-5)
    assert np.isclose(out_default["g_score"], out_cached["g_score"], atol=1e-5)


def test_compute_alignment_gated_unbiased_cka_cached_matches_default():
    """Test that unbiased CKA cached matches the non-cached implementation."""
    torch.manual_seed(6)
    repsA = [torch.randn(14, 6), torch.randn(14, 6)]
    repsB = [torch.randn(14, 6), torch.randn(14, 6)]
    out_default = compute_alignment_gated(
        repsA,
        repsB,
        metric="unbiased_cka",
        topk=4,
        normalize=True,
        num_permutations=10,
        alpha=0.1,
        seed=3,
    )
    x_grams = build_gram_cache(repsA, normalize=True, kernel="linear")
    y_grams = build_gram_cache(repsB, normalize=True, kernel="linear")
    out_cached = compute_alignment_gated_cka_cached(
        x_grams,
        y_grams,
        num_permutations=10,
        alpha=0.1,
        seed=3,
        unbiased=True,
    )
    assert np.isclose(out_default["raw_score"], out_cached["raw_score"], atol=1e-5)
    assert np.isclose(out_default["g_score"], out_cached["g_score"], atol=1e-5)


def test_compute_alignment_gated_cknna_cached_matches_default():
    """Test that CKNNA cached matches the non-cached implementation."""
    torch.manual_seed(7)
    repsA = [torch.randn(14, 6), torch.randn(14, 6)]
    repsB = [torch.randn(14, 6), torch.randn(14, 6)]
    out_default = compute_alignment_gated(
        repsA,
        repsB,
        metric="cknna",
        topk=4,
        normalize=True,
        num_permutations=10,
        alpha=0.1,
        seed=4,
    )
    x_grams = build_gram_cache(repsA, normalize=True, kernel="linear")
    y_grams = build_gram_cache(repsB, normalize=True, kernel="linear")
    out_cached = compute_alignment_gated_cknna_cached(
        x_grams,
        y_grams,
        topk=4,
        num_permutations=10,
        alpha=0.1,
        seed=4,
        unbiased=True,
    )
    assert np.isclose(out_default["raw_score"], out_cached["raw_score"], atol=1e-5)
    assert np.isclose(out_default["g_score"], out_cached["g_score"], atol=1e-5)


def test_compute_alignment_gated_cycle_knn_cached_matches_default():
    torch.manual_seed(8)
    repsA = [torch.randn(12, 6), torch.randn(12, 6)]
    repsB = [torch.randn(12, 6), torch.randn(12, 6)]
    repsA_norm = [torch.nn.functional.normalize(x, p=2, dim=-1) for x in repsA]
    repsB_norm = [torch.nn.functional.normalize(y, p=2, dim=-1) for y in repsB]
    out_default = compute_alignment_gated(
        repsA_norm,
        repsB_norm,
        metric="cycle_knn",
        topk=3,
        normalize=False,
        num_permutations=8,
        alpha=0.1,
        seed=5,
    )
    _, x_knn, _ = build_knn_cache_with_indices(repsA_norm, topk=3, normalize=False)
    _, y_knn, _ = build_knn_cache_with_indices(repsB_norm, topk=3, normalize=False)
    out_cached = compute_alignment_gated_cycle_knn_cached(
        x_knn,
        y_knn,
        num_permutations=8,
        alpha=0.1,
        seed=5,
    )
    assert np.isclose(out_default["raw_score"], out_cached["raw_score"], atol=1e-5)


def test_compute_alignment_gated_knn_cached_matches_default():
    torch.manual_seed(12)
    repsA = [torch.randn(12, 6), torch.randn(12, 6)]
    repsB = [torch.randn(12, 6), torch.randn(12, 6)]
    repsA_norm = [torch.nn.functional.normalize(x, p=2, dim=-1) for x in repsA]
    repsB_norm = [torch.nn.functional.normalize(y, p=2, dim=-1) for y in repsB]
    out_default = compute_alignment_gated(
        repsA_norm,
        repsB_norm,
        metric="knn",
        topk=3,
        normalize=False,
        num_permutations=6,
        alpha=0.1,
        seed=9,
    )
    _, x_knn, _ = build_knn_cache_with_indices(repsA_norm, topk=3, normalize=False)
    _, y_knn, _ = build_knn_cache_with_indices(repsB_norm, topk=3, normalize=False)
    out_cached = compute_alignment_gated_knn_cached(
        x_knn,
        y_knn,
        topk=3,
        num_permutations=6,
        alpha=0.1,
        seed=9,
    )
    assert np.isclose(out_default["raw_score"], out_cached["raw_score"], atol=1e-5)


def test_compute_alignment_gated_svcca_cached_matches_default():
    torch.manual_seed(9)
    np.random.seed(9)
    repsA = [torch.randn(10, 5), torch.randn(10, 5)]
    repsB = [torch.randn(10, 5), torch.randn(10, 5)]
    repsA_norm = [torch.nn.functional.normalize(x, p=2, dim=-1) for x in repsA]
    repsB_norm = [torch.nn.functional.normalize(y, p=2, dim=-1) for y in repsB]
    out_default = compute_alignment_gated(
        repsA_norm,
        repsB_norm,
        metric="svcca",
        topk=3,
        normalize=False,
        num_permutations=6,
        alpha=0.1,
        seed=6,
    )
    torch.manual_seed(9)
    np.random.seed(9)
    out_cached = compute_alignment_gated_svcca_cached(
        repsA_norm,
        repsB_norm,
        num_permutations=6,
        alpha=0.1,
        seed=6,
    )
    assert np.isclose(out_default["raw_score"], out_cached["raw_score"], atol=1e-5)
    assert np.isclose(out_default["g_score"], out_cached["g_score"], atol=1e-5)


def test_compute_alignment_gated_pwcca_cached_matches_default():
    torch.manual_seed(10)
    repsA = [torch.randn(10, 6), torch.randn(10, 6)]
    repsB = [torch.randn(10, 6), torch.randn(10, 6)]
    repsA_norm = [torch.nn.functional.normalize(x, p=2, dim=-1) for x in repsA]
    repsB_norm = [torch.nn.functional.normalize(y, p=2, dim=-1) for y in repsB]
    out_default = compute_alignment_gated(
        repsA_norm,
        repsB_norm,
        metric="pwcca",
        topk=3,
        normalize=False,
        num_permutations=6,
        alpha=0.1,
        seed=7,
    )
    out_cached = compute_alignment_gated_pwcca_cached(
        repsA_norm,
        repsB_norm,
        num_permutations=6,
        alpha=0.1,
        seed=7,
    )
    # Torch (class) vs numpy (cached) eigh can diverge ~1e-4 in float32
    assert np.isclose(out_default["raw_score"], out_cached["raw_score"], atol=1e-3)
    assert np.isclose(out_default["g_score"], out_cached["g_score"], atol=1e-3)


def test_compute_alignment_gated_procrustes_cached_matches_default():
    torch.manual_seed(11)
    repsA = [torch.randn(10, 6), torch.randn(10, 6)]
    repsB = [torch.randn(10, 6), torch.randn(10, 6)]
    repsA_norm = [torch.nn.functional.normalize(x, p=2, dim=-1) for x in repsA]
    repsB_norm = [torch.nn.functional.normalize(y, p=2, dim=-1) for y in repsB]
    out_default = compute_alignment_gated(
        repsA_norm,
        repsB_norm,
        metric="procrustes",
        topk=3,
        normalize=False,
        num_permutations=6,
        alpha=0.1,
        seed=8,
    )
    out_cached = compute_alignment_gated_procrustes_cached(
        repsA_norm,
        repsB_norm,
        num_permutations=6,
        alpha=0.1,
        seed=8,
    )
    assert np.isclose(out_default["raw_score"], out_cached["raw_score"], atol=1e-5)
    assert np.isclose(out_default["g_score"], out_cached["g_score"], atol=1e-5)


def test_compute_alignment_gated_cka_rbf_cached_returns_valid():
    """Test that CKA RBF cached returns valid scores."""
    torch.manual_seed(8)
    repsA = [torch.randn(14, 6), torch.randn(14, 6)]
    repsB = [torch.randn(14, 6), torch.randn(14, 6)]
    x_grams = build_gram_cache(repsA, normalize=True, kernel="rbf", rbf_sigma=1.0)
    y_grams = build_gram_cache(repsB, normalize=True, kernel="rbf", rbf_sigma=1.0)
    out_cached = compute_alignment_gated_cka_cached(
        x_grams,
        y_grams,
        num_permutations=10,
        alpha=0.1,
        seed=5,
        unbiased=False,
    )
    # RBF CKA should return valid scores in [0, 1] range
    assert 0.0 <= out_cached["raw_score"] <= 1.0
    assert 0.0 <= out_cached["g_score"] <= 1.0
    assert 0.0 <= out_cached["p_value"] <= 1.0


def test_build_gram_cache_linear_kernel():
    """Test that linear Gram cache produces correct Gram matrices."""
    torch.manual_seed(9)
    feats = [torch.randn(10, 5), torch.randn(10, 5)]
    grams = build_gram_cache(feats, normalize=True, kernel="linear")
    assert len(grams) == 2
    for g in grams:
        assert g.shape == (10, 10)
        # Linear kernel Gram matrix should be symmetric
        assert torch.allclose(g, g.T)


def test_build_gram_cache_rbf_kernel():
    """Test that RBF Gram cache produces correct Gram matrices."""
    torch.manual_seed(10)
    feats = [torch.randn(10, 5), torch.randn(10, 5)]
    grams = build_gram_cache(feats, normalize=True, kernel="rbf", rbf_sigma=1.0)
    assert len(grams) == 2
    for g in grams:
        assert g.shape == (10, 10)
        # RBF kernel Gram matrix should be symmetric
        assert torch.allclose(g, g.T)
        # RBF kernel diagonal should be 1 (K(x,x) = exp(0) = 1)
        assert torch.allclose(g.diag(), torch.ones(10), atol=1e-5)
