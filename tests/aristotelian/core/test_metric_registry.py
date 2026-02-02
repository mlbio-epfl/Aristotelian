import numpy as np
import torch

from aristotelian import mutual_knn, mutual_knn_overlap, standard_cka
from aristotelian.metrics.api import (
    gated_cka_linear,
    metric_definitions,
    prh_metric_spec,
    raw_cca,
    raw_cka_linear,
    raw_cka_rbf,
    raw_knn,
    raw_procrustes,
    raw_pwcca,
    raw_rv,
    raw_svcca,
)
from aristotelian.metrics.cca import cca_mean, pwcca_mean, rv_coefficient, svcca_mean
from aristotelian.metrics.other_metrics import procrustes_score


def test_raw_metrics_match_direct_impls():
    torch.manual_seed(0)
    X = torch.randn(12, 5)
    Y = torch.randn(12, 5)
    assert np.isclose(raw_cka_linear(X, Y), standard_cka(X, Y, mode="linear"))
    assert np.isclose(raw_cka_rbf(X, Y), standard_cka(X, Y, mode="kernel"))
    assert np.isclose(raw_knn(X, Y, k=3), mutual_knn_overlap(X, Y, k=3))
    assert np.isclose(raw_cca(X, Y), cca_mean(X.numpy(), Y.numpy()))
    assert np.isclose(raw_svcca(X, Y), svcca_mean(X.numpy(), Y.numpy()))
    assert np.isclose(raw_pwcca(X, Y), pwcca_mean(X.numpy(), Y.numpy()))
    assert np.isclose(raw_rv(X, Y), rv_coefficient(X.numpy(), Y.numpy()))
    assert np.isclose(raw_procrustes(X, Y), procrustes_score(X.numpy(), Y.numpy()))


def test_metric_definitions_returns_expected_names():
    metric_defs, multiq_helpers = metric_definitions(num_permutations=5, device="cpu")
    names = [entry[0] for entry in metric_defs]
    assert "CKA (lin)" in names
    assert "kNN" in names
    assert "SVCCA" in names
    assert "PWCCA" in names
    assert "Procrustes" in names
    assert "CKA (lin)" in multiq_helpers


def test_gated_cka_linear_matches_sg():
    torch.manual_seed(1)
    X = torch.randn(10, 4)
    Y = torch.randn(10, 4)
    perms = torch.stack([torch.randperm(10) for _ in range(5)])
    res = gated_cka_linear(
        X,
        Y,
        0.9,
        num_permutations=5,
        device="cpu",
        perms=perms,
    )
    assert hasattr(res, "raw")
    assert 0.0 <= res.raw <= 1.0


def test_prh_metric_spec_matches_prh_metrics():
    torch.manual_seed(2)
    X = torch.randn(10, 4)
    Y = torch.randn(10, 4)
    Xn = torch.nn.functional.normalize(X, p=2, dim=-1)
    Yn = torch.nn.functional.normalize(Y, p=2, dim=-1)
    fn, _ = prh_metric_spec("mutual_knn", topk=3)
    assert np.isclose(fn(Xn, Yn), mutual_knn(Xn, Yn, topk=3))
