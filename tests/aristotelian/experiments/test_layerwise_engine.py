import torch

from aristotelian import standard_cka
from aristotelian.experiments.layerwise_engine import (
    permutation_null_matrices_layerwise,
    similarity_matrix_layerwise,
)
from aristotelian.metrics.aggregation import SimpleMetric, compute_similarity_matrix


def _cka_metric() -> SimpleMetric:
    return SimpleMetric(
        name="cka_linear",
        max_value=1.0,
        compute=lambda X, Y: standard_cka(X, Y, mode="linear"),
    )


def test_similarity_matrix_layerwise_cka_matches_standard():
    torch.manual_seed(0)
    repsA = [torch.randn(12, 5), torch.randn(12, 5)]
    repsB = [torch.randn(12, 5), torch.randn(12, 5)]
    metric = _cka_metric()
    S_ref = compute_similarity_matrix(repsA, repsB, metric)
    S_cached = similarity_matrix_layerwise(
        repsA, repsB, metric, metric_name=metric.name
    )
    assert torch.allclose(S_ref, S_cached, atol=1e-6)


def test_permutation_null_matrices_layerwise_cka_matches_standard():
    torch.manual_seed(1)
    repsA = [torch.randn(10, 4), torch.randn(10, 4)]
    repsB = [torch.randn(10, 4), torch.randn(10, 4)]
    metric = _cka_metric()
    mats_ref = []
    n = repsA[0].shape[0]
    rng = torch.Generator()
    rng.manual_seed(5)
    for _ in range(3):
        perm = torch.randperm(n, generator=rng)
        repsB_perm = [Y[perm, :] for Y in repsB]
        mats_ref.append(compute_similarity_matrix(repsA, repsB_perm, metric))
    mats_cached = permutation_null_matrices_layerwise(
        repsA, repsB, metric, metric_name=metric.name, num_permutations=3, seed=5
    )
    assert len(mats_ref) == len(mats_cached)
    for ref, cached in zip(mats_ref, mats_cached):
        assert torch.allclose(ref, cached, atol=1e-6)
