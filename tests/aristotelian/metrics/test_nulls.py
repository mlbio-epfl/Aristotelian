import torch

from aristotelian.metrics.nulls import spectrum_matched_view


def test_spectrum_matched_preserves_singular_values():
    torch.manual_seed(0)
    X = torch.randn(20, 8)
    Xp = spectrum_matched_view(X, seed=123)
    s1 = torch.linalg.svdvals(X)
    s2 = torch.linalg.svdvals(Xp)
    assert torch.allclose(s1, s2, rtol=1e-4, atol=1e-5)


def test_spectrum_matched_reproducible_and_nontrivial():
    torch.manual_seed(1)
    X = torch.randn(16, 6)
    Xp1 = spectrum_matched_view(X, seed=42)
    Xp2 = spectrum_matched_view(X, seed=42)
    assert torch.allclose(Xp1, Xp2)
    assert torch.norm(Xp1 - X) > 1e-3
