import torch

from aristotelian.metrics.esr import compute_esr, run_esr_validation


def test_compute_esr_counts_above_tau():
    svals = torch.tensor([0.9, 0.5, 0.1])
    out = compute_esr(svals, tau=0.4)
    assert out.recovered_rank == 2
    assert out.effective_rank >= 1.0


def test_run_esr_validation_shapes():
    torch.manual_seed(0)
    res = run_esr_validation(
        n=40,
        d=20,
        ranks=(1, 2),
        snr_values=(0.5, 2.0),
        num_trials=3,
        num_permutations=10,
        quantile=0.9,
        seed=123,
    )
    assert set(res.keys()) == {1, 2}
    for rank_dict in res.values():
        assert set(rank_dict.keys()) == {0.5, 2.0}
        for entry in rank_dict.values():
            assert "recovered_rank_mean" in entry
            assert "effective_rank_mean" in entry
