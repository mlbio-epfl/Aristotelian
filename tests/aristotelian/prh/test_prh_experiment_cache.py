import torch

from aristotelian.prh.prh_experiment import _cached_feats_match


def test_cached_feats_match_detects_mismatch():
    payload = {"feats": torch.zeros(256, 3)}
    assert _cached_feats_match(payload, 256)
    assert not _cached_feats_match(payload, 1024)
