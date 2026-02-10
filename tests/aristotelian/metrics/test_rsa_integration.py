import os
import time

import numpy as np
import pytest
import torch

from aristotelian import rsa_vector, sg_rsa
from aristotelian.metrics.rsa import (
    _rankdata_torch,
    _rankdata_torch_batch,
    _spearman_corr_torch,
)


def _old_sg_rsa(X, Y, *, perms, batch_size):
    vx = rsa_vector(X)
    vy = rsa_vector(Y)
    raw = float(_spearman_corr_torch(vx, vy).item())
    dist_y = torch.cdist(Y, Y, p=2)
    idx0, idx1 = torch.triu_indices(dist_y.shape[0], dist_y.shape[0], offset=1)
    rx = _rankdata_torch(vx)
    rx_c = rx - rx.mean()
    rx_norm = torch.norm(rx_c) + 1e-8
    null_scores = []
    for start in range(0, perms.size(0), batch_size):
        batch = perms[start : start + batch_size]
        temp = dist_y[batch]
        perm_exp = batch.unsqueeze(1).expand(-1, dist_y.size(0), -1)
        dist_perm = temp.gather(2, perm_exp)
        vy_perm = dist_perm[:, idx0, idx1]
        ry = _rankdata_torch_batch(vy_perm)
        ry_c = ry - ry.mean(dim=1, keepdim=True)
        denom = (ry_c.norm(dim=1) * rx_norm) + 1e-8
        corr = (ry_c * rx_c).sum(dim=1) / denom
        null_scores.extend(corr.detach().cpu().tolist())
    return raw, null_scores


def _new_sg_rsa(X, Y, *, perms, batch_size):
    vx = rsa_vector(X)
    vy = rsa_vector(Y)
    raw = float(_spearman_corr_torch(vx, vy).item())
    dist_y = torch.cdist(Y, Y, p=2)
    idx0, idx1 = torch.triu_indices(dist_y.shape[0], dist_y.shape[0], offset=1)
    rx = _rankdata_torch(vx)
    rx_c = rx - rx.mean()
    rx_norm = torch.norm(rx_c) + 1e-8
    null_scores = []
    for start in range(0, perms.size(0), batch_size):
        batch = perms[start : start + batch_size]
        vy_perm = dist_y[batch[:, idx0], batch[:, idx1]]
        ry = _rankdata_torch_batch(vy_perm)
        ry_c = ry - ry.mean(dim=1, keepdim=True)
        denom = (ry_c.norm(dim=1) * rx_norm) + 1e-8
        corr = (ry_c * rx_c).sum(dim=1) / denom
        null_scores.extend(corr.detach().cpu().tolist())
    return raw, null_scores


def _env_int(name, default):
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_cases(name, default):
    raw = os.getenv(name)
    if not raw:
        return default
    cases = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        n_str, d_str = entry.lower().split("x")
        cases.append((int(n_str), int(d_str)))
    return cases


def _env_int_list(name, default):
    raw = os.getenv(name)
    if not raw:
        return default
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _bench_case(n, d, perms_n, batch_size, device):
    X = torch.randn(n, d, device=device)
    Y = torch.randn(n, d, device=device)
    perms = torch.stack([torch.randperm(n, device=device) for _ in range(perms_n)])

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    raw_old, null_old = _old_sg_rsa(X, Y, perms=perms, batch_size=batch_size)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    raw_new, null_new = _new_sg_rsa(X, Y, perms=perms, batch_size=batch_size)
    if device == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()

    raw_diff = abs(raw_old - raw_new)
    null_diff = (
        np.max(np.abs(np.array(null_old) - np.array(null_new))) if null_old else 0.0
    )
    speedup = (t1 - t0) / (t2 - t1) if (t2 - t1) > 0 else float("inf")
    return {
        "device": device,
        "n": n,
        "d": d,
        "perms": perms_n,
        "batch_size": batch_size,
        "old_s": t1 - t0,
        "new_s": t2 - t1,
        "speedup": speedup,
        "raw_diff": raw_diff,
        "max_null_diff": float(null_diff),
    }


@pytest.mark.parametrize(
    ("n", "d"),
    _env_cases("RSA_BENCH_CASES", [(128, 128), (256, 256), (512, 256)]),
)
def test_rsa_speed_and_equivalence_realistic(n, d):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    perms_n = _env_int("RSA_BENCH_PERMS", 200)
    batch_size = _env_int("RSA_BENCH_BATCH", 32)

    result = _bench_case(n, d, perms_n, batch_size, device)
    print(result)

    assert result["raw_diff"] <= 1e-8
    assert result["max_null_diff"] <= 1e-8


@pytest.mark.parametrize(
    ("n", "d"),
    _env_cases("RSA_APPROX_CASES", [(128, 128), (256, 256), (512, 256)]),
)
def test_rsa_approx_speed_and_diff(n, d):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    perms_n = _env_int("RSA_APPROX_PERMS", 200)
    batch_size = _env_int("RSA_APPROX_BATCH", 32)
    pair_samples_list = _env_int_list("RSA_APPROX_SAMPLES", [2048, 8192, 32768])

    X = torch.randn(n, d, device=device)
    Y = torch.randn(n, d, device=device)
    perms = torch.stack([torch.randperm(n, device=device) for _ in range(perms_n)])

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    res_full = sg_rsa(
        X,
        Y,
        num_permutations=perms_n,
        quantile=0.95,
        device=device,
        batch_size=batch_size,
        perms=perms,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    null_full = np.asarray(res_full.null_samples)
    full_s = t1 - t0
    for pair_samples in pair_samples_list:
        torch.manual_seed(0)
        res_approx = sg_rsa(
            X,
            Y,
            num_permutations=perms_n,
            quantile=0.95,
            device=device,
            batch_size=batch_size,
            pair_samples=pair_samples,
            perms=perms,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        raw_diff = abs(float(res_full.raw) - float(res_approx.raw))
        null_approx = np.asarray(res_approx.null_samples)
        null_mean_diff = float(np.mean(np.abs(null_full - null_approx)))
        speedup = full_s / (t2 - t1) if (t2 - t1) > 0 else float("inf")

        print(
            {
                "device": device,
                "n": n,
                "d": d,
                "perms": perms_n,
                "batch_size": batch_size,
                "pair_samples": pair_samples,
                "full_s": full_s,
                "approx_s": t2 - t1,
                "speedup": speedup,
                "raw_diff": raw_diff,
                "mean_null_diff": null_mean_diff,
            }
        )

        assert raw_diff <= 0.1
        assert null_mean_diff <= 0.1
