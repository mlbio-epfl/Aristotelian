"""Unit tests for experiment infrastructure utilities."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.experiments.infra.device import mp_context, parse_devices, resolve_device
from scripts.experiments.infra.io import (
    noise_output_path,
    save_array,
    should_skip,
    to_cpu,
    write_csv_rows,
)
from scripts.experiments.infra.parallel import (
    PARALLEL_SECTIONS,
    mp_chunksize,
    mp_limit_main_threads,
    mp_worker_init,
)
from scripts.experiments.infra.stats import (
    binomial_ci,
    bootstrap_spearman,
    mean_ci,
    rankdata,
    spearman,
)


class TestDeviceUtilities:
    def test_resolve_device_with_explicit(self):
        assert resolve_device("cpu") == "cpu"
        assert resolve_device("cuda:0") == "cuda:0"

    def test_resolve_device_default(self):
        device = resolve_device(None)
        assert device in ("cpu", "cuda")

    def test_parse_devices_empty(self):
        devices = parse_devices(None, "cpu")
        assert devices == ["cpu"]

    def test_parse_devices_single(self):
        devices = parse_devices("cpu", "cuda")
        assert devices == ["cpu"]

    def test_parse_devices_multiple(self):
        devices = parse_devices("cpu,cpu", "cuda")
        assert devices == ["cpu", "cpu"]

    def test_mp_context_returns_context(self):
        ctx = mp_context("cpu", None)
        assert hasattr(ctx, "Process")
        assert hasattr(ctx, "Queue")


class TestIOUtilities:
    def test_to_cpu_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = to_cpu(t)
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_to_cpu_dict(self):
        d = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
        result = to_cpu(d)
        assert isinstance(result, dict)
        assert result["a"] == 1.0
        assert result["b"] == 2.0

    def test_to_cpu_nested(self):
        d = {"a": {"b": torch.tensor([1.0, 2.0])}}
        result = to_cpu(d)
        assert isinstance(result["a"]["b"], np.ndarray)

    def test_save_array_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.npy"
            data = {"key": np.array([1, 2, 3])}
            save_array(path, data)
            assert path.exists()
            loaded = np.load(path, allow_pickle=True).item()
            assert np.array_equal(loaded["key"], [1, 2, 3])

    def test_should_skip_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.npy"
            assert not should_skip([path], force=False)

    def test_should_skip_exists_no_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exists.npy"
            path.touch()
            assert should_skip([path], force=False)

    def test_should_skip_exists_with_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exists.npy"
            path.touch()
            assert not should_skip([path], force=True)

    def test_write_csv_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.csv"
            rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
            write_csv_rows(path, rows)
            assert path.exists()
            content = path.read_text()
            assert "a,b" in content
            assert "1,2" in content

    def test_noise_output_path_default(self):
        base = Path("/tmp/assets/test.npy")
        # When noise_type equals default, return base unchanged
        path = noise_output_path(base, "gaussian", "gaussian")
        assert path == base

    def test_noise_output_path_non_default(self):
        base = Path("/tmp/assets/test.npy")
        # When noise_type differs from default, include noise type in name
        path = noise_output_path(base, "heavy", "gaussian")
        assert "heavy" in str(path)


class TestParallelUtilities:
    def test_parallel_sections_is_set(self):
        assert isinstance(PARALLEL_SECTIONS, set)
        assert len(PARALLEL_SECTIONS) > 0

    def test_mp_chunksize_basic(self):
        assert mp_chunksize(100, 4) > 0
        assert mp_chunksize(10, 4) >= 1

    def test_mp_chunksize_small_total(self):
        assert mp_chunksize(1, 4) == 1

    def test_mp_worker_init_runs(self):
        # Should not raise
        mp_worker_init()

    def test_mp_limit_main_threads_runs(self):
        # Should not raise
        mp_limit_main_threads()


class TestStatsUtilities:
    def test_mean_ci_basic(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, low, high = mean_ci(values)
        assert mean == pytest.approx(3.0)
        assert low < mean < high

    def test_mean_ci_single_value(self):
        mean, low, high = mean_ci([5.0])
        assert mean == 5.0

    def test_binomial_ci_basic(self):
        low, high = binomial_ci(50, 100)
        assert 0 <= low <= 0.5 <= high <= 1

    def test_binomial_ci_zero_successes(self):
        low, high = binomial_ci(0, 100)
        assert low == 0

    def test_binomial_ci_all_successes(self):
        low, high = binomial_ci(100, 100)
        assert high == pytest.approx(1.0, abs=1e-10)

    def test_rankdata_basic(self):
        x = np.array([3.0, 1.0, 2.0])
        ranks = rankdata(x)
        # 0-based ranks: smallest gets 0, largest gets 2
        assert ranks[0] == 2  # 3.0 is largest -> rank 2
        assert ranks[1] == 0  # 1.0 is smallest -> rank 0
        assert ranks[2] == 1  # 2.0 is middle -> rank 1

    def test_spearman_perfect(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        rho = spearman(x, y)
        assert rho == pytest.approx(1.0)

    def test_spearman_negative(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([3.0, 2.0, 1.0])
        rho = spearman(x, y)
        assert rho == pytest.approx(-1.0)

    def test_bootstrap_spearman_basic(self):
        np.random.seed(42)
        x = np.random.randn(50)
        y = x + 0.1 * np.random.randn(50)
        rho, low, high = bootstrap_spearman(x, y, num_boot=100, seed_val=42)
        assert low < rho < high
