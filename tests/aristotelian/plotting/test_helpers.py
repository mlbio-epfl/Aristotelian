"""Unit tests for pure helper functions in plot_experiments.py.

These tests verify helper functions that don't require matplotlib.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.plots.experiments import (
    DEFAULT_NOISE_TYPE,
    METRIC_LABELS,
    NOISE_TYPES,
    _collect_metric_variants,
    _default_quantile_key,
    _infer_llm_family,
    _noise_suffix,
    _noise_variants,
    _order_metrics,
    _parse_llm_size_b,
    _prettify_metric,
    _quantile_keys,
    _require_asset,
    _shorten_prh_label,
    _should_skip,
    _slug,
    _variant_title,
)


class TestPrettifyMetric:
    """Tests for _prettify_metric function."""

    def test_exact_match(self) -> None:
        """Test exact match returns mapped label."""
        assert _prettify_metric("sgcka_lin") == "CKA (linear)"
        assert _prettify_metric("sgknn") == "mKNN"

    def test_lowercase_match(self) -> None:
        """Test lowercase normalization."""
        assert _prettify_metric("CKA") == "CKA"  # Exact match first
        # KNN matches "knn" in METRIC_LABELS via lowercase normalization
        assert _prettify_metric("KNN") == "mKNN"

    def test_no_match_returns_original(self) -> None:
        """Test unknown metric returns original name."""
        assert _prettify_metric("unknown_metric") == "unknown_metric"
        assert _prettify_metric("CustomMetric") == "CustomMetric"

    def test_known_metrics(self) -> None:
        """Test all known metrics have valid mappings."""
        for code_name, pretty_name in METRIC_LABELS.items():
            assert _prettify_metric(code_name) == pretty_name


class TestSlug:
    """Tests for _slug function."""

    def test_lowercase(self) -> None:
        """Test conversion to lowercase."""
        assert _slug("CKA") == "cka"
        assert _slug("UPPER") == "upper"

    def test_remove_parentheses(self) -> None:
        """Test parentheses removal."""
        assert _slug("CKA (lin)") == "cka_lin"
        assert _slug("(test)") == "test"

    def test_replace_spaces_and_special(self) -> None:
        """Test replacement of spaces and special characters."""
        assert _slug("hello world") == "hello_world"
        assert _slug("a/b") == "a_b"
        assert _slug("a-b") == "a_b"

    def test_combined(self) -> None:
        """Test combined transformations."""
        # Space around "/" becomes "_ _" after transformation
        assert _slug("CKA (linear) / test") == "cka_linear___test"


class TestNoiseSuffix:
    """Tests for _noise_suffix function."""

    def test_default_noise_type(self) -> None:
        """Test default noise type returns empty string."""
        assert _noise_suffix(DEFAULT_NOISE_TYPE) == ""

    def test_other_noise_types(self) -> None:
        """Test other noise types return underscore prefix."""
        for noise_type in NOISE_TYPES:
            if noise_type != DEFAULT_NOISE_TYPE:
                assert _noise_suffix(noise_type) == f"_{noise_type}"


class TestOrderMetrics:
    """Tests for _order_metrics function."""

    def test_preferred_order(self) -> None:
        """Test metrics are ordered according to preference."""
        metrics = ["PWCCA", "CKA (lin)", "kNN"]
        result = _order_metrics(metrics)
        # CKA should come before kNN, which should come before PWCCA
        assert result.index("CKA (lin)") < result.index("kNN")
        assert result.index("kNN") < result.index("PWCCA")

    def test_unknown_metrics_sorted_alphabetically(self) -> None:
        """Test unknown metrics are sorted alphabetically at end."""
        metrics = ["ZMetric", "AMetric", "CKA (lin)"]
        result = _order_metrics(metrics)
        assert result[0] == "CKA (lin)"
        # Unknown metrics sorted alphabetically
        assert result.index("AMetric") < result.index("ZMetric")

    def test_empty_list(self) -> None:
        """Test empty list returns empty list."""
        assert _order_metrics([]) == []


class TestQuantileKeys:
    """Tests for _quantile_keys and _default_quantile_key functions."""

    def test_extract_quantile_keys(self) -> None:
        """Test extraction of quantile keys."""
        variants = ["raw", "q90", "q95", "z", "q80"]
        result = _quantile_keys(variants)
        assert result == ["q80", "q90", "q95"]  # Sorted by value

    def test_no_quantile_keys(self) -> None:
        """Test handling of no quantile keys."""
        variants = ["raw", "z", "ari"]
        assert _quantile_keys(variants) == []

    def test_default_quantile_key(self) -> None:
        """Test default quantile key is highest."""
        variants = ["raw", "q90", "q95", "z"]
        assert _default_quantile_key(variants) == "q95"

    def test_default_quantile_key_empty(self) -> None:
        """Test default returns None when no quantiles."""
        variants = ["raw", "z"]
        assert _default_quantile_key(variants) is None


class TestVariantTitle:
    """Tests for _variant_title function."""

    def test_raw(self) -> None:
        """Test raw variant."""
        assert _variant_title("raw", []) == "raw"

    def test_null_centered(self) -> None:
        """Test null-centered variant."""
        assert _variant_title("null_centered", []) == "null-centered"

    def test_z_score(self) -> None:
        """Test z-score variant."""
        assert _variant_title("z", []) == "z-score"

    def test_ari(self) -> None:
        """Test ARI variant."""
        assert _variant_title("ari", []) == "ARI-adjusted"

    def test_single_q95(self) -> None:
        """Test single q95 shortens to 'gated'."""
        assert _variant_title("q95", ["q95"]) == "gated"

    def test_multiple_quantiles(self) -> None:
        """Test multiple quantiles show full label."""
        q_keys = ["q90", "q95"]
        assert _variant_title("q95", q_keys) == "gated q95"
        assert _variant_title("q90", q_keys) == "gated q90"

    def test_unknown_variant(self) -> None:
        """Test unknown variant returns as-is."""
        assert _variant_title("custom", []) == "custom"


class TestShortenPrhLabel:
    """Tests for _shorten_prh_label function."""

    def test_extract_model_name(self) -> None:
        """Test extraction of model name from path."""
        assert "ViT" in _shorten_prh_label("models/vit_base")

    def test_shorten_replacements(self) -> None:
        """Test various shortenings."""
        assert "openllama" in _shorten_prh_label("open_llama_3b")
        assert "bloom" in _shorten_prh_label("bloomz-7b")

    def test_remove_suffixes(self) -> None:
        """Test removal of common suffixes."""
        result = _shorten_prh_label("vit_base.augreg_in21k")
        assert ".augreg_in21k" not in result


class TestParseLlmSizeB:
    """Tests for _parse_llm_size_b function."""

    def test_simple_billion(self) -> None:
        """Test simple billion parameter format."""
        assert _parse_llm_size_b("llama-7b") == 7.0
        assert _parse_llm_size_b("gpt-3.5b") == 3.5

    def test_million_to_billion(self) -> None:
        """Test million parameters converted to billion."""
        assert _parse_llm_size_b("bert-125m") == 0.125

    def test_mixture_format(self) -> None:
        """Test mixture of experts format like 8x7b."""
        assert _parse_llm_size_b("mixtral-8x7b") == 56.0

    def test_decimal_format(self) -> None:
        """Test decimal billion format like 7b1."""
        assert _parse_llm_size_b("model-7b1") == 7.1

    def test_no_size(self) -> None:
        """Test model without size returns None."""
        assert _parse_llm_size_b("bert-base") is None


class TestInferLlmFamily:
    """Tests for _infer_llm_family function."""

    def test_known_families(self) -> None:
        """Test known family detection."""
        assert _infer_llm_family("bloomz-7b") == "bloomz"
        assert _infer_llm_family("open_llama_3b") == "open-llama"
        assert _infer_llm_family("llama-2-7b") == "llama"
        assert _infer_llm_family("olmo-7b") == "olmo"
        assert _infer_llm_family("gemma-2b") == "gemma"
        assert _infer_llm_family("mistral-7b") == "mistral"
        assert _infer_llm_family("mixtral-8x7b") == "mistral"

    def test_unknown_family(self) -> None:
        """Test unknown models return 'other'."""
        assert _infer_llm_family("custom-model") == "other"


class TestNoiseVariants:
    """Tests for _noise_variants function."""

    def test_base_file_only(self) -> None:
        """Test when only base file exists."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            base_file = tmp_dir / "test.npy"
            base_file.touch()

            variants = _noise_variants(tmp_dir, "test.npy")
            assert len(variants) == 1
            assert variants[0][0] == DEFAULT_NOISE_TYPE
            assert variants[0][1] == base_file

    def test_with_noise_variants(self) -> None:
        """Test discovery of noise variants."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            # Create base and variant files
            (tmp_dir / "test.npy").touch()
            (tmp_dir / "test_student_t.npy").touch()
            (tmp_dir / "test_laplace.npy").touch()

            variants = _noise_variants(tmp_dir, "test.npy")
            assert len(variants) == 3
            noise_types = [v[0] for v in variants]
            assert DEFAULT_NOISE_TYPE in noise_types
            assert "student_t" in noise_types
            assert "laplace" in noise_types

    def test_no_files(self) -> None:
        """Test empty result when no files exist."""
        with tempfile.TemporaryDirectory() as tmp:
            variants = _noise_variants(Path(tmp), "nonexistent.npy")
            assert variants == []


class TestShouldSkip:
    """Tests for _should_skip function."""

    def test_force_never_skips(self) -> None:
        """Test force=True never skips."""
        assert not _should_skip([Path("/nonexistent")], force=True)

    def test_skip_when_exists(self) -> None:
        """Test skip when all outputs exist."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            (tmp_dir / "output.pdf").touch()
            assert _should_skip([tmp_dir / "output.pdf"], force=False)

    def test_no_skip_when_missing(self) -> None:
        """Test no skip when outputs missing."""
        with tempfile.TemporaryDirectory() as tmp:
            assert not _should_skip([Path(tmp) / "missing.pdf"], force=False)


class TestRequireAsset:
    """Tests for _require_asset function."""

    def test_exists(self) -> None:
        """Test returns True when file exists."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_file = Path(tmp) / "test.npy"
            tmp_file.touch()
            assert _require_asset(tmp_file)

    def test_missing(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test returns False and prints when file missing."""
        result = _require_asset(Path("/nonexistent/file.npy"))
        assert not result
        captured = capsys.readouterr()
        assert "skip missing" in captured.out


class TestCollectMetricVariants:
    """Tests for _collect_metric_variants function."""

    def test_collect_variants(self) -> None:
        """Test collection of variants by metric."""
        vals = {
            ("CKA", "raw"): np.array([1, 2]),
            ("CKA", "q95"): np.array([3, 4]),
            ("kNN", "raw"): np.array([5, 6]),
        }
        result = _collect_metric_variants(vals)
        assert "CKA" in result
        assert "kNN" in result
        assert result["CKA"] == {"raw", "q95"}
        assert result["kNN"] == {"raw"}
