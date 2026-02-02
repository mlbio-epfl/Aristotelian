"""Tests for video-to-text alignment experiment (VideoPRH)."""

import numpy as np
import pytest

from aristotelian.prh.v2t_experiment import (
    compute_v2t_family_alignment,
    fit_scaling_law,
)
from aristotelian.prh.video_models import (
    get_model_family,
    get_video_model_config,
    get_video_models,
    is_native_video_model,
)


class TestGetVideoModels:
    """Tests for the video model registry."""

    def test_default_modelset_returns_models(self):
        models = get_video_models("default")
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)

    def test_small_modelset_returns_fewer_models(self):
        default_models = get_video_models("default")
        small_models = get_video_models("small")
        assert len(small_models) < len(default_models)
        assert len(small_models) > 0

    def test_videomae_modelset_contains_videomae(self):
        models = get_video_models("videomae")
        assert all("videomae" in m.lower() for m in models)

    def test_dinov2_modelset_contains_dinov2(self):
        models = get_video_models("dinov2")
        assert all("dinov2" in m.lower() for m in models)

    def test_clip_modelset_contains_clip(self):
        models = get_video_models("clip")
        assert all("clip" in m.lower() for m in models)

    def test_extended_modelset_larger_than_default(self):
        default_models = get_video_models("default")
        extended_models = get_video_models("extended")
        assert len(extended_models) >= len(default_models)

    def test_unknown_modelset_raises(self):
        with pytest.raises(ValueError, match="Unknown video modelset"):
            get_video_models("nonexistent")


class TestIsNativeVideoModel:
    """Tests for native video model detection."""

    def test_videomae_is_native(self):
        assert is_native_video_model("MCG-NJU/videomae-base")
        assert is_native_video_model("MCG-NJU/videomae-large")
        assert is_native_video_model("MCG-NJU/videomae-huge")

    def test_dinov2_is_not_native(self):
        assert not is_native_video_model("vit_base_patch14_dinov2.lvd142m")
        assert not is_native_video_model("vit_large_patch14_dinov2.lvd142m")

    def test_clip_is_not_native(self):
        assert not is_native_video_model("vit_base_patch16_clip_224.laion2b")
        assert not is_native_video_model("vit_large_patch14_clip_224.laion2b")


class TestGetModelFamily:
    """Tests for model family detection."""

    def test_videomae_family(self):
        assert get_model_family("MCG-NJU/videomae-base") == "VideoMAE"
        assert get_model_family("MCG-NJU/videomae-large") == "VideoMAE"

    def test_dinov2_family(self):
        assert get_model_family("vit_base_patch14_dinov2.lvd142m") == "DINOv2"
        assert get_model_family("vit_large_patch14_dinov2.lvd142m") == "DINOv2"

    def test_clip_family(self):
        assert get_model_family("vit_base_patch16_clip_224.laion2b") == "CLIP"
        assert get_model_family("vit_large_patch14_clip_224.laion2b") == "CLIP"

    def test_clip_finetuned_family(self):
        assert (
            get_model_family("vit_base_patch16_clip_224.laion2b_ft_in12k")
            == "CLIP (finetuned)"
        )


class TestGetVideoModelConfig:
    """Tests for video model configuration."""

    def test_videomae_config(self):
        config = get_video_model_config("MCG-NJU/videomae-base")
        assert config["is_native"] is True
        assert config["family"] == "VideoMAE"
        assert config["expected_frames"] == 16

    def test_dinov2_config(self):
        config = get_video_model_config("vit_base_patch14_dinov2.lvd142m")
        assert config["is_native"] is False
        assert config["family"] == "DINOv2"
        assert config["expected_frames"] == 1
        assert config["patch_size"] == 14

    def test_clip_config(self):
        config = get_video_model_config("vit_base_patch16_clip_224.laion2b")
        assert config["is_native"] is False
        assert config["family"] == "CLIP"
        assert config["patch_size"] == 16


class TestPVDData:
    """Tests for PE-Video (PVD) data loading utilities."""

    def test_load_dataset_import(self):
        """Test that dataset loading function is importable."""
        from aristotelian.prh.pvd_data import load_pvd_dataset

        assert callable(load_pvd_dataset)

    def test_extract_frames_import(self):
        """Test that frame extraction function is importable."""
        from aristotelian.prh.pvd_data import extract_video_frames

        assert callable(extract_video_frames)

    def test_download_function_import(self):
        """Test that download function is importable."""
        from aristotelian.prh.pvd_data import download_pvd

        assert callable(download_pvd)

    def test_iter_samples_import(self):
        """Test that iterator function is importable."""
        from aristotelian.prh.pvd_data import iter_pvd_samples

        assert callable(iter_pvd_samples)

    def test_get_path_function_import(self):
        """Test that path helper function is importable."""
        from aristotelian.prh.pvd_data import get_pvd_path

        assert callable(get_pvd_path)


class TestFrameExtraction:
    """Tests for frame extraction utilities."""

    def test_frame_indices_uniform(self):
        """Test uniform frame sampling strategy."""
        from aristotelian.prh.pvd_data import _get_frame_indices

        indices = _get_frame_indices(100, 8, "uniform")
        assert len(indices) == 8
        # Should be evenly spaced
        assert indices[0] == 0
        assert indices[-1] == 99

    def test_frame_indices_start(self):
        """Test start frame sampling strategy."""
        from aristotelian.prh.pvd_data import _get_frame_indices

        indices = _get_frame_indices(100, 8, "start")
        assert len(indices) == 8
        assert indices == list(range(8))

    def test_frame_indices_middle(self):
        """Test middle frame sampling strategy."""
        from aristotelian.prh.pvd_data import _get_frame_indices

        indices = _get_frame_indices(100, 8, "middle")
        assert len(indices) == 8
        # Should start from the middle
        expected_start = (100 - 8) // 2
        assert indices[0] == expected_start

    def test_frame_indices_caps_at_total(self):
        """Test that num_frames is capped at total_frames."""
        from aristotelian.prh.pvd_data import _get_frame_indices

        indices = _get_frame_indices(5, 10, "uniform")
        assert len(indices) == 5


class TestScalingLaw:
    """Tests for scaling law fitting."""

    def test_fit_scaling_law_returns_expected_keys(self):
        """Test that fit_scaling_law returns all expected parameters."""
        scores = np.random.rand(3, 3) * 0.5 + 0.3
        frame_counts = [1, 4, 8]
        caption_counts = [1, 2, 4]

        result = fit_scaling_law(scores, frame_counts, caption_counts)

        assert "S_inf" in result
        assert "Cf" in result
        assert "alpha" in result
        assert "Cc" in result
        assert "beta" in result
        assert "residual" in result

    def test_fit_scaling_law_with_perfect_data(self):
        """Test fitting with data that follows the scaling law exactly."""
        # Create data that approximately follows the scaling law
        # score(nf, nc) = S_inf - (Cf * nf^(-alpha) + Cc * nc^(-beta))
        S_inf, Cf, alpha, Cc, beta = 0.8, 0.1, 0.5, 0.1, 0.5

        frame_counts = [1, 4, 8, 16]
        caption_counts = [1, 2, 4, 8]

        scores = np.zeros((len(frame_counts), len(caption_counts)))
        for i, nf in enumerate(frame_counts):
            for j, nc in enumerate(caption_counts):
                scores[i, j] = S_inf - (Cf * nf ** (-alpha) + Cc * nc ** (-beta))

        result = fit_scaling_law(scores, frame_counts, caption_counts)

        # Should fit well
        assert result["residual"] < 0.01
        assert 0 < result["S_inf"] < 1

    def test_fit_scaling_law_handles_edge_cases(self):
        """Test that fitting handles edge cases gracefully."""
        # Single point
        scores = np.array([[0.5]])
        frame_counts = [8]
        caption_counts = [1]

        result = fit_scaling_law(scores, frame_counts, caption_counts)

        # Should not crash, may return NaN
        assert isinstance(result, dict)


class TestV2TFamilyAlignment:
    """Tests for family-level alignment computation."""

    def test_returns_expected_keys(self):
        """Test that the function returns all expected keys."""
        scores = np.random.rand(4, 5)
        video_models = [
            "MCG-NJU/videomae-base",
            "vit_base_patch14_dinov2.lvd142m",
            "vit_base_patch16_clip_224.laion2b",
            "vit_large_patch14_clip_224.laion2b",
        ]
        text_models = [
            "bigscience/bloomz-560m",
            "bigscience/bloomz-1b1",
            "huggyllama/llama-7b",
            "huggyllama/llama-13b",
            "google/gemma-2b",
        ]

        result = compute_v2t_family_alignment(scores, video_models, text_models)

        assert "video_families" in result
        assert "unique_families" in result
        assert "family_alignment" in result
        assert "inter_family" in result

    def test_categorizes_video_models_correctly(self):
        """Test that video models are categorized by family."""
        scores = np.random.rand(4, 2)
        video_models = [
            "MCG-NJU/videomae-base",
            "MCG-NJU/videomae-large",
            "vit_base_patch14_dinov2.lvd142m",
            "vit_base_patch16_clip_224.laion2b",
        ]
        text_models = ["model1", "model2"]

        result = compute_v2t_family_alignment(scores, video_models, text_models)

        families = result["video_families"]
        assert families[0] == "VideoMAE"
        assert families[1] == "VideoMAE"
        assert families[2] == "DINOv2"
        assert families[3] == "CLIP"

    def test_family_alignment_computed(self):
        """Test that family alignment is computed correctly."""
        # Set specific scores for different families
        scores = np.array(
            [
                [0.8, 0.8],  # VideoMAE model 1
                [0.8, 0.8],  # VideoMAE model 2
                [0.5, 0.5],  # DINOv2 model
                [0.3, 0.3],  # CLIP model
            ]
        )
        video_models = [
            "MCG-NJU/videomae-base",
            "MCG-NJU/videomae-large",
            "vit_base_patch14_dinov2.lvd142m",
            "vit_base_patch16_clip_224.laion2b",
        ]
        text_models = ["model1", "model2"]

        result = compute_v2t_family_alignment(scores, video_models, text_models)

        # VideoMAE family should have mean 0.8
        assert np.isclose(result["family_alignment"]["VideoMAE"]["mean"], 0.8)
        # DINOv2 family should have mean 0.5
        assert np.isclose(result["family_alignment"]["DINOv2"]["mean"], 0.5)
        # CLIP family should have mean 0.3
        assert np.isclose(result["family_alignment"]["CLIP"]["mean"], 0.3)


class TestV2TExperimentImports:
    """Tests for V2T experiment module imports."""

    def test_run_v2t_experiment_importable(self):
        """Test that main experiment function is importable."""
        from aristotelian.prh.v2t_experiment import run_v2t_experiment

        assert callable(run_v2t_experiment)

    def test_run_v2t_scaling_experiment_importable(self):
        """Test that scaling experiment function is importable."""
        from aristotelian.prh.v2t_experiment import run_v2t_scaling_experiment

        assert callable(run_v2t_scaling_experiment)

    def test_prh_package_exports(self):
        """Test that V2T functions are exported from prh package."""
        from aristotelian.prh import (
            compute_v2t_family_alignment,
            get_video_models,
            load_video_model,
            run_v2t_experiment,
        )

        assert callable(run_v2t_experiment)
        assert callable(get_video_models)
        assert callable(load_video_model)
        assert callable(compute_v2t_family_alignment)


class TestVideoModelsImports:
    """Tests for video models module imports."""

    def test_load_video_model_importable(self):
        """Test that model loading function is importable."""
        from aristotelian.prh.video_models import load_video_model

        assert callable(load_video_model)

    def test_get_model_num_params_importable(self):
        """Test that parameter counting function is importable."""
        from aristotelian.prh.video_models import get_model_num_params

        assert callable(get_model_num_params)
