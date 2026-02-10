"""Tests for video-to-text alignment experiment (VideoPRH)."""

import pytest

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


class TestV2TExperimentImports:
    """Tests for V2T experiment module imports."""

    def test_run_v2t_experiment_importable(self):
        """Test that main experiment function is importable."""
        from aristotelian.prh.v2t_experiment import run_v2t_experiment

        assert callable(run_v2t_experiment)

    def test_prh_package_exports(self):
        """Test that V2T functions are exported from prh package."""
        from aristotelian.prh import (
            get_video_models,
            load_video_model,
            run_v2t_experiment,
        )

        assert callable(run_v2t_experiment)
        assert callable(get_video_models)
        assert callable(load_video_model)


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
