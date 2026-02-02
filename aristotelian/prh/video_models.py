"""Video model registry and loaders for video-to-text alignment experiments.

This module provides model loading utilities for video models following
the VideoPRH paper's methodology.

Supported model types:
- VideoMAE: Native video models (HuggingFace transformers)
- DINOv2: Frame-by-frame processing (reuse timm)
- CLIP: Frame-by-frame processing (reuse timm)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import timm
import torch
from loguru import logger

# Video model registry organized by type
VIDEO_MODELS: Dict[str, List[str]] = {
    "default": [
        # VideoMAE (HuggingFace transformers) - native video models
        "MCG-NJU/videomae-base",
        "MCG-NJU/videomae-large",
        # DINOv2 frame-by-frame (reuse timm)
        "vit_base_patch14_dinov2.lvd142m",
        "vit_large_patch14_dinov2.lvd142m",
        # CLIP frame-by-frame (reuse timm)
        "vit_base_patch16_clip_224.laion2b",
        "vit_large_patch14_clip_224.laion2b",
    ],
    "small": [
        # Smaller subset for quick experiments
        "vit_base_patch14_dinov2.lvd142m",
        "vit_base_patch16_clip_224.laion2b",
    ],
    "videomae": [
        # VideoMAE models only
        "MCG-NJU/videomae-base",
        "MCG-NJU/videomae-large",
        "MCG-NJU/videomae-huge",
    ],
    "dinov2": [
        # DINOv2 models for frame-by-frame processing
        "vit_small_patch14_dinov2.lvd142m",
        "vit_base_patch14_dinov2.lvd142m",
        "vit_large_patch14_dinov2.lvd142m",
        "vit_giant_patch14_dinov2.lvd142m",
    ],
    "clip": [
        # CLIP models for frame-by-frame processing
        "vit_base_patch16_clip_224.laion2b",
        "vit_large_patch14_clip_224.laion2b",
        "vit_huge_patch14_clip_224.laion2b",
    ],
    "extended": [
        # Extended set with more model variants
        # VideoMAE
        "MCG-NJU/videomae-base",
        "MCG-NJU/videomae-large",
        "MCG-NJU/videomae-huge",
        # DINOv2
        "vit_small_patch14_dinov2.lvd142m",
        "vit_base_patch14_dinov2.lvd142m",
        "vit_large_patch14_dinov2.lvd142m",
        "vit_giant_patch14_dinov2.lvd142m",
        # CLIP
        "vit_base_patch16_clip_224.laion2b",
        "vit_large_patch14_clip_224.laion2b",
        "vit_huge_patch14_clip_224.laion2b",
        # CLIP finetuned on ImageNet-12K
        "vit_base_patch16_clip_224.laion2b_ft_in12k",
        "vit_large_patch14_clip_224.laion2b_ft_in12k",
        "vit_huge_patch14_clip_224.laion2b_ft_in12k",
    ],
    "videoprh": [
        # Video-text PRH experiment models (using models that exist on HuggingFace)
        # VideoMAE models (base, large, huge)
        "MCG-NJU/videomae-base",
        "MCG-NJU/videomae-large",
        "MCG-NJU/videomae-huge-finetuned-kinetics",
        # DINOv2 frame-by-frame (small, base, large, giant)
        "vit_small_patch14_dinov2.lvd142m",
        "vit_base_patch14_dinov2.lvd142m",
        "vit_large_patch14_dinov2.lvd142m",
        "vit_giant_patch14_dinov2.lvd142m",
        # CLIP frame-by-frame (base, large, huge, giant)
        "vit_base_patch16_clip_224.laion2b",
        "vit_large_patch14_clip_224.laion2b",
        "vit_huge_patch14_clip_224.laion2b",
        "vit_giant_patch14_clip_224.laion2b",
    ],
}


def get_video_models(modelset: str = "default") -> List[str]:
    """Get video models for V2T experiments.

    Args:
        modelset: Which model set to use. Options:
            - "default": Main set of video models
            - "small": Smaller subset for quick experiments
            - "videomae": VideoMAE models only
            - "dinov2": DINOv2 models only
            - "clip": CLIP models only
            - "extended": Extended set with more variants

    Returns:
        List of model names/identifiers.

    Raises:
        ValueError: If unknown modelset is provided.
    """
    if modelset not in VIDEO_MODELS:
        available = ", ".join(VIDEO_MODELS.keys())
        raise ValueError(f"Unknown video modelset: {modelset}. Available: {available}")
    return VIDEO_MODELS[modelset].copy()


def is_native_video_model(model_name: str) -> bool:
    """Check if model processes video natively vs frame-by-frame.

    Native video models (like VideoMAE) process multiple frames at once
    and learn temporal dynamics. Frame-by-frame models (like DINOv2, CLIP)
    process each frame independently.

    Args:
        model_name: Name/identifier of the model.

    Returns:
        True if model is a native video model, False for frame-by-frame models.
    """
    # VideoMAE and similar HuggingFace video models
    native_prefixes = ("MCG-NJU/videomae", "facebook/timesformer", "google/vivit")
    return any(model_name.startswith(prefix) for prefix in native_prefixes)


def get_model_family(model_name: str) -> str:
    """Get the model family for a given model name.

    Args:
        model_name: Name/identifier of the model.

    Returns:
        Model family name (e.g., "VideoMAE", "DINOv2", "CLIP").
    """
    name_lower = model_name.lower()
    if "videomae" in name_lower:
        return "VideoMAE"
    elif "timesformer" in name_lower:
        return "TimeSformer"
    elif "vivit" in name_lower:
        return "ViViT"
    elif "dinov2" in name_lower:
        return "DINOv2"
    elif "clip" in name_lower:
        if "ft_in" in name_lower:
            return "CLIP (finetuned)"
        return "CLIP"
    elif "mae" in name_lower:
        return "MAE"
    else:
        return "Other"


def load_video_model(
    model_name: str,
    *,
    device: str = "cpu",
    num_frames: int = 16,
) -> Tuple[Any, Any]:
    """Load video model for feature extraction.

    For VideoMAE and native video models, uses HuggingFace transformers.
    For DINOv2/CLIP and frame-by-frame models, uses timm.

    Args:
        model_name: Name/identifier of the model.
        device: Device to load the model on.
        num_frames: Number of frames for native video models (ignored for frame-by-frame).

    Returns:
        Tuple of (model, processor/transform):
        - For native video models: (VideoMAEModel, VideoMAEImageProcessor)
        - For frame-by-frame models: (timm model, timm transform)

    Raises:
        ImportError: If required dependencies are not installed.
    """
    if is_native_video_model(model_name):
        return _load_videomae_model(model_name, device=device, num_frames=num_frames)
    else:
        return _load_frame_model(model_name, device=device)


def _load_videomae_model(
    model_name: str,
    *,
    device: str = "cpu",
    num_frames: int = 16,
) -> Tuple[Any, Any]:
    """Load VideoMAE model from HuggingFace transformers."""
    try:
        from transformers import VideoMAEImageProcessor, VideoMAEModel
    except ImportError:
        raise ImportError(
            "transformers is required to load VideoMAE models. "
            "Install with: pip install transformers"
        )

    logger.debug(f"Loading VideoMAE model: {model_name}")

    # Determine torch dtype
    torch_dtype = None
    if device != "cpu" and torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

    # Load processor and model
    processor = VideoMAEImageProcessor.from_pretrained(model_name)

    if torch_dtype is not None:
        model = VideoMAEModel.from_pretrained(model_name, torch_dtype=torch_dtype)
    else:
        model = VideoMAEModel.from_pretrained(model_name)

    model.to(device)
    model.eval()

    return model, processor


def _load_frame_model(
    model_name: str,
    *,
    device: str = "cpu",
) -> Tuple[Any, Any]:
    """Load frame-by-frame model from timm."""
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    logger.debug(f"Loading frame-by-frame model: {model_name}")

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.to(device)
    model.eval()

    # Create transform
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )

    return model, transform


def get_model_num_params(model: Any) -> int:
    """Get the number of parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Number of parameters.
    """
    return int(sum(p.numel() for p in model.parameters()))


def get_video_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration info for a video model.

    Args:
        model_name: Name/identifier of the model.

    Returns:
        Dictionary with model configuration:
        - is_native: Whether it's a native video model
        - family: Model family name
        - expected_frames: Expected number of frames (for native models)
        - patch_size: Patch size if applicable
    """
    is_native = is_native_video_model(model_name)
    family = get_model_family(model_name)

    config = {
        "is_native": is_native,
        "family": family,
        "expected_frames": 16 if is_native else 1,
    }

    # Extract patch size from model name
    name_lower = model_name.lower()
    if "patch14" in name_lower:
        config["patch_size"] = 14
    elif "patch16" in name_lower:
        config["patch_size"] = 16
    elif "patch32" in name_lower:
        config["patch_size"] = 32
    else:
        config["patch_size"] = None

    return config
