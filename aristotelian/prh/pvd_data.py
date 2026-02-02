"""PE-Video (PVD) dataset utilities for video-to-text alignment experiments.

This module provides data loading utilities for the PE-Video dataset,
following the VideoPRH paper's methodology for video-text alignment.

PE-Video contains ~1M diverse videos with 118k+ human-annotated captions.
Dataset: https://huggingface.co/datasets/facebook/PE-Video

Frame extraction supports multiple strategies:
- uniform: Evenly spaced frames across the video
- start: First N frames
- middle: N frames from the middle
"""

from __future__ import annotations

import io
import random
import tempfile
import warnings
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from loguru import logger
from PIL import Image

# Default cache directory for video data
PVD_CACHE_DIR = Path.home() / ".cache" / "pevideo"

# Default number of samples for V2T experiments
DEFAULT_NUM_SAMPLES = 1024


def download_pvd(
    cache_dir: str | Path | None = None,
) -> Path:
    """Download PE-Video dataset using HuggingFace datasets.

    Args:
        cache_dir: Directory to cache the dataset. Defaults to ~/.cache/pevideo

    Returns:
        Path to the downloaded dataset directory.

    Note:
        Requires datasets to be installed: pip install datasets
        The dataset will be cached by HuggingFace in its default location.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets is required to download PE-Video. "
            "Install it with: pip install datasets"
        )

    if cache_dir is None:
        cache_dir = PVD_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading PE-Video dataset from HuggingFace...")
    logger.info("This may take a while on first run")

    try:
        # Load a small portion to trigger caching
        _ = load_dataset(
            "facebook/PE-Video",
            split="test",
            streaming=False,
        )
        logger.info(f"PE-Video dataset ready (cache: {cache_dir})")
        return cache_dir
    except Exception as e:
        raise RuntimeError(
            f"Failed to download PE-Video from HuggingFace: {e}\n"
            "Make sure you have datasets installed: pip install datasets"
        )


def load_pvd_dataset(
    *,
    split: str = "test",
    max_samples: int | None = DEFAULT_NUM_SAMPLES,
    num_captions: int = 1,
    auto_download: bool = True,
    seed: int = 42,
    streaming_limit: int | None = None,
) -> Tuple[List[bytes], List[List[str]]]:
    """Load PE-Video video bytes and captions using efficient streaming.

    Uses streaming mode with reservoir sampling to efficiently sample from the
    dataset without downloading the entire ~1M video collection.

    Args:
        split: Which split to load ('test' or 'train').
        max_samples: Maximum number of samples to load. Defaults to 1024.
        num_captions: Number of captions per video (PE-Video has 1 human caption).
        auto_download: If True, auto-download from HuggingFace (unused with streaming).
        seed: Random seed for reproducible sampling.
        streaming_limit: Maximum number of valid samples to stream through before
            stopping. If None, streams through entire dataset for true random
            sampling. Set to e.g. 10000 for faster (but less random) sampling.

    Returns:
        Tuple of (video_bytes_list, captions) where:
        - video_bytes_list: List of video file bytes
        - captions: List of caption lists, captions[i] has caption strings
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets is required to load PE-Video. "
            "Install it with: pip install datasets"
        )

    if num_captions > 1:
        logger.warning(
            f"num_captions={num_captions} requested but PE-Video has 1 human caption. "
            "Will include description as second caption if available."
        )

    logger.info(
        f"Loading PE-Video {split} split with streaming (sampling {max_samples} videos)"
    )

    # Use streaming mode to avoid downloading the entire dataset
    ds = load_dataset("facebook/PE-Video", split=split, streaming=True)

    # Set up reservoir sampling for random selection
    random.seed(seed)
    reservoir: List[dict] = []
    seen_count = 0

    logger.info("Streaming dataset with reservoir sampling...")

    for row in ds:
        # PE-Video stores metadata in a nested 'json' field
        metadata = row.get("json", {})
        if not isinstance(metadata, dict):
            continue

        # Skip samples without valid human captions
        human_caption = metadata.get("human_caption")
        if not human_caption or not str(human_caption).strip():
            continue

        # Skip samples without video data
        video_data = row.get("mp4")
        if video_data is None:
            continue

        seen_count += 1

        # Reservoir sampling algorithm (Algorithm R)
        if max_samples is None or len(reservoir) < max_samples:
            reservoir.append(row)
        else:
            # Replace elements with decreasing probability
            j = random.randint(0, seen_count - 1)
            if j < max_samples:
                reservoir[j] = row

        # Log progress periodically
        if seen_count % 1000 == 0:
            logger.debug(
                f"Processed {seen_count} valid samples, reservoir size: {len(reservoir)}"
            )

        # Early exit if streaming_limit is set and we've seen enough samples
        if streaming_limit is not None and seen_count >= streaming_limit:
            logger.info(f"Reached streaming limit of {streaming_limit} samples")
            break

    logger.info(
        f"Streamed {seen_count} valid samples, selected {len(reservoir)} for reservoir"
    )

    # Sort reservoir by caption for deterministic ordering across runs
    # This ensures reproducibility even if streaming order varies slightly
    reservoir.sort(key=lambda x: str(x.get("json", {}).get("human_caption", "")))

    # Extract video bytes and captions from reservoir
    video_bytes_list = []
    captions_list = []

    for idx, row in enumerate(reservoir):
        # Get video bytes
        video_data = row.get("mp4")

        # Handle different video data formats
        if isinstance(video_data, dict) and "bytes" in video_data:
            video_bytes = video_data["bytes"]
        elif isinstance(video_data, bytes):
            video_bytes = video_data
        else:
            logger.warning(
                f"Sample {idx} has unexpected video format: {type(video_data)}"
            )
            continue

        # Get metadata from nested json field
        metadata = row.get("json", {})

        # Get captions
        sample_captions = []
        human_caption = metadata.get("human_caption")
        if human_caption and str(human_caption).strip():
            sample_captions.append(str(human_caption).strip())

        # Add description as second caption if requested
        if num_captions > 1:
            description = metadata.get("description")
            if description and str(description).strip():
                sample_captions.append(str(description).strip())

        if not sample_captions:
            continue

        video_bytes_list.append(video_bytes)
        captions_list.append(sample_captions)

    logger.info(f"Loaded {len(video_bytes_list)} videos with captions")
    return video_bytes_list, captions_list


def extract_video_frames(
    video_source: str | bytes,
    *,
    num_frames: int = 8,
    strategy: str = "uniform",
    size: Tuple[int, int] | None = None,
) -> List[Image.Image]:
    """Extract frames from a video file or bytes.

    Uses decord for fast video decoding, with opencv as fallback.

    Args:
        video_source: Path to video file or video bytes.
        num_frames: Number of frames to extract.
        strategy: Frame selection strategy:
            - "uniform": Evenly spaced frames across the video
            - "start": First N frames
            - "middle": N frames from the middle
        size: Optional (width, height) to resize frames.

    Returns:
        List of PIL Images.

    Raises:
        ImportError: If neither decord nor opencv is available.
        FileNotFoundError: If video file doesn't exist.
        ValueError: If video cannot be decoded.
    """
    if strategy not in {"uniform", "start", "middle"}:
        raise ValueError(
            f"Unknown strategy: {strategy}. Use 'uniform', 'start', or 'middle'"
        )

    # Try decord first (faster)
    try:
        return _extract_frames_decord(video_source, num_frames, strategy, size)
    except ImportError:
        pass

    # Fall back to opencv
    try:
        return _extract_frames_opencv(video_source, num_frames, strategy, size)
    except ImportError:
        raise ImportError(
            "Video frame extraction requires either decord or opencv. "
            "Install with: pip install decord  OR  pip install opencv-python"
        )


def _extract_frames_decord(
    video_source: str | bytes,
    num_frames: int,
    strategy: str,
    size: Tuple[int, int] | None,
) -> List[Image.Image]:
    """Extract frames using decord (fast GPU-accelerated decoding)."""
    from decord import VideoReader, cpu

    # Handle bytes input
    if isinstance(video_source, bytes):
        vr = VideoReader(io.BytesIO(video_source), ctx=cpu(0))
    else:
        vr = VideoReader(video_source, ctx=cpu(0))

    total_frames = len(vr)

    if total_frames == 0:
        raise ValueError("Video has no frames")

    # Calculate frame indices based on strategy
    frame_indices = _get_frame_indices(total_frames, num_frames, strategy)

    # Extract frames
    frames = vr.get_batch(frame_indices).asnumpy()

    # Convert to PIL Images
    images = []
    for frame in frames:
        img = Image.fromarray(frame)
        if size is not None:
            img = img.resize(size, Image.Resampling.BILINEAR)
        images.append(img)

    return images


def _extract_frames_opencv(
    video_source: str | bytes,
    num_frames: int,
    strategy: str,
    size: Tuple[int, int] | None,
) -> List[Image.Image]:
    """Extract frames using opencv (fallback, slower)."""
    # Handle bytes input - need to write to temp file for opencv
    if isinstance(video_source, bytes):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_source)
            temp_path = f.name
        try:
            return _extract_frames_opencv_from_path(
                temp_path, num_frames, strategy, size
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)
    else:
        return _extract_frames_opencv_from_path(
            video_source, num_frames, strategy, size
        )


def _extract_frames_opencv_from_path(
    video_path: str,
    num_frames: int,
    strategy: str,
    size: Tuple[int, int] | None,
) -> List[Image.Image]:
    """Extract frames from a video file path using opencv."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    # Calculate frame indices based on strategy
    frame_indices = _get_frame_indices(total_frames, num_frames, strategy)

    images = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            warnings.warn(f"Failed to read frame {idx} from {video_path}")
            continue

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        if size is not None:
            img = img.resize(size, Image.Resampling.BILINEAR)

        images.append(img)

    cap.release()
    return images


def _get_frame_indices(
    total_frames: int,
    num_frames: int,
    strategy: str,
) -> List[int]:
    """Calculate frame indices based on sampling strategy.

    Args:
        total_frames: Total number of frames in the video.
        num_frames: Number of frames to sample.
        strategy: Sampling strategy ("uniform", "start", "middle").

    Returns:
        List of frame indices to extract.
    """
    import numpy as np

    # Cap num_frames at total_frames
    num_frames = min(num_frames, total_frames)

    if strategy == "uniform":
        # Evenly spaced frames across the video
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    elif strategy == "start":
        # First N frames
        indices = np.arange(num_frames)
    elif strategy == "middle":
        # N frames from the middle
        start = max(0, (total_frames - num_frames) // 2)
        indices = np.arange(start, start + num_frames)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return indices.tolist()


def iter_pvd_samples(
    *,
    split: str = "test",
    max_samples: int | None = DEFAULT_NUM_SAMPLES,
    num_captions: int = 1,
    num_frames: int = 8,
    frame_strategy: str = "uniform",
    auto_download: bool = True,
    seed: int = 42,
) -> Iterator[Tuple[List[Image.Image], List[str]]]:
    """Iterate over PE-Video samples, yielding (frames, captions) tuples.

    Args:
        split: Which split to load.
        max_samples: Maximum number of samples to yield.
        num_captions: Number of captions per video.
        num_frames: Number of frames to extract per video.
        frame_strategy: Frame extraction strategy.
        auto_download: If True, auto-download from HuggingFace.
        seed: Random seed for reproducible sampling.

    Yields:
        Tuples of (frames, captions) where:
        - frames: List of PIL Images
        - captions: List of caption strings
    """
    video_bytes_list, captions_list = load_pvd_dataset(
        split=split,
        max_samples=max_samples,
        num_captions=num_captions,
        auto_download=auto_download,
        seed=seed,
    )

    for video_bytes, captions in zip(video_bytes_list, captions_list):
        try:
            frames = extract_video_frames(
                video_bytes,
                num_frames=num_frames,
                strategy=frame_strategy,
            )
            yield frames, captions
        except Exception as e:
            logger.warning(f"Failed to extract frames: {e}")
            continue


def get_pvd_path(cache_dir: str | Path | None = None) -> Optional[Path]:
    """Get the path to cached PE-Video data if it exists.

    Args:
        cache_dir: Directory to check. Defaults to ~/.cache/pevideo

    Returns:
        Path to the dataset if found, None otherwise.
    """
    if cache_dir is None:
        cache_dir = PVD_CACHE_DIR
    cache_dir = Path(cache_dir)

    if cache_dir.exists():
        return cache_dir

    # Also check HuggingFace cache
    try:
        from datasets import load_dataset

        # Try to load from cache without downloading
        load_dataset(
            "facebook/PE-Video",
            split="test",
            streaming=True,  # Use streaming to avoid full download
        )
        return cache_dir
    except Exception:
        pass

    return None
