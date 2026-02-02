"""VATEX dataset utilities for video-to-text alignment experiments.

This module provides data loading utilities for the VATEX dataset,
following the VideoPRH paper's methodology for video-text alignment.

VATEX (Video and Text) contains ~41K videos with 10 English captions each.
Dataset: https://huggingface.co/datasets/open-vision-language/vatex

Key specifications from VideoPRH paper:
- 1024 videos sampled from validation set
- 10 English captions per video
- ~15 words average per caption
- ~10 seconds average video length
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

# Default cache directory for VATEX data
VATEX_CACHE_DIR = Path.home() / ".cache" / "vatex"

# Default number of samples for VideoPRH experiments
DEFAULT_NUM_SAMPLES = 1024

# Number of English captions per video in VATEX
VATEX_NUM_CAPTIONS = 10


def load_vatex_dataset(
    *,
    split: str = "validation",
    max_samples: int | None = DEFAULT_NUM_SAMPLES,
    num_captions: int = VATEX_NUM_CAPTIONS,
    seed: int = 42,
    streaming_limit: int | None = None,
) -> Tuple[List[bytes], List[List[str]]]:
    """Load VATEX video bytes and captions.

    Uses streaming mode with reservoir sampling to efficiently sample from the
    dataset.

    Args:
        split: Which split to load ('validation' or 'train').
        max_samples: Maximum number of samples to load. Defaults to 1024.
        num_captions: Number of captions per video (VATEX has 10 English captions).
        seed: Random seed for reproducible sampling.
        streaming_limit: Maximum number of valid samples to stream through before
            stopping. If None, streams through entire dataset.

    Returns:
        Tuple of (video_bytes_list, captions) where:
        - video_bytes_list: List of video file bytes
        - captions: List of caption lists, captions[i] has num_captions strings
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets is required to load VATEX. "
            "Install it with: pip install datasets"
        )

    logger.info(
        f"Loading VATEX {split} split with streaming (sampling {max_samples} videos)"
    )

    # Load VATEX from HuggingFace
    ds = load_dataset("open-vision-language/vatex", split=split, streaming=True)

    # Set up reservoir sampling for random selection
    random.seed(seed)
    reservoir: List[dict] = []
    seen_count = 0

    logger.info("Streaming dataset with reservoir sampling...")

    for row in ds:
        # Check for English captions
        enCap = row.get("enCap")
        if not enCap or len(enCap) < num_captions:
            continue

        # Check for video data
        video_data = row.get("video")
        if video_data is None:
            continue

        seen_count += 1

        # Reservoir sampling algorithm (Algorithm R)
        if max_samples is None or len(reservoir) < max_samples:
            reservoir.append(row)
        else:
            j = random.randint(0, seen_count - 1)
            if j < max_samples:
                reservoir[j] = row

        # Log progress periodically
        if seen_count % 500 == 0:
            logger.debug(
                f"Processed {seen_count} valid samples, reservoir size: {len(reservoir)}"
            )

        # Early exit if streaming_limit is set
        if streaming_limit is not None and seen_count >= streaming_limit:
            logger.info(f"Reached streaming limit of {streaming_limit} samples")
            break

    logger.info(
        f"Streamed {seen_count} valid samples, selected {len(reservoir)} for reservoir"
    )

    # Sort reservoir by video ID for deterministic ordering
    reservoir.sort(key=lambda x: str(x.get("videoID", "")))

    # Extract video bytes and captions from reservoir
    video_bytes_list = []
    captions_list = []

    for idx, row in enumerate(reservoir):
        # Get video bytes
        video_data = row.get("video")

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

        # Get English captions (take first num_captions)
        enCap = row.get("enCap", [])
        sample_captions = [str(cap).strip() for cap in enCap[:num_captions]]

        if len(sample_captions) < num_captions:
            logger.warning(
                f"Sample {idx} has only {len(sample_captions)} captions, "
                f"expected {num_captions}"
            )
            # Pad with duplicates if needed
            while len(sample_captions) < num_captions:
                sample_captions.append(sample_captions[-1])

        video_bytes_list.append(video_bytes)
        captions_list.append(sample_captions)

    logger.info(
        f"Loaded {len(video_bytes_list)} videos with {num_captions} captions each"
    )
    return video_bytes_list, captions_list


def get_vatex_video_paths(
    *,
    split: str = "validation",
    max_samples: int | None = DEFAULT_NUM_SAMPLES,
    cache_dir: str | Path | None = None,
    seed: int = 42,
) -> List[str]:
    """Get video file paths from VATEX dataset.

    Note: This function downloads videos to a local cache if needed.

    Args:
        split: Which split to use ('validation' or 'train').
        max_samples: Maximum number of video paths to return.
        cache_dir: Directory to cache videos. Defaults to ~/.cache/vatex
        seed: Random seed for reproducible sampling.

    Returns:
        List of video file paths.
    """
    if cache_dir is None:
        cache_dir = VATEX_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    video_bytes_list, _ = load_vatex_dataset(
        split=split, max_samples=max_samples, seed=seed
    )

    video_paths = []
    for idx, video_bytes in enumerate(video_bytes_list):
        video_path = cache_dir / f"vatex_{split}_{idx:05d}.mp4"
        if not video_path.exists():
            video_path.write_bytes(video_bytes)
        video_paths.append(str(video_path))

    return video_paths


def get_vatex_captions(
    *,
    split: str = "validation",
    max_samples: int | None = DEFAULT_NUM_SAMPLES,
    num_captions: int = VATEX_NUM_CAPTIONS,
    seed: int = 42,
) -> List[List[str]]:
    """Get captions from VATEX dataset.

    Args:
        split: Which split to use ('validation' or 'train').
        max_samples: Maximum number of samples.
        num_captions: Number of captions per video.
        seed: Random seed for reproducible sampling.

    Returns:
        List of caption lists, where each inner list has num_captions strings.
    """
    _, captions_list = load_vatex_dataset(
        split=split,
        max_samples=max_samples,
        num_captions=num_captions,
        seed=seed,
    )
    return captions_list


def get_vatex_path(cache_dir: str | Path | None = None) -> Optional[Path]:
    """Get the path to cached VATEX data if it exists.

    Args:
        cache_dir: Directory to check. Defaults to ~/.cache/vatex

    Returns:
        Path to the dataset cache if found, None otherwise.
    """
    if cache_dir is None:
        cache_dir = VATEX_CACHE_DIR
    cache_dir = Path(cache_dir)

    if cache_dir.exists() and any(cache_dir.iterdir()):
        return cache_dir

    return None
