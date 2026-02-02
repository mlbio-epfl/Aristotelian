"""Places365 dataset utilities for vision-to-vision alignment experiments.

This module provides data loading utilities for the Places365 dataset,
following the PRH paper's vision-vision convergence experiments (Appendix C.1).

The validation set is downloaded from Kaggle:
https://www.kaggle.com/datasets/benjaminkz/places365
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional

from loguru import logger
from PIL import Image

# Default cache directory for Places365 data
PLACES365_CACHE_DIR = Path.home() / ".cache" / "places365"


def download_places365_kaggle(
    cache_dir: str | Path | None = None,
) -> Path:
    """Download Places365 validation set from Kaggle.

    Args:
        cache_dir: Directory to cache the dataset. Defaults to ~/.cache/places365

    Returns:
        Path to the downloaded dataset directory.

    Note:
        Requires kagglehub to be installed: pip install kagglehub
        You may need to authenticate with Kaggle credentials.
    """
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "kagglehub is required to download Places365 from Kaggle. "
            "Install it with: pip install kagglehub"
        )

    if cache_dir is None:
        cache_dir = PLACES365_CACHE_DIR
    cache_dir = Path(cache_dir)

    # Check if already downloaded
    val_dir = cache_dir / "val_256"
    if val_dir.exists() and any(val_dir.rglob("*.jpg")):
        logger.info(f"Places365 already cached at {cache_dir}")
        return cache_dir

    logger.info("Downloading Places365 validation set from Kaggle...")
    logger.info("This may take a while on first run (~1GB download)")

    # Download using kagglehub
    # The dataset is at: benjaminkz/places365
    try:
        download_path = kagglehub.dataset_download("benjaminkz/places365")
        download_path = Path(download_path)
        logger.info(f"Downloaded to: {download_path}")

        # The kagglehub downloads to its own cache, we'll use that directly
        # Check the structure - Kaggle dataset uses 'val/' not 'val_256/'
        if (download_path / "val").exists():
            return download_path
        elif (download_path / "val_256").exists():
            return download_path
        elif (download_path / "places365" / "val").exists():
            return download_path / "places365"
        else:
            # Search for val or val_256 in the download
            for dirname in ["val", "val_256"]:
                for p in download_path.rglob(dirname):
                    if p.is_dir():
                        return p.parent
            raise FileNotFoundError(
                f"Could not find val or val_256 directory in downloaded data at {download_path}"
            )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download Places365 from Kaggle: {e}\n"
            "Make sure you have kagglehub installed and Kaggle credentials configured.\n"
            "See: https://www.kaggle.com/docs/api#authentication"
        )


def load_places365_dataset(
    *,
    root: str | Path | None = None,
    split: str = "val",
    max_samples: int | None = None,
    auto_download: bool = True,
) -> List[Image.Image]:
    """Load Places365 images from disk.

    Args:
        root: Root directory containing Places365 dataset.
            If None and auto_download=True, downloads from Kaggle.
            Expected structure:
            - {root}/val_256/ for validation set
        split: Which split to load ('val' only supported).
        max_samples: Maximum number of samples to load.
        auto_download: If True and root is None, auto-download from Kaggle.

    Returns:
        List of PIL images.
    """
    if split != "val":
        raise ValueError(
            f"Only 'val' split is supported (got: {split}). "
            "The Kaggle dataset only contains the validation set."
        )

    if root is None:
        if auto_download:
            root = download_places365_kaggle()
        else:
            raise ValueError(
                "root must be provided when auto_download=False. "
                "Either provide a path to Places365 data or set auto_download=True."
            )

    root = Path(root)

    # Try both 'val' (Kaggle) and 'val_256' (official) directory names
    if (root / "val").exists():
        image_dir = root / "val"
    elif (root / "val_256").exists():
        image_dir = root / "val_256"
    else:
        raise FileNotFoundError(
            f"Places365 image directory not found: {root}/val or {root}/val_256\n"
            f"Please download Places365 from Kaggle: "
            f"https://www.kaggle.com/datasets/benjaminkz/places365"
        )

    images = []
    count = 0
    image_paths = sorted(image_dir.rglob("*.jpg"))
    total = (
        len(image_paths) if max_samples is None else min(max_samples, len(image_paths))
    )
    logger.info(f"Loading {total} images from {image_dir}")

    for img_path in image_paths:
        if max_samples is not None and count >= max_samples:
            break
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            count += 1
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            continue

    logger.info(f"Loaded {len(images)} images")
    return images


def iter_places365_samples(
    *,
    root: str | Path | None = None,
    split: str = "val",
    max_samples: int | None = None,
    auto_download: bool = True,
) -> Iterator[Image.Image]:
    """Iterate over Places365 images.

    Args:
        root: Root directory containing Places365 dataset.
            If None and auto_download=True, downloads from Kaggle.
        split: Which split to load ('val' only supported).
        max_samples: Maximum number of samples to yield.
        auto_download: If True and root is None, auto-download from Kaggle.

    Yields:
        PIL images one at a time.
    """
    if split != "val":
        raise ValueError(
            f"Only 'val' split is supported (got: {split}). "
            "The Kaggle dataset only contains the validation set."
        )

    if root is None:
        if auto_download:
            root = download_places365_kaggle()
        else:
            raise ValueError(
                "root must be provided when auto_download=False. "
                "Either provide a path to Places365 data or set auto_download=True."
            )

    root = Path(root)

    # Try both 'val' (Kaggle) and 'val_256' (official) directory names
    if (root / "val").exists():
        image_dir = root / "val"
    elif (root / "val_256").exists():
        image_dir = root / "val_256"
    else:
        raise FileNotFoundError(
            f"Places365 image directory not found: {root}/val or {root}/val_256\n"
            f"Please download Places365 from Kaggle: "
            f"https://www.kaggle.com/datasets/benjaminkz/places365"
        )

    count = 0
    for img_path in sorted(image_dir.rglob("*.jpg")):
        if max_samples is not None and count >= max_samples:
            break
        try:
            img = Image.open(img_path).convert("RGB")
            yield img
            count += 1
        except Exception:
            continue


def get_places365_path(cache_dir: str | Path | None = None) -> Optional[Path]:
    """Get the path to cached Places365 data if it exists.

    Args:
        cache_dir: Directory to check. Defaults to ~/.cache/places365

    Returns:
        Path to the dataset if found, None otherwise.
    """
    if cache_dir is None:
        cache_dir = PLACES365_CACHE_DIR
    cache_dir = Path(cache_dir)

    # Check for both 'val' (Kaggle) and 'val_256' (official) directory names
    if (cache_dir / "val").exists():
        return cache_dir
    if (cache_dir / "val_256").exists():
        return cache_dir

    # Also check kagglehub cache
    try:
        import kagglehub  # noqa: F401

        # kagglehub stores in ~/.cache/kagglehub/datasets
        kaggle_cache = Path.home() / ".cache" / "kagglehub" / "datasets"
        if kaggle_cache.exists():
            for dirname in ["val", "val_256"]:
                for p in kaggle_cache.rglob(dirname):
                    if p.is_dir() and any(p.rglob("*.jpg")):
                        return p.parent
    except ImportError:
        pass

    return None
