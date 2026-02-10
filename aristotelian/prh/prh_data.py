"""PRH dataset utilities (WIT variants)."""

from __future__ import annotations

from typing import Iterable, Iterator, Optional

from datasets import load_dataset

DATASET_NAME = "minhuh/prh"


def load_prh_dataset(
    *,
    config: str = "default",
    split: str = "train",
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    revision: Optional[str] = None,
    data_dir: Optional[str] = None,
):
    """Load the PRH dataset split."""
    return load_dataset(
        DATASET_NAME,
        config,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
        revision=revision,
        data_dir=data_dir,
    )


def iter_prh_samples(
    dataset: Iterable,
    *,
    text_key: str | None = None,
    image_key: str | None = None,
) -> Iterator[dict]:
    """Yield samples with canonical {text, image} keys."""
    for row in dataset:
        if text_key is None:
            text_key = "text" if "text" in row else "caption"
        if image_key is None:
            image_key = "image" if "image" in row else "jpg"
        yield {
            "text": row.get(text_key),
            "image": row.get(image_key),
        }
