"""PRH path helpers."""

from __future__ import annotations

import os


def prh_feature_filename(
    output_dir: str,
    dataset: str,
    subset: str,
    model_name: str,
    *,
    pool: str | None,
    prompt: bool | None,
    caption_idx: int | None,
) -> str:
    save_name = model_name.replace("/", "_")
    if pool:
        save_name += f"_pool-{pool}"
    if prompt is not None:
        save_name += f"_prompt-{prompt}"
    if caption_idx is not None:
        save_name += f"_cid-{caption_idx}"
    return os.path.join(output_dir, dataset, subset, f"{save_name}.pt")


def prh_alignment_filename(
    output_dir: str,
    dataset: str,
    modelset: str,
    modality_x: str,
    pool_x: str | None,
    prompt_x: bool,
    modality_y: str,
    pool_y: str | None,
    prompt_y: bool,
    metric: str,
    topk: int,
) -> str:
    return os.path.join(
        output_dir,
        dataset,
        modelset,
        f"{modality_x}_pool-{pool_x}_prompt-{prompt_x}_{modality_y}_pool-{pool_y}_prompt-{prompt_y}",
        f"{metric}_k{topk}.npy" if "knn" in metric else f"{metric}.npy",
    )
