"""
Vision-to-Vision alignment experiment following PRH paper Appendix C.1.

This module implements the vision model convergence experiment from the Platonic
Representation Hypothesis paper, showing that vision models with different
architectures and training objectives converge to aligned representations.

Key findings from PRH paper:
- Models that solve more VTAB tasks tend to be more aligned with each other
- More competent and general models have more similar representations
- VISION models converge as COMPETENCE increases

This uses Places365 dataset following the PRH paper methodology.
"""

from __future__ import annotations

import gc
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from ..experiments.multiple import bh_fdr
from ..utils.logging import ExperimentState, log_timing
from .cache import _cached_feats_match, build_gram_cache, build_knn_cache_with_indices
from .layers import _as_layers, _normalize_layers
from .paths import prh_alignment_filename, prh_feature_filename
from .places365_data import load_places365_dataset
from .preprocess import prepare_features
from .prh_experiment import (
    _prh_compute_pair_cka_cached,
    _prh_compute_pair_cknna_cached,
    _prh_compute_pair_cycle_knn_cached,
    _prh_compute_pair_generic,
    _prh_compute_pair_knn_cached,
    _prh_compute_pair_procrustes_cached,
    _prh_compute_pair_pwcca_cached,
    _prh_compute_pair_svcca_cached,
    _stack_layers,
)
from .prh_models import get_vision_models_v2v, load_vision_model
from .prh_pipeline import collect_vision_activations


def _extract_vision_features_v2v(
    images: Iterable,
    *,
    model_name: str,
    device: str,
    batch_size: int,
    pool: str,
) -> Tuple[torch.Tensor, int]:
    """Extract features from a vision model for v2v alignment."""
    model = load_vision_model(model_name, device=device)
    num_params = int(sum(p.numel() for p in model.parameters()))
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )

    # Determine return nodes based on model architecture
    if "vit" in model_name.lower():
        if hasattr(model, "blocks"):
            return_nodes = [f"blocks.{i}.add_1" for i in range(len(model.blocks))]
        else:
            # Fallback for models without blocks attribute
            return_nodes = None
    else:
        return_nodes = None

    if return_nodes:
        feat_model = create_feature_extractor(model, return_nodes=return_nodes)
    else:
        feat_model = model

    layers = collect_vision_activations(
        images,
        model=feat_model,
        device=device,
        batch_size=batch_size,
        pool=pool,
        transform=transform,
    )
    return _stack_layers(layers), num_params


def run_v2v_experiment(
    *,
    dataset: str = "places365",
    data_dir: Optional[str] = None,
    split: str = "val",
    modelset: str = "default",
    pool: str = "cls",
    max_samples: int | None = 1000,
    batch_size: int = 4,
    device: str = "cpu",
    fallback_device: str = "cpu",
    output_dir: str = "./results",
    k: int = 10,
    metric: str = "mutual_knn",
    rbf_sigma: float = 1.0,
    num_permutations: int = 200,
    alpha: float = 0.05,
    q_outlier: float = 0.95,
    force: bool = False,
    force_features: bool = False,
    num_workers: int = 1,
) -> Dict[str, np.ndarray]:
    """Run vision-to-vision alignment experiment following PRH paper C.1.

    This experiment measures alignment between different vision models using
    the Places365 dataset, following the methodology from the Platonic
    Representation Hypothesis paper.

    Args:
        dataset: Dataset name ('places365').
        data_dir: Directory containing Places365 images. If None, auto-downloads
            from Kaggle (requires kagglehub: pip install kagglehub).
        split: Dataset split ('val' only - the Kaggle dataset only has validation).
        modelset: Which vision model set to use ('default', 'small', etc.).
        pool: Pooling method for vision features ('cls' or 'mean').
        max_samples: Maximum number of samples to use.
        batch_size: Batch size for feature extraction.
        device: Device for computation ('cpu' or 'cuda').
        fallback_device: Fallback device if OOM occurs.
        output_dir: Directory to save results.
        k: Number of nearest neighbors for KNN-based metrics.
        metric: Alignment metric to use.
        num_permutations: Number of permutations for null distribution.
        alpha: Significance level.
        q_outlier: Quantile for outlier clipping.
        force: Force recomputation even if results exist.
        force_features: Force recomputation of features.
        num_workers: Number of parallel workers.

    Returns:
        Dictionary containing alignment scores, p-values, and gated scores.
    """
    if device != "cpu" and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    logger.info("=" * 80)
    logger.info("Vision-to-Vision Alignment Experiment (PRH C.1)")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset} ({split} split)")
    logger.info(f"Modelset: {modelset}")
    logger.info(f"Pooling: {pool}")
    if metric in {"mutual_knn", "knn", "cycle_knn"}:
        logger.info(f"Metric: {metric} (k={k})")
    elif metric == "cka_rbf":
        logger.info(f"Metric: {metric} (sigma={rbf_sigma})")
    else:
        logger.info(f"Metric: {metric}")
    logger.info(f"Max samples: {max_samples}, permutations: {num_permutations}")
    logger.info(f"Outlier quantile: {q_outlier}")
    logger.info("=" * 80)

    # Load images (auto-downloads from Kaggle if data_dir not provided)
    with log_timing("Loading data"):
        images = load_places365_dataset(
            root=data_dir,
            split=split,
            max_samples=max_samples,
            auto_download=(data_dir is None),
        )

    # Get vision models
    vision_models = get_vision_models_v2v(modelset)
    logger.info(f"Vision models: {len(vision_models)}")
    for model in vision_models:
        logger.debug(f"  - {model}")

    # Extract features for all models
    logger.info(f"State: {ExperimentState.GENERATING_DATA} - extracting features")
    feats_list = []
    model_params = []

    for model_name in tqdm(vision_models, desc="Extracting vision features"):
        out_path = prh_feature_filename(
            os.path.join(output_dir, "features_v2v"),
            dataset,
            f"{split}_{max_samples}",
            model_name,
            pool=pool,
            prompt=False,
            caption_idx=None,
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.exists(out_path) and not force_features:
            payload = torch.load(out_path, map_location=device)
            if _cached_feats_match(payload, len(images)):
                feats_list.append(payload["feats"])
                model_params.append(payload.get("num_params", 0))
                continue
            logger.warning(f"Cached features mismatch for {model_name}; recomputing")

        try:
            feats, num_params = _extract_vision_features_v2v(
                images,
                model_name=model_name,
                device=device,
                batch_size=batch_size,
                pool=pool,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            logger.warning(
                f"OOM while extracting {model_name}; retrying on {fallback_device}"
            )
            torch.cuda.empty_cache()
            gc.collect()
            feats, num_params = _extract_vision_features_v2v(
                images,
                model_name=model_name,
                device=fallback_device,
                batch_size=batch_size,
                pool=pool,
            )

        torch.save({"feats": feats, "num_params": num_params}, out_path)
        feats_list.append(feats)
        model_params.append(num_params)

        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    logger.info(f"Extracted features for {len(feats_list)} models")

    # Compute pairwise alignment
    logger.info(f"State: {ExperimentState.COMPUTING_SIMILARITIES}")
    n_models = len(feats_list)
    scores = np.zeros((n_models, n_models))
    indices = np.zeros((n_models, n_models, 2), dtype=int)
    gated = np.zeros((n_models, n_models))
    pvals = np.zeros((n_models, n_models))
    taus = np.zeros((n_models, n_models))

    # Preprocess features
    prepped_list = [
        prepare_features(f, q=q_outlier, exact=False, device=device) for f in feats_list
    ]

    # Build caches based on metric type
    cka_cached_metrics = {"cka", "cka_lin", "unbiased_cka", "cka_rbf"}
    knn_cached_metrics = {"mutual_knn", "knn", "cycle_knn"}
    gram_cached_metrics = cka_cached_metrics | {"cknna"}

    layers_list = []
    knn_list = []
    masks_list = []
    grams_list = []

    if metric in knn_cached_metrics:
        for prepped in prepped_list:
            layers, knn_idx, masks = build_knn_cache_with_indices(
                prepped, topk=k, normalize=True
            )
            layers_list.append(layers)
            knn_list.append(knn_idx)
            masks_list.append(masks)
    elif metric in gram_cached_metrics:
        kernel = "rbf" if metric == "cka_rbf" else "linear"
        if metric == "cka_rbf":
            logger.info(
                f"Building Gram cache for {metric} (kernel={kernel}, sigma={rbf_sigma})"
            )
        else:
            logger.info(f"Building Gram cache for {metric} (kernel={kernel})")
        for prepped in prepped_list:
            grams_list.append(
                build_gram_cache(
                    prepped, normalize=True, kernel=kernel, rbf_sigma=rbf_sigma
                )
            )
    else:
        for prepped in prepped_list:
            layers_list.append(_normalize_layers(_as_layers(prepped)))

    # Build configs for parallel execution
    if metric in {"mutual_knn", "knn"}:
        configs = [
            (i, j, knn_list[i], knn_list[j], k, num_permutations, alpha)
            for i in range(n_models)
            for j in range(n_models)
        ]
        compute_fn = _prh_compute_pair_knn_cached
    elif metric == "cycle_knn":
        configs = [
            (i, j, knn_list[i], knn_list[j], num_permutations, alpha)
            for i in range(n_models)
            for j in range(n_models)
        ]
        compute_fn = _prh_compute_pair_cycle_knn_cached
    elif metric in cka_cached_metrics:
        use_unbiased = metric == "unbiased_cka"
        configs = [
            (i, j, grams_list[i], grams_list[j], num_permutations, alpha, use_unbiased)
            for i in range(n_models)
            for j in range(n_models)
        ]
        compute_fn = _prh_compute_pair_cka_cached
    elif metric == "cknna":
        configs = [
            (i, j, grams_list[i], grams_list[j], k, num_permutations, alpha)
            for i in range(n_models)
            for j in range(n_models)
        ]
        compute_fn = _prh_compute_pair_cknna_cached
    elif metric == "svcca":
        configs = [
            (i, j, layers_list[i], layers_list[j], num_permutations, alpha)
            for i in range(n_models)
            for j in range(n_models)
        ]
        compute_fn = _prh_compute_pair_svcca_cached
    elif metric == "pwcca":
        configs = [
            (i, j, layers_list[i], layers_list[j], num_permutations, alpha)
            for i in range(n_models)
            for j in range(n_models)
        ]
        compute_fn = _prh_compute_pair_pwcca_cached
    elif metric == "procrustes":
        configs = [
            (i, j, layers_list[i], layers_list[j], num_permutations, alpha)
            for i in range(n_models)
            for j in range(n_models)
        ]
        compute_fn = _prh_compute_pair_procrustes_cached
    else:
        configs = [
            (i, j, layers_list[i], layers_list[j], metric, k, num_permutations, alpha)
            for i in range(n_models)
            for j in range(n_models)
        ]
        compute_fn = _prh_compute_pair_generic

    total_pairs = len(configs)
    if num_workers > 1 and device == "cpu":
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results_iter = executor.map(compute_fn, configs)
            for i, j, res in tqdm(
                results_iter, total=total_pairs, desc=f"Computing {metric}"
            ):
                scores[i, j] = res["raw_score"]
                indices[i, j] = res["best_indices"]
                gated[i, j] = res["g_score"]
                pvals[i, j] = res["p_value"]
                taus[i, j] = res["tau_alpha"]
    else:
        for config in tqdm(configs, desc=f"Computing {metric}"):
            i, j, res = compute_fn(config)
            scores[i, j] = res["raw_score"]
            indices[i, j] = res["best_indices"]
            gated[i, j] = res["g_score"]
            pvals[i, j] = res["p_value"]
            taus[i, j] = res["tau_alpha"]

    # FDR correction
    flat_p = pvals.reshape(-1)
    fdr_threshold, fdr_mask_flat = bh_fdr(flat_p, alpha=alpha)
    fdr_mask = fdr_mask_flat.reshape(pvals.shape)

    logger.info(f"Computed {n_models} x {n_models} = {n_models ** 2} alignments")

    # Save results
    logger.info(f"State: {ExperimentState.SAVING_RESULTS}")
    save_path = prh_alignment_filename(
        os.path.join(output_dir, "alignment_v2v"),
        dataset,
        modelset,
        "vision",
        pool,
        False,
        "vision",
        pool,
        False,
        metric,
        k,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    payload = {
        "scores": scores,
        "indices": indices,
        "gated": gated,
        "pvalues": pvals,
        "taus": taus,
        "metric": metric,
        "fdr_threshold": fdr_threshold,
        "fdr_mask": fdr_mask,
        "model_names": vision_models,
        "model_params": model_params,
    }
    np.save(save_path, payload)
    logger.success(f"Results saved to {save_path}")

    return payload


def compute_intra_bucket_alignment(
    scores: np.ndarray,
    model_names: List[str],
    *,
    num_buckets: int = 5,
) -> Dict:
    """Compute intra-bucket alignment as in PRH Figure 2.

    Groups models by training objective and computes average alignment
    within each group. Also computes bucket-level statistics by dividing
    models into buckets based on their mean alignment with other models.

    Args:
        scores: Pairwise alignment scores matrix [n_models, n_models].
        model_names: List of model names.
        num_buckets: Number of buckets to split models into.

    Returns:
        Dictionary with bucket statistics including:
        - categories: List of category labels for each model
        - intra_alignment: Dict of mean intra-category alignment
        - inter_alignment: Dict of mean inter-category alignment
        - unique_categories: List of unique category labels
        - bucket_means: Mean alignment within each bucket
        - bucket_stds: Std of alignment within each bucket
        - bucket_sizes: Number of models in each bucket
    """
    n = len(model_names)

    # Categorize models by training objective
    categories = []
    for name in model_names:
        if "dinov2" in name.lower():
            categories.append("DINOv2")
        elif "clip" in name.lower() and "ft_in" in name.lower():
            categories.append("CLIP (finetuned)")
        elif "clip" in name.lower():
            categories.append("CLIP")
        elif "mae" in name.lower():
            categories.append("MAE")
        elif "in21k" in name.lower() or "in1k" in name.lower():
            categories.append("Classification")
        else:
            categories.append("Other")

    unique_cats = sorted(set(categories))
    cat_indices = {cat: [] for cat in unique_cats}
    for i, cat in enumerate(categories):
        cat_indices[cat].append(i)

    # Compute intra-category alignment
    intra_alignment = {}
    for cat, idxs in cat_indices.items():
        if len(idxs) > 1:
            sub_scores = scores[np.ix_(idxs, idxs)]
            # Exclude diagonal
            mask = ~np.eye(len(idxs), dtype=bool)
            intra_alignment[cat] = float(np.mean(sub_scores[mask]))
        else:
            intra_alignment[cat] = float("nan")

    # Compute inter-category alignment
    inter_alignment = {}
    for cat1 in unique_cats:
        for cat2 in unique_cats:
            if cat1 < cat2:
                idxs1 = cat_indices[cat1]
                idxs2 = cat_indices[cat2]
                sub_scores = scores[np.ix_(idxs1, idxs2)]
                inter_alignment[f"{cat1}_vs_{cat2}"] = float(np.mean(sub_scores))

    # Compute bucket-level statistics (as in PRH Figure 2)
    # Use mean alignment with other models as a proxy for "competence"
    if n > 1:
        # Mean alignment of each model with all other models (excluding self)
        mask = ~np.eye(n, dtype=bool)
        model_mean_alignment = np.array([np.mean(scores[i, mask[i]]) for i in range(n)])
        # Sort models by mean alignment and divide into buckets
        sorted_indices = np.argsort(model_mean_alignment)
        bucket_size = max(1, n // num_buckets)

        bucket_means = []
        bucket_stds = []
        bucket_sizes = []

        for b in range(num_buckets):
            start_idx = b * bucket_size
            if b == num_buckets - 1:
                # Last bucket gets remaining models
                end_idx = n
            else:
                end_idx = start_idx + bucket_size

            bucket_indices = sorted_indices[start_idx:end_idx]
            if len(bucket_indices) > 1:
                # Compute mean alignment within this bucket
                sub_scores = scores[np.ix_(bucket_indices, bucket_indices)]
                sub_mask = ~np.eye(len(bucket_indices), dtype=bool)
                bucket_vals = sub_scores[sub_mask]
                bucket_means.append(float(np.mean(bucket_vals)))
                bucket_stds.append(float(np.std(bucket_vals)))
            elif len(bucket_indices) == 1:
                bucket_means.append(float("nan"))
                bucket_stds.append(float("nan"))
            else:
                continue
            bucket_sizes.append(len(bucket_indices))
    else:
        bucket_means = [float("nan")] * num_buckets
        bucket_stds = [float("nan")] * num_buckets
        bucket_sizes = [0] * num_buckets

    return {
        "categories": categories,
        "intra_alignment": intra_alignment,
        "inter_alignment": inter_alignment,
        "unique_categories": unique_cats,
        "bucket_means": np.array(bucket_means),
        "bucket_stds": np.array(bucket_stds),
        "bucket_sizes": np.array(bucket_sizes),
    }
