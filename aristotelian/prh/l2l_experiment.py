"""
Language-to-Language alignment experiment following PRH paper methodology.

This module implements language model convergence experiments, showing that
language models with different architectures converge to aligned representations.

This reuses features from the cross-modal PRH experiment (WIT dataset captions).
"""

from __future__ import annotations

import gc
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from ..experiments.multiple import bh_fdr
from ..utils.logging import ExperimentState
from .cache import build_gram_cache, build_knn_cache_with_indices
from .layers import _as_layers, _normalize_layers
from .paths import prh_alignment_filename, prh_feature_filename
from .preprocess import prepare_features
from .prh_experiment import (
    _extract_text_features,
    _load_prh_samples,
    _prh_compute_pair_cka_cached,
    _prh_compute_pair_cknna_cached,
    _prh_compute_pair_cycle_knn_cached,
    _prh_compute_pair_generic,
    _prh_compute_pair_knn_cached,
    _prh_compute_pair_procrustes_cached,
    _prh_compute_pair_pwcca_cached,
    _prh_compute_pair_svcca_cached,
)
from .prh_models import get_models


def run_l2l_experiment(
    *,
    dataset: str = "minhuh/prh",
    subset: str = "wit_1024",
    split: str = "train",
    modelset: str = "val",
    pool: str = "avg",
    prompt: bool = False,
    caption_idx: int = 0,
    max_samples: int | None = None,
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
    """Run language-to-language alignment experiment.

    This experiment measures alignment between different language models using
    the WIT dataset captions (same as PRH cross-modal experiment), following
    the Platonic Representation Hypothesis methodology.

    Args:
        dataset: HuggingFace dataset name.
        subset: Dataset subset to use.
        split: Dataset split.
        modelset: Which language model set to use ('val', 'test', 'custom').
        pool: Pooling method for language features ('avg' or 'last').
        prompt: Whether to use prompted inputs.
        caption_idx: Which caption to use (for multi-caption datasets).
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
    logger.info("Language-to-Language Alignment Experiment")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset}/{subset} ({split} split)")
    logger.info(f"Modelset: {modelset}")
    logger.info(f"Pooling: {pool}, Prompt: {prompt}")
    if metric in {"mutual_knn", "knn", "cycle_knn"}:
        logger.info(f"Metric: {metric} (k={k})")
    elif metric == "cka_rbf":
        logger.info(f"Metric: {metric} (sigma={rbf_sigma})")
    else:
        logger.info(f"Metric: {metric}")
    logger.info(f"Max samples: {max_samples}, permutations: {num_permutations}")
    logger.info(f"Outlier quantile: {q_outlier}")
    logger.info("=" * 80)

    # Get language models only
    llm_models, _ = get_models(modelset, modality="language")
    if not llm_models:
        raise ValueError(f"No language models found for modelset: {modelset}")

    logger.info(f"Language models: {len(llm_models)}")
    for model in llm_models:
        logger.debug(f"  - {model}")

    # Load or extract features for all language models
    logger.info(
        f"State: {ExperimentState.GENERATING_DATA} - loading/extracting features"
    )
    feats_list = []
    model_params = []

    # Check which features need to be extracted
    models_needing_extraction = []
    for model_name in llm_models:
        out_path = prh_feature_filename(
            os.path.join(output_dir, "features"),
            dataset,
            subset,
            model_name,
            pool=pool,
            prompt=prompt,
            caption_idx=caption_idx,
        )
        if not os.path.exists(out_path) or force_features:
            models_needing_extraction.append(model_name)

    # Load texts only if we need to extract features
    texts = None
    if models_needing_extraction:
        logger.info(
            f"Need to extract features for {len(models_needing_extraction)} models"
        )
        texts, _ = _load_prh_samples(
            dataset,
            subset,
            split=split,
            max_samples=max_samples,
            data_dir=None,
            caption_idx=caption_idx,
            prompt=prompt,
        )
        logger.info(f"Loaded {len(texts)} texts for feature extraction")

    for model_name in tqdm(llm_models, desc="Loading language features"):
        out_path = prh_feature_filename(
            os.path.join(output_dir, "features"),
            dataset,
            subset,
            model_name,
            pool=pool,
            prompt=prompt,
            caption_idx=caption_idx,
        )

        if os.path.exists(out_path) and not force_features:
            # Reuse existing features from PRH experiment
            payload = torch.load(out_path, map_location=device)
            feats_list.append(payload["feats"])
            model_params.append(payload.get("num_params", 0))
            logger.debug(f"Loaded cached features for {model_name}")
            continue

        # Extract features if not cached
        logger.info(f"Extracting features for {model_name} (not cached)")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        assert texts is not None, "texts should have been loaded for extraction"

        try:
            feats, num_params = _extract_text_features(
                texts,
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
            feats, num_params = _extract_text_features(
                texts,
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

    logger.info(f"Loaded features for {len(feats_list)} language models")

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
        os.path.join(output_dir, "alignment_l2l"),
        dataset,
        modelset,
        "language",
        pool,
        prompt,
        "language",
        pool,
        prompt,
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
        "model_names": llm_models,
        "model_params": model_params,
    }
    np.save(save_path, payload)
    logger.success(f"Results saved to {save_path}")

    return payload


def compute_l2l_intra_family_alignment(
    scores: np.ndarray,
    model_names: List[str],
) -> Dict:
    """Compute intra-family alignment for language models.

    Groups models by family (e.g., Llama, Bloom, OLMo) and computes
    average alignment within and between families.

    Args:
        scores: Pairwise alignment scores matrix [n_models, n_models].
        model_names: List of model names.

    Returns:
        Dictionary with family-level alignment statistics.
    """
    # Categorize models by family
    families = []
    for name in model_names:
        name_lower = name.lower()
        if "llama" in name_lower:
            families.append("Llama")
        elif "bloom" in name_lower:
            families.append("Bloom")
        elif "olmo" in name_lower:
            families.append("OLMo")
        elif "gemma" in name_lower:
            families.append("Gemma")
        elif "mistral" in name_lower or "mixtral" in name_lower:
            families.append("Mistral")
        elif "open_llama" in name_lower:
            families.append("OpenLlama")
        else:
            families.append("Other")

    unique_families = sorted(set(families))
    family_indices = {fam: [] for fam in unique_families}
    for i, fam in enumerate(families):
        family_indices[fam].append(i)

    # Compute intra-family alignment
    intra_alignment = {}
    for fam, idxs in family_indices.items():
        if len(idxs) > 1:
            sub_scores = scores[np.ix_(idxs, idxs)]
            mask = ~np.eye(len(idxs), dtype=bool)
            intra_alignment[fam] = float(np.mean(sub_scores[mask]))
        else:
            intra_alignment[fam] = float("nan")

    # Compute inter-family alignment
    inter_alignment = {}
    for fam1 in unique_families:
        for fam2 in unique_families:
            if fam1 < fam2:
                idxs1 = family_indices[fam1]
                idxs2 = family_indices[fam2]
                sub_scores = scores[np.ix_(idxs1, idxs2)]
                inter_alignment[f"{fam1}_vs_{fam2}"] = float(np.mean(sub_scores))

    # Compute size-based buckets (by parameter count proxy from model name)
    # Sort by typical model sizes based on naming conventions
    size_order = []
    for name in model_names:
        name_lower = name.lower()
        if "560m" in name_lower or "1b" in name_lower:
            size_order.append(1)
        elif "3b" in name_lower or "2b" in name_lower:
            size_order.append(2)
        elif "7b" in name_lower or "8b" in name_lower:
            size_order.append(3)
        elif "13b" in name_lower:
            size_order.append(4)
        elif "30b" in name_lower or "65b" in name_lower or "70b" in name_lower:
            size_order.append(5)
        else:
            size_order.append(3)  # Default to medium

    # Group by size and compute mean alignment within each size tier
    size_groups = {}
    for i, size in enumerate(size_order):
        if size not in size_groups:
            size_groups[size] = []
        size_groups[size].append(i)

    size_alignment = {}
    for size, idxs in sorted(size_groups.items()):
        if len(idxs) > 1:
            sub_scores = scores[np.ix_(idxs, idxs)]
            mask = ~np.eye(len(idxs), dtype=bool)
            size_alignment[f"size_{size}"] = float(np.mean(sub_scores[mask]))
        else:
            size_alignment[f"size_{size}"] = float("nan")

    return {
        "families": families,
        "intra_alignment": intra_alignment,
        "inter_alignment": inter_alignment,
        "unique_families": unique_families,
        "size_alignment": size_alignment,
    }
