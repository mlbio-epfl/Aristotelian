"""
Video-to-Text alignment experiment following VideoPRH paper methodology.

This module implements video-text alignment experiments, showing alignment
between video representations and text representations using the PE-Video (PVD) dataset.

Key features:
- Multi-frame video feature extraction with temporal averaging
- Multi-caption text aggregation
- Scaling law fitting: score(nf, nc) = S_inf - (Cf * nf^(-alpha) + Cc * nc^(-beta))
- Family-level alignment analysis (VideoMAE vs DINOv2 vs CLIP)
"""

from __future__ import annotations

import gc
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from loguru import logger
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from ..experiments.multiple import bh_fdr
from ..utils.logging import ExperimentState, log_timing
from .cache import _cached_feats_match, build_gram_cache, build_knn_cache_with_indices
from .layers import _as_layers, _normalize_layers
from .paths import prh_alignment_filename, prh_feature_filename
from .preprocess import prepare_features
from .prh_experiment import (
    _extract_text_features,
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
from .prh_models import get_models
from .prh_pipeline import collect_vision_activations
from .pvd_data import extract_video_frames, load_pvd_dataset
from .video_models import (
    get_model_family,
    get_video_model_config,
    get_video_models,
    is_native_video_model,
    load_video_model,
)


def _extract_video_features(
    video_sources: List[Any],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    pool: str,
    num_frames: int = 8,
    frame_strategy: str = "uniform",
    temporal_pooling: str = "single",
) -> Tuple[torch.Tensor, int]:
    """Extract video features with configurable temporal pooling.

    For VideoMAE: passes all frames to model at once (native video model)
    For DINOv2/CLIP: extracts per-frame features with configurable temporal pooling

    Args:
        video_sources: List of video sources (file paths or video bytes).
        model_name: Name of the video model to use.
        device: Device for computation.
        batch_size: Batch size for processing.
        pool: Pooling method ('cls' or 'mean').
        num_frames: Number of frames to extract per video.
        frame_strategy: Frame extraction strategy ('uniform', 'start', 'middle').
        temporal_pooling: Temporal pooling strategy for frame-by-frame models:
            - 'single': Use single middle frame (default, recommended for CKA).
            - 'mean': Average features across all frames.
            - 'max': Max pooling across frames.

    Returns:
        Tuple of (features, num_params) where features shape is (n_videos, n_layers, hidden_dim)
    """
    # For native video models (VideoMAE), use the model's expected frame count
    # (typically 16) since position embeddings are pretrained for that size
    if is_native_video_model(model_name):
        model_config = get_video_model_config(model_name)
        actual_num_frames = model_config["expected_frames"]
        if actual_num_frames != num_frames:
            logger.info(
                f"Using {actual_num_frames} frames for {model_name} "
                f"(model requires this for position embeddings)"
            )
    else:
        actual_num_frames = num_frames

    model, processor_or_transform = load_video_model(
        model_name, device=device, num_frames=actual_num_frames
    )
    num_params = int(sum(p.numel() for p in model.parameters()))

    if is_native_video_model(model_name):
        return (
            _extract_native_video_features(
                video_sources,
                model=model,
                processor=processor_or_transform,
                device=device,
                batch_size=batch_size,
                pool=pool,
                num_frames=actual_num_frames,
                frame_strategy=frame_strategy,
            ),
            num_params,
        )
    else:
        return (
            _extract_frame_by_frame_features(
                video_sources,
                model=model,
                transform=processor_or_transform,
                device=device,
                batch_size=batch_size,
                pool=pool,
                num_frames=actual_num_frames,
                frame_strategy=frame_strategy,
                temporal_pooling=temporal_pooling,
            ),
            num_params,
        )


def _extract_native_video_features(
    video_sources: List[Any],
    *,
    model: Any,
    processor: Any,
    device: str,
    batch_size: int,
    pool: str,
    num_frames: int,
    frame_strategy: str,
    subclip_averaging: bool = True,
) -> torch.Tensor:
    """Extract features from native video models (e.g., VideoMAE).

    Native video models process all frames at once and learn temporal dynamics.
    When subclip_averaging is True, for videos with >num_frames frames:
    - Extract multiples of num_frames (e.g., 32, 48, 64 for num_frames=16)
    - Pass each num_frames-length sub-clip through encoder
    - Average representations across sub-clips
    """
    all_layer_features = []

    for i in tqdm(
        range(0, len(video_sources), batch_size), desc="Extracting video features"
    ):
        batch_sources = video_sources[i : i + batch_size]

        for video_source in batch_sources:
            if subclip_averaging:
                # Extract features with sub-clip averaging for longer videos
                video_layers = _extract_subclip_features(
                    video_source,
                    model=model,
                    processor=processor,
                    device=device,
                    pool=pool,
                    native_frames=num_frames,
                    frame_strategy=frame_strategy,
                )
            else:
                # Original behavior: extract exactly num_frames
                frames = extract_video_frames(
                    video_source,
                    num_frames=num_frames,
                    strategy=frame_strategy,
                )
                video_layers = _process_single_clip(
                    [frames], model, processor, device, pool
                )[0]

            all_layer_features.append(video_layers)

    # Stack all videos: (n_videos, n_layers, hidden_dim)
    features = np.stack(all_layer_features, axis=0)
    return torch.from_numpy(features)


def _extract_subclip_features(
    video_source: Any,
    *,
    model: Any,
    processor: Any,
    device: str,
    pool: str,
    native_frames: int = 16,
    frame_strategy: str = "uniform",
    max_subclips: int = 4,
) -> np.ndarray:
    """Extract features with sub-clip averaging for longer videos.

    For videos with >native_frames usable frames:
    - Extract multiples of native_frames (up to max_subclips * native_frames)
    - Pass each native_frames-length sub-clip through encoder
    - Average CLS tokens across sub-clips

    Args:
        video_source: Video file path or bytes.
        model: VideoMAE model.
        processor: VideoMAE processor.
        device: Device for computation.
        pool: Pooling method ('cls' or 'mean').
        native_frames: Native frame count for the model (e.g., 16 for VideoMAE).
        frame_strategy: Frame extraction strategy.
        max_subclips: Maximum number of sub-clips to extract.

    Returns:
        Feature array of shape (n_layers, hidden_dim).
    """
    # Try to get total frame count
    try:
        import io

        from decord import VideoReader, cpu

        if isinstance(video_source, bytes):
            vr = VideoReader(io.BytesIO(video_source), ctx=cpu(0))
        else:
            vr = VideoReader(video_source, ctx=cpu(0))
        total_frames = len(vr)
    except Exception:
        # Fall back to single clip if we can't count frames
        total_frames = native_frames

    # Determine number of sub-clips
    num_subclips = min(max_subclips, max(1, total_frames // native_frames))
    total_extract_frames = num_subclips * native_frames

    # Extract all frames needed
    frames = extract_video_frames(
        video_source,
        num_frames=total_extract_frames,
        strategy=frame_strategy,
    )

    # Split into sub-clips
    subclips = []
    for j in range(num_subclips):
        start_idx = j * native_frames
        end_idx = start_idx + native_frames
        subclip_frames = frames[start_idx:end_idx]
        if len(subclip_frames) == native_frames:
            subclips.append(subclip_frames)

    if not subclips:
        # Fall back to whatever frames we have
        subclips = [frames[:native_frames]]
        # Pad if needed
        while len(subclips[0]) < native_frames:
            subclips[0].append(subclips[0][-1])

    # Process each sub-clip
    subclip_features = _process_single_clip(subclips, model, processor, device, pool)

    # Average across sub-clips: (n_subclips, n_layers, hidden_dim) -> (n_layers, hidden_dim)
    return np.mean(subclip_features, axis=0)


def _process_single_clip(
    clips: List[List[Any]],
    model: Any,
    processor: Any,
    device: str,
    pool: str,
) -> List[np.ndarray]:
    """Process a batch of video clips through the model.

    Args:
        clips: List of frame lists, each with native_frames frames.
        model: VideoMAE model.
        processor: VideoMAE processor.
        device: Device for computation.
        pool: Pooling method ('cls' or 'mean').

    Returns:
        List of feature arrays, each of shape (n_layers, hidden_dim).
    """
    # Process batch through model
    inputs = processor(clips, return_tensors="pt").to(device)
    # Match input dtype to model dtype (e.g., bfloat16)
    model_dtype = next(model.parameters()).dtype
    if model_dtype != torch.float32:
        for k, v in inputs.items():
            if hasattr(v, "dtype") and v.dtype.is_floating_point:
                inputs[k] = v.to(model_dtype)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get hidden states from all layers
    hidden_states = outputs.hidden_states

    # Pool across patches to get (batch, hidden_dim) per layer
    results = []
    for b in range(len(clips)):
        video_layers = []
        for layer_hidden in hidden_states:
            if pool == "cls":
                # Use CLS token (convert to float32 for numpy compatibility)
                feat = layer_hidden[b, 0, :].float().cpu().numpy()
            else:
                # Mean pool across all tokens
                feat = layer_hidden[b].mean(dim=0).float().cpu().numpy()
            video_layers.append(feat)
        results.append(np.stack(video_layers, axis=0))

    return results


def _extract_frame_by_frame_features(
    video_sources: List[Any],
    *,
    model: Any,
    transform: Any,
    device: str,
    batch_size: int,
    pool: str,
    num_frames: int,
    frame_strategy: str,
    temporal_pooling: str = "single",
) -> torch.Tensor:
    """Extract features from frame-by-frame models (e.g., DINOv2, CLIP).

    Args:
        video_sources: List of video sources.
        model: Vision model.
        transform: Image transform.
        device: Device for computation.
        batch_size: Batch size.
        pool: Spatial pooling method ('cls' or 'mean').
        num_frames: Number of frames to extract per video.
        frame_strategy: Frame extraction strategy.
        temporal_pooling: Temporal pooling strategy:
            - 'single': Use single middle frame (default, recommended for CKA).
            - 'mean': Average features across all frames.
            - 'max': Max pooling across frames.

    Returns:
        Features tensor of shape (n_videos, n_layers, hidden_dim).
    """
    # Determine return nodes based on model architecture
    if hasattr(model, "blocks"):
        return_nodes = [f"blocks.{i}.add_1" for i in range(len(model.blocks))]
        feat_model = create_feature_extractor(model, return_nodes=return_nodes)
    else:
        feat_model = model
        return_nodes = None

    all_video_features = []

    # For single-frame mode, we only need 1 frame from the middle
    effective_num_frames = 1 if temporal_pooling == "single" else num_frames
    effective_strategy = "middle" if temporal_pooling == "single" else frame_strategy

    for video_source in tqdm(video_sources, desc="Extracting frame features"):
        frames = extract_video_frames(
            video_source,
            num_frames=effective_num_frames,
            strategy=effective_strategy,
        )

        # Extract features for each frame
        frame_features = []
        for frame in frames:
            layers = collect_vision_activations(
                [frame],
                model=feat_model,
                device=device,
                batch_size=1,
                pool=pool,
                transform=transform,
            )
            frame_features.append(_stack_layers(layers))  # (1, n_layers, hidden_dim)

        # Apply temporal pooling
        stacked = torch.cat(frame_features, dim=0)  # (n_frames, n_layers, hidden_dim)

        if temporal_pooling == "single":
            # Single frame - no pooling needed (n_frames=1)
            video_feat = stacked  # (1, n_layers, hidden_dim)
        elif temporal_pooling == "mean":
            # Mean pooling across frames
            video_feat = stacked.mean(dim=0, keepdim=True)  # (1, n_layers, hidden_dim)
        elif temporal_pooling == "max":
            # Max pooling across frames
            video_feat = stacked.max(dim=0, keepdim=True)[
                0
            ]  # (1, n_layers, hidden_dim)
        else:
            raise ValueError(f"Unknown temporal_pooling: {temporal_pooling}")

        all_video_features.append(video_feat)

    # Concatenate all videos: (n_videos, n_layers, hidden_dim)
    return torch.cat(all_video_features, dim=0)


def _extract_multi_caption_features(
    captions_list: List[List[str]],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    pool: str,
    num_captions: int = 1,
) -> Tuple[torch.Tensor, int]:
    """Extract text features with multi-caption aggregation.

    Following VideoPRH paper: concatenates selected captions into a single string,
    then extracts features. The text encoder produces per-token embeddings which
    are averaged along the token dimension.

    Args:
        captions_list: List of caption lists, one per sample.
        model_name: Name of the text model to use.
        device: Device for computation.
        batch_size: Batch size for processing.
        pool: Pooling method ('avg' or 'last').
        num_captions: Number of captions to use per sample.

    Returns:
        Tuple of (features, num_params) where features shape is (n_samples, n_layers, hidden_dim)
    """
    # Concatenate captions into single strings (per VideoPRH paper methodology)
    # "We concatenate the set of selected captions into a single string and use
    # text-based encoders to extract their intermediate features."
    texts = []
    for caps in captions_list:
        # Take up to num_captions captions and concatenate with space separator
        sample_caps = caps[:num_captions]
        concatenated = " ".join(sample_caps)
        texts.append(concatenated)

    return _extract_text_features(
        texts,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        pool=pool,
    )


def run_v2t_experiment(
    *,
    dataset: str = "pvd",
    split: str = "test",
    video_modelset: str = "default",
    text_modelset: str = "val",
    pool_video: str = "cls",
    pool_text: str = "avg",
    num_frames: int = 8,
    num_captions: int = 1,
    frame_strategy: str = "uniform",
    temporal_pooling: str = "single",
    max_samples: int | None = 1000,
    streaming_limit: int | None = None,
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
    """Run video-to-text alignment experiment.

    This experiment measures alignment between video and text models using
    the PE-Video (PVD) dataset, following the VideoPRH paper methodology.

    Args:
        dataset: Dataset name ('pvd').
        split: Dataset split ('test' or 'train').
        video_modelset: Which video model set to use.
        text_modelset: Which text model set to use.
        pool_video: Pooling method for video features ('cls' or 'mean').
        pool_text: Pooling method for text features ('avg' or 'last').
        num_frames: Number of frames to extract per video.
        num_captions: Number of captions to use per sample.
        frame_strategy: Frame extraction strategy.
        temporal_pooling: Temporal pooling for frame-by-frame models ('single', 'mean', 'max').
            'single' uses middle frame only (default, recommended for CKA).
            'mean' averages features across frames.
            'max' uses max pooling across frames.
        max_samples: Maximum number of samples to use.
        streaming_limit: Maximum number of samples to stream through before stopping.
            If None, streams through entire dataset for true random sampling.
            Set to e.g. 5000 for faster (but less random) sampling.
        batch_size: Batch size for feature extraction.
        device: Device for computation.
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
    logger.info("Video-to-Text Alignment Experiment (VideoPRH)")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset} ({split} split)")
    logger.info(f"Video modelset: {video_modelset}, Text modelset: {text_modelset}")
    logger.info(
        f"Frames: {num_frames}, Captions: {num_captions}, Temporal pooling: {temporal_pooling}"
    )
    logger.info(f"Pooling: video={pool_video}, text={pool_text}")
    if metric in {"mutual_knn", "knn", "cycle_knn"}:
        logger.info(f"Metric: {metric} (k={k})")
    elif metric == "cka_rbf":
        logger.info(f"Metric: {metric} (sigma={rbf_sigma})")
    else:
        logger.info(f"Metric: {metric}")
    logger.info(f"Max samples: {max_samples}, permutations: {num_permutations}")
    logger.info(f"Outlier quantile: {q_outlier}")
    logger.info("=" * 80)

    # Load data
    with log_timing("Loading data"):
        video_sources, captions_list = load_pvd_dataset(
            split=split,
            max_samples=max_samples,
            num_captions=num_captions,
            auto_download=True,
            streaming_limit=streaming_limit,
        )
        logger.info(
            f"Loaded {len(video_sources)} videos with {num_captions} captions each"
        )

    # Get models
    video_models = get_video_models(video_modelset)
    text_models, _ = get_models(text_modelset, modality="language")

    logger.info(f"Video models: {len(video_models)}")
    for model in video_models:
        logger.debug(f"  - {model}")
    logger.info(f"Text models: {len(text_models)}")
    for model in text_models:
        logger.debug(f"  - {model}")

    # Extract features
    logger.info(f"State: {ExperimentState.GENERATING_DATA} - extracting features")

    # Video features
    video_feats_list = []
    video_params = []

    # Include temporal pooling in path (use 'nf1' for single-frame to indicate 1 frame used)
    effective_nf = 1 if temporal_pooling == "single" else num_frames
    tp_suffix = "" if temporal_pooling == "single" else f"_tp{temporal_pooling}"

    for model_name in tqdm(video_models, desc="Extracting video features"):
        out_path = prh_feature_filename(
            os.path.join(output_dir, "features_v2t"),
            dataset,
            f"{split}_{max_samples}_nf{effective_nf}{tp_suffix}",
            model_name,
            pool=pool_video,
            prompt=False,
            caption_idx=None,
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.exists(out_path) and not force_features:
            payload = torch.load(out_path, map_location=device)
            if _cached_feats_match(payload, len(video_sources)):
                logger.info(f"Using cached features for {model_name}")
                video_feats_list.append(payload["feats"])
                video_params.append(payload.get("num_params", 0))
                continue
            logger.warning(f"Cached features mismatch for {model_name}; recomputing")

        try:
            feats, num_p = _extract_video_features(
                video_sources,
                model_name=model_name,
                device=device,
                batch_size=batch_size,
                pool=pool_video,
                num_frames=num_frames,
                frame_strategy=frame_strategy,
                temporal_pooling=temporal_pooling,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            logger.warning(
                f"OOM while extracting {model_name}; retrying on {fallback_device}"
            )
            torch.cuda.empty_cache()
            gc.collect()
            feats, num_p = _extract_video_features(
                video_sources,
                model_name=model_name,
                device=fallback_device,
                batch_size=batch_size,
                pool=pool_video,
                num_frames=num_frames,
                frame_strategy=frame_strategy,
                temporal_pooling=temporal_pooling,
            )

        torch.save({"feats": feats, "num_params": num_p}, out_path)
        video_feats_list.append(feats)
        video_params.append(num_p)

        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    # Text features
    text_feats_list = []
    text_params = []

    for model_name in tqdm(text_models, desc="Extracting text features"):
        out_path = prh_feature_filename(
            os.path.join(output_dir, "features_v2t"),
            dataset,
            f"{split}_{max_samples}_nc{num_captions}",
            model_name,
            pool=pool_text,
            prompt=False,
            caption_idx=0 if num_captions == 1 else None,
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.exists(out_path) and not force_features:
            payload = torch.load(out_path, map_location=device)
            if _cached_feats_match(payload, len(captions_list)):
                logger.info(f"Using cached features for {model_name}")
                text_feats_list.append(payload["feats"])
                text_params.append(payload.get("num_params", 0))
                continue
            logger.warning(f"Cached features mismatch for {model_name}; recomputing")

        try:
            feats, num_p = _extract_multi_caption_features(
                captions_list,
                model_name=model_name,
                device=device,
                batch_size=batch_size,
                pool=pool_text,
                num_captions=num_captions,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            logger.warning(
                f"OOM while extracting {model_name}; retrying on {fallback_device}"
            )
            torch.cuda.empty_cache()
            gc.collect()
            feats, num_p = _extract_multi_caption_features(
                captions_list,
                model_name=model_name,
                device=fallback_device,
                batch_size=batch_size,
                pool=pool_text,
                num_captions=num_captions,
            )

        torch.save({"feats": feats, "num_params": num_p}, out_path)
        text_feats_list.append(feats)
        text_params.append(num_p)

        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    logger.info(f"Extracted features for {len(video_feats_list)} video models")
    logger.info(f"Extracted features for {len(text_feats_list)} text models")

    # Compute pairwise alignment
    logger.info(f"State: {ExperimentState.COMPUTING_SIMILARITIES}")
    n_video = len(video_feats_list)
    n_text = len(text_feats_list)
    scores = np.zeros((n_video, n_text))
    indices = np.zeros((n_video, n_text, 2), dtype=int)
    gated = np.zeros((n_video, n_text))
    pvals = np.zeros((n_video, n_text))
    taus = np.zeros((n_video, n_text))

    # Preprocess features
    video_prepped = [
        prepare_features(f, q=q_outlier, exact=False, device=device)
        for f in video_feats_list
    ]
    text_prepped = [
        prepare_features(f, q=q_outlier, exact=False, device=device)
        for f in text_feats_list
    ]

    # Build caches based on metric type
    cka_cached_metrics = {"cka", "cka_lin", "unbiased_cka", "cka_rbf"}
    knn_cached_metrics = {"mutual_knn", "knn", "cycle_knn"}
    gram_cached_metrics = cka_cached_metrics | {"cknna"}

    video_layers_list = []
    text_layers_list = []
    video_knn_list = []
    text_knn_list = []
    video_grams_list = []
    text_grams_list = []

    if metric in knn_cached_metrics:
        for prepped in video_prepped:
            layers, knn_idx, masks = build_knn_cache_with_indices(
                prepped, topk=k, normalize=True
            )
            video_layers_list.append(layers)
            video_knn_list.append(knn_idx)
        for prepped in text_prepped:
            layers, knn_idx, masks = build_knn_cache_with_indices(
                prepped, topk=k, normalize=True
            )
            text_layers_list.append(layers)
            text_knn_list.append(knn_idx)
    elif metric in gram_cached_metrics:
        kernel = "rbf" if metric == "cka_rbf" else "linear"
        if metric == "cka_rbf":
            logger.info(
                f"Building Gram cache for {metric} (kernel={kernel}, sigma={rbf_sigma})"
            )
        else:
            logger.info(f"Building Gram cache for {metric} (kernel={kernel})")
        for prepped in video_prepped:
            video_grams_list.append(
                build_gram_cache(
                    prepped, normalize=True, kernel=kernel, rbf_sigma=rbf_sigma
                )
            )
        for prepped in text_prepped:
            text_grams_list.append(
                build_gram_cache(
                    prepped, normalize=True, kernel=kernel, rbf_sigma=rbf_sigma
                )
            )
    else:
        for prepped in video_prepped:
            video_layers_list.append(_normalize_layers(_as_layers(prepped)))
        for prepped in text_prepped:
            text_layers_list.append(_normalize_layers(_as_layers(prepped)))

    # Build configs for computation
    if metric in {"mutual_knn", "knn"}:
        configs = [
            (i, j, video_knn_list[i], text_knn_list[j], k, num_permutations, alpha)
            for i in range(n_video)
            for j in range(n_text)
        ]
        compute_fn = _prh_compute_pair_knn_cached
    elif metric == "cycle_knn":
        configs = [
            (i, j, video_knn_list[i], text_knn_list[j], num_permutations, alpha)
            for i in range(n_video)
            for j in range(n_text)
        ]
        compute_fn = _prh_compute_pair_cycle_knn_cached
    elif metric in cka_cached_metrics:
        use_unbiased = metric == "unbiased_cka"
        configs = [
            (
                i,
                j,
                video_grams_list[i],
                text_grams_list[j],
                num_permutations,
                alpha,
                use_unbiased,
            )
            for i in range(n_video)
            for j in range(n_text)
        ]
        compute_fn = _prh_compute_pair_cka_cached
    elif metric == "cknna":
        configs = [
            (i, j, video_grams_list[i], text_grams_list[j], k, num_permutations, alpha)
            for i in range(n_video)
            for j in range(n_text)
        ]
        compute_fn = _prh_compute_pair_cknna_cached
    elif metric == "svcca":
        configs = [
            (i, j, video_layers_list[i], text_layers_list[j], num_permutations, alpha)
            for i in range(n_video)
            for j in range(n_text)
        ]
        compute_fn = _prh_compute_pair_svcca_cached
    elif metric == "pwcca":
        configs = [
            (i, j, video_layers_list[i], text_layers_list[j], num_permutations, alpha)
            for i in range(n_video)
            for j in range(n_text)
        ]
        compute_fn = _prh_compute_pair_pwcca_cached
    elif metric == "procrustes":
        configs = [
            (i, j, video_layers_list[i], text_layers_list[j], num_permutations, alpha)
            for i in range(n_video)
            for j in range(n_text)
        ]
        compute_fn = _prh_compute_pair_procrustes_cached
    else:
        configs = [
            (
                i,
                j,
                video_layers_list[i],
                text_layers_list[j],
                metric,
                k,
                num_permutations,
                alpha,
            )
            for i in range(n_video)
            for j in range(n_text)
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

    logger.info(f"Computed {n_video} x {n_text} = {n_video * n_text} alignments")

    # Save results
    logger.info(f"State: {ExperimentState.SAVING_RESULTS}")
    save_path = prh_alignment_filename(
        os.path.join(output_dir, "alignment_v2t"),
        dataset,
        f"{video_modelset}_{text_modelset}",
        "video",
        pool_video,
        False,
        "text",
        pool_text,
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
        "video_models": video_models,
        "text_models": text_models,
        "video_params": video_params,
        "text_params": text_params,
        "num_frames": num_frames,
        "num_captions": num_captions,
    }
    np.save(save_path, payload)
    logger.success(f"Results saved to {save_path}")

    return payload


def run_v2t_scaling_experiment(
    *,
    frame_counts: Sequence[int] = (1, 2, 4, 8, 16, 32, 48),
    caption_counts: Sequence[int] = (1, 2, 4, 6, 8, 10),
    video_model: str = "vit_base_patch14_dinov2.lvd142m",
    text_model: str = "bigscience/bloomz-1b1",
    **kwargs,
) -> Dict:
    """Run V2T across multiple frame/caption counts and fit scaling law.

    Follows the VideoPRH paper's scaling law:
    score(nf, nc) = S_inf - (Cf * nf^(-alpha) + Cc * nc^(-beta))

    Args:
        frame_counts: Sequence of frame counts to test.
        caption_counts: Sequence of caption counts to test.
        video_model: Video model to use for scaling experiment.
        text_model: Text model to use for scaling experiment.
        **kwargs: Additional arguments passed to run_v2t_experiment.

    Returns:
        Dictionary containing:
        - scores: 2D array of scores [len(frame_counts), len(caption_counts)]
        - frame_counts: Frame counts used
        - caption_counts: Caption counts used
        - scaling_params: Fitted scaling law parameters
    """
    scores_grid = np.zeros((len(frame_counts), len(caption_counts)))

    for i, nf in enumerate(frame_counts):
        for j, nc in enumerate(caption_counts):
            logger.info(f"Running scaling experiment: nf={nf}, nc={nc}")

            # Override specific models
            kwargs_copy = kwargs.copy()
            kwargs_copy["num_frames"] = nf
            kwargs_copy["num_captions"] = nc

            # Run with single video and text model
            result = run_v2t_experiment(
                video_modelset="small",  # Will be filtered to single model
                text_modelset="val",
                **kwargs_copy,
            )

            # Get score for the specific model pair
            scores_grid[i, j] = np.mean(result["scores"])

    # Fit scaling law
    scaling_params = fit_scaling_law(
        scores_grid, list(frame_counts), list(caption_counts)
    )

    return {
        "scores": scores_grid,
        "frame_counts": list(frame_counts),
        "caption_counts": list(caption_counts),
        "scaling_params": scaling_params,
    }


def fit_scaling_law(
    scores: np.ndarray,
    frame_counts: List[int],
    caption_counts: List[int],
) -> Dict:
    """Fit the VideoPRH scaling law to alignment scores.

    Fits: score(nf, nc) = S_inf - (Cf * nf^(-alpha) + Cc * nc^(-beta))

    Args:
        scores: 2D array of scores [len(frame_counts), len(caption_counts)].
        frame_counts: List of frame counts.
        caption_counts: List of caption counts.

    Returns:
        Dictionary with fitted parameters:
        - S_inf: Asymptotic score
        - Cf: Frame coefficient
        - Cc: Caption coefficient
        - alpha: Frame exponent
        - beta: Caption exponent
        - residual: Fit residual
    """
    from scipy.optimize import curve_fit

    # Flatten for fitting
    nf_grid, nc_grid = np.meshgrid(frame_counts, caption_counts, indexing="ij")
    nf_flat = nf_grid.flatten()
    nc_flat = nc_grid.flatten()
    scores_flat = scores.flatten()

    def scaling_law(X, S_inf, Cf, alpha, Cc, beta):
        nf, nc = X
        return S_inf - (Cf * np.power(nf, -alpha) + Cc * np.power(nc, -beta))

    try:
        # Initial guess
        p0 = [np.max(scores_flat), 0.1, 0.5, 0.1, 0.5]
        bounds = ([0, 0, 0, 0, 0], [1, np.inf, 2, np.inf, 2])

        popt, pcov = curve_fit(
            scaling_law,
            (nf_flat, nc_flat),
            scores_flat,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )

        S_inf, Cf, alpha, Cc, beta = popt

        # Compute residual
        pred = scaling_law((nf_flat, nc_flat), *popt)
        residual = float(np.mean((scores_flat - pred) ** 2))

        return {
            "S_inf": float(S_inf),
            "Cf": float(Cf),
            "alpha": float(alpha),
            "Cc": float(Cc),
            "beta": float(beta),
            "residual": residual,
        }
    except Exception as e:
        logger.warning(f"Failed to fit scaling law: {e}")
        return {
            "S_inf": float("nan"),
            "Cf": float("nan"),
            "alpha": float("nan"),
            "Cc": float("nan"),
            "beta": float("nan"),
            "residual": float("nan"),
        }


def compute_v2t_family_alignment(
    scores: np.ndarray,
    video_models: List[str],
    text_models: List[str],
) -> Dict:
    """Compute family-level alignment statistics for V2T results.

    Groups video models by family (VideoMAE, DINOv2, CLIP) and computes
    average alignment per family.

    Args:
        scores: Pairwise alignment scores [n_video, n_text].
        video_models: List of video model names.
        text_models: List of text model names.

    Returns:
        Dictionary with family-level statistics.
    """
    # Categorize video models by family
    video_families = [get_model_family(m) for m in video_models]
    unique_video_families = sorted(set(video_families))

    # Group video model indices by family
    family_indices = {fam: [] for fam in unique_video_families}
    for i, fam in enumerate(video_families):
        family_indices[fam].append(i)

    # Compute mean alignment per video family
    family_alignment = {}
    for fam, idxs in family_indices.items():
        if idxs:
            fam_scores = scores[idxs, :]
            family_alignment[fam] = {
                "mean": float(np.mean(fam_scores)),
                "std": float(np.std(fam_scores)),
                "n_models": len(idxs),
            }

    # Compute inter-family comparison
    inter_family = {}
    for fam1 in unique_video_families:
        for fam2 in unique_video_families:
            if fam1 < fam2:
                idxs1 = family_indices[fam1]
                idxs2 = family_indices[fam2]
                if idxs1 and idxs2:
                    scores1 = scores[idxs1, :].mean()
                    scores2 = scores[idxs2, :].mean()
                    inter_family[f"{fam1}_vs_{fam2}"] = {
                        fam1: float(scores1),
                        fam2: float(scores2),
                        "diff": float(scores1 - scores2),
                    }

    return {
        "video_families": video_families,
        "unique_families": unique_video_families,
        "family_alignment": family_alignment,
        "inter_family": inter_family,
    }


def _extract_averaged_caption_features(
    captions_list: List[List[str]],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    pool: str,
    num_captions: int = 10,
) -> Tuple[torch.Tensor, int]:
    """Extract text features by averaging across multiple captions.

    Different from _extract_multi_caption_features which concatenates captions,
    this function extracts features for each caption independently and averages
    the resulting representations.

    Args:
        captions_list: List of caption lists, one per sample.
        model_name: Name of the text model to use.
        device: Device for computation.
        batch_size: Batch size for processing.
        pool: Pooling method ('avg' or 'last').
        num_captions: Number of captions to use per sample.

    Returns:
        Tuple of (features, num_params) where features shape is (n_samples, n_layers, hidden_dim)
    """
    from .prh_models import load_text_model

    logger.info(f"Extracting averaged caption features using {model_name}")

    # Load model once
    tokenizer, model = load_text_model(model_name, device=device)
    num_params = int(sum(p.numel() for p in model.parameters()))

    all_sample_features = []

    for sample_captions in tqdm(captions_list, desc="Extracting caption features"):
        # Take up to num_captions
        caps_to_process = sample_captions[:num_captions]

        # Extract features for each caption
        caption_features = []
        for caption in caps_to_process:
            feats, _ = _extract_text_features(
                [caption],
                model_name=model_name,
                device=device,
                batch_size=1,
                pool=pool,
            )
            caption_features.append(feats)  # (1, n_layers, hidden_dim)

        # Stack and average across captions
        # (num_captions, n_layers, hidden_dim) -> (1, n_layers, hidden_dim)
        stacked = torch.cat(caption_features, dim=0)
        averaged = stacked.mean(dim=0, keepdim=True)
        all_sample_features.append(averaged)

    # Concatenate all samples: (n_samples, n_layers, hidden_dim)
    features = torch.cat(all_sample_features, dim=0)
    return features, num_params


def run_v2t_vatex_alignment(
    *,
    video_modelset: str = "videoprh",
    text_modelset: str = "videoprh",
    pool_video: str = "cls",
    pool_text: str = "avg",
    num_frames: int = 16,
    num_frames_framewise: int = 8,
    num_captions: int = 10,
    temporal_pooling: str = "single",
    max_samples: int = 1024,
    streaming_limit: int | None = None,
    batch_size: int = 4,
    device: str = "cpu",
    fallback_device: str = "cpu",
    output_dir: str = "./results",
    k: int = 10,
    metric: str = "mutual_knn",
    rbf_sigma: float = 1.0,
    num_permutations: int = 1000,
    alpha: float = 0.05,
    q_outlier: float = 0.95,
    force: bool = False,
    force_features: bool = False,
    num_workers: int = 1,
    average_caption_features: bool = True,
) -> Dict[str, np.ndarray]:
    """Run video-to-text alignment experiment on VATEX dataset.

    This function reproduces the VideoPRH paper results using:
    - VATEX dataset with 1024 videos and 10 captions per video
    - VideoMAEv2, DINOv2, and CLIP video models
    - Gemma2-9b-it text model
    - Sub-clip averaging for VideoMAE models

    Args:
        video_modelset: Which video model set to use (default: "videoprh").
        text_modelset: Which text model set to use (default: "videoprh").
        pool_video: Pooling method for video features ('cls' or 'mean').
        pool_text: Pooling method for text features ('avg' or 'last').
        num_frames: Number of frames for native video models (VideoMAE).
        num_frames_framewise: Number of frames for frame-by-frame models (DINOv2, CLIP).
        num_captions: Number of captions to use per sample.
        temporal_pooling: Temporal pooling for frame-by-frame models ('single', 'mean', 'max').
            'single' uses middle frame only (default, recommended for CKA).
        max_samples: Maximum number of samples to use.
        streaming_limit: Maximum samples to stream through for sampling.
        batch_size: Batch size for feature extraction.
        device: Device for computation.
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
        average_caption_features: If True, average features across captions.
            If False, concatenate captions into a single string.

    Returns:
        Dictionary containing alignment scores, p-values, and gated scores.
    """
    from .vatex_data import load_vatex_dataset

    if device != "cpu" and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    logger.info("=" * 80)
    logger.info("Video-to-Text Alignment Experiment (VideoPRH - VATEX)")
    logger.info("=" * 80)
    logger.info("Dataset: VATEX (validation split)")
    logger.info(f"Video modelset: {video_modelset}, Text modelset: {text_modelset}")
    logger.info(f"Frames: native={num_frames}, framewise={num_frames_framewise}")
    logger.info(f"Temporal pooling: {temporal_pooling}")
    logger.info(f"Captions: {num_captions} (averaged={average_caption_features})")
    logger.info(f"Pooling: video={pool_video}, text={pool_text}")
    if metric in {"mutual_knn", "knn", "cycle_knn"}:
        logger.info(f"Metric: {metric} (k={k})")
    elif metric == "cka_rbf":
        logger.info(f"Metric: {metric} (sigma={rbf_sigma})")
    else:
        logger.info(f"Metric: {metric}")
    logger.info(f"Max samples: {max_samples}, permutations: {num_permutations}")
    logger.info("=" * 80)

    # Load VATEX data
    with log_timing("Loading VATEX data"):
        video_bytes_list, captions_list = load_vatex_dataset(
            split="validation",
            max_samples=max_samples,
            num_captions=num_captions,
            streaming_limit=streaming_limit,
        )
        logger.info(
            f"Loaded {len(video_bytes_list)} videos with {num_captions} captions each"
        )

    # Get models
    video_models = get_video_models(video_modelset)
    text_models, _ = get_models(text_modelset, modality="language")

    logger.info(f"Video models: {len(video_models)}")
    for model in video_models:
        logger.debug(f"  - {model}")
    logger.info(f"Text models: {len(text_models)}")
    for model in text_models:
        logger.debug(f"  - {model}")

    # Extract features
    logger.info(f"State: {ExperimentState.GENERATING_DATA} - extracting features")

    # Video features
    video_feats_list = []
    video_params = []

    for model_name in tqdm(video_models, desc="Extracting video features"):
        # Determine frame count based on model type
        actual_num_frames = (
            num_frames if is_native_video_model(model_name) else num_frames_framewise
        )
        # Adjust frame count and path suffix for temporal pooling mode
        effective_nf = 1 if temporal_pooling == "single" else actual_num_frames
        tp_suffix = "" if temporal_pooling == "single" else f"_tp{temporal_pooling}"

        out_path = prh_feature_filename(
            os.path.join(output_dir, "features_vatex"),
            "vatex",
            f"validation_{max_samples}_nf{effective_nf}{tp_suffix}",
            model_name,
            pool=pool_video,
            prompt=False,
            caption_idx=None,
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.exists(out_path) and not force_features:
            payload = torch.load(out_path, map_location=device)
            if _cached_feats_match(payload, len(video_bytes_list)):
                logger.info(f"Using cached features for {model_name}")
                video_feats_list.append(payload["feats"])
                video_params.append(payload.get("num_params", 0))
                continue
            logger.warning(f"Cached features mismatch for {model_name}; recomputing")

        try:
            feats, num_p = _extract_video_features(
                video_bytes_list,
                model_name=model_name,
                device=device,
                batch_size=batch_size,
                pool=pool_video,
                num_frames=actual_num_frames,
                frame_strategy="uniform",
                temporal_pooling=temporal_pooling,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            logger.warning(
                f"OOM while extracting {model_name}; retrying on {fallback_device}"
            )
            torch.cuda.empty_cache()
            gc.collect()
            feats, num_p = _extract_video_features(
                video_bytes_list,
                model_name=model_name,
                device=fallback_device,
                batch_size=batch_size,
                pool=pool_video,
                num_frames=actual_num_frames,
                frame_strategy="uniform",
                temporal_pooling=temporal_pooling,
            )

        torch.save({"feats": feats, "num_params": num_p}, out_path)
        video_feats_list.append(feats)
        video_params.append(num_p)

        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    # Text features
    text_feats_list = []
    text_params = []

    for model_name in tqdm(text_models, desc="Extracting text features"):
        suffix = "avg" if average_caption_features else "concat"
        out_path = prh_feature_filename(
            os.path.join(output_dir, "features_vatex"),
            "vatex",
            f"validation_{max_samples}_nc{num_captions}_{suffix}",
            model_name,
            pool=pool_text,
            prompt=False,
            caption_idx=None,
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.exists(out_path) and not force_features:
            payload = torch.load(out_path, map_location=device)
            if _cached_feats_match(payload, len(captions_list)):
                logger.info(f"Using cached features for {model_name}")
                text_feats_list.append(payload["feats"])
                text_params.append(payload.get("num_params", 0))
                continue
            logger.warning(f"Cached features mismatch for {model_name}; recomputing")

        try:
            if average_caption_features:
                feats, num_p = _extract_averaged_caption_features(
                    captions_list,
                    model_name=model_name,
                    device=device,
                    batch_size=batch_size,
                    pool=pool_text,
                    num_captions=num_captions,
                )
            else:
                feats, num_p = _extract_multi_caption_features(
                    captions_list,
                    model_name=model_name,
                    device=device,
                    batch_size=batch_size,
                    pool=pool_text,
                    num_captions=num_captions,
                )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            logger.warning(
                f"OOM while extracting {model_name}; retrying on {fallback_device}"
            )
            torch.cuda.empty_cache()
            gc.collect()
            if average_caption_features:
                feats, num_p = _extract_averaged_caption_features(
                    captions_list,
                    model_name=model_name,
                    device=fallback_device,
                    batch_size=batch_size,
                    pool=pool_text,
                    num_captions=num_captions,
                )
            else:
                feats, num_p = _extract_multi_caption_features(
                    captions_list,
                    model_name=model_name,
                    device=fallback_device,
                    batch_size=batch_size,
                    pool=pool_text,
                    num_captions=num_captions,
                )

        torch.save({"feats": feats, "num_params": num_p}, out_path)
        text_feats_list.append(feats)
        text_params.append(num_p)

        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    logger.info(f"Extracted features for {len(video_feats_list)} video models")
    logger.info(f"Extracted features for {len(text_feats_list)} text models")

    # Compute pairwise alignment
    logger.info(f"State: {ExperimentState.COMPUTING_SIMILARITIES}")
    n_video = len(video_feats_list)
    n_text = len(text_feats_list)
    scores = np.zeros((n_video, n_text))
    indices = np.zeros((n_video, n_text, 2), dtype=int)
    gated = np.zeros((n_video, n_text))
    pvals = np.zeros((n_video, n_text))
    taus = np.zeros((n_video, n_text))

    # Preprocess features
    video_prepped = [
        prepare_features(f, q=q_outlier, exact=False, device=device)
        for f in video_feats_list
    ]
    text_prepped = [
        prepare_features(f, q=q_outlier, exact=False, device=device)
        for f in text_feats_list
    ]

    # Build caches based on metric type
    cka_cached_metrics = {"cka", "cka_lin", "unbiased_cka", "cka_rbf"}
    knn_cached_metrics = {"mutual_knn", "knn", "cycle_knn"}
    gram_cached_metrics = cka_cached_metrics | {"cknna"}

    video_layers_list = []
    text_layers_list = []
    video_knn_list = []
    text_knn_list = []
    video_grams_list = []
    text_grams_list = []

    if metric in knn_cached_metrics:
        for prepped in video_prepped:
            layers, knn_idx, masks = build_knn_cache_with_indices(
                prepped, topk=k, normalize=True
            )
            video_layers_list.append(layers)
            video_knn_list.append(knn_idx)
        for prepped in text_prepped:
            layers, knn_idx, masks = build_knn_cache_with_indices(
                prepped, topk=k, normalize=True
            )
            text_layers_list.append(layers)
            text_knn_list.append(knn_idx)
    elif metric in gram_cached_metrics:
        kernel = "rbf" if metric == "cka_rbf" else "linear"
        if metric == "cka_rbf":
            logger.info(
                f"Building Gram cache for {metric} (kernel={kernel}, sigma={rbf_sigma})"
            )
        else:
            logger.info(f"Building Gram cache for {metric} (kernel={kernel})")
        for prepped in video_prepped:
            video_grams_list.append(
                build_gram_cache(
                    prepped, normalize=True, kernel=kernel, rbf_sigma=rbf_sigma
                )
            )
        for prepped in text_prepped:
            text_grams_list.append(
                build_gram_cache(
                    prepped, normalize=True, kernel=kernel, rbf_sigma=rbf_sigma
                )
            )
    else:
        for prepped in video_prepped:
            video_layers_list.append(_normalize_layers(_as_layers(prepped)))
        for prepped in text_prepped:
            text_layers_list.append(_normalize_layers(_as_layers(prepped)))

    # Build configs for computation
    if metric in {"mutual_knn", "knn"}:
        configs = [
            (i, j, video_knn_list[i], text_knn_list[j], k, num_permutations, alpha)
            for i in range(n_video)
            for j in range(n_text)
        ]
        compute_fn = _prh_compute_pair_knn_cached
    elif metric == "cycle_knn":
        configs = [
            (i, j, video_knn_list[i], text_knn_list[j], num_permutations, alpha)
            for i in range(n_video)
            for j in range(n_text)
        ]
        compute_fn = _prh_compute_pair_cycle_knn_cached
    elif metric in cka_cached_metrics:
        use_unbiased = metric == "unbiased_cka"
        configs = [
            (
                i,
                j,
                video_grams_list[i],
                text_grams_list[j],
                num_permutations,
                alpha,
                use_unbiased,
            )
            for i in range(n_video)
            for j in range(n_text)
        ]
        compute_fn = _prh_compute_pair_cka_cached
    elif metric == "cknna":
        configs = [
            (i, j, video_grams_list[i], text_grams_list[j], k, num_permutations, alpha)
            for i in range(n_video)
            for j in range(n_text)
        ]
        compute_fn = _prh_compute_pair_cknna_cached
    elif metric == "svcca":
        configs = [
            (i, j, video_layers_list[i], text_layers_list[j], num_permutations, alpha)
            for i in range(n_video)
            for j in range(n_text)
        ]
        compute_fn = _prh_compute_pair_svcca_cached
    elif metric == "pwcca":
        configs = [
            (i, j, video_layers_list[i], text_layers_list[j], num_permutations, alpha)
            for i in range(n_video)
            for j in range(n_text)
        ]
        compute_fn = _prh_compute_pair_pwcca_cached
    elif metric == "procrustes":
        configs = [
            (i, j, video_layers_list[i], text_layers_list[j], num_permutations, alpha)
            for i in range(n_video)
            for j in range(n_text)
        ]
        compute_fn = _prh_compute_pair_procrustes_cached
    else:
        configs = [
            (
                i,
                j,
                video_layers_list[i],
                text_layers_list[j],
                metric,
                k,
                num_permutations,
                alpha,
            )
            for i in range(n_video)
            for j in range(n_text)
        ]
        compute_fn = _prh_compute_pair_generic

    total_pairs = len(configs)
    if num_workers > 1 and device == "cpu":
        from concurrent.futures import ProcessPoolExecutor

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

    logger.info(f"Computed {n_video} x {n_text} = {n_video * n_text} alignments")

    # Save results
    logger.info(f"State: {ExperimentState.SAVING_RESULTS}")
    save_path = prh_alignment_filename(
        os.path.join(output_dir, "alignment_vatex"),
        "vatex",
        f"{video_modelset}_{text_modelset}",
        "video",
        pool_video,
        False,
        "text",
        pool_text,
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
        "video_models": video_models,
        "text_models": text_models,
        "video_params": video_params,
        "text_params": text_params,
        "num_frames": num_frames,
        "num_frames_framewise": num_frames_framewise,
        "num_captions": num_captions,
        "average_caption_features": average_caption_features,
        "dataset": "vatex",
    }
    np.save(save_path, payload)
    logger.success(f"Results saved to {save_path}")

    return payload
