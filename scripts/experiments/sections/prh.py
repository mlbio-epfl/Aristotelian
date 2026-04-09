"""PRH (Platonic Representation Hypothesis) alignment experiments.

This module includes:
1. Cross-modal alignment (language <-> vision) following PRH main text
2. Video-to-text alignment (VideoPRH)
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from aristotelian.prh.prh_experiment import run_prh_experiment

from ..infra.io import save_array, should_skip

PRH_METRICS = (
    # Spectral / global — O(d/n) null baseline (Proposition 4.1)
    "cka_lin",
    "cka_rbf",
    "unbiased_cka",
    "rv_coefficient",
    "cca",
    # Geometric — d-dependent null baseline
    "procrustes",
    # Neighborhood / local — O(k/n) null baseline (Proposition 4.2)
    "mutual_knn",
    "cycle_knn",
    "cknna",
)
# k values for kNN-based metrics that sweep over k (mutual_knn, cknna)
PRH_K_VALUES = (10, 20, 50, 100)
# sigma values for RBF kernel CKA (small=local, large=global)
# Default is 1.0; sweep explores local-to-global spectrum
PRH_SIGMA_VALUES = (0.1, 0.5, 1.0, 2.0, 5.0)
PRH_Q_OUTLIER = 0.95
# Metrics that use k parameter
KNN_BASED_METRICS = ("mutual_knn", "cycle_knn", "cknna")
# Metrics that sweep over multiple k values (others use default k=10)
K_SWEEP_METRICS = ("mutual_knn", "cknna")
# Metrics that use sigma parameter (RBF kernel bandwidth)
RBF_SIGMA_METRICS = ("cka_rbf",)


def run_prh_alignment(
    assets_dir: Path,
    *,
    device: str,
    force: bool,
    force_features: bool = False,
    seed: int | None,
    num_workers: int = 1,
    prh_metrics: tuple = PRH_METRICS,
    prh_k_values: tuple = PRH_K_VALUES,
    prh_sigma_values: tuple = PRH_SIGMA_VALUES,
    prh_q_outlier: float = PRH_Q_OUTLIER,
) -> None:
    """Run PRH alignment experiment across multiple metrics.

    For mutual_knn and cknna, experiments are run for each k value in
    prh_k_values. For RBF kernel metrics (cka_rbf), experiments are run for
    each sigma value in prh_sigma_values. Other metrics (including cycle_knn)
    use the default k=10.
    """
    del seed  # unused but kept for consistent interface
    for metric in prh_metrics:
        # Determine k values to use for this metric
        # Only mutual_knn and cknna sweep over k values; others use default k=10
        if metric in K_SWEEP_METRICS:
            k_values = prh_k_values
        else:
            k_values = (10,)  # Default k for non-sweep metrics

        # Determine sigma values to use for this metric
        if metric in RBF_SIGMA_METRICS:
            sigma_values = prh_sigma_values
        else:
            sigma_values = (1.0,)  # Placeholder, sigma not used for non-RBF metrics

        for k in k_values:
            for sigma in sigma_values:
                # Build output filename
                if metric == "mutual_knn" and k == 10:
                    # Default case for backward compatibility
                    output_name = "prh_alignment.npy"
                elif metric in KNN_BASED_METRICS:
                    output_name = f"prh_alignment_{metric}_k{k}.npy"
                elif metric in RBF_SIGMA_METRICS and sigma == 1.0:
                    # Default sigma=1.0 case for backward compatibility
                    output_name = f"prh_alignment_{metric}.npy"
                elif metric in RBF_SIGMA_METRICS:
                    output_name = f"prh_alignment_{metric}_sigma{sigma}.npy"
                else:
                    output_name = f"prh_alignment_{metric}.npy"

                output = assets_dir / output_name
                if should_skip([output], force):
                    logger.info(
                        f"Skipping prh_alignment ({metric}, k={k}, sigma={sigma}, output exists: {output})"
                    )
                    continue

                out = run_prh_experiment(
                    dataset="minhuh/prh",
                    subset="wit_1024",
                    split="train",
                    modelset="val",
                    modality_x="language",
                    pool_x="avg",
                    prompt_x=False,
                    modality_y="vision",
                    pool_y="cls",
                    prompt_y=False,
                    caption_idx=0,
                    max_samples=None,
                    batch_size=4,
                    device=device,
                    output_dir="./results",
                    k=k,
                    metric=metric,
                    rbf_sigma=sigma,
                    num_permutations=500,
                    alpha=0.05,
                    q_outlier=prh_q_outlier,
                    force_features=force_features,
                    num_workers=num_workers,
                )
                save_array(output, out)


def run_v2t_alignment(
    assets_dir: Path,
    *,
    device: str,
    force: bool,
    force_features: bool = False,
    seed: int | None,
    num_workers: int = 1,
    v2t_metrics: tuple = PRH_METRICS,
    v2t_k_values: tuple = PRH_K_VALUES,
    v2t_sigma_values: tuple = PRH_SIGMA_VALUES,
    v2t_video_modelset: str = "videoprh",
    v2t_text_modelset: str = "videoprh",
    v2t_q_outlier: float = PRH_Q_OUTLIER,
    num_frames: int = 16,
    num_captions: int = 1,
    streaming_limit: int | None = 5000,
) -> None:
    """Run video-to-text alignment experiment (VideoPRH).

    This experiment compares video models to text models using the PE-Video (PVD)
    dataset, demonstrating PRH in the video-text domain.

    For mutual_knn and cknna, experiments are run for each k value in
    v2t_k_values. For RBF kernel metrics (cka_rbf), experiments are run for
    each sigma value in v2t_sigma_values. Other metrics (including cycle_knn)
    use the default k=10.
    """
    del seed  # unused but kept for consistent interface
    from aristotelian.prh.v2t_experiment import run_v2t_experiment

    for metric in v2t_metrics:
        # Determine k values to use for this metric
        # Only mutual_knn and cknna sweep over k values; others use default k=10
        if metric in K_SWEEP_METRICS:
            k_values = v2t_k_values
        else:
            k_values = (10,)  # Default k for non-sweep metrics

        # Determine sigma values to use for this metric
        if metric in RBF_SIGMA_METRICS:
            sigma_values = v2t_sigma_values
        else:
            sigma_values = (1.0,)  # Placeholder, sigma not used for non-RBF metrics

        for k in k_values:
            for sigma in sigma_values:
                # Build output filename
                if metric == "mutual_knn" and k == 10:
                    output_name = (
                        f"v2t_alignment_{v2t_video_modelset}_{v2t_text_modelset}.npy"
                    )
                elif metric in KNN_BASED_METRICS:
                    output_name = f"v2t_alignment_{v2t_video_modelset}_{v2t_text_modelset}_{metric}_k{k}.npy"
                elif metric in RBF_SIGMA_METRICS and sigma == 1.0:
                    output_name = f"v2t_alignment_{v2t_video_modelset}_{v2t_text_modelset}_{metric}.npy"
                elif metric in RBF_SIGMA_METRICS:
                    output_name = f"v2t_alignment_{v2t_video_modelset}_{v2t_text_modelset}_{metric}_sigma{sigma}.npy"
                else:
                    output_name = f"v2t_alignment_{v2t_video_modelset}_{v2t_text_modelset}_{metric}.npy"

                output = assets_dir / output_name
                if should_skip([output], force):
                    logger.info(
                        f"Skipping v2t_alignment ({metric}, k={k}, sigma={sigma}, output exists: {output})"
                    )
                    continue

                out = run_v2t_experiment(
                    dataset="pvd",
                    split="test",
                    video_modelset=v2t_video_modelset,
                    text_modelset=v2t_text_modelset,
                    pool_video="cls",
                    pool_text="avg",
                    num_frames=num_frames,
                    num_captions=num_captions,
                    max_samples=1024,  # Per VideoPRH paper
                    streaming_limit=streaming_limit,  # For faster data loading
                    batch_size=4,
                    device=device,
                    output_dir="./results",
                    k=k,
                    metric=metric,
                    rbf_sigma=sigma,
                    num_permutations=500,
                    alpha=0.05,
                    q_outlier=v2t_q_outlier,
                    force_features=force_features,
                    num_workers=num_workers,
                )
                save_array(output, out)
