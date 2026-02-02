"""PRH package exports."""

from .l2l_experiment import compute_l2l_intra_family_alignment, run_l2l_experiment
from .prh import run_prh_alignment
from .prh_experiment import run_prh_experiment
from .prh_models import get_models, get_vision_models_v2v
from .pvd_data import extract_video_frames, load_pvd_dataset
from .v2t_experiment import (
    compute_v2t_family_alignment,
    fit_scaling_law,
    run_v2t_experiment,
    run_v2t_scaling_experiment,
    run_v2t_vatex_alignment,
)
from .v2v_experiment import compute_intra_bucket_alignment, run_v2v_experiment
from .vatex_data import load_vatex_dataset
from .video_models import get_video_models, is_native_video_model, load_video_model

__all__ = [
    # PRH cross-modal
    "run_prh_alignment",
    "run_prh_experiment",
    "get_models",
    # V2V (vision-to-vision)
    "get_vision_models_v2v",
    "run_v2v_experiment",
    "compute_intra_bucket_alignment",
    # L2L (language-to-language)
    "run_l2l_experiment",
    "compute_l2l_intra_family_alignment",
    # V2T (video-to-text)
    "run_v2t_experiment",
    "run_v2t_scaling_experiment",
    "run_v2t_vatex_alignment",
    "compute_v2t_family_alignment",
    "fit_scaling_law",
    # Video models
    "get_video_models",
    "load_video_model",
    "is_native_video_model",
    # PVD data
    "load_pvd_dataset",
    "extract_video_frames",
    # VATEX data
    "load_vatex_dataset",
]
