"""Experiment implementations."""

from .aggregation import run_exp_b_aggregator_calibration
from .calibration import run_perm_budget, run_type1_calibration
from .null_drift import run_null_drift_gaussian, run_null_drift_heavy
from .prh import run_prh_alignment, run_v2t_alignment
from .snr import (
    SNR_NOISE_POINTS,
    SNR_NOISE_RANGE,
    SNR_PHASE_SIGMAS_POINTS,
    SNR_RANKS,
    SNR_STRENGTHS,
    SNR_STRENGTHS_FN,
    run_phase_diagram,
    run_snr_sweep,
)

__all__ = [
    # null_drift
    "run_null_drift_gaussian",
    "run_null_drift_heavy",
    # calibration
    "run_perm_budget",
    "run_type1_calibration",
    # snr
    "run_snr_sweep",
    "run_phase_diagram",
    # snr constants
    "SNR_NOISE_RANGE",
    "SNR_NOISE_POINTS",
    "SNR_STRENGTHS",
    "SNR_STRENGTHS_FN",
    "SNR_RANKS",
    "SNR_PHASE_SIGMAS_POINTS",
    # prh
    "run_prh_alignment",
    # v2t (video-to-text)
    "run_v2t_alignment",
    # aggregation
    "run_exp_b_aggregator_calibration",
]
