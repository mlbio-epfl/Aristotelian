"""Experiment registry mapping section names to runner functions."""

from __future__ import annotations

from typing import Callable, Dict, Sequence

from .sections import (
    run_exp_b_aggregator_calibration,
    run_null_drift_gaussian,
    run_null_drift_heavy,
    run_perm_budget,
    run_phase_diagram,
    run_prh_alignment,
    run_snr_sweep,
    run_type1_calibration,
    run_v2t_alignment,
)

# Map section names to their runner functions
SECTION_FUNCS: Dict[str, Callable] = {
    "null_drift_gaussian": run_null_drift_gaussian,
    "null_drift_heavy": run_null_drift_heavy,
    "perm_budget": run_perm_budget,
    "type1_calibration": run_type1_calibration,
    "snr_sweep": run_snr_sweep,
    "phase_diagram": run_phase_diagram,
    "prh_alignment": run_prh_alignment,
    "v2t_alignment": run_v2t_alignment,
    "exp_b_aggregator_calibration": run_exp_b_aggregator_calibration,
}


def parse_sections(raw_sections: Sequence[str]) -> Sequence[str]:
    """Parse section arguments into list of section names.

    Args:
        raw_sections: List of section names, possibly comma-separated.
                     Use "all" to get all sections.

    Returns:
        List of validated section names.

    Raises:
        ValueError: If unknown section names are provided.
    """
    if not raw_sections:
        return list(SECTION_FUNCS.keys())
    parts = []
    for item in raw_sections:
        parts.extend(p.strip() for p in item.split(","))
    parts = [p for p in parts if p]
    if "all" in parts:
        return list(SECTION_FUNCS.keys())
    unknown = sorted(set(parts) - set(SECTION_FUNCS.keys()))
    if unknown:
        raise ValueError(f"Unknown sections: {', '.join(unknown)}")
    return parts
