"""Data generators for experiments."""

from .common import (
    center_norm_features,
    knn_indicator,
    projection_matrix,
    projection_matrix_no_rng,
)
from .gen1 import gen1_mixture_invariance
from .gen2 import (
    gen2_geometry,
    gen2_geometry_state,
    gen2_linear,
    gen2_linear_from_state,
    gen2_linear_shared_state,
    gen2_linear_state,
    gen2_local,
    gen2_local_state,
    make_gen2_linear_signal,
)
from .layerwise import (
    make_random_layers,
    make_random_layers_batched,
    make_random_layers_with_rng,
    make_signal_layers,
    make_signal_layers_batched,
)
from .low_rank import (
    make_low_rank_signal,
    make_low_rank_signal_unitvar,
    make_pure_noise,
)
from .noise import (
    DEFAULT_NOISE_TYPE,
    MIXTURE_PROB,
    NOISE_TYPES,
    STUDENT_T_DF,
    sample_noise,
    strength_from_snr,
)

__all__ = [
    # noise
    "NOISE_TYPES",
    "DEFAULT_NOISE_TYPE",
    "STUDENT_T_DF",
    "MIXTURE_PROB",
    "sample_noise",
    "strength_from_snr",
    # low_rank
    "make_pure_noise",
    "make_low_rank_signal",
    "make_low_rank_signal_unitvar",
    # common
    "projection_matrix",
    "projection_matrix_no_rng",
    "center_norm_features",
    "knn_indicator",
    # gen2_linear
    "gen2_linear_shared_state",
    "gen2_linear_from_state",
    "make_gen2_linear_signal",
    "gen2_linear_state",
    "gen2_linear",
    # gen2_geometry
    "gen2_geometry_state",
    "gen2_geometry",
    # gen2_local
    "gen2_local_state",
    "gen2_local",
    # gen1
    "gen1_mixture_invariance",
    # layerwise
    "make_random_layers",
    "make_random_layers_batched",
    "make_random_layers_with_rng",
    "make_signal_layers",
    "make_signal_layers_batched",
]
