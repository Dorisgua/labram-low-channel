"""Clean, composable LaBraM token-disentanglement components."""

from .losses import (
    reconstruction_mse,
    swap_sub_reconstruction,
    swap_task_reconstruction,
    symmetric_info_nce,
)
from .modeling import (
    FULL_DIRECT_SPEC,
    FULL_PROTOTYPE_SPEC,
    MISSING_DIRECT_SPEC,
    MISSING_PROTOTYPE_SPEC,
    ComponentMode,
    CompositionMode,
    MissingFillMode,
    OutputBaseMode,
    ReconstructionModel,
    ReconstructionScope,
    ReconstructionSpec,
    StableCore,
    build_components,
    compose_prediction,
    get_output_base,
)
from .prototype import (
    PrototypeProvider,
    channel_positions_to_token_positions,
    select_positions,
)

__all__ = [
    "ComponentMode",
    "CompositionMode",
    "FULL_DIRECT_SPEC",
    "FULL_PROTOTYPE_SPEC",
    "MISSING_DIRECT_SPEC",
    "MISSING_PROTOTYPE_SPEC",
    "MissingFillMode",
    "OutputBaseMode",
    "PrototypeProvider",
    "ReconstructionModel",
    "ReconstructionScope",
    "ReconstructionSpec",
    "StableCore",
    "build_components",
    "channel_positions_to_token_positions",
    "compose_prediction",
    "get_output_base",
    "reconstruction_mse",
    "select_positions",
    "swap_sub_reconstruction",
    "swap_task_reconstruction",
    "symmetric_info_nce",
]
