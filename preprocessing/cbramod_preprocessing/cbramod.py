"""Preserve the existing CBraMod preprocessing imports."""

from .bids import (
    _dataset_description,
    _write_derivative_description,
    bids_output_path,
    channels_sidecar_path,
    read_bids_bad_channels,
    run_bids_dataset,
)
from .io import DataIO
from .processing import (
    PLAUSIBLE_MEDIAN_UV,
    CBraModConfig,
    CBraModProcessor,
    CBraModResult,
    reject_and_format_windows,
)

__all__ = [
    "PLAUSIBLE_MEDIAN_UV",
    "CBraModConfig",
    "CBraModProcessor",
    "CBraModResult",
    "DataIO",
    "_dataset_description",
    "_write_derivative_description",
    "bids_output_path",
    "channels_sidecar_path",
    "read_bids_bad_channels",
    "reject_and_format_windows",
    "run_bids_dataset",
]
