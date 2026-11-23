"""Prepare ds007526 resting-state EEG for CBraMod."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from collections.abc import Sequence

from cbramod_preprocessing import CBraModConfig, CBraModProcessor, run_bids_dataset


warnings.filterwarnings(
    "ignore",
    message="The data contains 'boundary' events",
    category=RuntimeWarning,
)


CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
)
AUXILIARY_TYPES = {
    "EOG1": "eog", "EOG2": "eog", "EOG3": "eog", "EOG4": "eog",
}
DEFAULT_EDGE_CROP_SECONDS = 15.0
DEFAULT_TASKS = ("rest",)


def make_processor(
    edge_crop_seconds: float,
    released_code_scale: bool,
    n_jobs: int = 1,
) -> CBraModProcessor:
    """Define the complete ds007526 CBraMod recipe in one place."""
    config = CBraModConfig(
        channels=CHANNELS,
        n_jobs=n_jobs,
        channel_types=AUXILIARY_TYPES,
        channel_aliases={"VREF": "Cz"},
        average_reference=True,
        sampling_rate=200.0,
        bandpass=(0.3, 75.0),
        notch_frequency=50.0,
        minimum_recording_seconds=60.0,
        edge_crop_seconds=edge_crop_seconds,
        window_seconds=30.0,
        amplitude_limit_volts=100e-6,
        normalize_by_limit=not released_code_scale,
    )
    return CBraModProcessor(config)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Path to the ds007526 BIDS root")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output root (default: DATASET_ROOT/derivatives/cbramod)",
    )
    parser.add_argument("--tasks", nargs="+", choices=("rest", "walk"), default=list(DEFAULT_TASKS))
    parser.add_argument("--edge-crop-seconds", type=float, default=DEFAULT_EDGE_CROP_SECONDS)
    parser.add_argument("--n-jobs", type=int, default=1, help="Workers for filtering and resampling")
    parser.add_argument(
        "--released-code-scale", action="store_true",
        help="Write microvolts instead of dividing accepted samples by the 100 uV limit",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing arrays")
    parser.add_argument("--limit", type=int, help="Process only the first N matching recordings")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Check the dataset and list planned outputs without writing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    output_dir = args.output_dir or args.dataset_root / "derivatives" / "cbramod"
    summary = run_bids_dataset(
        args.dataset_root,
        output_dir,
        make_processor(args.edge_crop_seconds, args.released_code_scale, n_jobs=args.n_jobs),
        tasks=args.tasks,
        expected_identity="ds007526",
        dataset_name="OpenNeuro ds007526",
        overwrite=args.overwrite,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"Dry run valid: {summary['files_seen']} matching recording(s).")
        print(f"Planned output root: {summary['output_dir']}")
    else:
        print(
            f"Finished: {summary['files_succeeded']} succeeded, "
            f"{summary['files_skipped']} skipped, {summary['files_failed']} failed, "
            f"{summary['retained_windows']} windows retained."
        )
        print(f"Manifest: {summary['manifest_path']}")


if __name__ == "__main__":
    main()
