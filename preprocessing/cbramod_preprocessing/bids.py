"""BIDS recording discovery, derivative export, and provenance manifests."""

from __future__ import annotations

import csv
import json
import platform
from pathlib import Path
from collections.abc import Sequence

import mne
import numpy as np
from tqdm import tqdm

from .io import DataIO
from .processing import CBraModProcessor


def _bids_stem(path: Path) -> str:
    """Return the file name up to the BIDS ``_eeg`` suffix (or the plain stem)."""
    marker = path.name.lower().rfind("_eeg.")
    return path.name[:marker] if marker >= 0 else path.stem


def channels_sidecar_path(input_path: Path) -> Path:
    return input_path.with_name(f"{_bids_stem(input_path)}_channels.tsv")


def read_bids_bad_channels(input_path: Path) -> list[str]:
    """Read bad-channel labels for auditing without changing the signal."""
    sidecar = channels_sidecar_path(input_path)
    if not sidecar.is_file():
        return []
    with sidecar.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"name", "status"}.issubset(reader.fieldnames):
            raise ValueError(f"BIDS channels sidecar lacks name/status columns: {sidecar}")
        return [
            row["name"].strip()
            for row in reader
            if row.get("status", "").strip().lower() == "bad"
        ]


def bids_output_path(input_path: Path, dataset_root: Path, output_dir: Path) -> Path:
    relative = input_path.relative_to(dataset_root)
    return output_dir / relative.parent / f"{_bids_stem(relative)}_desc-cbramod_eeg.npy"


def _dataset_description(dataset_root: Path, expected_identity: str) -> dict:
    path = dataset_root / "dataset_description.json"
    if not path.is_file():
        raise FileNotFoundError(f"BIDS dataset_description.json is missing: {dataset_root}")
    description = json.loads(path.read_text(encoding="utf-8-sig"))
    identity = f"{description.get('Name', '')} {description.get('DatasetDOI', '')}".lower()
    if expected_identity.lower() not in identity:
        raise ValueError(f"Dataset does not match expected identity {expected_identity!r}")
    return description


def _write_derivative_description(output_dir: Path, source: dict, dataset_name: str) -> Path:
    source_dataset = {}
    if doi := source.get("DatasetDOI"):
        source_dataset["DOI"] = doi
    path = output_dir / "dataset_description.json"
    path.write_text(json.dumps({
        "Name": dataset_name,
        "BIDSVersion": source.get("BIDSVersion", "1.10.0"),
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "EEG-MNE CBraMod preprocessing"}],
        "SourceDatasets": [source_dataset],
    }, indent=2), encoding="utf-8")
    return path


def run_bids_dataset(
    dataset_root: Path,
    output_dir: Path,
    processor: CBraModProcessor,
    *,
    tasks: Sequence[str],
    expected_identity: str,
    dataset_name: str,
    overwrite: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Discover BIDS EEGLAB recordings and export CBraMod arrays plus a manifest."""
    dataset_root, output_dir = dataset_root.resolve(), output_dir.resolve()
    source = _dataset_description(dataset_root, expected_identity)
    files = [
        path for path in DataIO([".set"]).discover_files(dataset_root, tasks=list(tasks))
        if "derivatives" not in {part.lower() for part in path.relative_to(dataset_root).parts}
    ]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        files = files[:limit]
    if not files:
        raise FileNotFoundError("No matching EEGLAB .set files were found")

    summary = {
        "dataset": dataset_name,
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "tasks": list(tasks),
        "files_seen": len(files),
        "dry_run": dry_run,
        "environment": {
            "python": platform.python_version(),
            "mne": mne.__version__,
            "numpy": np.__version__,
        },
        "settings": processor.config.manifest_settings(),
        "records": [],
    }
    if dry_run:
        summary["records"] = [{
            "input_path": str(path),
            "output_path": str(bids_output_path(path, dataset_root, output_dir)),
            "status": "planned",
        } for path in files]
        return summary

    output_dir.mkdir(parents=True, exist_ok=True)
    summary["derivative_description_path"] = str(
        _write_derivative_description(output_dir, source, f"{dataset_name} CBraMod windows")
    )
    data_io = DataIO([".set"])
    for input_path in tqdm(files, desc=f"{dataset_name} -> CBraMod"):
        output_path = bids_output_path(input_path, dataset_root, output_dir)
        if output_path.exists() and not overwrite:
            # Carry the existing window count forward so a resumed run still
            # produces a manifest that describes the whole derivative.
            shape = np.load(output_path, mmap_mode="r").shape
            entry = {"input_path": str(input_path), "output_path": str(output_path),
                     "status": "skipped", "reason": "output already exists",
                     "shape": list(shape), "retained_windows": int(shape[0])}
        else:
            try:
                raw = data_io.load_file(input_path, preload=True)
                source_duration = raw.n_times / float(raw.info["sfreq"])
                recorded_bads = list(dict.fromkeys(raw.info["bads"] + read_bids_bad_channels(input_path)))
                result = processor.run(raw)
                common = {
                    "input_path": str(input_path),
                    "duration_seconds": source_duration,
                    "candidate_windows": result.candidate_windows,
                    "retained_windows": len(result.data),
                    "rejected_windows": result.rejected_windows,
                    "channels": list(processor.config.channels),
                    "bids_bad_channels": recorded_bads,
                }
                if len(result.data) == 0:
                    if overwrite and output_path.exists():
                        output_path.unlink()
                    entry = {**common, "status": "skipped",
                             "reason": result.skip_reason or "no window passed the amplitude criterion"}
                else:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(output_path, result.data, allow_pickle=False)
                    entry = {**common, "output_path": str(output_path), "status": "success",
                             "shape": list(result.data.shape), "dtype": str(result.data.dtype)}
            except Exception as exc:
                entry = {"input_path": str(input_path), "status": "failed",
                         "error_type": type(exc).__name__, "error": str(exc)}
        summary["records"].append(entry)

    records = summary["records"]
    summary.update(
        files_succeeded=sum(row["status"] == "success" for row in records),
        files_skipped=sum(row["status"] == "skipped" for row in records),
        files_failed=sum(row["status"] == "failed" for row in records),
        retained_windows=sum(row.get("retained_windows", 0) for row in records),
    )
    manifest = output_dir / "cbramod_preprocessing_manifest.json"
    manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["manifest_path"] = str(manifest)
    return summary
