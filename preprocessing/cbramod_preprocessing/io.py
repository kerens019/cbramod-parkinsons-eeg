"""EEG file discovery with BIDS entity filtering, and format-specific readers."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import mne


class DataIO:
    """Handles file discovery and loading for EEG data."""

    def __init__(self, supported_extensions: list[str] | None = None) -> None:
        """Initialize the I/O helper with the file types this pipeline supports."""
        self.supported_extensions = supported_extensions or [
            ".set", ".edf", ".bdf", ".vhdr", ".fif", ".fif.gz"
        ]

    def _matches_supported_extension(self, path: Path) -> bool:
        """Check a file name against supported_extensions (handles multi-part suffixes like .fif.gz)."""
        name = path.name.lower()
        return any(name.endswith(ext.lower()) for ext in self.supported_extensions)

    @staticmethod
    def _bids_entities(path: Path, case_sensitive: bool = False) -> dict[str, str]:
        """Extract sub/ses/task entities from a BIDS-like file name."""
        entities = {
            key: value
            for key, value in re.findall(r"(?:^|_)(sub|ses|task)-([^_]+)", path.name)
        }
        if not case_sensitive:
            entities = {key.lower(): value.lower() for key, value in entities.items()}
        return entities

    @staticmethod
    def _normalize_entity(value: str, prefix: str, case_sensitive: bool) -> str:
        text = str(value)
        expected_prefix = f"{prefix}-"
        if text.lower().startswith(expected_prefix):
            text = text[len(expected_prefix):]
        return text if case_sensitive else text.lower()

    def discover_files(
        self,
        input_path: str | Path,
        *,
        tasks: list[str] | None = None,
        case_sensitive: bool = False,
    ) -> list[Path]:
        """Return supported files, optionally filtered by BIDS task."""
        path = Path(input_path)
        if path.is_file():
            if not self._matches_supported_extension(path):
                raise ValueError(f"Unsupported file type: {path}")
            return [path]
        if path.is_dir():
            files = sorted(
                p for p in path.rglob("*")
                if p.is_file() and self._matches_supported_extension(p)
            )
            task_values = (
                {self._normalize_entity(value, "task", case_sensitive) for value in tasks}
                if tasks else None
            )
            if task_values is None:
                return files
            return [
                candidate for candidate in files
                if self._bids_entities(candidate, case_sensitive=case_sensitive).get("task")
                in task_values
            ]
        raise FileNotFoundError(f"Input path not found: {path}")

    def load_file(self, file_path: str | Path, preload: bool = True):
        """Load EEG data from a supported file format."""
        path = Path(file_path)
        name = path.name.lower()
        suffix = path.suffix.lower()

        if suffix == ".set":
            return mne.io.read_raw_eeglab(path, preload=preload, verbose=False)
        if suffix == ".edf":
            return mne.io.read_raw_edf(path, preload=preload, verbose=False)
        if suffix == ".bdf":
            return mne.io.read_raw_bdf(path, preload=preload, verbose=False)
        if suffix == ".vhdr":
            return mne.io.read_raw_brainvision(path, preload=preload, verbose=False)
        if name.endswith(".fif") or name.endswith(".fif.gz"):
            try:
                # The pipeline supports both Raw and Epochs FIF objects. Suppress
                # naming warnings from the deliberately attempted first reader.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    return mne.io.read_raw_fif(path, preload=preload, verbose=False)
            except (OSError, TypeError, ValueError, RuntimeError):
                try:
                    return mne.read_epochs(path, preload=preload, verbose=False)
                except (OSError, TypeError, ValueError, RuntimeError) as epochs_error:
                    raise ValueError(
                        f"FIF file is neither readable Raw nor Epochs data: {path}"
                    ) from epochs_error

        raise ValueError(f"Unsupported file type: {path}")
