"""Signal conditioning and CBraMod window formatting."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Median |amplitude| a filtered EEG window must fall inside, as a guard against
# recordings whose units were misread (volts read as microvolts, and vice versa).
PLAUSIBLE_MEDIAN_UV = (0.05, 500.0)


@dataclass(frozen=True)
class CBraModConfig:
    """Parameters for the ordered CBraMod signal-processing recipe."""

    channels: tuple[str, ...]
    channel_types: dict[str, str] = field(default_factory=dict)
    channel_aliases: dict[str, str] = field(default_factory=dict)
    average_reference: bool = False
    sampling_rate: float = 200.0
    bandpass: tuple[float, float] = (0.3, 75.0)
    # Pinned explicitly so the realized filter does not drift with MNE's defaults.
    filter_transition_hz: tuple[float, float] = (0.3, 18.75)
    notch_frequency: float = 50.0
    notch_width_hz: float = 0.25
    n_jobs: int = 1
    minimum_recording_seconds: float = 300.0
    edge_crop_seconds: float = 60.0
    window_seconds: float = 30.0
    amplitude_limit_volts: float = 100e-6
    normalize_by_limit: bool = True

    def __post_init__(self) -> None:
        if not self.channels or len(self.channels) != len(set(self.channels)):
            raise ValueError("channels must be a non-empty ordered set")
        if self.sampling_rate <= 0 or self.window_seconds <= 0:
            raise ValueError("sampling_rate and window_seconds must be positive")
        if self.minimum_recording_seconds <= 0 or self.edge_crop_seconds < 0:
            raise ValueError("minimum duration must be positive and edge crop non-negative")
        if self.amplitude_limit_volts <= 0:
            raise ValueError("amplitude limit must be positive")
        if not 0 <= self.bandpass[0] < self.bandpass[1] < self.sampling_rate / 2:
            raise ValueError("bandpass must lie between 0 Hz and the target Nyquist frequency")
        if not 0 < self.notch_frequency < self.sampling_rate / 2:
            raise ValueError("notch_frequency must lie below the target Nyquist frequency")
        if not float(self.sampling_rate).is_integer() or not float(self.window_seconds).is_integer():
            raise ValueError("sampling_rate and window_seconds must be integers for 1-second patches")

    @property
    def samples_per_window(self) -> int:
        samples = self.window_seconds * self.sampling_rate
        if not float(samples).is_integer():
            raise ValueError("window_seconds * sampling_rate must be an integer")
        return int(samples)

    def manifest_settings(self) -> dict:
        return {
            "channels": list(self.channels),
            "channel_aliases": self.channel_aliases,
            "average_reference": self.average_reference,
            "sampling_rate_hz": self.sampling_rate,
            "bandpass_hz": list(self.bandpass),
            "filter_transition_hz": list(self.filter_transition_hz),
            "notch_hz": self.notch_frequency,
            "notch_width_hz": self.notch_width_hz,
            "minimum_recording_seconds": self.minimum_recording_seconds,
            "edge_crop_seconds_per_side": self.edge_crop_seconds,
            "window_seconds": self.window_seconds,
            "amplitude_limit_uv": self.amplitude_limit_volts * 1e6,
            "normalized_by_amplitude_limit": self.normalize_by_limit,
            "operation_order": [
                "duration_check", "reconstruct_reference", "average_reference_all_eeg",
                "select_channels", "bandpass", "notch", "resample", "crop_edges",
                "window", "reject_windows", "normalize",
            ],
        }


@dataclass
class CBraModResult:
    data: np.ndarray
    candidate_windows: int
    rejected_windows: int
    skip_reason: str | None = None


class CBraModProcessor:
    """Apply CBraMod's preprocessing operations in an explicit fixed order."""

    def __init__(self, config: CBraModConfig) -> None:
        self.config = config

    def prepare_raw(self, raw):
        """Select and filter cropped EEG, then resample it for CBraMod."""
        raw.load_data()
        types = {name: kind for name, kind in self.config.channel_types.items() if name in raw.ch_names}
        if types:
            raw.set_channel_types(types, on_unit_change="ignore", verbose=False)

        aliases = {
            source: target
            for source, target in self.config.channel_aliases.items()
            if source in raw.ch_names
        }
        collisions = [target for target in aliases.values() if target in raw.ch_names]
        if collisions:
            raise ValueError(f"Cannot reconstruct channel(s) already present: {collisions}")
        if aliases:
            raw.rename_channels(aliases)

        missing = [name for name in self.config.channels if name not in raw.ch_names]
        if missing:
            raise ValueError(f"Recording is missing required channel(s): {missing}")

        # Bad-channel labels are audit metadata here, not a channel rejection rule.
        raw.info["bads"] = []
        if self.config.average_reference:
            # A target channel typed anything but EEG is skipped by the average
            # reference, which would silently leave a reconstructed channel flat.
            not_eeg = [
                name for name, kind in
                zip(self.config.channels, raw.get_channel_types(picks=list(self.config.channels)))
                if kind != "eeg"
            ]
            if not_eeg:
                raise ValueError(f"Channel(s) must be typed 'eeg' before referencing: {not_eeg}")
            raw.set_eeg_reference("average", projection=False, verbose=False)
        raw.pick(list(self.config.channels))
        if raw.info["sfreq"] / 2 <= self.config.bandpass[1] + self.config.filter_transition_hz[1] / 2:
            raise ValueError("source sampling rate is too low for the requested band-pass")
        raw.filter(
            *self.config.bandpass, method="fir", phase="zero",
            l_trans_bandwidth=self.config.filter_transition_hz[0],
            h_trans_bandwidth=self.config.filter_transition_hz[1],
            n_jobs=self.config.n_jobs, verbose=False,
        )
        raw.notch_filter(
            [self.config.notch_frequency], method="fir",
            notch_widths=self.config.notch_width_hz,
            n_jobs=self.config.n_jobs, verbose=False,
        )
        raw.resample(
            self.config.sampling_rate, npad="auto", n_jobs=self.config.n_jobs, verbose=False
        )
        return raw

    def run(self, raw) -> CBraModResult:
        """Return accepted 30-second windows in channel/patch/sample layout."""
        cfg = self.config
        source_rate = float(raw.info["sfreq"])
        if raw.n_times < round(cfg.minimum_recording_seconds * source_rate):
            return CBraModResult(
                data=self._empty_output(),
                candidate_windows=0,
                rejected_windows=0,
                skip_reason=f"recording is shorter than {cfg.minimum_recording_seconds:g} seconds",
            )

        # Filter before cropping so the FIR transients land in the discarded edges.
        prepared = self.prepare_raw(raw)
        edge = int(round(cfg.edge_crop_seconds * cfg.sampling_rate))
        candidates = int(max(prepared.n_times - 2 * edge, 0) // cfg.samples_per_window)
        if candidates == 0:
            return CBraModResult(
                data=self._empty_output(),
                candidate_windows=0,
                rejected_windows=0,
                skip_reason="no complete window remains after edge cropping",
            )

        windows = prepared.get_data(
            start=edge, stop=edge + candidates * cfg.samples_per_window
        ).reshape(len(cfg.channels), candidates, cfg.samples_per_window).transpose(1, 0, 2)
        median_uv = float(np.median(np.abs(windows))) * 1e6
        if not PLAUSIBLE_MEDIAN_UV[0] <= median_uv <= PLAUSIBLE_MEDIAN_UV[1]:
            raise ValueError(
                f"implausible median amplitude {median_uv:.3g} uV; check the recording's units"
            )
        accepted, rejected = reject_and_format_windows(windows, cfg)
        return CBraModResult(accepted, candidates, rejected)

    def _empty_output(self) -> np.ndarray:
        cfg = self.config
        return np.empty(
            (0, len(cfg.channels), int(cfg.window_seconds), int(cfg.sampling_rate)),
            dtype=np.float32,
        )


def reject_and_format_windows(windows: np.ndarray, config: CBraModConfig) -> tuple[np.ndarray, int]:
    """Reject complete windows at the amplitude limit and format accepted data."""
    keep = np.all(np.abs(windows) <= config.amplitude_limit_volts, axis=(1, 2))
    accepted = np.asarray(windows[keep], dtype=np.float32)
    if config.normalize_by_limit:
        accepted /= np.float32(config.amplitude_limit_volts)
    else:
        accepted *= np.float32(1e6)
    accepted = accepted.reshape(
        len(accepted), len(config.channels), int(config.window_seconds), int(config.sampling_rate)
    )
    return accepted, int(np.count_nonzero(~keep))
