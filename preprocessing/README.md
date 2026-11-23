# CBraMod preprocessing for LEMON-PD-EEG

This repository prepares resting-state EEG from [OpenNeuro ds007526 v1.0.2](https://doi.org/10.18112/openneuro.ds007526.v1.0.2) for input to CBraMod. Download the BIDS dataset, including the original EEGLAB recordings and sidecars, before running the pipeline.

## Run the pipeline

Run these commands from this `preprocessing` folder. Use Python 3.11 and install the dependencies in a virtual environment.

```sh
python -m pip install -r requirements.txt
python preprocess_ds007526.py "path/to/ds007526" --output-dir "path/to/derivatives/cbramod/ds007526" --dry-run
python preprocess_ds007526.py "path/to/ds007526" --output-dir "path/to/derivatives/cbramod/ds007526"
```

Rest is the default task. The dry run checks dataset identity and discovers matching recordings without writing derivatives. Without an explicit output directory, the script writes under the dataset's own `derivatives/cbramod` directory. Existing arrays are reused unless you pass `--overwrite`. Use `--help` for all options.

## Processing and outputs

Recordings shorter than 60 seconds are skipped. The pipeline renames `VREF` to `Cz` when present, applies an average reference across all EEG channels, then selects and orders the 19 target channels. It applies a 0.3–75 Hz FIR band-pass filter, a 50 Hz notch filter, and resampling to 200 Hz. The pipeline trims 15 seconds from each end and divides the remaining samples into complete, nonoverlapping 30-second windows.

A window is rejected if any sample exceeds 100 µV in absolute amplitude. By default, accepted samples are divided by the 100 µV threshold. BIDS bad-channel labels are recorded in the manifest and do not drive channel exclusion.

For each recording with retained windows, the pipeline writes a `float32` NumPy array with shape (windows, 19 channels, 30 one-second patches, 200 samples per patch). The output mirrors the input subject/session layout and includes `dataset_description.json` and `cbramod_preprocessing_manifest.json`. The manifest reports package versions, settings, recording-level outcomes, and absolute paths from the local run.

## Code

- **preprocess_ds007526.py** defines the dataset-specific configuration and command-line interface.
- **cbramod_preprocessing/processing.py** implements signal processing and window formatting.
- **cbramod_preprocessing/bids.py** handles BIDS discovery, export, and provenance, using the file reader in **io.py**.
