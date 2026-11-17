"""Classical EEG-feature baseline for one task, on the same folds as CBraMod.

Needs no GPU. The subjects and their fold order come from the data folder,
exactly as in train_cbramod.py.

    python scripts/train_classical.py --task pd_hc --data-dir data
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cbramod_pd import config as C                                               # noqa: E402
from cbramod_pd.classical import align_features, load_subject_features, run_loso  # noqa: E402
from cbramod_pd.data import build_task_data                                       # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(C.TASKS))
    p.add_argument("--data-dir", default="data",
                   help="Folder with eeg/PD, eeg/HC, participants.csv and classical_features.csv")
    p.add_argument("--features", default=None, help="Feature table (default: <data-dir>/classical_features.csv)")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--npy-units", default="cbramod", choices=["cbramod", "uV", "V"])
    args = p.parse_args()

    task = C.TASKS[args.task]
    task_dir = os.path.join(args.results_dir, task.name)
    os.makedirs(task_dir, exist_ok=True)

    subjects, _, _ = build_task_data(task, args.data_dir, args.npy_units, load_eeg=False)
    cbramod_csv = os.path.join(task_dir, "cbramod_predictions.csv")
    if os.path.exists(cbramod_csv):      # guard: identical subjects and order as the CBraMod run
        cb = pd.read_csv(cbramod_csv, dtype={"subject_id": str}).sort_values("outer_fold")
        if cb["subject_id"].tolist() != subjects["subject_id"].tolist():
            raise RuntimeError(f"Subjects differ from the CBraMod run in {cbramod_csv}")

    features = args.features or os.path.join(args.data_dir, "classical_features.csv")
    aligned = align_features(load_subject_features(features), subjects)
    aligned.to_csv(os.path.join(task_dir, "classical_features.csv"), index=False)

    for name, df in run_loso(aligned, task.kind).items():
        df.to_csv(os.path.join(task_dir, f"classical_predictions_{name}.csv"), index=False)
        print(f"{task.title} - {name}: {len(df)} subjects")


if __name__ == "__main__":
    main()
