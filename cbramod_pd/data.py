"""Loading the preprocessed EEG and the clinical targets.

Expected input folder (see README):
    eeg/PD/<subject>.npy, eeg/HC/<subject>.npy   preprocessed recordings
    participants.csv                              subject_id, moca, updrs3, gait_speed
The subject ID of a recording is its file name up to the first "_"
(so "S001.npy" and "S001_task-rest_eeg.npy" both give "S001").
"""

import os

import numpy as np
import pandas as pd

from . import config as C


def subject_id_from_file(filename):
    return os.path.splitext(os.path.basename(filename))[0].split("_")[0]


def load_npy(path, units="cbramod", check_only=False):
    """Load one subject's preprocessed recording as CBraMod patches.

    The file holds the subject's clean 30-s segments, shaped
    (n_segments, 19, 6000) or already patched as (n_segments, 19, 30, 200).
    `units` says how the values are stored: "cbramod" (1.0 = 100 uV), "uV" or "V".
    Returns float32 patches of shape (n_segments, 19, 30, 200) in CBraMod units.
    With check_only=True the file is only validated (not read into memory)
    and None is returned.
    """
    data = np.load(path, mmap_mode="r" if check_only else None)
    if data.ndim == 2:                       # a single segment
        data = data[np.newaxis]
    if data.ndim == 3:
        n_seg, n_ch, n_t = data.shape
        if n_t != C.N_PATCHES * C.PATCH_LEN:
            raise ValueError(f"{path}: expected {C.N_PATCHES * C.PATCH_LEN} samples per segment, got {n_t}")
        data = data.reshape(n_seg, n_ch, C.N_PATCHES, C.PATCH_LEN)
    if data.ndim != 4 or data.shape[1:] != (C.N_CHANNELS, C.N_PATCHES, C.PATCH_LEN):
        raise ValueError(f"{path}: unexpected shape {data.shape}")
    if data.shape[0] == 0:
        raise ValueError(f"{path}: no segments left after artifact rejection")
    if check_only:
        return None

    scale = {"cbramod": 1.0, "uV": 1.0 / C.UV_PER_UNIT, "V": 1e6 / C.UV_PER_UNIT}[units]
    return (data.astype(np.float64) * scale).astype(np.float32)


def scan_folder(folder, units="cbramod", load_eeg=True):
    """Load every .npy file in `folder`, in sorted file-name order.

    Returns ({subject_id: patches}, [messages for files that could not be loaded]).
    With load_eeg=False the files are only validated and the values are None.
    """
    files = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {folder}")
    eeg, failed = {}, []
    for f in files:
        sid = subject_id_from_file(f)
        if sid in eeg:
            raise ValueError(f"Two recordings for subject {sid} in {folder}")
        try:
            eeg[sid] = load_npy(os.path.join(folder, f), units, check_only=not load_eeg)
        except Exception as e:                # report and skip unreadable recordings
            failed.append(f"{f}: {e}")
            print(f"  could not load {f}: {e}")
    return eeg, failed


def load_scores(participants_csv, column):
    """{subject_id: value} for one column of participants.csv (missing values dropped)."""
    df = pd.read_csv(participants_csv, dtype={"subject_id": str})
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {participants_csv}")
    df["subject_id"] = df["subject_id"].str.strip()
    if df["subject_id"].duplicated().any():
        raise ValueError(f"Duplicate subject_id values in {participants_csv}")
    df = df.dropna(subset=[column])
    return dict(zip(df["subject_id"], df[column]))


def build_task_data(task, data_dir, units="cbramod", load_eeg=True):
    """Load the EEG and targets of one prediction task.

    Returns
        subjects : DataFrame [subject_id, target], in the fixed order that
                   defines the cross-validation folds (sorted file names,
                   PD before HC)
        eeg      : {subject_id: float32 patches}
        failed   : files that could not be loaded
    With load_eeg=False the recordings are only validated (eeg values are None),
    which is enough to get the subjects and folds for the classical baseline.
    """
    eeg_dir = os.path.join(data_dir, "eeg")
    if task.name == "pd_hc":
        eeg, failed, rows = {}, [], []
        for group, label in (("PD", 0), ("HC", 1)):
            g_eeg, g_failed = scan_folder(os.path.join(eeg_dir, group), units, load_eeg)
            if set(g_eeg) & set(eeg):
                raise ValueError(f"Subject IDs found in both PD and HC folders: {sorted(set(g_eeg) & set(eeg))}")
            eeg.update(g_eeg)
            failed += g_failed
            rows += [(sid, label) for sid in g_eeg]
        subjects = pd.DataFrame(rows, columns=["subject_id", "target"])
        return subjects, eeg, failed

    # PD-only tasks: patients with an available score
    eeg, failed = scan_folder(os.path.join(eeg_dir, "PD"), units, load_eeg)
    scores = load_scores(os.path.join(data_dir, "participants.csv"), task.score_col)
    subjects = pd.DataFrame({"subject_id": list(eeg)})
    subjects["score"] = subjects["subject_id"].map(scores)
    n_missing = int(subjects["score"].isna().sum())
    subjects = subjects.dropna(subset=["score"]).reset_index(drop=True)
    print(f"  {n_missing} patients without a {task.score_col} score excluded")

    if task.name == "moca":
        n_low = int((subjects["score"] < C.MOCA_MIN).sum())
        subjects = subjects[subjects["score"] >= C.MOCA_MIN].reset_index(drop=True)
        print(f"  {n_low} patients with MoCA < {C.MOCA_MIN} excluded")
        subjects["target"] = (subjects["score"] >= C.MOCA_NORMAL).astype(int)
    else:
        subjects["target"] = subjects["score"].astype(float)

    eeg = {sid: eeg[sid] for sid in subjects["subject_id"]}
    return subjects[["subject_id", "target"]], eeg, failed
