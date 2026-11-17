"""Writing and merging the per-job outputs ("shards") of the CBraMod run."""

import glob
import json
import os

import pandas as pd

OUTPUTS = ("predictions", "embeddings", "inner_folds")


def shard_dir(results_dir, task_name, job_index):
    return os.path.join(results_dir, task_name, "cbramod_shards", f"job{job_index:02d}")


def merge(results_dir, task_name):
    """Combine all shards of a task; checks every outer fold is present exactly once.

    Writes results/<task>/cbramod_{predictions,embeddings,inner_folds}.csv.
    """
    task_dir = os.path.join(results_dir, task_name)
    shards = sorted(glob.glob(os.path.join(task_dir, "cbramod_shards", "job*")))
    if not shards:
        raise FileNotFoundError(f"No shards in {task_dir}/cbramod_shards")
    with open(os.path.join(shards[0], "manifest.json")) as f:
        n_folds = json.load(f)["n_outer_folds"]

    for name in OUTPUTS:
        df = pd.concat([pd.read_csv(os.path.join(s, f"{name}.csv")) for s in shards], ignore_index=True)
        df = df.sort_values([c for c in ("outer_fold", "inner_fold") if c in df]).reset_index(drop=True)
        if name == "predictions":
            folds = df["outer_fold"].tolist()
            missing = sorted(set(range(1, n_folds + 1)) - set(folds))
            dupes = sorted({f for f in folds if folds.count(f) > 1})
            if missing or dupes:
                raise RuntimeError(f"{task_name}: outer folds missing {missing}, duplicated {dupes}")
        df.to_csv(os.path.join(task_dir, f"cbramod_{name}.csv"), index=False)
    print(f"{task_name}: merged {len(shards)} job(s); all {n_folds} outer folds present.")
