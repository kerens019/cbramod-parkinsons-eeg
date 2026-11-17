"""Pooled performance of CBraMod and the classical models (the metrics of Tables 2-5).

Every metric is computed once over all held-out subjects of a task, on the
subjects that all three models predicted.

    python scripts/summarize.py
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cbramod_pd import config as C                                         # noqa: E402
from cbramod_pd.classical import make_models                               # noqa: E402
from cbramod_pd.metrics import classification_metrics, regression_metrics  # noqa: E402


def summarize_task(task, task_dir):
    files = {"CBraMod": "cbramod_predictions.csv",
             **{name: f"classical_predictions_{name}.csv" for name in make_models(task.kind)}}
    models = {k: pd.read_csv(os.path.join(task_dir, f), dtype={"subject_id": str}) for k, f in files.items()}
    common = set.intersection(*(set(df["subject_id"]) for df in models.values()))
    rows = []
    for name, df in models.items():
        df = df[df["subject_id"].isin(common)]
        if task.kind == "classification":
            m = classification_metrics(df["y_true"], df["y_prob"], df["y_pred"])
        else:
            m = regression_metrics(df["y_true"], df["y_pred"])
        rows.append({"task": task.name, "model": name, **m})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all", choices=["all"] + list(C.TASKS))
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    for t in (list(C.TASKS) if args.task == "all" else [args.task]):
        task_dir = os.path.join(args.results_dir, t)
        if not os.path.exists(os.path.join(task_dir, "cbramod_predictions.csv")):
            print(f"{t}: no results yet, skipped")
            continue
        table = summarize_task(C.TASKS[t], task_dir)
        table.to_csv(os.path.join(task_dir, "performance.csv"), index=False)
        print(f"\n=== {C.TASKS[t].title} ===")
        print(table.drop(columns="task").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
