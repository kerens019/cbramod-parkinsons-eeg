"""Statistical comparison of CBraMod with each classical model (Methods, "Statistical Analysis").

Run after training (train_cbramod.py and train_classical.py). For every task:
  * paired permutation test of CBraMod vs each classical model; statistic =
    CBraMod minus classical model, in balanced accuracy (from the predicted
    class labels) for classification or RMSE for regression; 1,000,000
    iterations, each swapping every subject's two predictions with
    probability 0.5 (seed 42); two-sided p = (1 + b) / (1 + N);
  * Benjamini-Hochberg FDR correction within the task's two comparisons.

    python scripts/permutation_tests.py                # all tasks
    python scripts/permutation_tests.py --task moca

Writes results/<task>/permutation_tests.csv.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cbramod_pd import config as C                                    # noqa: E402
from cbramod_pd.classical import make_models                          # noqa: E402
from cbramod_pd.stats import N_ITER, fdr_bh, paired_permutation_test  # noqa: E402


def test_task(task, task_dir, n_iter):
    read = lambda f: pd.read_csv(os.path.join(task_dir, f), dtype={"subject_id": str})  # noqa: E731
    cbramod = read("cbramod_predictions.csv")
    classical = {name: read(f"classical_predictions_{name}.csv") for name in make_models(task.kind)}

    # same subjects for all models, in CBraMod's outer-fold order
    # (the random swaps are drawn subject by subject in this order)
    common = set(cbramod["subject_id"]).intersection(*(set(df["subject_id"]) for df in classical.values()))
    order = [s for s in cbramod.sort_values("outer_fold")["subject_id"] if s in common]
    cb = cbramod.set_index("subject_id").loc[order]

    metric = "balanced_accuracy" if task.kind == "classification" else "rmse"
    rows = []
    for name, df in classical.items():
        cl = df.set_index("subject_id").loc[order]
        if not (cl["y_true"].values == cb["y_true"].values).all():
            raise RuntimeError(f"{task.name}: targets differ between CBraMod and {name}")
        diff, p = paired_permutation_test(cb["y_true"].values, cb["y_pred"].values,
                                          cl["y_pred"].values, metric, n_iter=n_iter)
        rows.append({"task": task.name, "comparison": f"CBraMod vs {name}", "n_subjects": len(order),
                     "metric": metric, "difference_cbramod_minus_classical": diff,
                     "n_iterations": n_iter, "p": p})
    table = pd.DataFrame(rows)
    table["p_fdr"] = fdr_bh(table["p"])
    table["significant_fdr_0.05"] = table["p_fdr"] < 0.05
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="all", choices=["all"] + list(C.TASKS))
    p.add_argument("--results-dir", default="results")
    p.add_argument("--n-iter", type=int, default=N_ITER, help=f"Permutations (paper: {N_ITER:,})")
    args = p.parse_args()

    for t in (list(C.TASKS) if args.task == "all" else [args.task]):
        task_dir = os.path.join(args.results_dir, t)
        if not os.path.exists(os.path.join(task_dir, "cbramod_predictions.csv")):
            print(f"{t}: no results yet, skipped")
            continue
        table = test_task(C.TASKS[t], task_dir, args.n_iter)
        table.to_csv(os.path.join(task_dir, "permutation_tests.csv"), index=False)
        print(f"\n=== {C.TASKS[t].title} ({table['metric'].iloc[0]}) ===")
        print(table[["comparison", "difference_cbramod_minus_classical", "p", "p_fdr"]]
              .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
