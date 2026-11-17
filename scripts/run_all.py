"""Run the whole training pipeline locally: CBraMod and the classical baseline, all four tasks.

    python scripts/run_all.py --data-dir data \
        --cbramod-dir CBraMod/models --weights pretrained_weights/pretrained_weights.pth

For each task: fine-tune CBraMod (train_cbramod.py), fit the classical
models on the same folds (train_classical.py); then print the pooled
performance of every model (summarize.py). Expect several days on one GPU.
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cbramod_pd import config as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, *args):
    cmd = [sys.executable, os.path.join(HERE, script), *args]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=list(C.TASKS), choices=list(C.TASKS))
    p.add_argument("--data-dir", default="data")
    p.add_argument("--cbramod-dir", default="CBraMod/models")
    p.add_argument("--weights", default="pretrained_weights/pretrained_weights.pth")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--skip-cbramod", action="store_true", help="Only the classical baseline (no GPU needed)")
    args = p.parse_args()

    common = ["--data-dir", args.data_dir, "--results-dir", args.results_dir]
    for task in args.tasks:
        if not args.skip_cbramod:
            run("train_cbramod.py", "--task", task, *common,
                "--cbramod-dir", args.cbramod_dir, "--weights", args.weights)
        run("train_classical.py", "--task", task, *common)
    if not args.skip_cbramod:
        run("summarize.py", "--results-dir", args.results_dir)


if __name__ == "__main__":
    main()
