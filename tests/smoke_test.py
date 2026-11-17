"""End-to-end check of the training pipeline on tiny synthetic data (CPU, ~2 minutes).

Uses random EEG and a stand-in for the CBraMod backbone, so it tests that
the code runs (local mode, single jobs + merge, classical baseline,
summary, statistics), not the model itself.

    python tests/smoke_test.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE_CBRAMOD = '''
import torch.nn as nn
class CBraMod(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Linear(200, 200)
        self.encoder = nn.Sequential(nn.GELU(), nn.Dropout(0.1), nn.Linear(200, 200))
'''
FEATURES = ["LZC_mean", "theta_relative_bandpower", "alpha_relative_bandpower",
            "beta_relative_bandpower", "gamma_relative_bandpower"]


def make_data(root):
    rng = np.random.default_rng(0)
    groups = {"PD": [f"P{i:02d}" for i in range(1, 15)], "HC": [f"C{i:02d}" for i in range(1, 8)]}
    for group, ids in groups.items():
        os.makedirs(f"{root}/data/eeg/{group}")
        for sid in ids:
            x = rng.normal(0, 0.1, (int(rng.integers(1, 4)), 19, 6000)).astype(np.float32)
            np.save(f"{root}/data/eeg/{group}/{sid}_task-rest_eeg.npy", x)
    pd.DataFrame({"subject_id": groups["PD"],
                  "moca": rng.integers(17, 31, 14),
                  "updrs3": np.where(rng.random(14) < 0.1, np.nan, rng.integers(5, 40, 14)),
                  "gait_speed": rng.uniform(0.7, 1.4, 14)}).to_csv(f"{root}/data/participants.csv", index=False)
    rows = [{"subject_id": sid, "electrode": e, **dict(zip(FEATURES, rng.uniform(0, 1, 5)))}
            for ids in groups.values() for sid in ids for e in ("Fz", "Cz")]
    pd.DataFrame(rows).to_csv(f"{root}/data/classical_features.csv", index=False)
    os.makedirs(f"{root}/cbramod")
    with open(f"{root}/cbramod/cbramod.py", "w") as f:
        f.write(FAKE_CBRAMOD)
    subprocess.run([sys.executable, "-c",
                    f"import sys,torch; sys.path.insert(0,'{root}/cbramod'); from cbramod import CBraMod; "
                    f"torch.save(CBraMod().state_dict(), '{root}/cbramod/weights.pth')"], check=True)


def run(*args):
    print("$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True, cwd=REPO)


def main():
    with tempfile.TemporaryDirectory() as root:
        make_data(root)
        res = f"{root}/results"
        common = ["--data-dir", f"{root}/data", "--results-dir", res]
        model = ["--cbramod-dir", f"{root}/cbramod", "--weights", f"{root}/cbramod/weights.pth",
                 "--max-epochs", "2", "--allow-cpu"]
        for task in ("pd_hc", "moca", "updrs3", "gait_speed"):
            if task == "moca":       # the AWS path: separate jobs, then merge
                for job in ("0", "1"):
                    run("scripts/train_cbramod.py", "--task", task, *common, *model,
                        "--num-jobs", "2", "--job-index", job)
                run("scripts/train_cbramod.py", "--task", task, "--results-dir", res, "--merge-only")
            else:                    # the local path: all jobs in one run
                run("scripts/train_cbramod.py", "--task", task, *common, *model, "--num-jobs", "2")
            run("scripts/train_classical.py", "--task", task, *common)
        run("scripts/summarize.py", "--results-dir", res)
        run("scripts/permutation_tests.py", "--results-dir", res, "--n-iter", "2000")
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
