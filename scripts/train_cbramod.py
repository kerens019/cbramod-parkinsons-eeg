"""Fine-tune CBraMod with leave-one-subject-out cross-validation for one task.

Local run (default): all outer folds on this machine, then the outputs are
merged into results/<task>/cbramod_*.csv.

    python scripts/train_cbramod.py --task pd_hc --data-dir data \
        --cbramod-dir CBraMod/models --weights pretrained_weights/pretrained_weights.pth

The outer folds are processed in "jobs" - consecutive chunks of folds, each
starting from a freshly seeded random state. By default the chunks are the
ones of the paper's run (20 for pd_hc, 5 for the other tasks), so a local
run reproduces the paper's results without any parallel infrastructure.

Single job (--job-index k): runs only chunk k and writes its shard. This is
how the work is split over several GPUs, machines or AWS SageMaker jobs
(see aws/launch_sagemaker.py); merge afterwards with --merge-only.
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cbramod_pd import config as C                                   # noqa: E402
from cbramod_pd import shards                                       # noqa: E402
from cbramod_pd.cv import job_slice, loso_splits                    # noqa: E402
from cbramod_pd.data import build_task_data                         # noqa: E402
from cbramod_pd.model import register_cbramod                       # noqa: E402
from cbramod_pd.training import run_outer_fold, set_seed            # noqa: E402


def log(msg, path):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def run_job(task, job_index, num_jobs, splits, eeg, targets, failed, args, device, upload=None):
    """Run one chunk of outer folds and write its shard (saved after every fold)."""
    out_dir = shards.shard_dir(args.results_dir, task.name, job_index)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "progress_log.txt")
    my_splits = job_slice(splits, job_index, num_jobs)
    log(f"{task.name} job {job_index + 1}/{num_jobs}: outer folds "
        f"{my_splits[0]['outer_fold']}-{my_splits[-1]['outer_fold']} of {len(splits)}", log_path)

    set_seed(C.SEED)                    # every job starts from the same random state
    preds, embs, inner = [], [], []
    for split in my_splits:
        row, emb_row, inner_rows, model = run_outer_fold(
            split, eeg, targets, task.kind, args.weights, device, args.inner_num_workers)
        preds.append(row)
        embs.append(emb_row)
        inner += inner_rows
        if args.save_weights:
            os.makedirs(os.path.join(out_dir, "weights"), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(out_dir, "weights", f"fold{split['outer_fold']:03d}.pt"))

        pd.DataFrame(preds).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
        pd.DataFrame(embs).to_csv(os.path.join(out_dir, "embeddings.csv"), index=False)
        pd.DataFrame(inner).to_csv(os.path.join(out_dir, "inner_folds.csv"), index=False)
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump({"task": task.name, "job_index": job_index, "num_jobs": num_jobs,
                       "n_outer_folds": len(splits),
                       "assigned": [s["outer_fold"] for s in my_splits],
                       "done": [r["outer_fold"] for r in preds], "failed_files": failed}, f, indent=2)
        log(f"outer fold {split['outer_fold']}/{len(splits)} done", log_path)
        if upload:
            upload()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=list(C.TASKS))
    p.add_argument("--data-dir", default="data", help="Folder with eeg/PD, eeg/HC and participants.csv")
    p.add_argument("--cbramod-dir", default="CBraMod/models", help="Folder containing cbramod.py")
    p.add_argument("--weights", default="pretrained_weights/pretrained_weights.pth",
                   help="CBraMod pretrained weights (.pth)")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--num-jobs", type=int, default=None,
                   help="Number of chunks the outer folds are split into (default: as in the paper)")
    p.add_argument("--job-index", type=int, default=None,
                   help="Run only this chunk (0-based) and do not merge; default: run all chunks")
    p.add_argument("--merge-only", action="store_true", help="Only merge existing shards")
    p.add_argument("--npy-units", default="cbramod", choices=["cbramod", "uV", "V"],
                   help="Units of the stored EEG values (cbramod: 1.0 = 100 uV)")
    p.add_argument("--inner-num-workers", type=int, default=0,
                   help="DataLoader workers for the inner-fold loaders (0 is fastest)")
    p.add_argument("--save-weights", action="store_true", help="Also save every outer fold's fine-tuned model")
    p.add_argument("--max-epochs", type=int, default=C.MAX_EPOCHS,
                   help=f"Maximum training epochs (paper: {C.MAX_EPOCHS}; change only for testing)")
    p.add_argument("--allow-cpu", action="store_true", help="Run without a GPU (very slow; for testing only)")
    p.add_argument("--s3-bucket", default=None,
                   help="AWS only: copy s3://BUCKET/<--s3-data-prefix> to --data-dir first, and "
                        "copy --results-dir to s3://BUCKET/<--s3-results-prefix> after every fold")
    p.add_argument("--s3-data-prefix", default="data")
    p.add_argument("--s3-results-prefix", default="results")
    args = p.parse_args()

    task = C.TASKS[args.task]
    C.MAX_EPOCHS = args.max_epochs
    if args.merge_only:
        shards.merge(args.results_dir, task.name)
        return

    upload = None
    if args.s3_bucket:
        from cbramod_pd import aws
        aws.download(args.s3_bucket, args.s3_data_prefix, args.data_dir)
        upload = lambda: aws.upload(args.results_dir, args.s3_bucket, args.s3_results_prefix)  # noqa: E731

    num_jobs = args.num_jobs or task.paper_num_jobs
    jobs = range(num_jobs) if args.job_index is None else [args.job_index]
    if not all(0 <= j < num_jobs for j in jobs):
        raise ValueError("--job-index must be between 0 and --num-jobs - 1")

    set_seed(C.SEED)
    subjects, eeg, failed = build_task_data(task, args.data_dir, args.npy_units)
    targets = dict(zip(subjects["subject_id"], subjects["target"]))
    labels = subjects["target"].tolist() if task.kind == "classification" else None
    splits = loso_splits(subjects["subject_id"], labels)
    print(f"{task.title}: {len(subjects)} subjects, {len(failed)} files could not be loaded")

    register_cbramod(args.cbramod_dir)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("No GPU found. Use --allow-cpu only for small tests.")

    for j in jobs:
        run_job(task, j, num_jobs, splits, eeg, targets, failed, args, device, upload)
    if args.job_index is None:
        shards.merge(args.results_dir, task.name)


if __name__ == "__main__":
    main()
