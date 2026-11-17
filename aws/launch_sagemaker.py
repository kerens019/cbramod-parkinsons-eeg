"""Optional: run the CBraMod fine-tuning as parallel AWS SageMaker training jobs.

Each SageMaker job runs one chunk of outer folds (scripts/train_cbramod.py
--job-index k) on its own GPU instance, so a task finishes in hours instead
of days. With the default split (20 jobs for pd_hc, 5 for the other tasks)
the results are identical to the paper's and to a local run.

Before launching:
  1. Upload the data folder (same layout as data/ in the README) to
     s3://<bucket>/<data-prefix>/, plus the CBraMod code and weights as
     <data-prefix>/CBraMod/models/... and <data-prefix>/pretrained_weights.pth
  2. Run from an environment with AWS credentials and a SageMaker execution
     role (e.g. a SageMaker notebook): pip install sagemaker boto3

    python aws/launch_sagemaker.py --bucket <your-bucket> --tasks pd_hc moca updrs3 gait_speed

When the jobs have finished:
    python aws/collect_results.py --bucket <your-bucket>
then run scripts/train_classical.py and scripts/summarize.py locally.
Written for the SageMaker Python SDK v3.
"""

import argparse
import datetime
import os
import sys

try:
    from sagemaker.train import ModelTrainer
    from sagemaker.train.configs import Compute, SourceCode, StoppingCondition
except ImportError:  # older v3 import paths
    from sagemaker.modules.train import ModelTrainer
    from sagemaker.modules.configs import Compute, SourceCode, StoppingCondition
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session, get_execution_role

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from cbramod_pd import config as C  # noqa: E402

LOCAL = "/opt/ml/local"   # writable scratch folder inside the training container


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bucket", required=True)
    p.add_argument("--tasks", nargs="+", default=list(C.TASKS), choices=list(C.TASKS))
    p.add_argument("--num-jobs", type=int, default=None, help="Jobs per task (default: as in the paper)")
    p.add_argument("--data-prefix", default="data")
    p.add_argument("--results-prefix", default="results")
    p.add_argument("--instance-type", default="ml.g5.2xlarge")
    p.add_argument("--max-hours", type=int, default=10, help="Time limit per job")
    p.add_argument("--on-demand", action="store_true", help="On-demand instead of (cheaper) Spot instances")
    args = p.parse_args()

    session = Session()
    image = image_uris.retrieve(framework="pytorch", region=session.boto_region_name, version="2.3",
                                py_version="py311", instance_type=args.instance_type, image_scope="training")
    source = SourceCode(source_dir=REPO_ROOT, entry_script="scripts/train_cbramod.py",
                        requirements="requirements.txt")
    role = get_execution_role()
    spot = not args.on_demand
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")

    for task in args.tasks:
        num_jobs = args.num_jobs or C.TASKS[task].paper_num_jobs
        for job in range(num_jobs):
            trainer = ModelTrainer(
                training_image=image, source_code=source, role=role,
                base_job_name=f"cbramod-{task.replace('_', '-')}-{stamp}-j{job:02d}",
                compute=Compute(instance_type=args.instance_type, instance_count=1,
                                enable_managed_spot_training=spot),
                stopping_condition=StoppingCondition(
                    max_runtime_in_seconds=args.max_hours * 3600,
                    **({"max_wait_time_in_seconds": args.max_hours * 3600 + 1800} if spot else {})),
                hyperparameters={
                    "task": task, "job-index": job, "num-jobs": num_jobs,
                    "data-dir": f"{LOCAL}/data", "results-dir": f"{LOCAL}/results",
                    "cbramod-dir": f"{LOCAL}/data/CBraMod/models",
                    "weights": f"{LOCAL}/data/pretrained_weights.pth",
                    "s3-bucket": args.bucket, "s3-data-prefix": args.data_prefix,
                    "s3-results-prefix": args.results_prefix,
                },
                sagemaker_session=session,
            )
            trainer.train(wait=False, logs=False)
            print(f"submitted {task} job {job + 1}/{num_jobs}")


if __name__ == "__main__":
    main()
