"""Optional: download the SageMaker jobs' outputs from S3 and merge them per task.

    python aws/collect_results.py --bucket <your-bucket>

Needs the AWS CLI (`aws`) with credentials for the bucket.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from cbramod_pd import aws, config as C, shards  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--tasks", nargs="+", default=list(C.TASKS), choices=list(C.TASKS))
    p.add_argument("--results-prefix", default="results")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    aws.download(args.bucket, args.results_prefix, args.results_dir)
    for task in args.tasks:
        shards.merge(args.results_dir, task)


if __name__ == "__main__":
    main()
