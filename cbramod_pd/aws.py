"""Optional helpers for running on AWS (copying data from / results to S3 with the AWS CLI)."""

import subprocess


def s3_sync(src, dst):
    subprocess.run(["aws", "s3", "sync", src, dst, "--quiet"], check=True)


def download(bucket, prefix, local_dir):
    s3_sync(f"s3://{bucket}/{prefix.strip('/')}/", f"{local_dir}/")


def upload(local_dir, bucket, prefix):
    s3_sync(f"{local_dir}/", f"s3://{bucket}/{prefix.strip('/')}/")
