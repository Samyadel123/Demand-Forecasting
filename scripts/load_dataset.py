"""
load_dataset.py
───────────────
Downloads the Kaggle "Product Demand Forecasting" dataset and
uploads every file to the MinIO `raw` bucket.

Runs once at container startup; safe to re-run (idempotent – it
skips files that are already present in the bucket).
"""

import os
import pathlib
import sys
import json
import zipfile
import time

import boto3
from botocore.client import Config

# ─── Configuration (from environment) ────────────────────────────────────────
MINIO_ENDPOINT  = os.environ["MINIO_ENDPOINT"]          # e.g. http://minio:9000
MINIO_USER      = os.environ["MINIO_ROOT_USER"]
MINIO_PASSWORD  = os.environ["MINIO_ROOT_PASSWORD"]
BUCKET_NAME     = os.environ.get("BUCKET_NAME", "raw")
DATASET_SLUG    = os.environ.get("DATASET_SLUG",
                                  "felixzhao/productdemandforecasting")
KAGGLE_USERNAME = os.environ["KAGGLE_USERNAME"]
KAGGLE_KEY      = os.environ["KAGGLE_KEY"]
DOWNLOAD_DIR    = pathlib.Path(os.environ.get("DOWNLOAD_DIR", "/tmp/kaggle"))

# ─── Write Kaggle credentials ─────────────────────────────────────────────────
kaggle_dir = pathlib.Path.home() / ".config" / "kaggle"
kaggle_dir.mkdir(parents=True, exist_ok=True)
kaggle_cred = kaggle_dir / "kaggle.json"
kaggle_cred.write_text(
    json.dumps({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY})
)
kaggle_cred.chmod(0o600)
print(f"Kaggle credentials written to {kaggle_cred}")

# ─── Download dataset ─────────────────────────────────────────────────────────
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402 (import after cred file written)

api = KaggleApi()
api.authenticate()

print(f"Downloading dataset '{DATASET_SLUG}' → {DOWNLOAD_DIR} ...")
api.dataset_download_files(
    DATASET_SLUG,
    path=str(DOWNLOAD_DIR),
    unzip=False,   # we'll unzip ourselves so we control the path
    quiet=False,
)
print("Download complete.")

# ─── Unzip ────────────────────────────────────────────────────────────────────
zip_files = list(DOWNLOAD_DIR.glob("*.zip"))
if zip_files:
    for zf in zip_files:
        print(f"Extracting {zf.name} ...")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(DOWNLOAD_DIR)
        zf.unlink()   # remove the zip after extraction
    print("Extraction complete.")

# ─── Connect to MinIO (S3-compatible) ────────────────────────────────────────
s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_USER,
    aws_secret_access_key=MINIO_PASSWORD,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)

# Ensure bucket exists (mc init.sh creates it, but be defensive)
existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
if BUCKET_NAME not in existing:
    s3.create_bucket(Bucket=BUCKET_NAME)
    print(f"Bucket '{BUCKET_NAME}' created by loader.")

# ─── Upload every file ───────────────────────────────────────────────────────
all_files = [p for p in DOWNLOAD_DIR.rglob("*") if p.is_file()]
if not all_files:
    print("No files found after download/extraction – nothing to upload.")
    sys.exit(1)

for local_path in all_files:
    # Preserve subfolder structure inside the bucket
    s3_key = local_path.relative_to(DOWNLOAD_DIR).as_posix()

    # Idempotency check
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=s3_key)
        print(f"Already exists – skipping: {s3_key}")
        continue
    except s3.exceptions.ClientError:
        pass  # object doesn't exist yet

    print(f"Uploading {local_path.name} → s3://{BUCKET_NAME}/{s3_key}")
    s3.upload_file(
        str(local_path),
        BUCKET_NAME,
        s3_key,
        ExtraArgs={"ContentType": "text/csv"},
    )

print(f"\n All files uploaded to MinIO bucket '{BUCKET_NAME}'.")
print(f"     Access via:  {MINIO_ENDPOINT}/{BUCKET_NAME}/")