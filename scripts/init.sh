#!/bin/sh
set -e

echo " Waiting for MinIO to be ready..."
until mc alias set local http://minio:9000 \
        "${MINIO_ROOT_USER}" "${MINIO_ROOT_PASSWORD}" 2>/dev/null; do
  echo "   MinIO not ready yet – retrying in 3s..."
  sleep 3
done

echo "Connected to MinIO."

# Create buckets if they don't already exist (Added mlflow)
for bucket in raw processed mlflow; do
  if mc ls "local/${bucket}" > /dev/null 2>&1; then
    echo "Bucket '${bucket}' already exists – skipping creation."
  else
    mc mb "local/${bucket}"
    echo "Bucket '${bucket}' created."
  fi
done

# Set anonymous read policy so the data is always accessible
mc anonymous set download local/raw
mc anonymous set download local/processed
mc anonymous set download local/mlflow
echo "Buckets are publicly readable."

echo "MinIO initialisation complete."
