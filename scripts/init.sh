#!/bin/sh
set -e
 
echo " Waiting for MinIO to be ready..."
until mc alias set local http://minio:9000 \
        "${MINIO_ROOT_USER}" "${MINIO_ROOT_PASSWORD}" 2>/dev/null; do
  echo "   MinIO not ready yet – retrying in 3s..."
  sleep 3
done
 
echo "Connected to MinIO."
 
# Create the 'raw' bucket if it doesn't already exist
if mc ls local/raw > /dev/null 2>&1; then
  echo "Bucket 'raw' already exists – skipping creation."
else
  mc mb local/raw
  echo "Bucket 'raw' created."
fi
 
# Set anonymous read policy so the data is always accessible
mc anonymous set download local/raw
echo "Bucket 'raw' is publicly readable."
 
echo "MinIO initialisation complete."