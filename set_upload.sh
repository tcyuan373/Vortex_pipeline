#!/bin/bash

# Set static variables
RESOURCE_GROUP="Vortex"
STORAGE_ACCOUNT=""
CONTAINER_NAME="microbenchmark"

# Get file path and blob name from arguments or prompt user
FILE_PATH="$1"
BLOB_NAME="$2"

if [ -z "$FILE_PATH" ]; then
  read -p "Enter the full path of the file to upload: " FILE_PATH
fi

if [ -z "$BLOB_NAME" ]; then
  read -p "Enter the name to use for the blob in Azure: " BLOB_NAME
fi

# Upload the blob
az storage blob upload \
  --account-name "$STORAGE_ACCOUNT" \
  --container-name "$CONTAINER_NAME" \
  --name "$BLOB_NAME" \
  --file "$FILE_PATH" \
  --auth-mode login
