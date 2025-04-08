#!/bin/bash

# Set static variables
RESOURCE_GROUP="Vortex"
STORAGE_ACCOUNT=""
CONTAINER_NAME="microbenchmark"

# Get file or folder path and optional blob name
FILE_PATH="$1"
BLOB_NAME="$2"

if [ -z "$FILE_PATH" ]; then
  read -p "Enter the full path of the file or folder to upload: " FILE_PATH
fi

if [ -f "$FILE_PATH" ]; then
  # It's a file
  if [ -z "$BLOB_NAME" ]; then
    read -p "Enter the name to use for the blob in Azure: " BLOB_NAME
  fi

  az storage blob upload \
    --account-name "$STORAGE_ACCOUNT" \
    --container-name "$CONTAINER_NAME" \
    --name "$BLOB_NAME" \
    --file "$FILE_PATH" \
    --auth-mode login

elif [ -d "$FILE_PATH" ]; then
  # It's a folder
  echo "Uploading folder '$FILE_PATH' to container '$CONTAINER_NAME'..."

  az storage blob upload-batch \
    --account-name "$STORAGE_ACCOUNT" \
    --destination "$CONTAINER_NAME" \
    --source "$FILE_PATH" \
    --auth-mode login

else
  echo "Error: '$FILE_PATH' is not a valid file or directory."
  exit 1
fi
