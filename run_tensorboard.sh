#!/bin/bash

# Find the latest folder in .output/train
LATEST_DIR=$(ls -td .output/train/*/ 2>/dev/null | head -n 1)

if [ -z "$LATEST_DIR" ]; then
    echo "Error: No folders found in .output/train"
    exit 1
fi

echo "Starting TensorBoard for: $LATEST_DIR"
tensorboard --logdir="$LATEST_DIR"
