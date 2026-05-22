#!/bin/bash

# WARNING! This script wipes out your current vLLM installation!
# Run this and install_vllm.sh to get back to a good state!

# path to where your vllm repo is
VLLM_PATH=$1

# virtual environment path to run deactivate
VLLM_VENV_PATH=$2

if [ -z "$VLLM_PATH" ] || [ -z "$VLLM_VENV_PATH" ]; then
    echo "Usage: $0 <vllm_path> <venv_path>"
    exit 1
fi

if [ ! -d "$VLLM_PATH" ]; then
    echo "Error: vLLM path $VLLM_PATH does not exist."
    exit 1
fi

if [ ! -d "$VLLM_VENV_PATH" ]; then
    echo "Error: Venv path $VLLM_VENV_PATH does not exist."
    exit 1
fi

"$VLLM_VENV_PATH/bin/deactivate"

rm -rf "$VLLM_PATH/vllm"