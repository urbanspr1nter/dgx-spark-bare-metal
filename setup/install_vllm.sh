#!/bin/bash

# Setup the environment variables that will influence: 
# 1. the version of VLLM
# 2. relative locations of things

# tip: How to use
# ./install.sh $HOME/models v0.21.0

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <vllm_path> <vllm_version>"
    echo "Example: ./install.sh $HOME/models v0.21.0"
    exit 1
fi

TORCH_CUDA_ARCH_LIST=12.1a

VLLM_PATH=$1
VLLM_VERSION=$2

if [ ! -d "$(dirname "$VLLM_PATH")" ]; then
    echo "Error: Directory of $VLLM_PATH does not exist."
    exit 1
fi


# Clone the repo to the specified path
mkdir -p "$VLLM_PATH"

cd "$VLLM_PATH"
git clone https://github.com/urbanspr1nter/vllm

cd "$VLLM_PATH/vllm"

git checkout $VLLM_VERSION
git clean -fdx
git reset HEAD --hard

# Setup the Python virtual environment and install dependencies
cd "$VLLM_PATH/vllm"

python3 -m venv .venv

VLLM_VENV_PATH="$VLLM_PATH/vllm/.venv"
VLLM_VENV_PYTHON="$VLLM_VENV_PATH/bin/python"

"$VLLM_VENV_PYTHON" -m pip install uv setuptools wheel
"$VLLM_VENV_PYTHON" -m uv pip install torch torchvision torchaudio triton --index-url https://download.pytorch.org/whl/nightly/cu130

# Important to use the existing torch install. Don't have vLLM compile with a different torch
"$VLLM_VENV_PYTHON" use_existing_torch.py
"$VLLM_VENV_PYTHON" -m pip install -r requirements/build/cuda.txt

# New in 0.20.0+
"$VLLM_VENV_PYTHON" -m pip install pybind11

MAX_JOBS=$(nproc)

# VERY IMPORTANT: Ensure vLLM is built with the same python environment you just set up.
"$VLLM_VENV_PYTHON" -m uv pip install --no-build-isolation -e .

"$VLLM_VENV_PYTHON" -m pip install "ray[default]"