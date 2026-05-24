#!/bin/bash

# Setup the environment variables that will influence:
# 1. the version of VLLM
# 2. relative locations of things
#
# tip: How to use
#   Install from a tag or release:
#     ./install.sh $HOME/models v0.21.0
#
#   Install from a branch (e.g. a fork with patches):
#     ./install.sh $HOME/models v0.21.0-sm121-fix --branch
#
# The --branch flag tells the script to use `git checkout` to the given ref
# as a branch rather than a tag. Without it, the script does a hard reset
# to the ref (safe for tags, destructive for branches with local changes).

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <vllm_path> <vllm_ref> [--branch]"
    echo ""
    echo "  vllm_ref   A git tag (e.g. v0.21.0) or branch name (e.g. v0.21.0-sm121-fix)."
    echo "  --branch   Treat vllm_ref as a branch. Without this flag, the script"
    echo "             does a hard reset to the ref (safe for tags, but will wipe"
    echo "             local changes on a branch)."
    echo ""
    echo "Examples:"
    echo "  $0 $HOME/models v0.21.0"
    echo "  $0 $HOME/models v0.21.0-sm121-fix --branch"
    exit 1
fi

# TORCH_CUDA_ARCH_LIST controls which GPU architectures get compiled.
# For DGX Spark (sm_121 / GB10) you need BOTH 12.0 and 12.1:
#   - 12.0 for kernels that use sm_120-only instructions (e.g. NVFP4/MXFP4 e2m1x2)
#   - 12.1 for native sm_121 cubins (sm_120 cubins fail at runtime on sm_121
#     with cutlass::Status::kErrorInternal for FP8 block-scaled GEMM)
# Building with only 12.1 will cause ptxas errors on sm_120-only instructions.
export TORCH_CUDA_ARCH_LIST="12.0 12.1"

VLLM_PATH=$1
VLLM_VERSION=$2
BRANCH_MODE=false
if [ "$3" = "--branch" ]; then
    BRANCH_MODE=true
fi

if [ ! -d "$(dirname "$VLLM_PATH")" ]; then
    echo "Error: Directory of $VLLM_PATH does not exist."
    exit 1
fi

# Clone the repo to the specified path
mkdir -p "$VLLM_PATH"

cd "$VLLM_PATH"

if [ ! -d "vllm/.git" ]; then
    git clone https://github.com/urbanspr1nter/vllm
fi

cd "$VLLM_PATH/vllm"

if [ "$BRANCH_MODE" = true ]; then
    git checkout "$VLLM_VERSION"
else
    git checkout "$VLLM_VERSION"
    git clean -fdx
    git reset HEAD --hard
fi

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