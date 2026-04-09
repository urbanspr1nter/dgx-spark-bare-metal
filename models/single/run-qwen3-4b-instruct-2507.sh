#!/bin/bash

# Model: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507

# Note the legacy location!
$HOME/models/legacy/vllm/.venv/bin/vllm serve \
        --enforce-eager \
        --served-model-name Qwen3-4B-Instruct-2507 \
        --model /home/rngo/models/Qwen3-4B-Instruct-2507 \
        --gpu-memory-utilization 0.8 \
        --max-model-len 262144 \
        --host 0.0.0.0 \
        --port 8000 \
        --attention-backend triton_attn \
        --tensor-parallel-size 1  \
        --data-parallel-size 1