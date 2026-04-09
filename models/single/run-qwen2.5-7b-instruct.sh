#!/bin/bash

# Model: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

# Note the legacy location!
$HOME/models/legacy/vllm/.venv/bin/vllm serve \
        --enforce-eager \
        --served-model-name Qwen2.5-7B-Instruct \
        --model /home/rngo/models/Qwen2.5-7B-Instruct \
        --gpu-memory-utilization 0.8 \
        --max-model-len 32768 \
        --host 0.0.0.0 \
        --port 8000 \
        --attention-backend triton_attn \
        --tensor-parallel-size 1  \
        --data-parallel-size 1