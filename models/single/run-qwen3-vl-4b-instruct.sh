#!/bin/bash

# HuggingFace: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

# Note the legacy location!
$HOME/models/legacy/vllm/.venv/bin/vllm serve \
        --enforce-eager \
        --served-model-name Qwen3-VL-4B-Instruct \
        --model /home/rngo/models/Qwen3-VL-4B-Instruct \
        --gpu-memory-utilization 0.8 \
        --max-model-len 262144 \
        --host 0.0.0.0 \
        --port 8000 \
        --attention-backend triton_attn \
        --mm-encoder-attn-backend torch_sdpa \
        --tensor-parallel-size 1  \
        --data-parallel-size 1