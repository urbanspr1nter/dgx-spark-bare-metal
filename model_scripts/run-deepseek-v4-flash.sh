#!/bin/bash

export VLLM_IFACE=enp1s0f0np0
export NCCL_SOCKET_IFNAME=$VLLM_IFACE
export GLOO_SOCKET_IFNAME=$VLLM_IFACE
export NCCL_IB_HCA=rocep1s0f0
export NCCL_IB_DISABLE=0
export VLLM_HOST_IP=169.254.195.229

export MODEL_PATH=$HOME/models/DeepSeek-V4-Flash
export CONTEXT_LENGTH=262144

$HOME/models/vllm/.venv/bin/vllm serve --model "$MODEL_PATH" \
        --served-model-name DeepSeek-V4-Flash \
        --enforce-eager \
        --distributed-executor-backend ray \
        --data-parallel-address 169.254.195.229 \
        --data-parallel-rpc-port 13345 \
        --gpu-memory-utilization 0.85 \
        --max-model-len $CONTEXT_LENGTH \
        --host 0.0.0.0 \
        --port 8000 \
        --block-size 256 \
        --kv-cache-dtype fp8 \
        --tokenizer-mode deepseek_v4 \
        --reasoning-parser deepseek_v4 \
        --tool-call-parser deepseek_v4 \
        --attention-backend triton_attn \
        --enable-auto-tool-choice \
        --enable-prefix-caching \
        --tensor-parallel-size 4 \
        --data-parallel-size 1 \
        --trust-remote-code \
        --enable-expert-parallel
