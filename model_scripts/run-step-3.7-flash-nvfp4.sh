#!/bin/bash

# Defensive: ensure no stale kernel overrides from previous debugging sessions
unset VLLM_DISABLED_KERNELS

export VLLM_IFACE=enp1s0f0np0
export NCCL_SOCKET_IFNAME=$VLLM_IFACE
export GLOO_SOCKET_IFNAME=$VLLM_IFACE
export NCCL_IB_HCA=rocep1s0f0
export NCCL_IB_DISABLE=0
export VLLM_HOST_IP=169.254.122.182

export MODEL_PATH=/home/rngo/models/Step-3.7-Flash-NVFP4
export CONTEXT_LENGTH=262144

# Note the location of vLLM!
$HOME/models/vllm/.venv/bin/vllm serve \
    --distributed-executor-backend ray \
    --data-parallel-address $VLLM_HOST_IP \
    --data-parallel-rpc-port 13345 \
    --served-model-name Step-3.7-Flash \
    --model $MODEL_PATH \
    --gpu-memory-utilization 0.8 \
    --max-model-len $CONTEXT_LENGTH \
    --host 0.0.0.0 \
    --port 8000 \
    --attention-backend triton_attn \
    --tool-call-parser step3p5 \
    --reasoning-parser step3p5 \
    --async-scheduling \
    --enable-auto-tool-choice \
    --enable-prefix-caching \
    --trust-remote-code \
    --enable-expert-parallel \
    --tensor-parallel-size 2 \
    --kv-cache-dtype fp8 \
    --data-parallel-size 1