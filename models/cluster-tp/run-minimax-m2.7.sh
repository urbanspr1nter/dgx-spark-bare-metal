#!/bin/bash

# HuggingFace: https://huggingface.co/MiniMaxAI/MiniMax-M2.7

export VLLM_IFACE=enp1s0f0np0
export NCCL_SOCKET_IFNAME=$VLLM_IFACE
export GLOO_SOCKET_IFNAME=$VLLM_IFACE
export NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0
export NCCL_IB_DISABLE=0
export VLLM_HOST_IP=169.254.195.229

export MODEL_PATH=/home/rngo/models/MiniMax-M2.7
export CONTEXT_LENGTH=196608

# Note the location of vLLM!
$HOME/models/vllm/.venv/bin/vllm serve \
    --enforce-eager \
    --distributed-executor-backend ray \
    --data-parallel-address $VLLM_HOST_IP \
    --data-parallel-rpc-port 13345 \
    --served-model-name MiniMax-M2.7 \
    --model $MODEL_PATH \
    --gpu-memory-utilization 0.8 \
    --max-model-len $CONTEXT_LENGTH \
    --host 0.0.0.0 \
    --port 8000 \
    --attention-backend triton_attn \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2 \
    --enable-auto-tool-choice \
    --enable-prefix-caching \
    --tensor-parallel-size 4 \
    --data-parallel-size 1 \
    --enable-expert-parallel
