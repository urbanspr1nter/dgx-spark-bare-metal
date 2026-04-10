#!/bin/bash

# HuggingFace: https://huggingface.co/google/gemma-4-31B-it

export VLLM_IFACE=enp1s0f0np0
export NCCL_SOCKET_IFNAME=$VLLM_IFACE
export GLOO_SOCKET_IFNAME=$VLLM_IFACE
export NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0
export NCCL_IB_DISABLE=0
export VLLM_HOST_IP=169.254.195.229

# Note the modern location!
$HOME/models/modern/vllm/.venv/bin/vllm serve \
        --enforce-eager \
        --distributed-executor-backend ray \
        --data-parallel-address 169.254.195.229 \
        --data-parallel-rpc-port 13345 \
        --served-model-name gemma-4-31b-it \
        --model /home/rngo/models/gemma-4-31B-it \
        --reasoning-parser gemma4 \
        --tool-call-parser gemma4 \
        --enable-auto-tool-choice \
        --default-chat-template-kwargs '{"enable_thinking": true}' \
        --mm-processor-kwargs '{"max_soft_tokens": 1120}' \
        --gpu-memory-utilization 0.9 \
        --max-model-len 262144 \
        --host 0.0.0.0 \
        --port 8000 \
        --attention-backend triton_attn \
        --mm-encoder-attn-backend triton_attn \
        --tensor-parallel-size 2  \
        --data-parallel-size 1