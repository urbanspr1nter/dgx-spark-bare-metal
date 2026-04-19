#!/bin/bash

# HuggingFace: https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8

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
        --served-model-name qwen3.5-397b-a17b \
        --model /home/rngo/models/Qwen3.5-397B-A17B-FP8 \
        --gpu-memory-utilization 0.88 \
        --max-model-len 262144 \
        --host 0.0.0.0 \
        --port 8000 \
        --attention-backend triton_attn \
        --mm-encoder-attn-backend triton_attn \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --enable-prefix-caching \
        --tensor-parallel-size 4  \
        --data-parallel-size 1 \
        --enable-expert-parallel