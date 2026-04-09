#!/bin/bash

# HuggingFace: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

export VLLM_IFACE=enp1s0f0np0
export NCCL_SOCKET_IFNAME=$VLLM_IFACE
export GLOO_SOCKET_IFNAME=$VLLM_IFACE
export NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0
export NCCL_IB_DISABLE=0
export VLLM_HOST_IP=169.254.195.229

# Note the legacy path!
$HOME/models/legacy/vllm/.venv/bin/vllm serve \
        --enforce-eager \
        --distributed-executor-backend ray \
        --data-parallel-address 169.254.195.229 \
        --data-parallel-rpc-port 13345 \
        --served-model-name Qwen3-VL-4B-Instruct \
        --model /home/rngo/models/Qwen3-VL-4B-Instruct \
        --gpu-memory-utilization 0.8 \
        --max-model-len 262144 \
        --host 0.0.0.0 \
        --port 8000 \
        --attention-backend triton_attn \
        --mm-encoder-attn-backend torch_sdpa \
        --tensor-parallel-size 2  \
        --data-parallel-size 1