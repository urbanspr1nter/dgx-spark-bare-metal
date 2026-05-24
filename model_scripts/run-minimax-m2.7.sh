#!/bin/bash

# HuggingFace: https://huggingface.co/MiniMaxAI/MiniMax-M2.7
# vLLM Recipe: https://docs.vllm.ai/projects/recipes/en/latest/MiniMax/MiniMax-M2.html
# Deploy Guide: https://github.com/MiniMax-AI/MiniMax-M2.7/blob/main/docs/vllm_deploy_guide.md
#
# NOTE: --enable-expert-parallel must NOT be used with TP4 (causes garbled output).
#       EP is only for TP8+ or DP8+ configurations.
# NOTE: --compilation-config with fuse_minimax_qk_norm uses CUDA IPC which does not
#       work across multi-node setups. Use --enforce-eager instead.
# NOTE: CutlassFp8BlockScaledMMKernel is required for FP8 MoE models like MiniMax-M2.7.
#       Disabling it (via VLLM_DISABLED_KERNELS) causes garbled/corrupted output.
#       This env var must be unset BEFORE starting Ray, not just before launching vLLM,
#       because Ray inherits the environment of the shell that starts it.

# Defensive: ensure no stale kernel overrides from previous debugging sessions
unset VLLM_DISABLED_KERNELS

export VLLM_IFACE=enp1s0f0np0
export NCCL_SOCKET_IFNAME=$VLLM_IFACE
export GLOO_SOCKET_IFNAME=$VLLM_IFACE
export NCCL_IB_HCA=rocep1s0f0
export NCCL_IB_DISABLE=0
export VLLM_HOST_IP=169.254.195.229
# export SAFETENSORS_FAST_GPU=1

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
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --data-parallel-size 1