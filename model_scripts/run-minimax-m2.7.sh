#!/bin/bash

# HuggingFace: https://huggingface.co/MiniMaxAI/MiniMax-M2.7
# vLLM Recipe: https://docs.vllm.ai/projects/recipes/en/latest/MiniMax/MiniMax-M2.html
# Deploy Guide: https://github.com/MiniMax-AI/MiniMax-M2.7/blob/main/docs/vllm_deploy_guide.md
#
# NOTE: --enable-expert-parallel was previously disabled with TP4 because it caused
#       garbled output. That turned out to be caused by VLLM_DISABLED_KERNELS forcing
#       the Triton fallback kernel, not EP itself. With CutlassFp8BlockScaledMMKernel
#       active, EP with TP4 works correctly and significantly improves MoE throughput.
# NOTE: --compilation-config with fuse_minimax_qk_norm uses CUDA IPC which does not
#       work across multi-node setups. Do NOT use it here.
# NOTE: --enforce-eager is NOT used. CUDA graphs provide a significant decode throughput
#       boost (~9 tk/s → ~22 tk/s under load). Only use --enforce-eager for debugging.
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
    --enable-expert-parallel \
    --tensor-parallel-size 4 \
    --data-parallel-size 1