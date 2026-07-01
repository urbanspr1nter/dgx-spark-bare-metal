#!/bin/bash

HOST_1=192.168.100.11
HOST_2=192.168.100.12
HOST_3=192.168.100.13
HOST_4=192.168.100.14
PORT=8000
CONTEXT_LENGTH=262144
TP=4
BASE_PATH=/home/rngo/code/spark-vllm-docker
"$BASE_PATH/launch-cluster.sh" \
    --nccl-debug INFO \
    --ib-if rocep1s0f0 \
    --eth-if enp1s0f0np0 \
    -n $HOST_1,$HOST_2,$HOST_3,$HOST_4 \
    -e HF_TOKEN=$HF_TOKEN \
    -e HF_HUB_DISABLE_XET="1" \
    -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_IB_TC=160 \
    -e NCCL_NVLS_ENABLE=0 \
    exec vllm serve \
        --model Qwen/Qwen3.5-397B-A17B-FP8 \
        --host 0.0.0.0 \
        --port $PORT \
        --tensor-parallel-size $TP \
        --max-model-len $CONTEXT_LENGTH \
        --gpu-memory-utilization 0.9 \
        --enable-auto-tool-choice \
        --trust-remote-code \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --enable-prefix-caching \
        --max-num-seqs 4 \
        --distributed-executor-backend ray