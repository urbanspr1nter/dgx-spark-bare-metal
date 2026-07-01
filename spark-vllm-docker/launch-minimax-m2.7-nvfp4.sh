#!/bin/bash

# spark-05
HOST_1=192.168.100.15

# spark-06
HOST_2=192.168.100.16
PORT=8000
CONTEXT_LENGTH=196608
TP=2

BASE_PATH=/home/rngo/code/spark-vllm-docker

"$BASE_PATH/launch-cluster.sh" \
    -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_IB_TC=160 \
    -e NCCL_NVLS_ENABLE=0 \
    -e HF_TOKEN=$HF_TOKEN \
    -e HF_DISABLE_XET="1" \
    -n $HOST_1,$HOST_2 \
    exec vllm serve \
    --model nvidia/MiniMax-M2.7-NVFP4 \
    --host 0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --max-model-len $CONTEXT_LENGTH \
    --gpu-memory-utilization 0.8 \
    --enable-auto-tool-choice \
    --trust-remote-code \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --distributed-executor-backend ray
