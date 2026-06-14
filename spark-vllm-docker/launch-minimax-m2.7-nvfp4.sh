#!/bin/bash

HOST_1=169.254.122.182
HOST_2=169.254.114.56
PORT=8000
CONTEXT_LENGTH=196608
TP=2

BASE_PATH=/home/rngo/code/spark-vllm-docker

"$BASE_PATH/launch-cluster.sh" -n $HOST_1,$HOST_2 exec vllm serve \
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
