#!/bin/bash

HOST_1=169.254.122.182
# HOST_2=169.254.114.56
PORT=8000
CONTEXT_LENGTH=262144
TP=1

BASE_PATH=/home/rngo/code/spark-vllm-docker

"$BASE_PATH/launch-cluster.sh" -n $HOST_1 exec vllm serve \
    --model Qwen/Qwen3.6-35B-A3B \
    --host 0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --max-model-len $CONTEXT_LENGTH \
    --gpu-memory-utilization 0.9 \
    --enable-auto-tool-choice \
    --trust-remote-code \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-prefix-caching
