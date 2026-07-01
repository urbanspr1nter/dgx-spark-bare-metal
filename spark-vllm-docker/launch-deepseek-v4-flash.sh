#!/bin/bash

# spark-07
HOST_1=192.168.100.17
# spark-08
HOST_2=192.168.100.18
BASE_PATH="$HOME/code/spark-vllm-docker"

"$BASE_PATH/run-recipe.sh" deepseek-v4-flash \
    --no-ray \
    -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_IB_TC=160 \
    -e NCCL_NVLS_ENABLE=0 \
    --nccl-debug INFO \
    --ib-if rocep1s0f0 \
    --eth-if enp1s0f0np0 \
    --max-model-len 1000000 \
    -n $HOST_1,$HOST_2 