#!/bin/bash

export VLLM_IFACE=enp1s0f0np0
export NCCL_SOCKET_IFNAME=$VLLM_IFACE
export GLOO_SOCKET_IFNAME=$VLLM_IFACE
export NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0
export NCCL_IB_DISABLE=0
export VLLM_HOST_IP=169.254.204.215
export RAY_memory_usage_threshold=0.99

ray stop -f || true
ray start --address=169.254.195.229:6379