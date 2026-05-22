# dgx-spark-bare-metal

vLLM, llama.cpp, fine-tuning, etc all on bare metal with the DGX Spark

See the blog for more details. 


# Cheat Sheet

## Stop Ray Cluster 

Spark A: 

```sh
ray stop -f
```

Spark B: 

```sh
ray stop -f
```

## Inference

First, if not already done, create start the cluster:

Spark A:

```sh
start-ray-spark-a.sh
```

Spark B:

```sh
start-ray-spark-b.sh
```

Going back to Spark A, start `vllm` to serve the model.

Spark A:

```sh
./run-qwen3.5-122b-a10b-fp8.sh
```

# My Personal Setup

- I have 4x DGX Sparks connected to a Mikrotik switch.

|Name|IP Address|
|----|----------|
|spark-01|192.168.1.21|
|spark-02|192.168.1.40|
|spark-03|192.168.1.24|
|spark-04|192.168.1.48|
