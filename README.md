# dgx-spark-bare-metal

Run `vllm` and load up large models in a 4x DGX Spark cluster!

# Clean vLLM installation

To go from a clean state, assume the path where we want to clone `vllm` to be `$HOME/models/modern`. 

We can execute the `setup/clean_vllm.sh` script. It will deactivate the current virtual environment, and remove the repo. You can run it like:

```bash
./setup/clean_vllm.sh $HOME/models/modern
```

Then clone the `vllm` repo specific to your version tag (e.g., v0.21.0) with the parent directory which you should clone the `vllm` repo into. For example, the following below will clone `vllm` into `$HOME/models/modern` and checkout the `v0.21.0` tag.

```bash
./setup/install_vllm.sh $HOME/models/modern v0.21.0
```

The above script will:

- Clone, checkout the tag
- Create a virtual environment
- Install build dependencies
- Build `vllm` and install to the environment
- Install `ray` distributed backend

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
