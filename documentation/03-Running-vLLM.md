# Running Models with vLLM

**CRITICAL** - Follow these patterns.

Essentially you need to do the following:

1. Start `ray` cluster. First on `spark-01` then `spark-02`, `spark-03` and `spark-04` in that exact order. `spark-01` is the main node.
2. Run one of the model scripts found in `model_scripts`.

## Where are my models?

Models are in `$HOME/models`. 

Tip: `ls -la` on `$HOME/models`.

## Starting Ray Cluster

Use the `ray-start` skill script (see [02-Start-Ray-Cluster.md](./02-Start-Ray-Cluster.md)):

```bash
./agents/skills/ray-start/scripts/start_cluster.sh
```

**IMPORTANT** - If starting manually, you must run `start-ray.sh` in the same directory as the `vllm` virtual environment (`$HOME/models/vllm`). The skill script handles this automatically.

## Running a Model

Run any model found in `model_scripts`. Just execute the shell script but do it in the same directory as the `vllm` virtual environment! Same as `start-ray.sh`, do it relatively!

So to run MiniMax-M2.7, you can use this example to see what I mean:

```bash
# lets say model script is in $HOME/models, but you are in $HOME/models/modern/vllm since .venv is there
../../run-minimax-m2.7.sh
```
