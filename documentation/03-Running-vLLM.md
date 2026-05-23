# Running Models with vLLM

**CRITICAL** - Follow these patterns.

Essentially you need to do the following:

1. Start `ray` cluster. First on `spark-01` then `spark-02`, `spark-03` and `spark-04` in that exact order. `spark-01` is the main node.
2. Run one of the model scripts found in `model_scripts`.

## Where are my models?

Models are in `$HOME/models`. 

Tip: `ls -la` on `$HOME/models`.

## Starting Ray Cluster

**IMPORTANT** - You must run `start-ray.sh` in the same directory as the `vllm` virtual environment! So take into consideration in running it relatively if you are invoking it from another place other than the `vllm` repo.

Here is the order in which you should start the `ray` cluster.

I am referring to the Sparks by their logical name here, but alway use the IPv4 address found in the [00-Infrastructure.md](./00-Infrastructure.md) table.

- Run `start-ray.sh` in `spark-01`
- Wait 10 seconds.
- Run `start-ray.sh` in `spark-02`
- Wait 10 seconds.
- Run `start-ray.sh` in `spark-03`
- Wait 10 seconds.
- Run `start-ray.sh` in `spark-04`.
- Wait 10 seconds.

## Running a Model

Run any model found in `model_scripts`. Just execute the shell script but do it in the same directory as the `vllm` virtual environment! Same as `start-ray.sh`, do it relatively!

So to run MiniMax-M2.7, you can use this example to see what I mean:

```bash
# lets say model script is in $HOME/models, but you are in $HOME/models/modern/vllm since .venv is there
../../run-minimax-m2.7.sh
```
