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

## Environment variables and Ray

**CRITICAL** — Environment variables set in the shell where you start Ray are inherited by all Ray worker processes. This means variables like `VLLM_DISABLED_KERNELS` will affect vLLM even if they are not set in the model launch script itself.

If you previously set `VLLM_DISABLED_KERNELS` for debugging (e.g. to work around a broken kernel), it will persist in your shell and be picked up by Ray. This silently causes vLLM to fall back to slower or incorrect kernels.

**Always unset `VLLM_DISABLED_KERNELS` before starting the Ray cluster:**

```bash
unset VLLM_DISABLED_KERNELS
ray start --head ...
```

Unsetting it only in the model launch script is **not sufficient** — Ray already inherited it when it started.

**How to diagnose:** If the Ray worker log shows `Selected TritonFp8BlockScaledMMKernel` instead of `Selected CutlassFp8BlockScaledMMKernel`, check:

1. `env | grep VLLM_DISABLED_KERNELS` in your shell
2. `cat /proc/$(pgrep -f raylet | head -1)/environ | tr '\0' '\n' | grep VLLM_DISABLED` in the Ray head process

Model scripts include `unset VLLM_DISABLED_KERNELS` at the top as a defensive measure, but the Ray startup environment must also be clean.
