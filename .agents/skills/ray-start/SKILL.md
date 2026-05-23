---
name: ray-start
description: Start the Ray cluster across all 4 DGX Spark nodes. Copies the per-node start-ray scripts, starts spark-01 as head, then joins spark-02/03/04 with 10-second pauses between each. Verifies cluster health with ray status afterward.
---

# Ray Cluster Start

Start the Ray cluster on all 4 DGX Spark nodes in the correct order with proper pacing.

## What It Does

1. **Copy scripts** — Copies each `cluster/start-ray-spark-*.sh` to `$HOME/models/start-ray.sh` on the corresponding node.
2. **Start spark-01 (head)** — Runs `start-ray.sh` from the vllm directory (where `.venv` lives) on spark-01.
3. **Wait 10 seconds.**
4. **Start spark-02** — Same pattern, via SSH.
5. **Wait 10 seconds.** Repeat for spark-03, then spark-04.
6. **Verify** — Runs `ray status` on spark-01 to confirm all nodes are active.

## Usage

```bash
./scripts/start_cluster.sh
```

Run from the skill directory, or from anywhere — the script locates the repo root and `env.sh` automatically.

## Stopping the Cluster

To stop Ray on all nodes:

```bash
./scripts/stop_cluster.sh
```

## Prerequisites

- vLLM installed at `$HOME/models/vllm/.venv` on all nodes.
- Passwordless SSH to all Sparks.
- `env.sh` at repo root with `SPARK_01_IP` through `SPARK_04_IP` defined.
- `cluster/start-ray-spark-{a,b,c,d}.sh` scripts present in the repo.