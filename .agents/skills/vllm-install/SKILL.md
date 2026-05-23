---
name: vllm-install
description: Install or reinstall vLLM across the entire 4× DGX Spark cluster. Builds vLLM from source on all nodes, using the setup/install_vllm.sh script. Handles rsync-ing the repo, kicking off builds via tmux, and verifying the binary exists on every node afterward.
---

# vLLM Cluster Install

Install vLLM from source on **all 4 DGX Spark nodes** in parallel, so that `$HOME/models/vllm/.venv/bin/vllm` exists everywhere and model scripts can run.

## What It Does

1. **spark-01 (local)** — Creates a new tmux window in session `0` and runs `setup/install_vllm.sh` there. This keeps the existing pi window untouched.
2. **Rsync repo** — Copies `$HOME/code/dgx-spark-bare-metal/` to spark-02, spark-03, spark-04 (same path, `--mkpath`).
3. **spark-02/03/04 (remote)** — Sends the install command into the existing tmux session `0` on each node via `tmux send-keys`. The user can attach to watch.
4. **Verification** — After builds complete, confirms `$HOME/models/vllm/.venv/bin/vllm` exists on all nodes.

## Usage

```bash
./scripts/install_cluster.sh <vllm_version>
```

For example:

```bash
./scripts/install_cluster.sh v0.21.0
```

The install path is always `$HOME/models` (which produces `$HOME/models/vllm/.venv/bin/vllm`), matching what the model scripts expect.

Run from the skill directory, or from anywhere — the script locates the repo root and `env.sh` automatically.

## Tmux Details

- **spark-01 (local):** A new window is created in tmux session `0` (e.g. `0:2` named `vllm-install`). This avoids disrupting the pi window. Attach with `tmux attach -t 0` and switch to the `vllm-install` window.
- **spark-02/03/04:** The install command is sent to the **existing** window in tmux session `0` on each node. Attach via `ssh <node>` then `tmux attach -t 0`.

## Prerequisites

- **Ray cluster must be stopped.** If Ray is running, stop it first using the `ray-start` skill (`./scripts/stop_cluster.sh`). Running `clean_vllm.sh` or reinstalling vLLM while Ray is active will leave the cluster in a broken state.
- Passwordless SSH to all Sparks (spark-01 already has this).
- `env.sh` at repo root with `SPARK_01_IP` through `SPARK_04_IP` defined.
- Python 3, CUDA toolkit, and build tools on each node.
- Tmux session `0` exists on each remote node (default on DGX Sparks).