# vLLM Installation

**CRITICAL** — vLLM must be installed on **all 4 nodes** in the cluster. Model scripts reference `$HOME/models/vllm/.venv/bin/vllm`, so the install path must be `$HOME/models` on every Spark.

## Cluster-wide Install (Recommended)

Use the `vllm-install` skill script to install across all 4 nodes at once:

```bash
./agents/skills/vllm-install/scripts/install_cluster.sh v0.21.0
```

This script will:

1. **Rsync** the repo to spark-02/03/04 (ensures the install script is available on every node).
2. **spark-01 (local)** — Create a **new tmux window** in session `0` (named `vllm-install`) so it doesn't disrupt your existing pi window. The build runs there.
3. **spark-02/03/04 (remote)** — Send the install command into the **existing** tmux session `0` on each node so you can attach and watch.
4. Print a summary showing where to attach tmux to monitor progress.

This is a full source compile — expect 15-30 minutes per node. Each node prints `=== <NODE> INSTALL COMPLETE ===` when finished.

See the [vllm-install skill](../.agents/skills/vllm-install/SKILL.md) for full details.

## Single-node Install

To install on just one node (e.g., for testing), use the setup script directly:

```bash
./setup/install_vllm.sh $HOME/models v0.21.0
```

The first argument is the parent directory — the repo gets cloned into `$1/vllm` and the venv lives at `$1/vllm/.venv`.

## Clean Install

**CRITICAL** — You must stop the Ray cluster before cleaning or reinstalling vLLM. If Ray is running with vLLM loaded, wiping the venv will leave the cluster in a broken state. Stop Ray first:

```bash
./agents/skills/ray-start/scripts/stop_cluster.sh
```

Then to wipe an existing vLLM install and start fresh:

```bash
./setup/clean_vllm.sh $HOME/models $HOME/models/vllm/.venv
```

The first argument is the parent directory containing the `vllm/` repo. The second is the path to the virtual environment (needed to deactivate it first). Must be repeated on each node manually.
