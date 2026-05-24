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

### Install from a tag (standard)

```bash
./scripts/install_cluster.sh v0.21.0
```

### Install from a branch (e.g. a fork with patches)

```bash
./scripts/install_cluster.sh v0.21.0-sm121-fix --branch
```

The `--branch` flag is passed through to `setup/install_vllm.sh`, which skips the `git clean -fdx && git reset HEAD --hard` that would discard local commits on a branch. Use this when building from a fork branch that has patches not yet in an upstream release.

The install path is always `$HOME/models` (which produces `$HOME/models/vllm/.venv/bin/vllm`), matching what the model scripts expect.

Run from the skill directory, or from anywhere — the script locates the repo root and `env.sh` automatically.

## Tmux Details

- **spark-01 (local):** A new window is created in tmux session `0` (e.g. `0:2` named `vllm-install`). This avoids disrupting the pi window. Attach with `tmux attach -t 0` and switch to the `vllm-install` window.
- **spark-02/03/04:** The install command is sent to the **existing** window in tmux session `0` on each node. Attach via `ssh <node>` then `tmux attach -t 0`.

**CRITICAL:** Always run builds in tmux windows, never in the foreground of a pi shell. Builds take 15-30+ minutes and the shell session can be lost or interrupted. Using tmux ensures you can monitor progress and the build survives disconnection.

## Cluster build pattern

When building vLLM from a fork with local patches (branches, not just tags), the correct workflow is:

1. **Commit your changes** on spark-01 first. The repo at `$HOME/models/vllm` should be on the correct branch with all patches committed. Do not rely on uncommitted changes — they will not survive the rsync.

2. **Clean and reset on spark-01** (`git clean -fdx && git reset HEAD --hard`) to remove build artifacts before rsyncing. This ensures remote nodes get a clean source tree, not a tree littered with `.so` files, `.deps/`, and `.venv/`. Do NOT use `--exclude` tricks to rsync partial trees — missing directories (like `requirements/build/`) will cause build failures on remote nodes. A clean source tree rsynced in full is reliable.

3. **Rsync the entire repo** from spark-01 to all remote nodes. No excludes for `.git/objects/pack/` or other directories — disk space is not a concern and completeness matters more than transfer time.

4. **Build on all 4 nodes in parallel**, each in its own tmux window. Spark-01 rebuilds too — since all nodes build in parallel, there is no time lost. Starting from a clean state is always better than copying individual files around.

5. **Verify** that the vllm binary and expected GPU architecture cubins exist on every node after the build completes.

## Prerequisites

- **Ray cluster must be stopped.** If Ray is running, stop it first using the `ray-start` skill (`./scripts/stop_cluster.sh`). Running `clean_vllm.sh` or reinstalling vLLM while Ray is active will leave the cluster in a broken state.
- Passwordless SSH to all Sparks (spark-01 already has this).
- `env.sh` at repo root with `SPARK_01_IP` through `SPARK_04_IP` defined.
- Python 3, CUDA toolkit, and build tools on each node.
- Tmux session `0` exists on each remote node (default on DGX Sparks).

## Environment variable hygiene

**CRITICAL** — Before starting the Ray cluster or launching vLLM, ensure that `VLLM_DISABLED_KERNELS` is not set in your shell environment. This variable is inherited by Ray and propagated to all worker processes. If it was set during a previous debugging session (e.g. `VLLM_DISABLED_KERNELS=CutlassFp8BlockScaledMMKernel`), it will silently cause vLLM to fall back to slower or incorrect kernels — even if the variable is not explicitly set in the model launch script.

**The fix:** Unset it **before** starting Ray, not just before launching vLLM:

```bash
unset VLLM_DISABLED_KERNELS
ray start --head ...
```

**How to diagnose:** If you see `Selected TritonFp8BlockScaledMMKernel` instead of `Selected CutlassFp8BlockScaledMMKernel` in the Ray worker logs, check:

1. `env | grep VLLM_DISABLED_KERNELS` in your shell
2. `cat /proc/$(pgrep -f raylet | head -1)/environ | tr '\0' '\n' | grep VLLM_DISABLED` in the Ray head process
3. The Ray worker log will show which kernel was selected

Model scripts should include `unset VLLM_DISABLED_KERNELS` at the top as a defensive measure.

## TORCH_CUDA_ARCH_LIST

`setup/install_vllm.sh` sets `TORCH_CUDA_ARCH_LIST="12.0 12.1"` which is required for DGX Spark (sm_121 / GB10):

- **`12.0`** is needed for kernels that use sm_120-only instructions (e.g. NVFP4/MXFP4 `e2m1x2`). Building with only `12.1` causes ptxas errors on these kernels.
- **`12.1`** is needed for native sm_121 cubins. Without it, vLLM compiles sm_120 cubins only, which fail at runtime on sm_121 hardware with `cutlass::Status::kErrorInternal` for FP8 block-scaled GEMM operations.

The old value `12.1a` (single arch) is insufficient. Both architectures must be present.