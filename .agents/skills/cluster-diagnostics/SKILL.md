---
name: cluster-diagnostics
description: Run diagnostics on the 4x DGX Spark cluster. Checks node reachability via SSH, verifies model directories exist on all nodes, and validates that paths referenced in model run scripts actually exist on each node. Use before starting Ray or running models, or when troubleshooting cluster issues.
---

# Cluster Diagnostics

Run basic health checks against the 4× DGX Spark cluster.

**Important:** All communication uses IPv4 addresses (not hostnames). Node IPs are sourced from `env.sh` at the repo root.

## Checks

1. **Reachability** — SSH to each Spark and confirm the node responds.
2. **Model availability** — List `$HOME/models` on every node and compare against spark-01 (the reference node). Flags any missing or extra directories.
3. **Script path consistency** — Extracts paths from model run scripts (vLLM binary, model directories) and verifies they exist on every node. Catches issues like scripts pointing to a vLLM install that isn't deployed everywhere.

## Usage

```bash
./scripts/diagnostics.sh
```

Run from the skill directory, or from anywhere — the script locates the repo root and `env.sh` automatically.

## Prerequisites

- Passwordless SSH to all Sparks (spark-01 already has this).
- `env.sh` at repo root with `SPARK_01_IP` through `SPARK_04_IP` defined.