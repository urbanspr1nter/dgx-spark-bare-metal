# Starting Ray Cluster

**CRITICAL** — spark-01 must start first as the head node, then workers join in order with 10-second pauses between each.

## Cluster-wide Start (Recommended)

Use the `ray-start` skill script to start Ray on all 4 nodes:

```bash
./agents/skills/ray-start/scripts/start_cluster.sh
```

This script will:

1. Copy each `cluster/start-ray-spark-*.sh` to `$HOME/models/start-ray.sh` on the corresponding node.
2. Start spark-01 as head node (from the vllm directory with venv activated).
3. Wait 10 seconds, then start spark-02. Repeat for spark-03, spark-04.
4. Run `ray status` to verify all nodes are active.

See the [ray-start skill](../.agents/skills/ray-start/SKILL.md) for full details.

## Stopping the Cluster

```bash
./agents/skills/ray-start/scripts/stop_cluster.sh
```

Stops workers first, then the head node.

## Manual Start

If you need to start Ray manually, you must:

1. Copy the per-node script to `$HOME/models/start-ray.sh` on each node.
2. Run it from `$HOME/models/vllm` (where the venv lives).

|Script|DGX Spark|
|------|---------|
|`start-ray-spark-a.sh`|spark-01|
|`start-ray-spark-b.sh`|spark-02|
|`start-ray-spark-c.sh`|spark-03|
|`start-ray-spark-d.sh`|spark-04|

Order: spark-01 → wait 10s → spark-02 → wait 10s → spark-03 → wait 10s → spark-04 → wait 10s.