#!/bin/bash

# Start the Ray cluster across all 4 DGX Spark nodes.
# Copies per-node start-ray scripts, starts spark-01 as head,
# then joins the remaining nodes with 10-second pauses.
# Verifies cluster health with ray status at the end.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

source "$REPO_ROOT/env.sh"

USER="$(whoami)"
NODES=("$SPARK_01_IP" "$SPARK_02_IP" "$SPARK_03_IP" "$SPARK_04_IP")
NODE_NAMES=("spark-01" "spark-02" "spark-03" "spark-04")
SCRIPTS=("a" "b" "c" "d")
REMOTE_NODES=("${NODES[@]:1}")
REMOTE_NAMES=("${NODE_NAMES[@]:1}")
REMOTE_SCRIPTS=("${SCRIPTS[@]:1}")

SSH_OPTS="-o ConnectTimeout=5 -o BatchMode=yes"
WAIT_SECS=10
VLLM_DIR="\$HOME/models/vllm"
START_RAY_REMOTE="\$HOME/models/start-ray.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }

echo "========================================"
echo "  Ray Cluster Start"
echo "========================================"
echo ""

# --- 1. Check reachability ---
echo "--- 1. Checking reachability ---"

ALL_OK=true
for i in "${!NODES[@]}"; do
    ip="${NODES[$i]}"
    name="${NODE_NAMES[$i]}"
    if [ "$i" -eq 0 ]; then
        pass "$name ($ip) [local]"
    elif ssh $SSH_OPTS "$USER@$ip" 'true' 2>/dev/null; then
        pass "$name ($ip)"
    else
        fail "$name ($ip) — UNREACHABLE"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    echo ""
    fail "Not all nodes are reachable. Aborting."
    exit 1
fi

echo ""

# --- 2. Copy start-ray scripts to each node ---
echo "--- 2. Copying start-ray scripts ---"

# spark-01 (local)
SCRIPT_FILE="$REPO_ROOT/cluster/start-ray-spark-${SCRIPTS[0]}.sh"
cp "$SCRIPT_FILE" "$HOME/models/start-ray.sh"
chmod +x "$HOME/models/start-ray.sh"
pass "spark-01: copied start-ray-spark-${SCRIPTS[0]}.sh → ~/models/start-ray.sh"

# Remote nodes
for i in "${!REMOTE_NODES[@]}"; do
    ip="${REMOTE_NODES[$i]}"
    name="${REMOTE_NAMES[$i]}"
    letter="${REMOTE_SCRIPTS[$i]}"
    SCRIPT_FILE="$REPO_ROOT/cluster/start-ray-spark-${letter}.sh"
    scp -q "$SCRIPT_FILE" "$USER@$ip:$START_RAY_REMOTE" 2>/dev/null
    ssh $SSH_OPTS "$USER@$ip" "chmod +x $START_RAY_REMOTE" 2>/dev/null
    pass "$name: copied start-ray-spark-${letter}.sh → ~/models/start-ray.sh"
done

echo ""

# --- 3. Start spark-01 (head node) ---
echo "--- 3. Starting spark-01 (head node) ---"

cd "$HOME/models/vllm"
source .venv/bin/activate
bash ../start-ray.sh

pass "spark-01: Ray head started"
echo ""

# --- 4. Start remote nodes with 10-second pauses ---
echo "--- 4. Joining worker nodes ---"

for i in "${!REMOTE_NODES[@]}"; do
    ip="${REMOTE_NODES[$i]}"
    name="${REMOTE_NAMES[$i]}"

    echo "  Waiting ${WAIT_SECS}s before starting $name..."
    sleep $WAIT_SECS

    ssh $SSH_OPTS "$USER@$ip" "cd $VLLM_DIR && source .venv/bin/activate && bash ../start-ray.sh" 2>&1
    pass "$name: Ray worker joined"
done

echo ""

# --- 5. Verify cluster ---
echo "--- 5. Cluster status ---"

ray status

echo ""
echo "========================================"
echo "  Ray cluster is up"
echo "========================================"