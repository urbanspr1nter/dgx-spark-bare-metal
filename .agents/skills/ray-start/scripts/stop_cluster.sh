#!/bin/bash

# Stop Ray on all 4 DGX Spark nodes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

source "$REPO_ROOT/env.sh"

USER="$(whoami)"
NODES=("$SPARK_01_IP" "$SPARK_02_IP" "$SPARK_03_IP" "$SPARK_04_IP")
NODE_NAMES=("spark-01" "spark-02" "spark-03" "spark-04")

SSH_OPTS="-o ConnectTimeout=5 -o BatchMode=yes"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }

echo "========================================"
echo "  Ray Cluster Stop"
echo "========================================"
echo ""

# Stop remote nodes first, then head
for i in $(seq $((${#NODES[@]} - 1)) -1 0); do
    ip="${NODES[$i]}"
    name="${NODE_NAMES[$i]}"
    if [ "$i" -eq 0 ]; then
        cd "$HOME/models/vllm"
        source .venv/bin/activate
        ray stop -f 2>/dev/null || true
        pass "$name [local]: Ray stopped"
    else
        ssh $SSH_OPTS "$USER@$ip" "cd \$HOME/models/vllm && source .venv/bin/activate && ray stop -f" 2>/dev/null || true
        pass "$name: Ray stopped"
    fi
done

echo ""
echo "========================================"
echo "  Ray cluster stopped"
echo "========================================"