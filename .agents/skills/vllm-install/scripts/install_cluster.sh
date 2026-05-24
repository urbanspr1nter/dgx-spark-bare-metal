#!/bin/bash

# Install vLLM across the entire 4× DGX Spark cluster.
# Builds from source on all nodes in parallel using tmux.
#
# Usage: ./install_cluster.sh <vllm_version>
# Example: ./install_cluster.sh v0.21.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

source "$REPO_ROOT/env.sh"

USER="$(whoami)"
NODES=("$SPARK_01_IP" "$SPARK_02_IP" "$SPARK_03_IP" "$SPARK_04_IP")
NODE_NAMES=("spark-01" "spark-02" "spark-03" "spark-04")
REMOTE_NODES=("${NODES[@]:1}")
REMOTE_NAMES=("${NODE_NAMES[@]:1}")

SSH_OPTS="-o ConnectTimeout=5 -o BatchMode=yes"

VLLM_VERSION="${1:?Usage: $0 <vllm_ref> [--branch] (e.g. v0.21.0 or v0.21.0-sm121-fix --branch)}"
BRANCH_FLAG="${2:-}"
VLLM_PATH="\$HOME/models"
REPO_PATH="\$HOME/code/dgx-spark-bare-metal"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }

echo "========================================"
echo "  vLLM Cluster Install — $VLLM_VERSION"
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

# --- 2. Rsync repo to remote nodes ---
echo "--- 2. Rsync-ing repo to remote nodes ---"

for i in "${!REMOTE_NODES[@]}"; do
    ip="${REMOTE_NODES[$i]}"
    name="${REMOTE_NAMES[$i]}"
    if rsync -avz --mkpath "$REPO_ROOT/" "$USER@$ip:$REPO_PATH/" > /dev/null 2>&1; then
        pass "$name: repo synced"
    else
        fail "$name: rsync FAILED"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = false ]; then
    echo ""
    fail "Rsync failed on some nodes. Aborting."
    exit 1
fi

echo ""

# --- 3. Start local build (spark-01) in a new tmux window ---
echo "--- 3. Launching builds via tmux ---"

# Create a new tmux window in session 0 for the local build.
# This avoids disrupting the existing pi window.
TMUX_SESSION="0"
WINDOW_NAME="vllm-install"

# Find the next available window index
EXISTING_INDICES=$(tmux list-windows -t "$TMUX_SESSION" -F '#{window_index}' 2>/dev/null | sort -n)
NEXT_INDEX=0
for idx in $EXISTING_INDICES; do
    if [ "$idx" -ge "$NEXT_INDEX" ]; then
        NEXT_INDEX=$((idx + 1))
    fi
done

tmux new-window -t "${TMUX_SESSION}:${NEXT_INDEX}" -n "$WINDOW_NAME" -c "$REPO_ROOT" 2>/dev/null || {
    # Fallback: if window creation fails, just use the next index we found
    tmux new-window -t "$TMUX_SESSION" -n "$WINDOW_NAME" -c "$REPO_ROOT"
}

tmux send-keys -t "${TMUX_SESSION}:${NEXT_INDEX}" \
    "cd $REPO_ROOT && bash setup/install_vllm.sh $VLLM_PATH $VLLM_VERSION $BRANCH_FLAG; echo '=== SPARK-01 INSTALL COMPLETE ==='" Enter

pass "spark-01: build started in tmux window ${TMUX_SESSION}:${NEXT_INDEX} (named '$WINDOW_NAME')"

# --- 4. Start remote builds in existing tmux session 0 ---
for i in "${!REMOTE_NODES[@]}"; do
    ip="${REMOTE_NODES[$i]}"
    name="${REMOTE_NAMES[$i]}"
    # Send the install command into the existing tmux session 0 on the remote node
    ssh $SSH_OPTS "$USER@$ip" \
        "tmux send-keys -t 0 'cd $REPO_PATH && bash setup/install_vllm.sh $VLLM_PATH $VLLM_VERSION $BRANCH_FLAG; echo === ${name^^} INSTALL COMPLETE ===' Enter" 2>/dev/null
    pass "$name: build started in tmux session 0"
done

echo ""
echo "========================================"
echo "  Builds launched on all 4 nodes"
echo "========================================"
echo ""
echo "  spark-01:  tmux attach -t 0, then switch to window '$WINDOW_NAME'"
echo "  spark-02:  ssh $USER@${NODES[1]}  then tmux attach -t 0"
echo "  spark-03:  ssh $USER@${NODES[2]}  then tmux attach -t 0"
echo "  spark-04:  ssh $USER@${NODES[3]}  then tmux attach -t 0"
echo ""
echo "  Each node will print '=== <NODE> INSTALL COMPLETE ===' when done."
echo "  This is a full source build — expect 15-30 minutes per node."
echo ""
if [ "$BRANCH_FLAG" = "--branch" ]; then
    echo "  Branch mode: git clean/reset skipped. Building from branch '$VLLM_VERSION'."
fi
echo ""