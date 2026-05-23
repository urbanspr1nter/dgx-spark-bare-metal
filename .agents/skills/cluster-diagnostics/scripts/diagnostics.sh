#!/bin/bash

# Cluster diagnostics for the DGX Spark bare-metal setup.
# Uses IPv4 addresses only (not hostnames).
# Sources env.sh for node IPs.

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
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }

# Assumes this script is run FROM spark-01. spark-01 is checked locally,
# not via SSH. All other nodes are checked via SSH.

echo "========================================"
echo "  DGX Spark Cluster Diagnostics"
echo "========================================"
echo ""

# --- 1. Reachability ---
echo "--- 1. Reachability ---"

REACHABLE_INDICES=()

for i in "${!NODES[@]}"; do
    ip="${NODES[$i]}"
    name="${NODE_NAMES[$i]}"
    if [ "$i" -eq 0 ]; then
        pass "$name ($ip) [local]"
        REACHABLE_INDICES+=("$i")
    elif ssh $SSH_OPTS "$USER@$ip" 'true' 2>/dev/null; then
        pass "$name ($ip)"
        REACHABLE_INDICES+=("$i")
    else
        fail "$name ($ip) — UNREACHABLE"
    fi
done

if [ ${#REACHABLE_INDICES[@]} -eq 0 ]; then
    echo ""
    echo -e "${RED}No nodes are reachable. Aborting.${NC}"
    exit 1
fi

echo ""

# --- 2. Model Availability ---
echo "--- 2. Model Availability ---"

declare -A NODE_MODELS
for i in "${REACHABLE_INDICES[@]}"; do
    ip="${NODES[$i]}"
    name="${NODE_NAMES[$i]}"
    if [ "$i" -eq 0 ]; then
        NODE_MODELS[$name]=$(ls -1 $HOME/models 2>/dev/null | sort)
    else
        NODE_MODELS[$name]=$(ssh $SSH_OPTS "$USER@$ip" 'ls -1 $HOME/models' 2>/dev/null | sort)
    fi
done

if [ -n "${NODE_MODELS[spark-01]:-}" ]; then
    REF="${NODE_MODELS[spark-01]}"
    for i in "${REACHABLE_INDICES[@]}"; do
        name="${NODE_NAMES[$i]}"
        if [ "${NODE_MODELS[$name]}" = "$REF" ]; then
            pass "$name: model directories match spark-01"
        else
            fail "$name: model directories differ from spark-01"
            diff <(echo "$REF") <(echo "${NODE_MODELS[$name]}") || true
        fi
    done
else
    warn "spark-01 not reachable, cannot compare model directories"
fi

echo ""

# --- 3. Script Path Consistency ---
echo "--- 3. Script Path Consistency ---"

# Resolve home directory from spark-01 (all nodes share the same filesystem layout)
HOME_DIR=$HOME

# Check vllm binary path consistency across scripts
echo "Checking vllm binary paths across scripts..."
declare -A VLLM_PATHS
MODEL_SCRIPTS_DIR="$REPO_ROOT/model_scripts"
for script in "$MODEL_SCRIPTS_DIR"/*.sh; do
    script_name=$(basename "$script")
    vllm_path=$(grep -oE '[^[:space:]]+\.venv/bin/vllm' "$script" 2>/dev/null | head -1 || true)
    if [ -n "$vllm_path" ]; then
        resolved_path=$(echo "$vllm_path" | sed "s|\$HOME|$HOME_DIR|g")
        VLLM_PATHS["$resolved_path"]=""
        echo "  $script_name: $resolved_path"
    fi
done

if [ ${#VLLM_PATHS[@]} -gt 1 ]; then
    fail "vllm binary paths are INCONSISTENT across scripts:"
    for path in "${!VLLM_PATHS[@]}"; do
        echo "    $path"
    done
elif [ ${#VLLM_PATHS[@]} -eq 1 ]; then
    pass "All scripts use the same vllm binary path"
fi

# Check vllm binary exists on all reachable nodes
for path in "${!VLLM_PATHS[@]}"; do
    echo ""
    echo "Verifying vllm binary on nodes: $path"
    for i in "${REACHABLE_INDICES[@]}"; do
        ip="${NODES[$i]}"
        name="${NODE_NAMES[$i]}"
        if [ "$i" -eq 0 ]; then
            if test -f "$path"; then
                pass "$name [local]: exists"
            else
                fail "$name [local]: NOT found"
            fi
        elif ssh $SSH_OPTS "$USER@$ip" "test -f '$path'" 2>/dev/null; then
            pass "$name: exists"
        else
            fail "$name: NOT found"
        fi
    done
done

echo ""

# Check model directory paths from each run script
echo "Checking model directory paths from scripts..."
for script in "$MODEL_SCRIPTS_DIR"/*.sh; do
    script_name=$(basename "$script")

    # Resolve model path: prefer MODEL_PATH variable, fall back to --model argument
    model_path=""

    # Try MODEL_PATH export line
    mp_line=$(grep -E '^export MODEL_PATH=' "$script" 2>/dev/null | head -1 || true)
    if [ -n "$mp_line" ]; then
        model_path=$(echo "$mp_line" | sed 's/^export MODEL_PATH=//' | sed "s|\$HOME|$HOME_DIR|g")
    else
        # Fall back: extract literal path from --model argument
        model_arg=$(grep -E '\-\-model[[:space:]]+' "$script" 2>/dev/null | grep -vE '\$MODEL_PATH' | head -1 | awk '{print $2}' || true)
        if [ -n "$model_arg" ]; then
            model_path=$(echo "$model_arg" | sed "s|\$HOME|$HOME_DIR|g")
        fi
    fi

    if [ -n "$model_path" ]; then
        echo ""
        echo "  $script_name → $model_path"
        for i in "${REACHABLE_INDICES[@]}"; do
            ip="${NODES[$i]}"
            name="${NODE_NAMES[$i]}"
            if [ "$i" -eq 0 ]; then
                if test -d "$model_path"; then
                    pass "$name [local]: exists"
                else
                    fail "$name [local]: NOT found"
                fi
            elif ssh $SSH_OPTS "$USER@$ip" "test -d '$model_path'" 2>/dev/null; then
                pass "$name: exists"
            else
                fail "$name: NOT found"
            fi
        done
    fi
done

echo ""
echo "========================================"
echo "  Diagnostics Complete"
echo "========================================"