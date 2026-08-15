#!/usr/bin/env bash
# cluster_env.sh - cluster-specific paths (never import into committed scripts directly).
# Source this file at the top of every Slurm script:
#   source "$(dirname "$0")/../cluster_env.sh"

# -- Fill in your cluster UPI once -------------------------------------------
export UPI="ztay632"           # e.g. ztay123  (no /home/<UPI> - use /data)

# -- Derived paths (edit only if your cluster layout differs) ----------------
export PROJECT_ROOT="/data/${UPI}/dissertation"
export VENV_DIR="/data/${UPI}/Dissertation-HRPO/.venv"
export WANDB_DIR="${PROJECT_ROOT}/wandb"

# -- Proxy (required for outbound HTTP on this cluster) ----------------------
export HOME=/data/ztay632/
export TMPDIR=/data/ztay632/
export HF_HOME=/data/ztay632/.cache/huggingface
export http_proxy="http://squid.auckland.ac.nz:3128"
export https_proxy="http://squid.auckland.ac.nz:3128"
export HTTP_PROXY="${http_proxy}"
export HTTPS_PROXY="${https_proxy}"

# -- HuggingFace / Wandb tokens (set these or pre-login on the login node) ---
# export HF_TOKEN="hf_..."
# export WANDB_API_KEY="..."

# -- Sanity check -----------------------------------------------------------
if [[ "${UPI}" == "YOUR_UPI_HERE" ]]; then
    echo "[cluster_env] ERROR: set UPI in cluster_env.sh before running." >&2
    exit 1
fi
