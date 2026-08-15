#!/usr/bin/env bash
# sync_to_cluster.sh - push the whole project to the cluster via rsync.
# Excludes only regenerated/local artifacts (the venv is recreated by setup_env.sh on the
# cluster; the vendored transformers/ trl/ unsloth/ dirs ARE synced - they are required).
#
# Usage:
#   bash sync_to_cluster.sh                 # dry-run first is recommended:
#   DRYRUN=1 bash sync_to_cluster.sh
# Override defaults via env: REMOTE_USER, REMOTE_HOST, REMOTE_DIR
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-ztay632}"
REMOTE_HOST="${REMOTE_HOST:-foscsmlprd01.its.auckland.ac.nz}"
REMOTE_DIR="${REMOTE_DIR:-/data/ztay632/dissertation}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/"

EXCLUDES=(
  --exclude='venv/'
  --exclude='.venv/'
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='.pytest_cache/'
  --exclude='unsloth_compiled_cache/'
  --exclude='wandb/'
)

DRY=""
[[ "${DRYRUN:-0}" == "1" ]] && DRY="--dry-run"

echo "[sync] ${SRC}  ->  ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/  ${DRY:+(dry-run)}"
rsync -avz ${DRY} --info=progress2 "${EXCLUDES[@]}" -e ssh \
  "${SRC}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
echo "[sync] done."
