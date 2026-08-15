#!/usr/bin/env bash
# sync_from_cluster.sh - pull run outputs from the cluster to local for inspection.
# Run LOCALLY (on your laptop/WSL, NOT inside an ssh session to the cluster); enter your token.
#
# Usage:
#   bash sync_from_cluster.sh            # logs + experiments (metrics/eval_results/run_card),
#                                        # EXCLUDING the big checkpoint-*/ and wandb/ dirs
#   WITH_CHECKPOINTS=1 bash sync_from_cluster.sh   # also pull the LoRA adapter checkpoints
# Override defaults via env: REMOTE_USER, REMOTE_HOST, REMOTE_DIR
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-ztay632}"
REMOTE_HOST="${REMOTE_HOST:-foscsmlprd01.its.auckland.ac.nz}"
REMOTE_DIR="${REMOTE_DIR:-/data/ztay632/dissertation}"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXCLUDES=( --exclude='wandb/' )
if [[ "${WITH_CHECKPOINTS:-0}" != "1" ]]; then
  # keep it light: skip the big adapter checkpoints, keep eval_results.json / run_card.json /
  # trainer_state.json / config etc. so you can read accuracy + curves locally.
  EXCLUDES+=( --exclude='checkpoint-*/' )
fi

# Pull a remote subdir if it exists; a missing/empty source must NOT abort the whole sync
# (set -e would otherwise kill it on the first non-existent dir, e.g. experiments/).
pull() {
  local sub="$1"
  echo "[sync] ${sub} ${WITH_CHECKPOINTS:+(with checkpoints) }..."
  mkdir -p "${DEST}/${sub}"
  # --ignore-missing-args + a trailing || keep a missing remote dir from being fatal.
  if rsync -avz -e ssh "${EXCLUDES[@]}" \
       "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${sub}/" "${DEST}/${sub}/"; then
    :
  else
    echo "[sync]   -> ${sub}/ not pulled (missing on cluster or rsync error; see above). Continuing."
  fi
}

pull logs           # Slurm stdout/stderr
pull runs           # the fusion x debias factorial (eval_results.json lives here; checkpoints excluded)
pull equiv          # equivalence-harness dumps (if any)
pull experiments    # legacy dir (older runs); harmless if absent

echo "[sync] done. Read factorial accuracy with:"
echo "        for f in runs/*/eval_results.json; do echo \"\$f\"; grep -o '\"accuracy\": [0-9.]*' \"\$f\"; done"
