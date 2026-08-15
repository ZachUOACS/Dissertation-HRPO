#!/bin/bash
# cluster/cleanup.sh — reclaim disk WITHOUT losing any analysis data.
#
# For every experiment whose training is COMPLETE (global_step == max_steps):
#   1. delete all but the highest-step checkpoint;
#   2. delete optimizer.pt / scheduler.pt / rng_state.pth from the survivor
#      (these are resume-only state; a finished run never resumes).
# Kept: adapter_model.safetensors (gate analysis reads Lambda), eval_results.json,
#       trainer_state.json (full step log), gradient_log.jsonl, run_config.json.
#
# NOTHING is deleted for an incomplete/running experiment — its optimizer state
# is needed to resume at the wall-time limit.
#
# Dry run by default (prints what it WOULD remove). Add --apply to delete.
#
# Usage (from /data/ztay632/Dissertation-HRPO):
#   bash cluster/cleanup.sh            # dry run
#   bash cluster/cleanup.sh --apply    # actually delete

set -u
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

is_complete() {
    "$PYTHON" -c "import sys; from variant_common import experiment_is_complete; \
sys.exit(0 if experiment_is_complete('$1') else 1)"
}

freed=0
for d in experiments/*/; do
    d="${d%/}"
    if ! is_complete "$d"; then
        echo "SKIP (incomplete/running): $d"
        continue
    fi
    # highest-step checkpoint = the one to keep
    last=$(ls -d "$d"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
    [[ -z "$last" ]] && continue
    keep="$d/checkpoint-$last"

    # 1. non-final checkpoints
    for c in "$d"/checkpoint-*; do
        [[ "$c" == "$keep" ]] && continue
        sz=$(du -sm "$c" 2>/dev/null | cut -f1); freed=$((freed + sz))
        echo "$([[ $APPLY == 1 ]] && echo DELETE || echo would-delete) $c  (${sz} MB)"
        [[ $APPLY == 1 ]] && rm -rf "$c"
    done
    # 2. resume-only state in the survivor
    for f in optimizer.pt scheduler.pt rng_state.pth; do
        p="$keep/$f"
        [[ -f "$p" ]] || continue
        sz=$(du -sm "$p" 2>/dev/null | cut -f1); freed=$((freed + sz))
        echo "$([[ $APPLY == 1 ]] && echo DELETE || echo would-delete) $p  (${sz} MB)"
        [[ $APPLY == 1 ]] && rm -f "$p"
    done
done

echo "-------------------------------------------------------------"
echo "$([[ $APPLY == 1 ]] && echo 'Freed' || echo 'Would free') ~${freed} MB"
[[ $APPLY == 0 ]] && echo "(dry run — re-run with --apply to delete)"
