#!/bin/bash
# Submit the Section 6 scale x variant matrix as independent SLURM jobs.
#
# DEFAULT_SIZES below is what a bare `bash cluster/submit_all.sh` submits:
#   1.5b -> v1 v2 v3 v4 v5 v6   (full causal chain)
#   3b   -> v1 v2 v4 v6         (spec Section 6 anchors)
# 0.5b is in the spec matrix but is NOT submitted by default; run it explicitly
# with `bash cluster/submit_all.sh 0.5b` when you want it.
#
# Any run whose experiment dir already holds a completed checkpoint exits
# within minutes via the skip logic (e.g. 1.5b-v1).
#
# WALL TIMES assume the HRPO paper's GSM8k config (Table 4/5): per_device 8,
# grad-accum 4, group 4, 512/512 lengths -> 7473 x 4 / (8 x 4) = 934 optimizer
# steps at ~57 s/step ~= 15 h for 1.5B (measured). 3b scales roughly 2x.
# If a job still hits the limit, just resubmit it: training resumes from the
# newest checkpoint (saved every 250 steps), so no progress is lost.
#
# Usage (from /data/ztay632/Dissertation-HRPO):
#   bash cluster/submit_all.sh            # 1.5b (6 jobs) + 3b (4 jobs)
#   bash cluster/submit_all.sh 0.5b       # add the 0.5b scale later
#   bash cluster/submit_all.sh 3b v5      # one job only (off-matrix is allowed)
#   MAX_WALLTIME=72:00:00 bash cluster/submit_all.sh   # clamp to partition limit
#
# Check the partition's hard limit first:  sinfo -o "%P %l"

set -u
cd "$(dirname "$0")/.."
mkdir -p slurm_logs

DEFAULT_SIZES="1.5b 3b"

ONLY_SIZE="${1:-}"
ONLY_VARIANT="${2:-}"   # e.g. v4

walltime_for() {
    case "$1" in
        0.5b) echo "${MAX_WALLTIME:-30:00:00}" ;;
        1.5b) echo "${MAX_WALLTIME:-72:00:00}" ;;
        3b)   echo "${MAX_WALLTIME:-120:00:00}" ;;
    esac
}

variants_for() {
    case "$1" in
        0.5b) echo "1 2 4 6" ;;
        1.5b) echo "1 2 3 4 5 6" ;;
        3b) echo "1 2 4 6" ;;
    esac
}

echo "Submitting HRPO ablation jobs..."
echo ""

for MODEL_SIZE in 0.5b 1.5b 3b; do
    [[ -n "$ONLY_SIZE" && "$MODEL_SIZE" != "$ONLY_SIZE" ]] && continue
    [[ -z "$ONLY_SIZE" && " $DEFAULT_SIZES " != *" $MODEL_SIZE "* ]] && continue
    TIME="$(walltime_for "$MODEL_SIZE")"
    # An explicitly requested variant is submitted even if it is outside that
    # size's default matrix (e.g. `submit_all.sh 3b v3`, the follow-up run that
    # tests whether v3's 1.5B advantage persists at 3B).
    if [[ -n "$ONLY_VARIANT" ]]; then
        VARIANT_LIST="${ONLY_VARIANT#v}"
    else
        VARIANT_LIST="$(variants_for "$MODEL_SIZE")"
    fi
    for V in $VARIANT_LIST; do
        JOBNAME="hrpo-${MODEL_SIZE}-v${V}"
        JOB_ID=$(sbatch \
            --job-name="$JOBNAME" \
            --time="$TIME" \
            --output="slurm_logs/${JOBNAME}-%j.log" \
            --error="slurm_logs/${JOBNAME}-%j.err" \
            --export="ALL,MODEL_SIZE=${MODEL_SIZE},VARIANT_NUM=${V}" \
            cluster/run_variant.slurm | awk '{print $4}')
        echo "  Submitted ${JOBNAME} -> job ${JOB_ID} (wall time: ${TIME})"
    done
done

echo ""
echo "Monitor with: squeue -u \$USER"
echo "After all runs finish: sbatch cluster/run_eval_and_analysis.slurm"
