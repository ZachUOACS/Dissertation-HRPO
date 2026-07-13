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
# Wall times are sized from the measured 18.5 h for the completed 1.5B run on
# an RTX 3060 Ti; the cluster GPUs should finish earlier and jobs simply end.
#
# Usage (from /data/ztay632/Dissertation-HRPO):
#   bash cluster/submit_all.sh            # 1.5b (6 jobs) + 3b (4 jobs)
#   bash cluster/submit_all.sh 0.5b       # add the 0.5b scale later
#   bash cluster/submit_all.sh 3b v5      # one job only (off-matrix is allowed)

set -u
cd "$(dirname "$0")/.."
mkdir -p slurm_logs

DEFAULT_SIZES="1.5b 3b"

ONLY_SIZE="${1:-}"
ONLY_VARIANT="${2:-}"   # e.g. v4

walltime_for() {
    case "$1" in
        1.5b) echo "30:00:00" ;;
        3b)   echo "48:00:00" ;;
    esac
}

variants_for() {
    case "$1" in
        1.5b) echo "1 2 3 4 5 6" ;;
        3b) echo "1 2 4 6" ;;
    esac
}

echo "Submitting HRPO ablation jobs..."
echo ""

for MODEL_SIZE in 1.5b 3b; do
    [[ -n "$ONLY_SIZE" && "$MODEL_SIZE" != "$ONLY_SIZE" ]] && continue
    TIME="$(walltime_for "$MODEL_SIZE")"
    for V in $(variants_for "$MODEL_SIZE"); do
        [[ -n "$ONLY_VARIANT" && "v$V" != "$ONLY_VARIANT" ]] && continue
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
