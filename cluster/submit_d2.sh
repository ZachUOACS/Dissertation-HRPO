#!/bin/bash
# Submit the D2 programme: exploration controls and rollout parameter noise, GSM8K, Dr. GRPO.
#
#   probe    Cheap controls that any noise result must beat (1.5B, seed 42, 4 jobs):
#              v1 @ tau 0.8, v1 @ tau 1.0   rollout temperature (exploration without noise)
#              v1 @ lr 1e-5                  is the baseline under-trained at the paper's lr?
#              v3 @ tau 0.8                  temperature also sharpens/flattens the Eq.3 latent
#   d2       QeRL-style RMSNorm noise (sigma 5e-2 -> 5e-3, 10 stages) on v1 and v3, seeds 42 43 44
#            (6 jobs). Compare with the settled Dr. GRPO baselines of the same seed.
#   d2sweep  Noise scale, v1 seed 42 only (2 jobs): 2e-2 -> 2e-3 and 1e-1 -> 1e-2.
#   math     The same noise on MATH, v1 and v3, seed 43 (2 jobs, ~32 h each). MATH_SEEDS="43 44" for more.
#
# Usage (from /data/ztay632/Dissertation-HRPO):
#   bash cluster/submit_d2.sh probe
#   bash cluster/submit_d2.sh d2
#   SEEDS="42" bash cluster/submit_d2.sh d2        # throttle
#   MAX_WALLTIME=36:00:00 bash cluster/submit_d2.sh d2
#
# Every run is resumable: a job that is pre-empted or hits --time resumes from its newest
# checkpoint (saved every 250 steps) when the SAME command is submitted again. The cluster
# gives non-PhD users low priority, so expect to resubmit; bash cluster/submit_d2.sh <tier>
# is idempotent for finished runs (exit_if_complete).
#
# All runs pin --repetition_penalty 1.0 and --dr_grpo both explicitly, even though 1.0 is now
# the parser default. Directories: experiments/new-<size>-<variant>-rp1-drboth-<axis>...
# Evaluate with: sbatch cluster/run_eval_d2.slurm ; compare with: python analyse_d2.py

set -u
cd "$(dirname "$0")/.."
mkdir -p slurm_logs

WHICH="${1:-probe}"
WALLTIME="${MAX_WALLTIME:-24:00:00}"     # --time is required by the cluster; 1.5B GSM8K runs take ~6-10 h
SEEDS="${SEEDS:-42 43 44}"
BASE="--repetition_penalty 1.0 --dr_grpo both"

submit() {
    local tag="$1" size="$2" vnum="$3" extra="$4"
    local jobname="hrpo-d2-${tag}"
    export EXTRA_ARGS="$extra"
    local job_id
    job_id=$(sbatch \
        --job-name="$jobname" \
        --time="$WALLTIME" \
        --output="slurm_logs/${jobname}-%j.log" \
        --error="slurm_logs/${jobname}-%j.err" \
        --export="ALL,MODEL_SIZE=${size},VARIANT_NUM=${vnum}" \
        cluster/run_variant.slurm | awk '{print $4}')
    printf "  %-34s -> job %-10s [%s]\n" "$jobname" "$job_id" "$extra"
    unset EXTRA_ARGS
}

case "$WHICH" in
    probe)
        echo "== D2 probes (1.5B, seed 42) =="
        submit "v1-tau0.8-s42"  1.5b 1 "$BASE --seed 42 --temperature 0.8"
        submit "v1-tau1.0-s42"  1.5b 1 "$BASE --seed 42 --temperature 1.0"
        submit "v1-lr1e-5-s42"  1.5b 1 "$BASE --seed 42 --lr 1e-5"
        submit "v3-tau0.8-s42"  1.5b 3 "$BASE --seed 42 --temperature 0.8"
        ;;
    d2)
        echo "== D2 noise 5e-2 -> 5e-3 (1.5B) =="
        for seed in $SEEDS; do
            submit "v1-noise-s${seed}" 1.5b 1 "$BASE --seed ${seed} --noise_sigma_start 0.05 --noise_sigma_end 0.005"
            submit "v3-noise-s${seed}" 1.5b 3 "$BASE --seed ${seed} --noise_sigma_start 0.05 --noise_sigma_end 0.005"
        done
        ;;
    d2sweep)
        echo "== D2 noise scale sweep (1.5B v1, seed 42) =="
        submit "v1-noise-lo-s42" 1.5b 1 "$BASE --seed 42 --noise_sigma_start 0.02 --noise_sigma_end 0.002"
        submit "v1-noise-hi-s42" 1.5b 1 "$BASE --seed 42 --noise_sigma_start 0.1  --noise_sigma_end 0.01"
        ;;
    math)
        # MATH is where the gaps between methods are large (cold Dr. GRPO: v1 45.07, v3 51.20) and
        # where exploration methods report most of their gains, so a noise effect is likelier to be
        # visible here than on GSM8K. Same arguments as the settled MATH Dr. GRPO runs (tier10 of
        # submit_followups.sh) plus noise. ~32 h per run, so one seed first.
        # Default seed 43, NOT 42: the v1 seed-42 MATH baseline is the run that drifted (37.40
        # against 49.20 / 48.60), so a seed-42 comparison would flatter any treatment.
        MATH_SEEDS="${MATH_SEEDS:-43}"
        WALLTIME="${MAX_WALLTIME:-48:00:00}"
        MATH_ARGS="--dataset math --repetition_penalty 1.0 --max_completion_length 1024"
        MATH_ARGS="$MATH_ARGS --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --dr_grpo both"
        echo "== D2 noise 5e-2 -> 5e-3 on MATH (1.5B), seeds: ${MATH_SEEDS} =="
        for seed in $MATH_SEEDS; do
            submit "math-v1-noise-s${seed}" 1.5b 1 "$MATH_ARGS --seed ${seed} --noise_sigma_start 0.05 --noise_sigma_end 0.005"
            submit "math-v3-noise-s${seed}" 1.5b 3 "$MATH_ARGS --seed ${seed} --noise_sigma_start 0.05 --noise_sigma_end 0.005"
        done
        ;;
    *)
        echo "unknown tier '$WHICH' (probe | d2 | d2sweep | math)" >&2
        exit 1
        ;;
esac

echo
echo "Monitor: squeue -u \$USER     Then:  sbatch cluster/run_eval_d2.slurm  &&  python analyse_d2.py"
