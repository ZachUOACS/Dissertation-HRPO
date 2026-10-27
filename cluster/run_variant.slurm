#!/bin/bash
# ======== SLURM Job Configuration ========
# job-name / time / output / error are overridden per job by submit_all.sh;
# the values below are safe fallbacks for a manual `sbatch cluster/run_variant.slurm`.
#SBATCH --job-name=hrpo-ablation
#SBATCH --time=30:00:00
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/hrpo-ablation-%j.log
#SBATCH --error=slurm_logs/hrpo-ablation-%j.err
#SBATCH --gres=gpu:1
# Host memory. Without this every job gets the cluster's 4 GB default, and a GSM8K
# run peaks at ~4.0 GB (MATH ~6.2 GB): tier-11 v3 seed 43 hit 4,208,084K and was
# killed twice with no traceback. 16 GB is ~2.5x the largest peak seen.
#SBATCH --mem=16G

# Expects (via sbatch --export="ALL,MODEL_SIZE=...,VARIANT_NUM=..."):
#   MODEL_SIZE   0.5b | 1.5b | 3b
#   VARIANT_NUM  1..7  (7 = v7_constant_vector, tier-8 control C1)

# ======== Environment Setup ========
export HOME=/data/ztay632/
export TMPDIR=/data/ztay632/
export HF_HOME=/data/ztay632/.cache/huggingface
export https_proxy=http://squid.auckland.ac.nz:3128
export http_proxy=http://squid.auckland.ac.nz:3128
export TORCHDYNAMO_DISABLE=1
# Unbuffered stdout: otherwise Python block-buffers when writing to a file and
# a job cancelled before its first flush leaves an empty .log (tqdm goes to
# stderr, so only the .err would have content).
export PYTHONUNBUFFERED=1
# wandb: cluster account is logged in, so jobs log online (project
# "latent-reasoning", set by the variant scripts). If the proxy ever blocks
# it, export WANDB_MODE=offline before submitting and `wandb sync` later —
# all analyses read trainer_state.json, so nothing depends on wandb.

# ======== Job Execution Steps ========
cd /data/ztay632/Dissertation-HRPO
source .venv/bin/activate

: "${MODEL_SIZE:?Set MODEL_SIZE via --export (0.5b|1.5b|3b)}"
: "${VARIANT_NUM:?Set VARIANT_NUM via --export (1..7)}"

echo "=== $(date -Is) job ${SLURM_JOB_ID:-?} v${VARIANT_NUM} @ ${MODEL_SIZE} on $(hostname) ==="
echo "logs: $(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -E 'StdOut|StdErr' | tr -s ' ')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# run_ablations.sh handles: variant -> script mapping, skip-if-complete,
# and merging start/end time + peak GPU memory into training_log.json
# (flock-guarded, safe under concurrent jobs).
# EXTRA_ARGS (optional, propagated via sbatch --export=ALL) carries follow-up
# run overrides such as "--seed 43" or "--repetition_penalty 1.0 --lora_targets
# attention". Unquoted on purpose so it word-splits into separate arguments.
echo "extra args: ${EXTRA_ARGS:-<none>}"
./run_ablations.sh --model_size "$MODEL_SIZE" --variant "v${VARIANT_NUM}" -- ${EXTRA_ARGS:-}
train_status=$?

# D2 programme: score finished experiments/new-* runs on this GPU before releasing it, so a result
# never waits behind days of queued training. Only for runs that move the D2 axes (noise,
# temperature, learning rate, latent temperature); eval_d2.py skips anything already evaluated or unfinished, so a
# resubmitted or resumed job does no duplicate work. A failed evaluation never fails the job.
case " ${EXTRA_ARGS:-} " in
    *" --noise_sigma_start "*|*" --temperature "*|*" --lr "*|*" --latent_temperature "*|*" --blend_gate "*|*" --stochastic_latent "*|*" --length_penalty_alpha "*)
        echo "=== $(date -Is) D2 evaluation of finished new-* runs ==="
        python eval_d2.py || echo "eval_d2.py exited with $? (training result unaffected)"
        ;;
esac

# Baseline runs submitted with AUTO_EVAL_BASELINE (submit_d2.sh scale3b): score this run, and for a
# latent arm also with the blend off, on this GPU once training has finished. Exact id plus size,
# because a run id carries no size. A job killed at --time never reaches this point, and a failed
# training (non-zero status) is not scored.
if [[ -n "${AUTO_EVAL_BASELINE:-}" && $train_status -eq 0 ]]; then
    echo "=== $(date -Is) scoring ${MODEL_SIZE} ${AUTO_EVAL_BASELINE} ==="
    python eval_d2.py --baselines --sizes "$MODEL_SIZE" --exact --variants "$AUTO_EVAL_BASELINE" \
        || echo "eval_d2.py exited with $? (training result unaffected)"
    if [[ "$VARIANT_NUM" != 1 ]]; then
        python eval_d2.py --baselines --sizes "$MODEL_SIZE" --exact --variants "$AUTO_EVAL_BASELINE" \
            --latent_off || echo "eval_d2.py --latent_off exited with $? (training result unaffected)"
    fi
fi

echo "=== $(date -Is) done (exit ${train_status}) ==="
