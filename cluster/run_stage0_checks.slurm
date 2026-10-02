#!/bin/bash
#SBATCH --job-name=hrpo-stage0-checks
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/hrpo-stage0-checks-%j.log
#SBATCH --error=slurm_logs/hrpo-stage0-checks-%j.err
#SBATCH --gres=gpu:1
#
# Stage-0 brief Phase 4, verification checks 3 and 4.
#   check 3: an RL run with --init_from loads the warm start without key/shape
#            mismatches and its step-0 Lambda equals the Stage-0 final Lambda.
#            --init_only exits straight after the load, so this costs seconds.
#   check 4: zero-shot GSM8K of both warm starts, reported against base 61.11.

export HOME=/data/ztay632/
export TMPDIR=/data/ztay632/
export HF_HOME=/data/ztay632/.cache/huggingface
export https_proxy=http://squid.auckland.ac.nz:3128
export http_proxy=http://squid.auckland.ac.nz:3128
export TORCHDYNAMO_DISABLE=1

SEED="${SEED:-42}"
SIZE="${SIZE:-1.5b}"

mkdir -p "$HF_HOME" slurm_logs
cd /data/ztay632/Dissertation-HRPO
source .venv/bin/activate

echo "########## CHECK 3a: v3 (T1) from the --blend on warm start ##########"
python hrpo_learned_lambda.py --model_size "$SIZE" --repetition_penalty 1.0 \
    --seed "$SEED" --report_to none --init_only \
    --init_from "experiments/${SIZE}-stage0-on-rmin0.98-seed${SEED}"

echo "########## CHECK 3b: v1 (C3) from the --blend off warm start ##########"
python hrpo_grpo_baseline.py --model_size "$SIZE" --repetition_penalty 1.0 \
    --seed "$SEED" --report_to none --init_only \
    --init_from "experiments/${SIZE}-stage0-off-rmin0.98-seed${SEED}"

echo "########## CHECK 4: zero-shot GSM8K of both warm starts ##########"
python eval_stage0.py --model_size "$SIZE" --seed "$SEED" --batch_size 32

echo "=== $(date -Is) stage0 checks complete ==="
