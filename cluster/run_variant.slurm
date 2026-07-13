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

# Expects (via sbatch --export="ALL,MODEL_SIZE=...,VARIANT_NUM=..."):
#   MODEL_SIZE   0.5b | 1.5b | 3b
#   VARIANT_NUM  1..6

# ======== Environment Setup ========
export HOME=/data/ztay632/
export TMPDIR=/data/ztay632/
export HF_HOME=/data/ztay632/.cache/huggingface
export https_proxy=http://squid.auckland.ac.nz:3128
export http_proxy=http://squid.auckland.ac.nz:3128
export TORCHDYNAMO_DISABLE=1
# wandb: cluster account is logged in, so jobs log online (project
# "latent-reasoning", set by the variant scripts). If the proxy ever blocks
# it, export WANDB_MODE=offline before submitting and `wandb sync` later —
# all analyses read trainer_state.json, so nothing depends on wandb.

# ======== Job Execution Steps ========
cd /data/ztay632/Dissertation-HRPO
source .venv/bin/activate

: "${MODEL_SIZE:?Set MODEL_SIZE via --export (0.5b|1.5b|3b)}"
: "${VARIANT_NUM:?Set VARIANT_NUM via --export (1..6)}"

echo "=== $(date -Is) job ${SLURM_JOB_ID:-?} v${VARIANT_NUM} @ ${MODEL_SIZE} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# run_ablations.sh handles: variant -> script mapping, skip-if-complete,
# and merging start/end time + peak GPU memory into training_log.json
# (flock-guarded, safe under concurrent jobs).
./run_ablations.sh --model_size "$MODEL_SIZE" --variant "v${VARIANT_NUM}"

echo "=== $(date -Is) done (exit $?) ==="
