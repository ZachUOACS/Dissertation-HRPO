#!/bin/bash
# ======== SLURM Job Configuration ========
#SBATCH --job-name=hrpo-latent-off
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/hrpo-latent-off-%j.log
#SBATCH --error=slurm_logs/hrpo-latent-off-%j.err
#SBATCH --gres=gpu:1

# Latent-off evaluation (Stage-0 brief, Phase 2).
#
# Re-evaluates trained checkpoints with the Eq.4 blend switched OFF
# (gate_mode="disabled" -> disable_gate sets answer_start="", so is_thinking is
# False for every token and the model decodes on pure token embeddings). Weights
# are untouched. Results go to eval_results_latent_off.json next to each
# checkpoint's eval_results.json — the headline file is never overwritten — and
# results/latent_off_summary.json gets the paired normal-vs-off table.
#
# MUST run as a batch job: it needs a GPU, and the env block below (HF_HOME +
# squid proxy) — running it bare on the login node fails at model load with
# "Connection reset by peer" and an unwritable ~/.cache/huggingface.

# ======== Environment Setup ========
export HOME=/data/ztay632/
export TMPDIR=/data/ztay632/
export HF_HOME=/data/ztay632/.cache/huggingface
export https_proxy=http://squid.auckland.ac.nz:3128
export http_proxy=http://squid.auckland.ac.nz:3128
export TORCHDYNAMO_DISABLE=1

# Which arms to evaluate. Override at submit time, e.g.
#   VARIANTS="v3_learned_lambda v4_full_hrpo_gate" sbatch cluster/run_latent_off_eval.slurm
SIZES="${SIZES:-1.5b}"
RP="${RP:-1.0}"
VARIANTS="${VARIANTS:-v1_grpo_baseline v3_learned_lambda v4_full_hrpo_gate}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# ======== Job Execution ========
mkdir -p "$HF_HOME" slurm_logs
cd /data/ztay632/Dissertation-HRPO
source .venv/bin/activate

set -x
python eval_all.py --latent_off \
    --batch_size "$BATCH_SIZE" \
    --sizes $SIZES \
    --rp "$RP" \
    --variants $VARIANTS

echo "=== $(date -Is) latent-off eval complete — see results/latent_off_summary.json ==="
