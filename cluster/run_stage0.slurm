#!/bin/bash
# ======== SLURM Job Configuration ========
#SBATCH --job-name=hrpo-stage0
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/hrpo-stage0-%j.log
#SBATCH --error=slurm_logs/hrpo-stage0-%j.err
#SBATCH --gres=gpu:1
# Host memory: the 4 GB default is too tight for training jobs (see run_variant.slurm).
#SBATCH --mem=16G
#
# Stage-0 warm start (Stage-0 brief, Phase 4).
#
# Smoke test first — validates the 2-pass blend for pennies before a full epoch:
#   SMOKE=1 BLEND=on FREEZE_LAMBDA=1 sbatch cluster/run_stage0.slurm   # check 2a
#   SMOKE=1 BLEND=on                 sbatch cluster/run_stage0.slurm   # check 2b
#   SMOKE=1 BLEND=off                sbatch cluster/run_stage0.slurm   # check 1
#
# Full warm starts (one per seed, per blend setting):
#   BLEND=on  SEED=42 sbatch cluster/run_stage0.slurm      # feeds arm T1 (v3)
#   BLEND=off SEED=42 sbatch cluster/run_stage0.slurm      # feeds arm C3 (v1)
#
# If pass 1 OOMs, the culprit is the (batch, seq, 152k) logits tensor: lower
# BATCH_SIZE first, then PROJ_CHUNK.

# ======== Environment Setup ========
export HOME=/data/ztay632/
export TMPDIR=/data/ztay632/
export HF_HOME=/data/ztay632/.cache/huggingface
export https_proxy=http://squid.auckland.ac.nz:3128
export http_proxy=http://squid.auckland.ac.nz:3128
export TORCHDYNAMO_DISABLE=1

MODEL_SIZE="${MODEL_SIZE:-1.5b}"
BLEND="${BLEND:-on}"
SEED="${SEED:-42}"
R_MIN="${R_MIN:-0.98}"          # matches the existing -rp1 arms (NOT the brief's 0.95)
BATCH_SIZE="${BATCH_SIZE:-4}"   # lower than RL's 8: pass 1 holds (B, S, vocab) logits
GRAD_ACCUM="${GRAD_ACCUM:-8}"   # keep the effective batch at 32
PROJ_CHUNK="${PROJ_CHUNK:-64}"
GRAD_CKPT="${GRAD_CKPT:-off}"   # off | unsloth  (see --grad_checkpoint)
REPORT_TO="${REPORT_TO:-none}"

# Smoke mode: a handful of steps on a few examples.
if [[ "${SMOKE:-0}" == "1" ]]; then
    LIMIT="${LIMIT:-64}"
    MAX_STEPS="${MAX_STEPS:-10}"
else
    LIMIT="${LIMIT:-}"
    MAX_STEPS="${MAX_STEPS:--1}"
fi

# ======== Job Execution ========
mkdir -p "$HF_HOME" slurm_logs
cd /data/ztay632/Dissertation-HRPO
source .venv/bin/activate

# ---- Preflight: the vendored forks must win over any pip-installed copy ----
# transformers/unsloth/trl in this repo carry the HRPO gate; a pip install would
# silently shadow them with stock, gate-less packages and every run would train
# a model with no latent path at all.
python - <<'PY' || { echo "PREFLIGHT FAILED — vendored packages are shadowed"; exit 1; }
import os, sys
import transformers, unsloth, trl
repo = os.path.abspath(".")
bad = [m.__name__ for m in (transformers, unsloth, trl)
       if not os.path.abspath(m.__file__).startswith(repo)]
if bad:
    sys.exit(f"NOT resolving to the repo: {bad}")
from transformers.generation.utils import hrpo_project_latent  # extraction present
print("preflight OK: vendored transformers/unsloth/trl + Eq.3 helper present")
PY

set -x

CMD=(python stage0_warmstart.py
     --model_size "$MODEL_SIZE"
     --blend "$BLEND"
     --seed "$SEED"
     --residual_r_min "$R_MIN"
     --per_device_train_batch_size "$BATCH_SIZE"
     --gradient_accumulation_steps "$GRAD_ACCUM"
     --proj_chunk "$PROJ_CHUNK"
     --grad_checkpoint "$GRAD_CKPT"
     --report_to "$REPORT_TO")

[[ -n "$LIMIT" ]] && CMD+=(--limit "$LIMIT")
[[ "$MAX_STEPS" != "-1" ]] && CMD+=(--max_steps "$MAX_STEPS")
[[ "${FREEZE_LAMBDA:-0}" == "1" ]] && CMD+=(--freeze_lambda)

# MATH warm start: 1,024-token completions as in the MATH RL runs, and its own
# -dsmath output directory so it can never be mistaken for a GSM8K warm start.
#   DATASET=math BLEND=on SEED=42 sbatch cluster/run_stage0.slurm
# If pass 1 OOMs on MATH's longer sequences: BATCH_SIZE=2 GRAD_ACCUM=16.
DATASET="${DATASET:-gsm8k}"
DS_SUFFIX=""
if [[ "$DATASET" != "gsm8k" ]]; then
    CMD+=(--dataset "$DATASET" --max_completion_length 1024)
    DS_SUFFIX="-ds${DATASET}"
fi

"${CMD[@]}"

set +x
OUT="experiments/${MODEL_SIZE}-stage0-${BLEND}-rmin${R_MIN}-seed${SEED}${DS_SUFFIX}"
echo "=== $(date -Is) stage0 complete -> ${OUT} ==="
[[ "$BLEND" == "on" ]] && echo "    diagnostics: ${OUT}/stage0_log.jsonl"
echo "    feed to RL with:  --init_from ${OUT}"
