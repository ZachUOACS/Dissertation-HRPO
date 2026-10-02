#!/bin/bash
#SBATCH --job-name=hrpo-blend-debug
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/hrpo-blend-debug-%j.log
#SBATCH --error=slurm_logs/hrpo-blend-debug-%j.err
#SBATCH --gres=gpu:1
#
# Standalone gradient-path diagnostic for the Stage-0 blend. Seconds, not a
# training run. Tests checkpointing off/unsloth x Stage-0-style/RL-style call
# in one go and prints where the autograd chain breaks.

export HOME=/data/ztay632/
export TMPDIR=/data/ztay632/
export HF_HOME=/data/ztay632/.cache/huggingface
export https_proxy=http://squid.auckland.ac.nz:3128
export http_proxy=http://squid.auckland.ac.nz:3128
export TORCHDYNAMO_DISABLE=1

mkdir -p "$HF_HOME" slurm_logs
cd /data/ztay632/Dissertation-HRPO
source .venv/bin/activate

python tests/debug_blend_gradient.py
