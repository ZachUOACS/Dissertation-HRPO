#!/bin/bash
# ======== SLURM Job Configuration ========
#SBATCH --job-name=hrpo-eval-analysis
#SBATCH --time=36:00:00
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/hrpo-eval-analysis-%j.log
#SBATCH --error=slurm_logs/hrpo-eval-analysis-%j.err
#SBATCH --gres=gpu:1

# Post-training pipeline (run AFTER all ablation jobs finish):
#   1. eval_all.py            — full 1319-question GSM8K eval of every final
#                               checkpoint + base models; comparison table
#   2. analyse_gate.py        — static plots (2-6, gradient flow) for all
#                               sizes, then dynamic plots (1/7/8, variance)
#                               per size
#   3. analyse_failures.py    — v4-vs-v1 (and v6-vs-v1) failure analysis per size
#   4. hrpo_overhead.py       — size x variant overhead matrix
#   5. generate_report.py     — LaTeX tables, summary figures, key findings

# ======== Environment Setup ========
export HOME=/data/ztay632/
export TMPDIR=/data/ztay632/
export HF_HOME=/data/ztay632/.cache/huggingface
export https_proxy=http://squid.auckland.ac.nz:3128
export http_proxy=http://squid.auckland.ac.nz:3128
export TORCHDYNAMO_DISABLE=1
# (No training happens here; wandb is untouched. Analyses read local JSON.)

# ======== Job Execution Steps ========
cd /data/ztay632/Dissertation-HRPO
source .venv/bin/activate
set -x

# 1. Evaluations (skips anything already fully evaluated).
python eval_all.py --batch_size 32

# 2. Gate behaviour analysis.
python analyse_gate.py --static
for SIZE in 0.5b 1.5b 3b; do
    python analyse_gate.py --dynamic --model_size "$SIZE" --n_examples 200 \
        --batch_size 16 || echo "dynamic gate analysis skipped/failed for $SIZE"
done

# 3. Failure-mode analysis (every size with both v4/v6 and v1 evaluated).
for SIZE in 0.5b 1.5b 3b; do
    for VAR in v4_full_hrpo_gate v6_hrpo_full; do
        HRPO_CKPT=$(ls -d experiments/${SIZE}-${VAR}/checkpoint-* 2>/dev/null | sort -V | tail -1)
        GRPO_CKPT=$(ls -d experiments/${SIZE}-v1_grpo_baseline/checkpoint-* 2>/dev/null | sort -V | tail -1)
        if [[ -n "$HRPO_CKPT" && -n "$GRPO_CKPT" ]]; then
            python analyse_failures.py "$HRPO_CKPT" "$GRPO_CKPT" \
                || echo "failure analysis skipped for ${SIZE}-${VAR}"
        fi
    done
done

# 4. Overhead matrix (isolated subprocess timing per size x variant).
python hrpo_overhead.py --variant_matrix

# 5. Report.
python generate_report.py

echo "=== $(date -Is) pipeline complete — see results/ ==="
