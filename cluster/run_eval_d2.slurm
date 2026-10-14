#!/bin/bash
# ======== SLURM Job Configuration ========
#SBATCH --job-name=hrpo-eval-d2
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/hrpo-eval-d2-%j.log
#SBATCH --error=slurm_logs/hrpo-eval-d2-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# Evaluate every finished experiments/new-* run (the D2 programme arms), then summarise.
# Latent-off scoring: submit again with LATENT_OFF=1.

export HOME=/data/ztay632/
export TMPDIR=/data/ztay632/
export HF_HOME=/data/ztay632/.cache/huggingface
export https_proxy=http://squid.auckland.ac.nz:3128
export http_proxy=http://squid.auckland.ac.nz:3128
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME"
cd /data/ztay632/Dissertation-HRPO
source .venv/bin/activate

# Modes (choose with --export):
#   default        greedy single-decode eval of every finished new-* run, then the summary
#   LATENT_OFF=1   also score each new-* checkpoint with the Eq.4 blend switched off
#   PASSK=8        sampled eval (avg@8 / pass@8 / maj@8 / tokens) of the new-* runs AND
#                  their Dr. GRPO baselines. k times the cost of a greedy eval, so give it
#                  time:  sbatch --time=36:00:00 --export=ALL,PASSK=8 cluster/run_eval_d2.slurm
#                  Already-finished files are skipped, so it is safe to resubmit.
#   QUANT=1        weight-only quantisation study (eval_quant.py): the Dr. GRPO arms with their
#                  weights merged and quantised (bf16 reference, int8, int4, nf4), blend on and,
#                  for latent arms, off; then analyse_quant.py. Runs tests/test_weight_quant.py
#                  first and stops if it fails. About 30 min per MATH evaluation.
if [[ "${LATENT_OFF:-0}" == "1" ]]; then
    python eval_d2.py --latent_off ${EXTRA_EVAL_ARGS:-}
elif [[ "${QUANT:-0}" == "1" ]]; then
    python -m pytest tests/test_weight_quant.py -q || exit 1
    python eval_quant.py ${EXTRA_EVAL_ARGS:-}
    python analyse_quant.py
elif [[ "${PASSK:-0}" != "0" ]]; then
    python eval_d2.py --passk "${PASSK}" ${EXTRA_EVAL_ARGS:-}
    python analyse_d2.py
else
    python eval_d2.py ${EXTRA_EVAL_ARGS:-}
    python analyse_d2.py
fi
