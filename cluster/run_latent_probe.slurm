#!/bin/bash
# ======== SLURM Job Configuration ========
#SBATCH --job-name=hrpo-latent-probe
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/hrpo-latent-probe-%j.log
#SBATCH --error=slurm_logs/hrpo-latent-probe-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# Go/no-go check for the decoupled latent temperature (probe_latent_temp.py). No training.
#   First (~15 min):  check_eval_refactor.py, the default path must reproduce a settled result.
#   Part A (~20 min): how many tokens HRPO's Eq.3 mixture effectively combines on MATH and GSM8K,
#                     as trained (temperature 0.5) and re-tempered to 1.0.
#   Part B (~1.5 h):  the three cold v3 MATH checkpoints re-scored with the mixture at 1.0.
# Part A only:  sbatch --export=ALL,PROBE_ARGS=--skip_rescore cluster/run_latent_probe.slurm
# D3 gate calibration (~15 min):  SKIP_REGRESSION=1 PROBE_ARGS=--calibrate sbatch cluster/run_latent_probe.slurm
# Output: results/latent_temp_probe.json, plus eval_results_lt1.json in each MATH checkpoint.

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

# Regression first: hrpo_latent_probs sits on every arm's rollout path, so the default (no latent
# temperature) must reproduce a settled greedy result exactly (968/1319, batch 32). ~15 min.
if [[ "${SKIP_REGRESSION:-0}" != "1" ]]; then
    echo "=== $(date -Is) regression check of the default path ==="
    python check_eval_refactor.py experiments/1.5b-v3_learned_lambda-rp1-drboth \
        || echo "!!! REGRESSION CHECK FAILED: do not train with this code until it is explained !!!"
fi

echo "=== $(date -Is) latent-temperature probe ==="
python probe_latent_temp.py ${PROBE_ARGS:-}
echo "=== $(date -Is) latent-temperature probe done (exit $?) ==="
