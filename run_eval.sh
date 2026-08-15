#!/usr/bin/env bash
# run_eval.sh - fusion-aware GSM8k evaluation for finished factorial runs.
#
# Generates + sbatch-submits one eval job per run dir (never runs python directly). Each job writes
# eval_results.json (with per-question qid for McNemar) INTO its run dir. Everything under
# PROJECT_ROOT (/data/${UPI}/dissertation); nothing in /data/${UPI}/hrpo.
#
# Usage:
#   ./run_eval.sh                       # eval every run dir under ${PROJECT_ROOT}/runs
#   ./run_eval.sh <run_dir> [<run_dir> ...]   # eval specific run dirs
#   ./run_eval.sh --dry-run [...]       # write scripts, don't submit
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster_env.sh"

DRY_RUN=0
RUN_DIRS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift;;
        -h|--help) sed -n '2,14p' "$0"; exit 0;;
        *) RUN_DIRS+=("$1"); shift;;
    esac
done

RUNS_ROOT="${PROJECT_ROOT}/runs"
LOGS_DIR="${PROJECT_ROOT}/logs"
SLURM_DIR="${PROJECT_ROOT}/slurm_gen"
mkdir -p "${LOGS_DIR}" "${SLURM_DIR}"

# Default: every immediate subdir of runs/ that has a checkpoint.
if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
    shopt -s nullglob
    for d in "${RUNS_ROOT}"/*/; do
        if compgen -G "${d}checkpoint-*" > /dev/null || [[ -f "${d}adapter_config.json" ]]; then
            RUN_DIRS+=("${d%/}")
        fi
    done
    shopt -u nullglob
fi
[[ ${#RUN_DIRS[@]} -eq 0 ]] && { echo "No run dirs with checkpoints under ${RUNS_ROOT}." >&2; exit 1; }

for run_dir in "${RUN_DIRS[@]}"; do
    name="$(basename "${run_dir%/}")"
    script="${SLURM_DIR}/eval_${name}.sh"
    cat > "${script}" <<EOF
#!/bin/bash
#SBATCH --job-name=eval_${name}
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=${LOGS_DIR}/%x-%j.out
#SBATCH --error=${LOGS_DIR}/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --requeue

set -euo pipefail
source "${SCRIPT_DIR}/cluster_env.sh"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd "${PROJECT_ROOT}"
source "${VENV_DIR}/bin/activate"

python3 eval_ablation.py --run_dir "${run_dir}" --batch_size 32
EOF
    echo "eval -> ${run_dir}   (script ${script})"
    if [[ ${DRY_RUN} -eq 1 ]]; then echo "  dry-run: not submitted."; else
        ( cd "${SCRIPT_DIR}" && sbatch "${script}" ); fi
done
