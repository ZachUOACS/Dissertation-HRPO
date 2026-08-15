#!/usr/bin/env bash
# run_ablation.sh - one-command runner for the fusion x debias 2x2 factorial.
#
# For each named arm it GENERATES a Slurm script and submits it with `sbatch` (never runs python
# directly - compute happens on the cluster only). Each arm writes to its own run dir so the 2x2
# comparison is a directory diff, and arms are independent single-GPU jobs the scheduler
# parallelizes.
#
# Usage:
#   ./run_ablation.sh baseline            # arm 1: vanilla HRPO
#   ./run_ablation.sh fusion              # arm 2: +fusion (gated on h_I)
#   ./run_ablation.sh debias              # arm 3: +debias (both Dr. GRPO fixes)
#   ./run_ablation.sh fusion_debias       # arm 4: +fusion+debias
#   ./run_ablation.sh all                 # submit arms 1-4 as independent jobs
#   ./run_ablation.sh fusion_token        # secondary: fusion gated on e_hat (on top of fusion)
#   ./run_ablation.sh fusion_adapter      # secondary: fusion + Eq. 20 adapter
#   ./run_ablation.sh len_only            # Change B sub-arm: constant length norm only
#   ./run_ablation.sh std_only            # Change B sub-arm: drop std norm only
#   ./run_ablation.sh equiv               # numerical arm-1 equivalence vs pre-change HEAD (test #1)
#
# Flags:
#   --smoke            short wall time + few steps (Phase 3 smoke); frequent small checkpoints
#   --model NAME       backbone (default Qwen/Qwen2.5-1.5B-Instruct, the smallest configured)
#   --seed N           seed (default 42)
#   --dry-run          write the Slurm script but do not sbatch it
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster_env.sh"

# ---- defaults -------------------------------------------------------------
SMOKE=0
DRY_RUN=0
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
SEED=42

RMIN=""       # residual_r_min; empty -> hrpo default (0.99, paper GSM8k-1.5B). Use --rmin 0.95
              # to give fusion a substantial latent channel (hidden_ratio ~0.20).
BETA_INIT=""  # fusion b_beta init; empty -> hrpo default (-4, ~baseline but too closed to open in
              # ~934 steps). Use --beta-init -1 (beta~0.27) or 0 (beta~0.5) so fusion is active.
ARMS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)     SMOKE=1; shift;;
        --dry-run)   DRY_RUN=1; shift;;
        --model)     MODEL="$2"; shift 2;;
        --seed)      SEED="$2"; shift 2;;
        --rmin)      RMIN="$2"; shift 2;;
        --beta-init) BETA_INIT="$2"; shift 2;;
        -h|--help) sed -n '2,32p' "$0"; exit 0;;
        *)           ARMS+=("$1"); shift;;
    esac
done
[[ ${#ARMS[@]} -eq 0 ]] && { echo "ERROR: name at least one arm (or 'all'). See --help." >&2; exit 1; }

# 'all' expands to the four factorial arms only (secondaries are opt-in).
EXPANDED=()
for a in "${ARMS[@]}"; do
    if [[ "$a" == "all" ]]; then
        EXPANDED+=(baseline fusion debias fusion_debias)
    else
        EXPANDED+=("$a")
    fi
done

# ---- per-arm flag sets (the ONLY thing that differs between arms) ----------
# fusion_enabled toggles Change A; length_norm/advantage_std_norm toggle Change B.
arm_flags() {
    case "$1" in
        baseline)      echo "";;
        fusion)        echo "--fusion_enabled";;
        debias)        echo "--length_norm constant --no-advantage_std_norm";;
        fusion_debias) echo "--fusion_enabled --length_norm constant --no-advantage_std_norm";;
        # secondary fusion arms (run on top of fusion, not part of the 2x2 grid)
        fusion_token)   echo "--fusion_enabled --fusion_gate_input token";;
        fusion_adapter) echo "--fusion_enabled --fusion_use_adapter";;
        # Change B sub-arms (Dr. GRPO Fig. 8 grid) for attributing which bias moves the metrics
        len_only)      echo "--length_norm constant";;
        std_only)      echo "--no-advantage_std_norm";;
        *) echo "__INVALID__";;
    esac
}

MODEL_TAG="$(basename "${MODEL}")"
# Everything lives under PROJECT_ROOT (/data/${UPI}/dissertation); nothing in /data/${UPI}/hrpo.
RUNS_ROOT="${PROJECT_ROOT}/runs"         # per-arm checkpoints/diagnostics
LOGS_DIR="${PROJECT_ROOT}/logs"          # Slurm stdout/stderr
SLURM_DIR="${PROJECT_ROOT}/slurm_gen"    # generated per-arm sbatch scripts (kept out of tracked slurm/)
mkdir -p "${RUNS_ROOT}" "${LOGS_DIR}" "${SLURM_DIR}"

# ---- smoke vs full run knobs ----------------------------------------------
if [[ ${SMOKE} -eq 1 ]]; then
    WALLTIME="00:30:00"; MAXSTEPS=20; SAVESTEPS=10; TAG="smoke"
else
    WALLTIME="24:00:00"; MAXSTEPS=-1; SAVESTEPS=250; TAG="full"
fi

submit_arm() {
    local arm="$1"
    # Special case: the numerical arm-1 equivalence check has its own (worktree-based) Slurm script.
    if [[ "$arm" == "equiv" ]]; then
        echo "arm      : equiv (numerical arm-1 equivalence vs pre-change HEAD)"
        echo "script   : ${SCRIPT_DIR}/slurm/equivalence.sbatch"
        if [[ ${DRY_RUN} -eq 1 ]]; then echo "dry-run  : not submitted."; else
            ( cd "${SCRIPT_DIR}" && sbatch slurm/equivalence.sbatch ); fi
        return 0
    fi
    local flags; flags="$(arm_flags "$arm")"
    if [[ "$flags" == "__INVALID__" ]]; then
        echo "ERROR: unknown arm '$arm'." >&2; return 1
    fi
    # r_min / beta-init tags keep variant runs in separate dirs (AFTER -s<seed> so eval parsing works).
    local rmin_tag=""; local rmin_flag=""
    if [[ -n "${RMIN}" ]]; then rmin_tag="-rmin${RMIN}"; rmin_flag="--residual_r_min ${RMIN}"; fi
    local beta_tag=""; local beta_flag=""
    if [[ -n "${BETA_INIT}" ]]; then beta_tag="-b${BETA_INIT}"; beta_flag="--fusion_beta_init ${BETA_INIT}"; fi
    local run_dir="${RUNS_ROOT}/${MODEL_TAG}-${arm}-s${SEED}${rmin_tag}${beta_tag}-${TAG}"
    local job="hrpo_${arm}_${TAG}"
    local script="${SLURM_DIR}/run_${arm}_${TAG}_s${SEED}${rmin_tag}${beta_tag}.sh"

    cat > "${script}" <<EOF
#!/bin/bash
#SBATCH --job-name=${job}
#SBATCH --time=${WALLTIME}
#SBATCH --open-mode=append
#SBATCH --output=${LOGS_DIR}/%x-%j.out
#SBATCH --error=${LOGS_DIR}/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --requeue

set -euo pipefail
source "${SCRIPT_DIR}/cluster_env.sh"      # UPI, PROJECT_ROOT, proxy, HOME/TMPDIR redirects
export HF_HUB_OFFLINE=1                     # weights pre-downloaded on the login node
export PIP_CACHE_DIR=/data/${UPI}/.cache/pip
export TRITON_CACHE_DIR=/data/${UPI}/.cache/triton
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
source "${VENV_DIR}/bin/activate"

echo "[quota] project dir usage before run:"
du -sh "${PROJECT_ROOT}" 2>/dev/null || true
du -sh "${RUNS_ROOT}" 2>/dev/null || true

# Arm '${arm}': fusion x debias flags below are the ONLY difference between arms.
python3 hrpo_gsm8k.py \\
    --model_name "${MODEL}" \\
    --seed ${SEED} \\
    --exp_name "${run_dir}" \\
    --max_steps ${MAXSTEPS} \\
    --save_steps ${SAVESTEPS} \\
    --resume \\
    ${rmin_flag} ${beta_flag} \\
    ${flags}

echo "[quota] project dir usage after run:"
du -sh "${RUNS_ROOT}" 2>/dev/null || true
EOF

    echo "----------------------------------------------------------------"
    echo "arm      : ${arm}"
    echo "flags    : ${flags:-<none> (baseline)}"
    echo "run_dir  : ${run_dir}"
    echo "script   : ${script}"
    echo "resume   : sbatch ${script}   (re-submit to continue from latest checkpoint)"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "dry-run  : script written, not submitted."
    else
        sbatch "${script}"
    fi
}

for arm in "${EXPANDED[@]}"; do
    submit_arm "${arm}"
done
