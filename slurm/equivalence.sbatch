#!/bin/bash
# equivalence.sbatch - one-command numerical arm-1 equivalence check (Phase 3 test #1).
#
# Dumps a forward+backward summary from a CLEAN PRE-CHANGE git worktree (HEAD, before the
# uncommitted fusion/debias edits) and from the CURRENT tree, then compares:
#   - new(baseline)      == ref             : default flags reproduce pre-change code
#   - new(beta_neg_inf)  == ref (float tol) : fusion with b_beta=-inf reduces to baseline
#
# Submit:  sbatch slurm/equivalence.sbatch
# (Runs on 1 GPU. The pre-change ref uses the OLD vendored transformers/trl/unsloth from HEAD,
#  picked up because the worktree dir is the cwd / on sys.path - same as how HRPO runs.)
#SBATCH --job-name=hrpo_equiv
#SBATCH --time=00:40:00
#SBATCH --open-mode=append
#SBATCH --output=/data/ztay632/dissertation/logs/%x-%j.out
#SBATCH --error=/data/ztay632/dissertation/logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --requeue
# NOTE: submit from PROJECT_ROOT so $SLURM_SUBMIT_DIR/cluster_env.sh resolves. The log dir
# /data/ztay632/dissertation/logs must exist; `mkdir -p` it once if submitting standalone.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/cluster_env.sh"

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
mkdir -p "${PROJECT_ROOT}/logs"

# Test artifacts live under PROJECT_ROOT (/data/${UPI}/dissertation). The git worktree, however,
# CANNOT live inside the repo (git refuses to nest a worktree in its own working tree), so it goes
# to a temp sibling under /data/${UPI} (TMPDIR) - not in /data/${UPI}/hrpo.
OUT="${PROJECT_ROOT}/equiv"
REF_DIR="${TMPDIR%/}/dissertation_equiv_ref"
mkdir -p "${OUT}"

cd "${PROJECT_ROOT}"
source "${VENV_DIR}/bin/activate"

echo "[equiv] quota before:"; du -sh "${PROJECT_ROOT}" 2>/dev/null || true

# --- 1. Clean pre-change reference worktree at HEAD (before the uncommitted edits) ----------
git -C "${PROJECT_ROOT}" worktree remove --force "${REF_DIR}" 2>/dev/null || true
rm -rf "${REF_DIR}"
git -C "${PROJECT_ROOT}" worktree add --force --detach "${REF_DIR}" HEAD
# The harness is untracked (new), so copy it into the ref worktree; it only uses baseline-path
# calls, which the pre-change code supports.
cp "${PROJECT_ROOT}/equivalence_harness.py" "${REF_DIR}/equivalence_harness.py"

# --- 2. Dump ref (old code, baseline) and new (baseline + beta_neg_inf) ---------------------
( cd "${REF_DIR}"     && python3 equivalence_harness.py --arm baseline     --out "${OUT}/ref.pt" )
( cd "${PROJECT_ROOT}" && python3 equivalence_harness.py --arm baseline     --out "${OUT}/new_baseline.pt" )
( cd "${PROJECT_ROOT}" && python3 equivalence_harness.py --arm beta_neg_inf --out "${OUT}/new_betaneginf.pt" )

# --- 3. Compare -----------------------------------------------------------------------------
echo "=================== arm-1 (baseline) vs pre-change ==================="
python3 "${PROJECT_ROOT}/equivalence_harness.py" --compare "${OUT}/ref.pt" "${OUT}/new_baseline.pt" --rtol 1e-2 --atol 1e-3
RC1=$?
echo "=================== fusion b_beta=-inf vs pre-change =================="
python3 "${PROJECT_ROOT}/equivalence_harness.py" --compare "${OUT}/ref.pt" "${OUT}/new_betaneginf.pt" --rtol 1e-2 --atol 1e-3
RC2=$?

# --- 4. Cleanup worktree --------------------------------------------------------------------
git -C "${PROJECT_ROOT}" worktree remove --force "${REF_DIR}" 2>/dev/null || true

if [[ ${RC1} -eq 0 && ${RC2} -eq 0 ]]; then
    echo "EQUIVALENCE: PASS (both comparisons within tolerance)"
else
    echo "EQUIVALENCE: FAIL (rc_baseline=${RC1} rc_betaneginf=${RC2})" >&2
    exit 1
fi
