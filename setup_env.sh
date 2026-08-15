#!/usr/bin/env bash
# setup_env.sh - run ONCE on a login/interactive node to create the venv
# and pre-download model weights.
#
# Usage:
#   bash setup_env.sh
#
# Prerequisites:
#   - cluster_env.sh filled in with your UPI
#   - requirements_train.txt present in the project root
#   - module system has python3.10+, cuda 12.x

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster_env.sh"

echo "[setup] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[setup] VENV_DIR=${VENV_DIR}"

# -- 1. Create venv ----------------------------------------------------------
if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
    echo "[setup] venv created at ${VENV_DIR}"
else
    echo "[setup] venv already exists, skipping creation"
fi

source "${VENV_DIR}/bin/activate"

# -- 2. Install dependencies -------------------------------------------------
pip install --upgrade pip

# requirements.txt is the original HRPO full environment dump - don't use it.
# transformers/ and unsloth/ are vendored locally with patches; don't pip-install them.
pip install -r "${PROJECT_ROOT}/requirements_train.txt"

# unsloth_zoo is a runtime dep of the vendored unsloth (unsloth/__init__.py:200 requires
# >= 2025.3.17; vendored unsloth is 2025.3.19). Install it WITHOUT deps so it cannot drag a
# different torch/transformers/trl over the built env (this is how unsloth upgrades it too).
pip install --no-deps --no-cache-dir "unsloth_zoo==2025.3.17"

# -- 3. Pre-download model weights to HF cache -------------------------------
export HF_HOME="${HF_HOME}"
python3 - <<'PYEOF'
from huggingface_hub import snapshot_download
for repo in [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]:
    print(f"[setup] downloading {repo} ...")
    snapshot_download(repo_id=repo, ignore_patterns=["*.msgpack", "*.h5", "flax_*"])
    print(f"[setup] done: {repo}")
PYEOF

echo "[setup] All done. Activate with: source ${VENV_DIR}/bin/activate"
