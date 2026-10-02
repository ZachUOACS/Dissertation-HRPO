#!/usr/bin/env bash
# run_ablations.sh — Task 1 ablation driver (spec Section 7).
#
# Usage:
#   ./run_ablations.sh --model_size [0.5b|1.5b|3b|all] --variant [v1|v2|v3|v4|v5|v6|v7|all] [-- extra args]
#
# Anything after `--` is passed through to the variant training scripts
# (e.g. `-- --report_to none`).
#
# Scale x variant matrix (spec Section 6): all six variants at 1.5b;
# v1, v2, v4, v6 only at 0.5b and 3b.
#
# Each run's start/end time and peak GPU memory are merged into the shared
# training_log.json. Runs whose experiment directory already contains a
# completed checkpoint (trainer_state.json with global_step == max_steps)
# are skipped and logged as such.

set -uo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

MODEL_SIZE=""
VARIANT=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_size) MODEL_SIZE="$2"; shift 2 ;;
        --variant)    VARIANT="$2";    shift 2 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$MODEL_SIZE" || -z "$VARIANT" ]]; then
    echo "Usage: $0 --model_size [0.5b|1.5b|3b|all] --variant [v1..v7|all] [-- extra args]" >&2
    exit 1
fi

declare -A SCRIPTS=(
    [v1]=hrpo_grpo_baseline.py
    [v2]=hrpo_fixed_blend.py
    [v3]=hrpo_learned_lambda.py
    [v4]=hrpo_full_gate.py
    [v5]=hrpo_format_reward.py
    [v6]=hrpo_full.py
    [v7]=hrpo_constant_vector.py
)
declare -A VARIANT_IDS=(
    [v1]=v1_grpo_baseline
    [v2]=v2_fixed_blend
    [v3]=v3_learned_lambda
    [v4]=v4_full_hrpo_gate
    [v5]=v5_hrpo_format_reward
    [v6]=v6_hrpo_full
    [v7]=v7_constant_vector
)

# Section 6 scale x variant matrix. v7 is deliberately absent: it is a tier-8
# CONTROL (arm C1), not a link in the v1..v6 ablation chain, so `--variant all`
# must never sweep it into the main results. Requesting it explicitly still
# works (see the NOTE branch below).
variants_for_size() {
    case "$1" in
        1.5b) echo "v1 v2 v3 v4 v5 v6" ;;
        0.5b|3b) echo "v1 v2 v4 v6" ;;
        *) echo "Unknown model size: $1" >&2; return 1 ;;
    esac
}

if [[ "$MODEL_SIZE" == "all" ]]; then
    SIZES=(1.5b 0.5b 3b)
else
    SIZES=("$MODEL_SIZE")
fi

run_one() {
    local size="$1" v="$2"
    local variant_id="${VARIANT_IDS[$v]}"
    local script="${SCRIPTS[$v]}"
    # Derive the experiment directory the SAME way the training script will
    # (variant_common.experiment_dir), so that follow-up runs carrying extra
    # args — --seed / --repetition_penalty / --lora_targets — resolve to their
    # own suffixed directory. Using the bare "{size}-{variant}" here would make
    # every such run skip against the completed main-ablation run.
    local exp_dir
    exp_dir=$("$PYTHON" - "$size" "$variant_id" "${EXTRA_ARGS[@]:-}" <<'PYEOF'
import sys
from variant_common import build_arg_parser, experiment_dir
size, variant_id, extra = sys.argv[1], sys.argv[2], [a for a in sys.argv[3:] if a]
args = build_arg_parser(variant_id).parse_args(["--model_size", size] + extra)
print(experiment_dir(size, variant_id, args).lstrip("./"))
PYEOF
)

    if "$PYTHON" -c "import sys; from variant_common import experiment_is_complete; sys.exit(0 if experiment_is_complete('$exp_dir') else 1)"; then
        echo ">>> SKIP ${size}-${variant_id} (completed checkpoint exists)"
        "$PYTHON" -c "from variant_common import merge_training_log; merge_training_log('$exp_dir', '$size', '$variant_id', '', '', 'skipped_complete')"
        return 0
    fi

    echo ">>> RUN  ${size}-${variant_id}  ($script --model_size $size ${EXTRA_ARGS[*]:-})"
    local start_time end_time status
    start_time="$(date -Iseconds)"
    if "$PYTHON" "$script" --model_size "$size" "${EXTRA_ARGS[@]}"; then
        status="completed"
    else
        status="failed"
    fi
    end_time="$(date -Iseconds)"
    "$PYTHON" -c "from variant_common import merge_training_log; merge_training_log('$exp_dir', '$size', '$variant_id', '$start_time', '$end_time', '$status')"
    if [[ "$status" == "failed" ]]; then
        echo ">>> FAILED ${size}-${variant_id} — continuing with remaining runs" >&2
    fi
}

for size in "${SIZES[@]}"; do
    allowed="$(variants_for_size "$size")" || exit 1
    if [[ "$VARIANT" == "all" ]]; then
        selected="$allowed"
    else
        if [[ " $allowed " != *" $VARIANT "* ]]; then
            echo ">>> NOTE: $VARIANT is not in the Section 6 matrix for $size (matrix: $allowed) — running anyway since it was requested explicitly."
        fi
        selected="$VARIANT"
    fi
    for v in $selected; do
        run_one "$size" "$v"
    done
done

echo ">>> All requested runs processed. See training_log.json."
