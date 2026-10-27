#!/bin/bash
# Smoke test for a new backbone in the cross-model study (../CROSS_MODEL_PLAN.md, Phase 1).
# Trains every arm the study uses for SMOKE_STEPS optimizer steps (with the real batch settings,
# so memory is tested too), saves a checkpoint, scores it on 32 questions with the latent on and,
# for latent arms, off, and prints one verdict table (smoke_family.py). Nothing here touches a
# real run: smoke runs live under experiments/smoke-*, which no analysis or evaluation discovers.
#
#   sbatch cluster/run_smoke_family.slurm                                  # llama1b, 4 steps
#   sbatch --export=ALL,SMOKE_SIZE=llama3b cluster/run_smoke_family.slurm
#   sbatch --export=ALL,SMOKE_ARMS="v1mmlu v3mmlu" cluster/run_smoke_family.slurm   # after prepare_mmlu_arc.py
#
#SBATCH --job-name=hrpo-smoke-family
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/hrpo-smoke-family-%j.log
#SBATCH --error=slurm_logs/hrpo-smoke-family-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

export HOME=/data/ztay632/
export TMPDIR=/data/ztay632/
export HF_HOME=/data/ztay632/.cache/huggingface
export https_proxy=http://squid.auckland.ac.nz:3128
export http_proxy=http://squid.auckland.ac.nz:3128
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

cd /data/ztay632/Dissertation-HRPO
source .venv/bin/activate

SIZE="${SMOKE_SIZE:-llama1b}"
STEPS="${SMOKE_STEPS:-4}"
SMOKE_ARMS="${SMOKE_ARMS:-v1 v3 v4 c1 lt v3math}"
echo "=== $(date -Is) smoke test ${SIZE}, ${STEPS} steps, job ${SLURM_JOB_ID:-?} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# No tests here: they run on the login node before this job is submitted (CROSS_MODEL_PLAN.md,
# Phase 1b), and a test that misbehaves must never run unattended on a compute node.

GSM8K_ARGS="--repetition_penalty 1.0 --dr_grpo both"
MATH_ARGS="--dataset math --repetition_penalty 1.0 --max_completion_length 1024 --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --dr_grpo both"
MMLU_ARGS="--dataset mmlu --max_prompt_length 1024 --repetition_penalty 1.0 --dr_grpo both"
COMMON="--model_size ${SIZE} --max_steps ${STEPS} --report_to none"

# arm | script | task | extra
ARMS=(
    "v1|hrpo_grpo_baseline.py|gsm8k|"
    "v3|hrpo_learned_lambda.py|gsm8k|"
    "v4|hrpo_full_gate.py|gsm8k|"
    "c1|hrpo_learned_lambda.py|gsm8k|--stochastic_latent 0.5"
    "lt|hrpo_learned_lambda.py|gsm8k|--latent_temperature 1.0"
    "v3math|hrpo_learned_lambda.py|math|"
    "v1mmlu|hrpo_grpo_baseline.py|mmlu|"
    "v3mmlu|hrpo_learned_lambda.py|mmlu|"
)
for spec in "${ARMS[@]}"; do
    IFS='|' read -r arm script task extra <<< "$spec"
    [[ " ${SMOKE_ARMS} " == *" ${arm} "* ]] || continue
    case "$task" in
        math) targs="$MATH_ARGS" ;;
        mmlu) targs="$MMLU_ARGS" ;;
        *)    targs="$GSM8K_ARGS" ;;
    esac
    echo "=== $(date -Is) train ${arm} (${script}, ${task}) ==="
    # shellcheck disable=SC2086
    python "$script" $COMMON $targs $extra || echo "!!! train ${arm} exited with $?"
done

echo "=== $(date -Is) score and summarise ==="
python smoke_family.py --size "$SIZE" --steps "$STEPS" --arms ${SMOKE_ARMS}
echo "=== $(date -Is) done ==="
