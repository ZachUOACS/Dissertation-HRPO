#!/bin/bash
# Submit the D2 programme: exploration controls and rollout parameter noise, GSM8K, Dr. GRPO.
#
#   probe    Cheap controls that any noise result must beat (1.5B, seed 42, 4 jobs):
#              v1 @ tau 0.8, v1 @ tau 1.0   rollout temperature (exploration without noise)
#              v1 @ lr 1e-5                  is the baseline under-trained at the paper's lr?
#              v3 @ tau 0.8                  temperature also sharpens/flattens the Eq.3 latent
#   d2       QeRL-style RMSNorm noise (sigma 5e-2 -> 5e-3, 10 stages) on v1 and v3, seeds 42 43 44
#            (6 jobs). Compare with the settled Dr. GRPO baselines of the same seed.
#   d2sweep  Noise scale, v1 seed 42 only (2 jobs): 2e-2 -> 2e-3 and 1e-1 -> 1e-2.
#   math     The same noise on MATH, v1 and v3, seed 43 (2 jobs, ~32 h each). MATH_SEEDS="43 44" for more.
#   latent   Decoupled latent temperature (--latent_temperature, default 1.0) on MATH, v3, seeds 43 44
#            (2 jobs, ~32 h each). LATENT_SEEDS="42" for the third seed.
#   d3       Uncertainty-gated blend (D3) on MATH v3, seeds 43 44, with latent temperature 1.0 by
#            default. Needs D3_D0 and D3_D1 from the calibration job, e.g.
#            D3_D0=0.02 D3_D1=0.1 bash cluster/submit_d2.sh d3
#   c1       Stochastic latent with its own gradient (after LEPO) on MATH v3, seeds 43 44,
#            Gumbel temperature C1_TAU (default 0.5).
#   length   Length-penalised reward (Arora & Zanette 2025) on MATH, v1 AND v3, seeds 43 44
#            (4 jobs), alpha LP_ALPHA (default 0.2). LP_VARIANTS="3" for v3 only.
#   scale3b  3B scale check: cold Dr. GRPO v1 and v3 at Qwen2.5-3B, GSM8K and MATH, seed 42 (4 jobs;
#            GSM8K 48 h, MATH 96 h wall time). S3B_TASKS="math" or S3B_SEEDS="43" to change.
#   family   Cross-model study (../CROSS_MODEL_PLAN.md): the same configuration on another backbone.
#            FAM_MODELS (size keys from model_registry.py, default llama1b), FAM_TASKS (gsm8k math
#            mmlu, default gsm8k; mmlu needs prepare_mmlu_arc.py first), FAM_ARMS (v1 v3 v4 = baselines, c1 = v3 + stochastic latent 0.5,
#            lt = v3 + latent temperature 1.0; default v1 v3), FAM_SEEDS (default 42 43 44).
#            e.g. FAM_TASKS="gsm8k math" FAM_ARMS="v1 v3" bash cluster/submit_d2.sh family
#
# Usage (from /data/ztay632/Dissertation-HRPO):
#   bash cluster/submit_d2.sh probe
#   bash cluster/submit_d2.sh d2
#   SEEDS="42" bash cluster/submit_d2.sh d2        # throttle
#   MAX_WALLTIME=36:00:00 bash cluster/submit_d2.sh d2
#
# Every run is resumable: a job that is pre-empted or hits --time resumes from its newest
# checkpoint (saved every 250 steps) when the SAME command is submitted again. The cluster
# gives non-PhD users low priority, so expect to resubmit; bash cluster/submit_d2.sh <tier>
# is idempotent for finished runs (exit_if_complete).
#
# All runs pin --repetition_penalty 1.0 and --dr_grpo both explicitly, even though 1.0 is now
# the parser default. Directories: experiments/new-<size>-<variant>-rp1-drboth-<axis>...
# Evaluate with: sbatch cluster/run_eval_d2.slurm ; compare with: python analyse_d2.py

set -u
cd "$(dirname "$0")/.."
mkdir -p slurm_logs

WHICH="${1:-probe}"
WALLTIME="${MAX_WALLTIME:-24:00:00}"     # --time is required by the cluster; 1.5B GSM8K runs take ~6-10 h
SEEDS="${SEEDS:-42 43 44}"
BASE="--repetition_penalty 1.0 --dr_grpo both"

submit() {
    local tag="$1" size="$2" vnum="$3" extra="$4"
    local jobname="${JOB_PREFIX:-hrpo-d2}-${tag}"
    export EXTRA_ARGS="$extra"
    local job_id
    if [[ -n "${SUBMIT_DRY_RUN:-}" ]]; then
        # Tests only (tests/test_family.py): append the job to the file SUBMIT_DRY_RUN names and never
        # call sbatch. A fake sbatch on the PATH is not safe: on 23 October a test's fake in /tmp
        # (mounted noexec) was skipped by the PATH search and the real sbatch submitted 40 jobs.
        printf '%s\t%s\t%s\n' \
            "--job-name=${jobname} --time=${WALLTIME} ${JOB_MEM:+--mem=$JOB_MEM} --export=ALL,MODEL_SIZE=${size},VARIANT_NUM=${vnum}" \
            "${AUTO_EVAL_BASELINE:-}" "$extra" >> "$SUBMIT_DRY_RUN"
        job_id="DRYRUN"
    else
        job_id=$(sbatch \
            --job-name="$jobname" \
            --time="$WALLTIME" \
            ${JOB_MEM:+--mem=$JOB_MEM} \
            --output="slurm_logs/${jobname}-%j.log" \
            --error="slurm_logs/${jobname}-%j.err" \
            --export="ALL,MODEL_SIZE=${size},VARIANT_NUM=${vnum}" \
            cluster/run_variant.slurm | awk '{print $4}')
    fi
    printf "  %-34s -> job %-10s [%s]\n" "$jobname" "$job_id" "$extra"
    unset EXTRA_ARGS
}

# Run id of a cold Dr. GRPO baseline, as variant_common.experiment_dir names it (without the size
# prefix): baseline_id 3 42 math -> v3_learned_lambda-rp1-drboth-dsmath. tests/test_scale3b.py
# checks it against experiment_dir.
baseline_id() {
    local vnum="$1" seed="$2" task="$3" id
    case "$vnum" in
        1) id="v1_grpo_baseline" ;;
        3) id="v3_learned_lambda" ;;
        4) id="v4_full_hrpo_gate" ;;
        *) echo "baseline_id: variant $vnum not supported" >&2; return 1 ;;
    esac
    [[ "$seed" != 42 ]] && id="${id}-seed${seed}"
    id="${id}-rp1-drboth"
    [[ "$task" == math ]] && id="${id}-dsmath"
    echo "$id"
}

case "$WHICH" in
    probe)
        echo "== D2 probes (1.5B, seed 42) =="
        submit "v1-tau0.8-s42"  1.5b 1 "$BASE --seed 42 --temperature 0.8"
        submit "v1-tau1.0-s42"  1.5b 1 "$BASE --seed 42 --temperature 1.0"
        submit "v1-lr1e-5-s42"  1.5b 1 "$BASE --seed 42 --lr 1e-5"
        submit "v3-tau0.8-s42"  1.5b 3 "$BASE --seed 42 --temperature 0.8"
        ;;
    d2)
        echo "== D2 noise 5e-2 -> 5e-3 (1.5B) =="
        for seed in $SEEDS; do
            submit "v1-noise-s${seed}" 1.5b 1 "$BASE --seed ${seed} --noise_sigma_start 0.05 --noise_sigma_end 0.005"
            submit "v3-noise-s${seed}" 1.5b 3 "$BASE --seed ${seed} --noise_sigma_start 0.05 --noise_sigma_end 0.005"
        done
        ;;
    d2sweep)
        echo "== D2 noise scale sweep (1.5B v1, seed 42) =="
        submit "v1-noise-lo-s42" 1.5b 1 "$BASE --seed 42 --noise_sigma_start 0.02 --noise_sigma_end 0.002"
        submit "v1-noise-hi-s42" 1.5b 1 "$BASE --seed 42 --noise_sigma_start 0.1  --noise_sigma_end 0.01"
        ;;
    math)
        # MATH is where the gaps between methods are large (cold Dr. GRPO: v1 45.07, v3 51.20) and
        # where exploration methods report most of their gains, so a noise effect is likelier to be
        # visible here than on GSM8K. Same arguments as the settled MATH Dr. GRPO runs (tier10 of
        # submit_followups.sh) plus noise. ~32 h per run, so one seed first.
        # Default seed 43, NOT 42: the v1 seed-42 MATH baseline is the run that drifted (37.40
        # against 49.20 / 48.60), so a seed-42 comparison would flatter any treatment.
        MATH_SEEDS="${MATH_SEEDS:-43}"
        WALLTIME="${MAX_WALLTIME:-48:00:00}"
        MATH_ARGS="--dataset math --repetition_penalty 1.0 --max_completion_length 1024"
        MATH_ARGS="$MATH_ARGS --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --dr_grpo both"
        echo "== D2 noise 5e-2 -> 5e-3 on MATH (1.5B), seeds: ${MATH_SEEDS} =="
        for seed in $MATH_SEEDS; do
            submit "math-v1-noise-s${seed}" 1.5b 1 "$MATH_ARGS --seed ${seed} --noise_sigma_start 0.05 --noise_sigma_end 0.005"
            submit "math-v3-noise-s${seed}" 1.5b 3 "$MATH_ARGS --seed ${seed} --noise_sigma_start 0.05 --noise_sigma_end 0.005"
        done
        ;;
    latent)
        # Decoupled latent temperature on MATH: tokens still sampled at 0.5, the Eq.3 mixture formed
        # at LATENT_TEMP (default 1.0). v3 only: v1 has no latent mixture (set_latent_temperature
        # refuses it). Controls are the settled cold Dr. GRPO MATH runs of the same seed (v3 49.80 /
        # 51.60 / 52.20, v1 37.40 / 49.20 / 48.60). Seeds 43 and 44 first (the two stable v1
        # seeds), seed 42 after. Submit only after probe_latent_temp.py says PROCEED.
        LATENT_SEEDS="${LATENT_SEEDS:-43 44}"
        LATENT_TEMP="${LATENT_TEMP:-1.0}"
        WALLTIME="${MAX_WALLTIME:-48:00:00}"
        MATH_ARGS="--dataset math --repetition_penalty 1.0 --max_completion_length 1024"
        MATH_ARGS="$MATH_ARGS --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --dr_grpo both"
        echo "== latent temperature ${LATENT_TEMP} on MATH (1.5B v3), seeds: ${LATENT_SEEDS} =="
        for seed in $LATENT_SEEDS; do
            submit "math-v3-lt${LATENT_TEMP}-s${seed}" 1.5b 3 "$MATH_ARGS --seed ${seed} --latent_temperature ${LATENT_TEMP}"
        done
        ;;
    d3)
        # D3, the uncertainty-gated blend (variant_common.set_blend_gate), on MATH v3. D3_D0 and
        # D3_D1 are REQUIRED and come from the calibration job (probe_latent_temp.py --calibrate,
        # results/d3_gate_calibration.json) for the latent temperature used here. D3_LT is the Eq.3
        # mixture temperature (default 1.0: the full uncertainty-targeted latent mixing, richer
        # latent AND blended only where uncertain; D3_LT=off for the gate alone). Job names start
        # hrpo-d3-. Controls: the settled cold Dr. GRPO MATH v3 runs of the same seed.
        : "${D3_D0:?set D3_D0 from results/d3_gate_calibration.json}"
        : "${D3_D1:?set D3_D1 from results/d3_gate_calibration.json}"
        D3_SEEDS="${D3_SEEDS:-43 44}"
        D3_MAX="${D3_MAX:-0.5}"
        D3_LT="${D3_LT:-1.0}"
        WALLTIME="${MAX_WALLTIME:-48:00:00}"
        MATH_ARGS="--dataset math --repetition_penalty 1.0 --max_completion_length 1024"
        MATH_ARGS="$MATH_ARGS --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --dr_grpo both"
        GATE_ARGS="--blend_gate divergence --blend_max ${D3_MAX} --blend_d0 ${D3_D0} --blend_d1 ${D3_D1}"
        LT_ARGS=""
        LT_TAG=""
        if [[ "$D3_LT" != "off" ]]; then
            LT_ARGS="--latent_temperature ${D3_LT}"
            LT_TAG="-lt${D3_LT}"
        fi
        export JOB_PREFIX="hrpo-d3"
        echo "== D3 gated blend (max ${D3_MAX}, d ${D3_D0} -> ${D3_D1}, latent temperature ${D3_LT}) on MATH v3, seeds: ${D3_SEEDS} =="
        for seed in $D3_SEEDS; do
            submit "math-v3${LT_TAG}-dg${D3_MAX}-s${seed}" 1.5b 3 "$MATH_ARGS --seed ${seed} ${GATE_ARGS} ${LT_ARGS}"
        done
        unset JOB_PREFIX
        ;;
    c1)
        # C1, stochastic latent with its own gradient (after LEPO): Gumbel latent at temperature
        # C1_TAU (default 0.5; LEPO used 0.3 at 3B and 0.5 at 7B) and the soft-label policy-gradient
        # term at latent positions. MATH v3, seeds 43 44. Job names start hrpo-c1-. Controls: the
        # settled cold Dr. GRPO MATH v3 runs of the same seed.
        C1_SEEDS="${C1_SEEDS:-43 44}"
        C1_TAU="${C1_TAU:-0.5}"
        WALLTIME="${MAX_WALLTIME:-48:00:00}"
        MATH_ARGS="--dataset math --repetition_penalty 1.0 --max_completion_length 1024"
        MATH_ARGS="$MATH_ARGS --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --dr_grpo both"
        export JOB_PREFIX="hrpo-c1"
        echo "== C1 stochastic latent (Gumbel temperature ${C1_TAU}) on MATH v3, seeds: ${C1_SEEDS} =="
        for seed in $C1_SEEDS; do
            submit "math-v3-sl${C1_TAU}-s${seed}" 1.5b 3 "$MATH_ARGS --seed ${seed} --stochastic_latent ${C1_TAU}"
        done
        unset JOB_PREFIX
        ;;
    length)
        # Length penalty: a correct response earns 1 - alpha * sigmoid(normalised length among the
        # group's correct responses). v1 and v3 under the same penalty, so the question is whether
        # the latent lets HRPO shorten at less accuracy cost than token-only GRPO. Seeds 43 44: the
        # seed-42 v1 MATH baseline drifted. Controls: the settled cold Dr. GRPO MATH runs.
        LP_SEEDS="${LP_SEEDS:-43 44}"
        LP_ALPHA="${LP_ALPHA:-0.2}"
        LP_VARIANTS="${LP_VARIANTS:-1 3}"
        WALLTIME="${MAX_WALLTIME:-48:00:00}"
        MATH_ARGS="--dataset math --repetition_penalty 1.0 --max_completion_length 1024"
        MATH_ARGS="$MATH_ARGS --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --dr_grpo both"
        export JOB_PREFIX="hrpo-lp"
        echo "== Length penalty (alpha ${LP_ALPHA}) on MATH, variants: ${LP_VARIANTS}, seeds: ${LP_SEEDS} =="
        for v in $LP_VARIANTS; do
            for seed in $LP_SEEDS; do
                submit "math-v${v}-lp${LP_ALPHA}-s${seed}" 1.5b "$v" "$MATH_ARGS --seed ${seed} --length_penalty_alpha ${LP_ALPHA}"
            done
        done
        unset JOB_PREFIX
        ;;
    scale3b)
        # Scale check: the settled cold Dr. GRPO configuration at Qwen2.5-3B-Instruct, v1 and v3,
        # GSM8K and MATH, one seed (S3B_SEEDS, default 42): 4 jobs. Everything except the model
        # size is the 1.5B baselines' configuration, so the directories are baseline names
        # (experiments/3b-v{1,3}_...-rp1-drboth[-dsmath]) and any later 3B arm pairs with them.
        # Each job scores its own run when training ends (AUTO_EVAL_BASELINE, run_variant.slurm);
        # v3 also with the latent off. Wall times: 3B GSM8K took 17.4 to 20.1 h before; 1.5B MATH
        # takes 30 to 33 h, so 3B MATH is expected near 3 days. The cluster has no 48 h cap (job
        # 17550 trained 58.3 h). Host memory 32G: a 4 GB host OOM once killed a MATH job silently.
        S3B_SEEDS="${S3B_SEEDS:-42}"
        S3B_TASKS="${S3B_TASKS:-gsm8k math}"
        export JOB_PREFIX="hrpo-3b" JOB_MEM="${JOB_MEM:-32G}"
        GSM8K_ARGS="--repetition_penalty 1.0 --dr_grpo both"
        MATH_ARGS="--dataset math --repetition_penalty 1.0 --max_completion_length 1024"
        MATH_ARGS="$MATH_ARGS --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --dr_grpo both"
        echo "== 3B scale check (Dr. GRPO), v1 and v3, tasks: ${S3B_TASKS}, seeds: ${S3B_SEEDS} =="
        for task in $S3B_TASKS; do
            if [[ "$task" == math ]]; then
                WALLTIME="${MATH_WALLTIME:-96:00:00}"; args="$MATH_ARGS"
            else
                WALLTIME="${GSM8K_WALLTIME:-48:00:00}"; args="$GSM8K_ARGS"
            fi
            for v in 1 3; do
                for seed in $S3B_SEEDS; do
                    export AUTO_EVAL_BASELINE
                    AUTO_EVAL_BASELINE="$(baseline_id "$v" "$seed" "$task")"
                    submit "${task}-v${v}-s${seed}" 3b "$v" "$args --seed ${seed}"
                    echo "      scores ${AUTO_EVAL_BASELINE} (3b) when training ends"
                done
            done
        done
        unset JOB_PREFIX JOB_MEM AUTO_EVAL_BASELINE
        ;;
    family)
        # Cross-model study (../CROSS_MODEL_PLAN.md). Exactly the settled cold Dr. GRPO configuration
        # of the Qwen runs (the same arguments as scale3b) on another backbone, one job per
        # (model, task, arm, seed). Baselines (v1 v3 v4) are plain baseline directories
        # (experiments/<key>-v{1,3,4}_...-rp1-drboth[-dsmath]) and score themselves when training
        # ends (AUTO_EVAL_BASELINE; v3 and v4 also with the latent off). Treatments are new-* runs
        # (c1 -> -sl0.5, lt -> -lt1), scored by run_variant.slurm's D2 branch, and are compared
        # with the v3 baseline of the same model, task and seed: submit the baselines with them.
        FAM_MODELS="${FAM_MODELS:-llama1b}"
        FAM_TASKS="${FAM_TASKS:-gsm8k}"
        FAM_ARMS="${FAM_ARMS:-v1 v3}"
        FAM_SEEDS="${FAM_SEEDS:-42 43 44}"
        export JOB_MEM="${JOB_MEM:-32G}"
        GSM8K_ARGS="--repetition_penalty 1.0 --dr_grpo both"
        MATH_ARGS="--dataset math --repetition_penalty 1.0 --max_completion_length 1024"
        MATH_ARGS="$MATH_ARGS --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --dr_grpo both"
        # mmlu: HRPO's merged MMLU auxiliary + ARC-C set (GSM8K-sized subset, mmlu_data.py). Prompts up to
        # 1024 tokens (HRPO's hrpo_mmlu.py; prepare_mmlu_arc.py keeps only prompts that fit for every
        # backbone); completions 512 as on GSM8K; batch settings are the parser defaults (8 x 4).
        MMLU_ARGS="--dataset mmlu --max_prompt_length 1024 --repetition_penalty 1.0 --dr_grpo both"
        if [[ " ${FAM_TASKS} " == *" mmlu "* && ! -d data/mmlu_arc_train_7473 ]]; then
            echo "family: data/mmlu_arc_train_7473 missing; run python prepare_mmlu_arc.py first" >&2; exit 1
        fi
        echo "== Cross-model study: models ${FAM_MODELS}, tasks ${FAM_TASKS}, arms ${FAM_ARMS}, seeds ${FAM_SEEDS} =="
        for model in $FAM_MODELS; do
            grep -q "^    \"${model}\":" model_registry.py \
                || { echo "family: unknown model '${model}' (keys are in model_registry.py)" >&2; exit 1; }
            export JOB_PREFIX="hrpo-fam-${model}"
            for task in $FAM_TASKS; do
                case "$task" in
                    gsm8k) args="$GSM8K_ARGS" ;;
                    math)  args="$MATH_ARGS" ;;
                    mmlu)  args="$MMLU_ARGS" ;;
                    *) echo "family: unknown task '$task' (gsm8k math mmlu)" >&2; exit 1 ;;
                esac
                # Generous limits: Qwen 1.5B takes ~8 h (GSM8K) and 30-33 h (MATH), Qwen 3B ~19 h
                # and ~30 h. A run that hits its limit resumes from its last checkpoint when the
                # same command is submitted again.
                case "${model}:${task}" in
                    *3b:math)  WALLTIME="${MAX_WALLTIME:-96:00:00}" ;;
                    *:math)    WALLTIME="${MAX_WALLTIME:-60:00:00}" ;;
                    *3b:gsm8k) WALLTIME="${MAX_WALLTIME:-48:00:00}" ;;
                    *3b:mmlu)  WALLTIME="${MAX_WALLTIME:-60:00:00}" ;;
                    *:mmlu)    WALLTIME="${MAX_WALLTIME:-36:00:00}" ;;
                    *)         WALLTIME="${MAX_WALLTIME:-24:00:00}" ;;
                esac
                for arm in $FAM_ARMS; do
                    for seed in $FAM_SEEDS; do
                        case "$arm" in
                            v1|v3|v4)
                                export AUTO_EVAL_BASELINE
                                AUTO_EVAL_BASELINE="$(baseline_id "${arm#v}" "$seed" "$task")" || exit 1
                                submit "${task}-${arm}-s${seed}" "$model" "${arm#v}" "$args --seed ${seed}"
                                echo "      scores ${AUTO_EVAL_BASELINE} (${model}) when training ends"
                                unset AUTO_EVAL_BASELINE
                                ;;
                            c1) submit "${task}-v3-sl0.5-s${seed}" "$model" 3 "$args --seed ${seed} --stochastic_latent 0.5" ;;
                            lt) submit "${task}-v3-lt1.0-s${seed}" "$model" 3 "$args --seed ${seed} --latent_temperature 1.0" ;;
                            *)  echo "family: unknown arm '$arm' (v1 v3 v4 c1 lt)" >&2; exit 1 ;;
                        esac
                    done
                done
            done
        done
        unset JOB_PREFIX JOB_MEM
        ;;
    *)
        echo "unknown tier '$WHICH' (probe | d2 | d2sweep | math | latent | d3 | c1 | length | scale3b | family)" >&2
        exit 1
        ;;
esac

echo
echo "Monitor: squeue -u \$USER     Then:  sbatch cluster/run_eval_d2.slurm  &&  python analyse_d2.py"
