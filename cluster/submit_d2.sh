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
    job_id=$(sbatch \
        --job-name="$jobname" \
        --time="$WALLTIME" \
        --output="slurm_logs/${jobname}-%j.log" \
        --error="slurm_logs/${jobname}-%j.err" \
        --export="ALL,MODEL_SIZE=${size},VARIANT_NUM=${vnum}" \
        cluster/run_variant.slurm | awk '{print $4}')
    printf "  %-34s -> job %-10s [%s]\n" "$jobname" "$job_id" "$extra"
    unset EXTRA_ARGS
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
    *)
        echo "unknown tier '$WHICH' (probe | d2 | d2sweep | math | latent | d3 | c1 | length)" >&2
        exit 1
        ;;
esac

echo
echo "Monitor: squeue -u \$USER     Then:  sbatch cluster/run_eval_d2.slurm  &&  python analyse_d2.py"
