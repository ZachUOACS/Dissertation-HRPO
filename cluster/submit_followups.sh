#!/bin/bash
# Submit the follow-up runs that harden the main ablation's conclusions.
#
#   tier1  Seed replicates of the central comparison (v3 vs v4) at 1.5B.
#          The headline result — v3 > v1 (+4.55, p=0.0004) and v3 > v4
#          (+4.40, p=0.0006) — currently rests on a single seed. McNemar shows
#          these two MODELS differ; it cannot show the METHODS differ. Seeds 43
#          and 44 give 3 seeds per arm, enough to report "replicates in k/3".
#          -> 4 jobs
#
#   tier3  Faithful reproduction of published HRPO at 1.5B: repetition_penalty
#          1.0 (the paper specifies none; ours perturbs the Eq.3 projection)
#          AND attention-only LoRA (paper Table 4: "query, key, value, dense",
#          vs the attention+MLP set used throughout this study). Tests whether
#          the ~8.6-point absolute offset from the paper is explained by these
#          two deviations. Sits OUTSIDE the ablation chain — it changes two
#          things at once — so it is a reproducibility check, not a chain link.
#          -> 1 job (+1 optional single-factor run, see tier3b)
#
#   tier3b Single-factor control: repetition_penalty 1.0 only, LoRA unchanged.
#          Isolates the penalty from the LoRA change, so tier3's result can be
#          attributed. -> 1 job
#
# Usage (from /data/ztay632/Dissertation-HRPO):
#   bash cluster/submit_followups.sh            # tier1 + tier3 + tier3b (6 jobs)
#   bash cluster/submit_followups.sh tier1      # seeds only (4 jobs)
#   bash cluster/submit_followups.sh tier3      # reproduction only (2 jobs)
#
# Each run writes to its own experiments/ directory (suffixed -seedN / -rpX /
# -loraattention by variant_common.experiment_dir), so nothing already
# completed is touched or overwritten.

set -u
cd "$(dirname "$0")/.."
mkdir -p slurm_logs

WHICH="${1:-all}"
WALLTIME="${MAX_WALLTIME:-72:00:00}"
# Seeds for the rp=1.0 tiers (6, 7, 8). Override to throttle, e.g.
#   SEEDS=42 bash cluster/submit_followups.sh tier7   # single-seed 3B pass
SEEDS="${SEEDS:-42 43 44}"

submit() {
    local tag="$1" size="$2" vnum="$3" extra="$4"
    local dep="${5:-}"                  # optional: job id this run must wait for
    local jobname="hrpo-${tag}"
    export EXTRA_ARGS="$extra"          # picked up via --export=ALL
    local job_id
    job_id=$(sbatch \
        --job-name="$jobname" \
        --time="$WALLTIME" \
        ${dep:+--dependency=afterok:$dep} \
        --output="slurm_logs/${jobname}-%j.log" \
        --error="slurm_logs/${jobname}-%j.err" \
        --export="ALL,MODEL_SIZE=${size},VARIANT_NUM=${vnum}" \
        cluster/run_variant.slurm | awk '{print $4}')
    printf "  %-34s -> job %-10s [%s]\n" "$jobname" "$job_id" "$extra"
    unset EXTRA_ARGS
}

echo "Submitting HRPO follow-up runs..."
echo ""

if [[ "$WHICH" == "all" || "$WHICH" == "tier1" ]]; then
    echo "TIER 1 — seed replicates of the central comparison (1.5B):"
    for SEED in 43 44; do
        submit "1.5b-v3-seed${SEED}" 1.5b 3 "--seed ${SEED}"
        submit "1.5b-v4-seed${SEED}" 1.5b 4 "--seed ${SEED}"
    done
    echo ""
fi

if [[ "$WHICH" == "all" || "$WHICH" == "tier3" ]]; then
    echo "TIER 3 — faithful published-HRPO reproduction (1.5B):"
    submit "1.5b-v4-paper-faithful" 1.5b 4 "--repetition_penalty 1.0 --lora_targets attention"
    echo ""
    echo "TIER 3b — single-factor control (penalty only):"
    submit "1.5b-v4-rp1.0" 1.5b 4 "--repetition_penalty 1.0"
    echo ""
fi

if [[ "$WHICH" == "all" || "$WHICH" == "tier4" ]]; then
    # The decisive missing cell. The Tier 1 seeds + Tier 3 controls showed the
    # "v3 > v4" result was driven by repetition_penalty=1.2 (which v4 alone was
    # run at both settings): v4 goes 62.62 -> 67.40 when the penalty is removed.
    # But v3 exists ONLY at rp=1.2. To compare the gate on a level field, run v3
    # at rp=1.0 across all three seeds, and complete v4's rp=1.0 seed set so both
    # arms have 3 seeds at the fair setting. Then v3-rp1.0 vs v4-rp1.0 (3 seeds
    # each) is the clean test of whether the input gate helps, hurts or is neutral.
    echo "TIER 4 — level-field gate test: v3 & v4 at rp=1.0, 3 seeds each (1.5B):"
    for SEED in 42 43 44; do
        submit "1.5b-v3-rp1.0-seed${SEED}" 1.5b 3 "--repetition_penalty 1.0 --seed ${SEED}"
    done
    # v4 rp=1.0 seed 42 already exists (1.5b-v4_full_hrpo_gate-rp1); add 43, 44.
    for SEED in 43 44; do
        submit "1.5b-v4-rp1.0-seed${SEED}" 1.5b 4 "--repetition_penalty 1.0 --seed ${SEED}"
    done
    echo ""
fi

if [[ "$WHICH" == "all" || "$WHICH" == "tier5" ]]; then
    # Tier 4 showed the penalty lifts the LATENT variants (v3, v4) by 6-8 pts to ~70%.
    # But the baselines (v1 GRPO, v2 fixed blend) were only run at rp=1.2. v1 has no
    # Eq.3 projection, so the penalty may barely affect it — meaning the fair
    # "does HRPO beat GRPO?" comparison needs v1 (and v2) at rp=1.0 too. These are
    # the last cells before the numbers are final.
    echo "TIER 5 — fair baselines at rp=1.0, 3 seeds each (1.5B):"
    for SEED in 42 43 44; do
        submit "1.5b-v1-rp1.0-seed${SEED}" 1.5b 1 "--repetition_penalty 1.0 --seed ${SEED}"
        submit "1.5b-v2-rp1.0-seed${SEED}" 1.5b 2 "--repetition_penalty 1.0 --seed ${SEED}"
    done
    echo ""
fi

# Tiers 6 and 7 move the *entire* study to the corrected rp=1.0 setting so the
# results can be reported without any confounded (rp=1.2) cell. They are gated
# on their explicit names only (NOT "all") because tier 7 is the expensive 3B
# model — run them deliberately.
if [[ "$WHICH" == "tier6" ]]; then
    # Reward-shaping variants (v5 +format reward, v6 +repetition handling) exist
    # ONLY at the confounded rp=1.2. Run both at rp=1.0 across the same seeds so
    # they sit on the same footing as the now-settled v1..v4 gate result. This
    # completes the 1.5B ablation chain v1..v6 at the faithful setting.
    echo "TIER 6 — reward-shaping variants at rp=1.0 (1.5B), seeds: ${SEEDS}:"
    for SEED in $SEEDS; do
        submit "1.5b-v5-rp1.0-seed${SEED}" 1.5b 5 "--repetition_penalty 1.0 --seed ${SEED}"
        submit "1.5b-v6-rp1.0-seed${SEED}" 1.5b 6 "--repetition_penalty 1.0 --seed ${SEED}"
    done
    echo ""
fi

if [[ "$WHICH" == "tier7" ]]; then
    # 3B chain at rp=1.0. The 3B runs so far are single-seed rp=1.2. Run the core
    # chain (v1,v2,v3,v4) plus the reward-shaping endpoint (v6) at rp=1.0 so the
    # scale comparison is drawn at the corrected setting. 3B is the expensive
    # model: default SEEDS="42 43 44" => 15 jobs. Throttle with e.g.
    #   SEEDS=42 bash cluster/submit_followups.sh tier7   # 5-job single-seed pass
    echo "TIER 7 — 3B chain at rp=1.0, seeds: ${SEEDS}:"
    for SEED in $SEEDS; do
        submit "3b-v1-rp1.0-seed${SEED}" 3b 1 "--repetition_penalty 1.0 --seed ${SEED}"
        submit "3b-v2-rp1.0-seed${SEED}" 3b 2 "--repetition_penalty 1.0 --seed ${SEED}"
        submit "3b-v3-rp1.0-seed${SEED}" 3b 3 "--repetition_penalty 1.0 --seed ${SEED}"
        submit "3b-v4-rp1.0-seed${SEED}" 3b 4 "--repetition_penalty 1.0 --seed ${SEED}"
        submit "3b-v6-rp1.0-seed${SEED}" 3b 6 "--repetition_penalty 1.0 --seed ${SEED}"
    done
    echo ""
fi

if [[ "$WHICH" == "tier8" ]]; then
    # Stage-0 brief, Phase 5. Tests whether the settled null (all six arms tie at
    # rp=1.0) is partly a STARTING-POINT artefact: RL begins from a cold LoRA, so
    # Lambda spends the run climbing before the latent channel carries anything.
    #
    #   T1  v3 warm-started from the --blend on  (blended SFT) Stage 0
    #   C3  v1 warm-started from the --blend off (plain LoRA SFT) Stage 0
    #       -> C3 is the control that separates "the warm start helped" from
    #          "any extra epoch of supervised data helped".
    #   C1  v7_constant_vector: v3 with the Eq.3 projection replaced by ONE
    #       learned vector u. Seed 42 only for now — it can only CONFIRM the
    #       null (T1 is the arm that could change the headline), so it runs as a
    #       single first data point rather than a 3-seed set.
    #
    # C2 (Dr. GRPO flags) is NOT here — not implemented yet.
    #
    # Warm-started runs land in their own directories via experiment_dir's
    # -warm{on,off} suffix (e.g. 1.5b-v3_learned_lambda-rp1-warmon), so they can
    # never overwrite or be pooled with the settled cold arms.
    #
    #   bash cluster/submit_followups.sh tier8
    #   SEEDS=42 bash cluster/submit_followups.sh tier8     # single-seed pass
    STAGE0_SIZE="${STAGE0_SIZE:-1.5b}"
    STAGE0_RMIN="${STAGE0_RMIN:-0.98}"

    # A missing warm start must stop the arm, not silently cold-start it: an RL
    # run whose --init_from does not exist would raise, but one pointed at a
    # half-written directory is worse. Require the adapter the loader reads.
    warm_ok() {
        local blend="$1" seed="$2"
        [[ -f "experiments/${STAGE0_SIZE}-stage0-${blend}-rmin${STAGE0_RMIN}-seed${seed}/adapter_model.safetensors" ]]
    }

    echo "TIER 8 — Stage-0 warm start + latent-source control (1.5B), seeds: ${SEEDS}:"
    for SEED in $SEEDS; do
        for pair in "on:3:T1-v3" "off:1:C3-v1"; do
            BLEND="${pair%%:*}"; rest="${pair#*:}"
            VNUM="${rest%%:*}"; LABEL="${rest#*:}"
            WARM="experiments/${STAGE0_SIZE}-stage0-${BLEND}-rmin${STAGE0_RMIN}-seed${SEED}"
            if warm_ok "$BLEND" "$SEED"; then
                submit "1.5b-${LABEL}-warm${BLEND}-seed${SEED}" "$STAGE0_SIZE" "$VNUM" \
                    "--repetition_penalty 1.0 --seed ${SEED} --init_from ${WARM}"
            else
                echo "  SKIP ${LABEL} seed ${SEED}: no warm start at ${WARM}"
                echo "       produce it first:  BLEND=${BLEND} SEED=${SEED} sbatch cluster/run_stage0.slurm"
            fi
        done
    done
    echo ""

    # C1 needs no Stage 0 — it is a training-time control, not a warm start.
    C1_SEEDS="${C1_SEEDS:-42}"
    echo "TIER 8 — C1 constant-vector control (1.5B), seeds: ${C1_SEEDS}:"
    for SEED in $C1_SEEDS; do
        submit "1.5b-C1-constant-seed${SEED}" "$STAGE0_SIZE" 7 \
            "--repetition_penalty 1.0 --latent_source constant --seed ${SEED}"
    done
    echo ""
fi

if [[ "$WHICH" == "tier9" ]]; then
    # Control C2 — Dr. GRPO de-bias (Liu et al. 2025): constant length
    # normaliser + no /std in the advantage (--dr_grpo both). Run on BOTH arms:
    #   v1+C2 vs v1      does removing GRPO's biases change GRPO?
    #   v3+C2 vs A1      ... and HRPO?
    #   v3+C2 vs v1+C2   the headline question (latent vs GRPO) under the
    #                    de-biased objective — does GRPO's length bias explain
    #                    why HRPO ties GRPO?
    # rp=1.0, cold start, full LoRA: identical to the settled arms except the
    # objective. Directories get -drboth, so nothing settled is touched.
    #
    # There is no --max_steps, so verify on the first seed before the rest:
    #   SEEDS=42 bash cluster/submit_followups.sh tier9
    #   (check the log for "[dr_grpo] mode=both", then)
    #   SEEDS="43 44" bash cluster/submit_followups.sh tier9
    echo "TIER 9 — C2 Dr. GRPO de-bias on v1 and v3 (1.5B), seeds: ${SEEDS}:"
    for SEED in $SEEDS; do
        submit "1.5b-C2-v1-drgrpo-seed${SEED}" 1.5b 1 \
            "--repetition_penalty 1.0 --dr_grpo both --seed ${SEED}"
        submit "1.5b-C2-v3-drgrpo-seed${SEED}" 1.5b 3 \
            "--repetition_penalty 1.0 --dr_grpo both --seed ${SEED}"
    done
    echo ""
fi

if [[ "$WHICH" == "tier10" ]]; then
    # Second task. The headline question — does the latent channel beat plain
    # GRPO? — asked on MATH. v1 vs v3 only: the two arms that differ solely in
    # the latent pathway, at rp=1.0, three seeds. The other four variants add
    # nothing on GSM8K and are not worth MATH's compute.
    #
    # Lengths follow the HRPO paper's MATH row (1024-token completions against
    # GSM8K's 512) and are passed EXPLICITLY so run_config.json records what
    # each run trained with. per_device 4 x grad_accum 8 keeps the total train
    # batch at 32 while offsetting the memory cost of the longer completions.
    #
    # MATH runs land in -dsmath directories; no GSM8K result is touched.
    # Training data comes from a local MATH copy if one exists, else the HF
    # mirror (utils.load_math_raw); evaluation is always MATH-500.
    #   SEEDS=42 bash cluster/submit_followups.sh tier10    # verify one seed first
    # 2026-09-20, after the first MATH pass: BOTH changes below are load-bearing.
    #
    # (a) --dr_grpo both. Under the standard objective MATH COLLAPSES: completion
    #     length drifts up all run (>700 tokens by step 175), hits the 1024 cap
    #     around step 750, and a truncated completion has no "####" — which the
    #     reward requires — so it scores 0 however good the reasoning. Reward
    #     fell 0.56 -> 0.14, KL rose 0.09 -> 2.2, grad_norm 0.5 -> 11 (clip 0.1).
    #     That is GRPO's length bias, which is exactly what Dr. GRPO removes. It
    #     never fired on GSM8K: completions sat at ~250 tokens against a 512 cap.
    #     MATH_DR=off reproduces the original (collapsing) standard-objective run.
    #
    # (b) per_device 8 x accum 4, not 4 x 8. GRPO generates once per MICRO-batch,
    #     so 4 x 8 doubled the generate() calls per optimizer step and halved the
    #     sequences per call — generation dominates, and the first pass took 58h
    #     per run against GSM8K's ~6h. Same 32 completions per step either way.
    MATH_DR="${MATH_DR:-both}"
    MATH_ARGS="--dataset math --repetition_penalty 1.0 --max_completion_length 1024"
    MATH_ARGS="$MATH_ARGS --per_device_train_batch_size 8 --gradient_accumulation_steps 4"
    [[ "$MATH_DR" != "off" ]] && MATH_ARGS="$MATH_ARGS --dr_grpo $MATH_DR"
    echo "TIER 10 — MATH: v1 vs v3 at rp=1.0 (1.5B), seeds: ${SEEDS}:"
    for SEED in $SEEDS; do
        submit "1.5b-math-v1-seed${SEED}" 1.5b 1 "$MATH_ARGS --seed ${SEED}"
        submit "1.5b-math-v3-seed${SEED}" 1.5b 3 "$MATH_ARGS --seed ${SEED}"
    done
    echo ""
fi

if [[ "$WHICH" == "tier11" ]]; then
    # The missing cell of the 2x2x2 (latent x warm start x objective).
    #
    # The warm start was measured ONLY under the standard objective, whose
    # length bias acts on exactly what the warm start changes most: completion
    # length (91 tokens at RL start vs 258 cold). Under per-response
    # normalisation every token of a 91-token completion carries ~2.8x the
    # weight of a token in a 258-token one, so T1's -2.05 was measured under an
    # objective biased on that axis. Dr. GRPO's constant normaliser removes it.
    #
    #   T1+C2 vs C2-v3   does the warm start help once the objective is fixed?
    #   T1+C2 vs T1      does de-biasing rescue the warm start?
    #   T1+C2 vs C3+C2   latent vs GRPO, both warm AND de-biased
    #
    # C3+C2 is not optional: without it a gain cannot be separated from "the
    # extra supervised epoch helped" — the same reason C3 exists in tier8.
    # Requires the Stage-0 warm starts (tier8 made them for seeds 42/43/44).
    STAGE0_SIZE="${STAGE0_SIZE:-1.5b}"
    STAGE0_RMIN="${STAGE0_RMIN:-0.98}"
    echo "TIER 11 — warm start x Dr. GRPO (1.5B), seeds: ${SEEDS}:"
    for SEED in $SEEDS; do
        for pair in "on:3:T1C2-v3" "off:1:C3C2-v1"; do
            BLEND="${pair%%:*}"; rest="${pair#*:}"
            VNUM="${rest%%:*}"; LABEL="${rest#*:}"
            WARM="experiments/${STAGE0_SIZE}-stage0-${BLEND}-rmin${STAGE0_RMIN}-seed${SEED}"
            if [[ -f "${WARM}/adapter_model.safetensors" ]]; then
                submit "1.5b-${LABEL}-warm${BLEND}-drboth-seed${SEED}" "$STAGE0_SIZE" "$VNUM" \
                    "--repetition_penalty 1.0 --dr_grpo both --seed ${SEED} --init_from ${WARM}"
            else
                echo "  SKIP ${LABEL} seed ${SEED}: no warm start at ${WARM}"
                echo "       produce it first:  BLEND=${BLEND} SEED=${SEED} sbatch cluster/run_stage0.slurm"
            fi
        done
    done
    echo ""
fi

if [[ "$WHICH" == "tier12" ]]; then
    # Warm start on MATH. The GSM8K warm starts cannot be reused: they are a
    # supervised epoch on GSM8K's solutions. This builds a MATH Stage 0 (one
    # teacher-forced epoch on MATH's reference solutions, answer kept boxed so
    # the RL reward scores it), then runs RL from it:
    #
    #   v3 from the --blend on  MATH Stage 0   vs cold de-biased v3 (tier10)
    #   v1 from the --blend off MATH Stage 0   vs cold de-biased v1 (tier10)
    #   and warm v3 vs warm v1 — latent vs GRPO, both warm
    #
    # Dr. GRPO throughout (the standard objective collapses on MATH) and the
    # same batching as tier10, so the only difference from tier10 is the start.
    #
    # Stage 0 is submitted here and each RL job waits on its own Stage-0 job
    # (afterok), so one command runs the lot. A Stage 0 that already exists is
    # reused, not re-run. If a Stage 0 fails, its RL job stays pending with
    # reason DependencyNeverSatisfied — scancel it and fix the Stage 0 first.
    #   SEEDS=42 bash cluster/submit_followups.sh tier12   # verify one seed first
    STAGE0_SIZE="${STAGE0_SIZE:-1.5b}"
    STAGE0_RMIN="${STAGE0_RMIN:-0.98}"
    MATH_RL="--dataset math --repetition_penalty 1.0 --max_completion_length 1024"
    MATH_RL="$MATH_RL --per_device_train_batch_size 8 --gradient_accumulation_steps 4"
    MATH_RL="$MATH_RL --dr_grpo both"
    echo "TIER 12 — MATH warm start (Stage 0 + RL, Dr. GRPO), seeds: ${SEEDS}:"
    for SEED in $SEEDS; do
        for pair in "on:3:v3" "off:1:v1"; do
            BLEND="${pair%%:*}"; rest="${pair#*:}"
            VNUM="${rest%%:*}"; LABEL="${rest#*:}"
            WARM="experiments/${STAGE0_SIZE}-stage0-${BLEND}-rmin${STAGE0_RMIN}-seed${SEED}-dsmath"
            DEP=""
            if [[ -f "${WARM}/adapter_model.safetensors" ]]; then
                echo "  stage 0 ${BLEND} seed ${SEED}: already built, reusing ${WARM}"
            else
                DEP=$(sbatch --parsable --job-name="hrpo-stage0-math-${BLEND}-seed${SEED}" \
                        --export="ALL,BLEND=${BLEND},SEED=${SEED},DATASET=math" \
                        cluster/run_stage0.slurm | cut -d';' -f1)
                printf "  %-34s -> job %-10s [stage 0 -> %s]\n" \
                    "hrpo-stage0-math-${BLEND}-seed${SEED}" "$DEP" "$WARM"
            fi
            submit "1.5b-math-${LABEL}-warm${BLEND}-seed${SEED}" "$STAGE0_SIZE" "$VNUM" \
                "$MATH_RL --seed ${SEED} --init_from ${WARM}" "$DEP"
        done
    done
    echo ""
fi

echo "Monitor with: squeue -u \$USER"
echo "When finished:  sbatch cluster/run_eval_and_analysis.slurm"
echo "                (eval_all.py discovers the new directories automatically;"
echo "                 completed evaluations are skipped)"
