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
# Seeds for the rp=1.0 tiers (6, 7). Override to throttle, e.g.
#   SEEDS=42 bash cluster/submit_followups.sh tier7   # single-seed 3B pass
SEEDS="${SEEDS:-42 43 44}"

submit() {
    local tag="$1" size="$2" vnum="$3" extra="$4"
    local jobname="hrpo-${tag}"
    export EXTRA_ARGS="$extra"          # picked up via --export=ALL
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

echo "Monitor with: squeue -u \$USER"
echo "When finished:  sbatch cluster/run_eval_and_analysis.slurm"
echo "                (eval_all.py discovers the new directories automatically;"
echo "                 completed evaluations are skipped)"
