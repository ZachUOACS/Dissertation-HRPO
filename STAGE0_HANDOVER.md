# Stage-0 warm start / tier-8 controls — handoff

Context for an agent continuing the **Stage-0 brief** (`CLAUDE_CODE_BRIEF_stage0.md`).
Verified against the repo on 2026-09-10. Branch: `Ablations`.

**This is the SECOND workstream in this repo.** The first — the six-variant HRPO ablation —
is finished and handed off separately in **`HANDOVER.md`**. Read that first for the base
context (variants, gate location, cluster layout, the rp=1.0 story). This file covers only
the Stage-0 / tier-8 work built on top of it.

---

## 1. What this workstream is

The ablation concluded that at rp=1.0 all six 1.5B arms tie (~69–71%) — HRPO ≈ GRPO, gate
neutral. The Stage-0 brief asks whether that null is partly a *starting-point* problem: RL
begins from a cold LoRA, so Λ spends the run climbing before the latent channel carries
anything. Stage 0 gives the blend a supervised warm start first, and adds controls that
isolate what the latent channel is actually contributing.

Four tier-8 arms:

| Arm | What it is | Depends on Stage 0? |
|---|---|---|
| **C1** | `v7_constant_vector` — v3 with the Eq.3 projection replaced by one learned vector `u` | no |
| **C2** | Dr. GRPO flags (constant-length norm, no ÷std) — **not implemented yet** | no |
| **C3** | v1 warm-started from `--blend off` (plain SFT) | yes (`off`) |
| **T1** | v3 warm-started from `--blend on` (blended SFT) | yes (`on`) |

---

## 2. Status

| Phase | State |
|---|---|
| 1 — discovery | ✅ done (findings in §3) |
| 2 — `--latent_off` eval | ✅ done, produced a real result (§4) |
| 3 — C1 constant vector | ✅ done, 8/8 tests pass |
| 4 — Stage-0 warm start | ✅ **complete** — checks 1, 2a, 2b, 3, 4 all pass; warm starts exist for seeds 42/43/44 |
| 5 — `tier8` | ✅ **run and analysed** — T1, C3 × 3 seeds, C1 seed 42 (results §6) |
| C2 — `tier9` | ✅ **run and analysed** — Dr. GRPO lifts BOTH arms to 72.00; the tie survives (§6) |
| MATH — `tier10` | ✅ **run and analysed** — standard GRPO COLLAPSES on MATH; de-biased it trains, and v3 > v1 (§6) |
| Warm + Dr. GRPO, GSM8K — `tier11` | 🟡 **run 2026-09-28**, v3 seed 43 died at step 535/934 (killed by the node, no traceback); resumable from checkpoint-500 |
| MATH warm start — `tier12` | 🟡 **seed 42 run 2026-09-28**; seeds 43/44 not submitted |

**Tier 11 results (GSM8K, warm start + Dr. GRPO; `results/replicate_analysis.json`):**
warm v3 71.19 (n=2: 70.58, 71.80) · warm v1 68.16 (66.79, 67.85, 69.83). Even with the objective
fixed, the warm start costs v1 **−3.84** (2/3 seeds p<0.001) and v3 **−0.81** (n=2, one seed
p=0.029). So warm v3 beats warm v1 by **+3.03** (seed 42 p=0.0013, seed 44 p=0.073) — but that gap
comes from the warm start hurting GRPO more, not from HRPO improving: warm v3 still sits below cold
de-biased v3 (72.00). Dr. GRPO on top of the warm start helps v3 +2.75.

**Tier 12 results (MATH, warm start + Dr. GRPO, seed 42 only):** warm v3 **38.00** vs cold v3 49.80
(**−11.80**, p≈0); warm v1 41.20 vs cold v1 37.40 (+3.80, p=0.099 — but cold v1 seed 42 is the one
drifted run; against the cold v1 mean of 45.07 it is −3.87). Warm v3 vs warm v1: −3.20 (p=0.094).
Mechanism: the MATH reference solutions are terse (median 171 tokens), so warm runs start RL at
~230-token completions against ~490 cold and only reach ~290–300 by the end (cold v3: 492). They are
stable — no drift, grad norm ≤ 0.13 — but reasoning stays short, and on MATH that costs accuracy.
The same shortening happened on GSM8K (91 → 130 tokens vs ~250 cold), where it cost ~2 points.

**None of tier 9 or tier 10 is warm-started** — only T1/C3 (tier 8) are. The MATH results
(tier 10) are cold starts; the existing GSM8K warm starts cannot be reused for MATH, so tier 12
builds a MATH Stage 0. `load_stage0_init` refuses a warm start from the wrong task.

**MATH Stage 0 (tier 12)**: `DATASET=math sbatch cluster/run_stage0.slurm`, output
`…-stage0-{on,off}-rmin0.98-seedN-dsmath`. Keeps 7,354/7,500 examples — drops 2 empty-box
solutions, 79 prompts over 512 tokens and 65 targets over 1,024 (dropped, never truncated: a
target cut before `####` would teach the model never to answer). Median target 171 tokens; 229
optimizer steps. Tier 12 submits each Stage 0 and chains its RL job on it with `afterok`; the RL
config is identical to tier 10 except `--init_from`.

**Scope decision (2026-09-10):** tier8 runs **T1 + C3 at seeds 42/43/44, and C1 at seed 42
only**. Rationale: C1 can only *confirm* the null, whereas T1 is the one arm that could change
the headline, so C1 runs as a single first data point rather than a 3-seed set. Override with
`C1_SEEDS="42 43 44"`.

---

## 3. Phase 1 discovery (settled answers — do not re-derive)

1. **`r_min`**: flag `--residual_r_min`, default **0.98**; init at
   `modeling_qwen2.py:476-483` (`uniform_(Λ, r_min, r_max)` then
   `Λ = -log(Λ^(-1/c) - 1)`, c=8). The existing `-rp1` arms used **0.98** (read from their
   `run_config.json`). The brief proposed 0.95; the human chose **match existing (0.98)**.
2. **Think span**: `generation/utils.py:3395`
   `is_thinking = [self.answer_start not in s for s in strs]` with `ANSWER_START = "####"`.
   String-level, recomputed per decode step per sequence — everything before `####` appears
   in the decoded text. NOT tag/token-id delimited.
3. **Eq.3 projection**: was inlined in the `_sample` loop; now extracted to
   `transformers.generation.utils.hrpo_project_latent` (§5). Generation is bit-identical
   (tests pin it).
4. **Data**: `load_dataset('openai/gsm8k','main')`; prompt = `SYSTEM_PROMPT` + question
   (`utils.process_gsm8k`). GSM8K `answer` = reasoning + `#### ans`, so the reference
   reasoning IS available as a teacher-forced target.
5. **Λ logging**: `GateGradientLogger` logs gradient NORMS only — it does **not** log Λ mean,
   CV or hidden ratio. The brief assumed it did. Stage 0 has its own `lambda_diagnostics()`.
6. **Eval hook**: `gate_mode="disabled"` → `disable_gate()` sets `model.answer_start = ""`, so
   `is_thinking` is False for every token and the blend is skipped in both rollout and loss
   forward, weights untouched. That is exactly "latent off".

---

## 4. Results so far

**Phase 2 — latent-off eval** (`results/latent_off_summary.json`). Switching the latent
channel off *at inference* costs nothing:

| arm | normal | latent-off | Δ |
|---|---|---|---|
| v1 (sanity) | 69.72 | identical vectors | +0.00 |
| v3 | 70.48 | 71.06 | **+0.58** (all 3 seeds up) |
| v4 (full-LoRA seeds) | 70.05 | 70.18 | +0.13 |

This is a **causal/interventional** result and is stronger than the ablation's correlational
mechanism evidence: the trained models carry no functional dependence on the latent pathway.
Nuance worth preserving: per-question correctness vectors are NOT identical for v3/v4 — answers
flip in both directions — so the latent is **informationally** inert, not numerically inert.

**Phase 4 — Stage-0 warm starts** (r_min 0.98, 1 epoch = 233 steps). Seed 42's `--blend on` run
logged 1864 Λ gradient arrivals (= 233 × 8 accum), max |grad| 1.689e-02, 1864 blend executions,
641,075 masked rows, train_loss 3.785. All three seeds move Λ down / hidden ratio up
consistently:

| seed | Λ init → final | hidden ratio init → final | CV final |
|---|---|---|---|
| 42 | 6.833611 → **6.764175** | 0.138199 → **0.142826** (+3.3%) | 0.29712 |
| 43 | 6.810340 → **6.740487** | 0.139564 → **0.144275** (+3.4%) | 0.29496 |
| 44 | 6.824298 → **6.754582** | 0.138707 → **0.143354** (+3.4%) | 0.29713 |

For scale: v3 RL converges at hidden ratio 0.1671, so Stage 0 covers ~10–13% of that gap.
`--blend off` runs: `final_lambda None` for all three (correct — no Λ at all).

**Check 3 — the warm start loads into its RL script** (`--init_only`, seed 42):
- 3a (v3 ← `--blend on`): 393 tensors, max |ΔΛ| 0.134846, `lambda_mean = 6.7641754150390625`
  — bit-exact with `stage0_summary.json`. **Pass.**
- 3b (v1 ← `--blend off`): 392 tensors, Λ reported `nan` — see §7.11; expected and inert.

Both report "344 missing key(s)". That is the **frozen base model**, which an adapter never
contains, not missing adapter content: 392 = 28 layers × 7 `LORA_TARGETS['full']` modules × 2
(A+B), i.e. the complete LoRA set, `load_stage0_init` raises on unexpected keys and did not, and
the count is *identical* in 3a and 3b even though the two files differ by exactly the Λ key.

**Check 4 — zero-shot GSM8K of the warm starts** vs base 61.11:
`--blend on` **59.06** (−2.05), `--blend off` **58.98** (−2.13). Both slightly *below* base and
0.08 apart. This is expected, not a failure: one SFT epoch optimises next-token likelihood, not
answer correctness, and at hidden ratio ~0.14 the blend contributes little. **The warm start's
value is whether it improves the RL starting point — only T1 vs A1 (Phase 5) answers that.**
Do not report check 4 as evidence for or against the warm start.

---

## 5. What was built

**New files**
| File | Purpose |
|---|---|
| `stage0_warmstart.py` | the warm start: teacher-forced epoch, two-pass blend |
| `hrpo_constant_vector.py` | arm C1 (`v7_constant_vector`) |
| `eval_stage0.py` | check 4 — Stage-0 dirs are invisible to `eval_all` by design |
| `tests/test_projection_extraction.py` | pins the Eq.3 extraction bit-identical |
| `tests/test_constant_vector.py` | C1 algebra, gradient paths, ordering guard |
| `tests/debug_blend_gradient.py` | standalone gradient-path diagnostic (see §7) |
| `cluster/run_stage0.slurm` | warm start (`SMOKE=1` for a 10-step check) |
| `cluster/run_stage0_checks.slurm` | checks 3 + 4 |
| `cluster/run_latent_off_eval.slurm` | Phase 2 |
| `cluster/run_blend_debug.slurm` | runs the standalone diagnostic |

**Modified**
- `transformers/generation/utils.py` — **only** the Eq.3 extraction into
  `hrpo_project_latent()`. Generation stays bit-identical; the tests assert it.
- `variant_common.py` — `--latent_source`, `--constant_init`, `--init_from`, `--init_only`;
  `v7_constant_vector` in `GATE_MODES` and in a separate **`CONTROL_VARIANTS`** list — deliberately
  NOT in `VARIANT_CHAIN`, which `analyse_gate`'s Λ plots and `generate_report`'s primary
  table/figure iterate (it was briefly there, which would have drawn C1 as a 7th ablation step);
  `register_constant_latent_vector`, `apply_constant_latent_patch`,
  `_constant_latent_thinking_residual`, `constant_init_stats`, `load_stage0_init`;
  `experiment_dir` warm suffix; `EXPERIMENT_NAME_RE` accepts `warm`; `GateGradientLogger`
  buckets `u` separately from Λ; fingerprint gained `latent_source`, `constant_init`,
  `init_from`.
- `analyse_replicates.py` — arm key is now
  **`(size, variant, rp, lora, warm)`**; `parse_suffix` returns 4 values; `arm_label` takes
  `warm`; added T1-vs-A1 and C3-vs-v1 comparisons.
- `generate_report.py` — updated for the new key arity; `_primary_arms` **excludes warm arms**
  from the cold ablation cells.
- `eval_all.py` — `--latent_off`, `--variants`, `--rp`, `write_latent_off_summary`.
- `hrpo_grpo_baseline.py`, `hrpo_learned_lambda.py` — `--init_from` / `--init_only`, applied
  **after** `get_peft_model`.
- `eval_gsm8k.py`, `analyse_gate.py` — new `gate_mode="constant"` (unit gates **+** the learned
  `u`), with `u` registered *before* `load_adapter` and a post-load assertion that the
  checkpoint's `u` actually arrived. See §7.9 — without this C1 is scored with the projection
  it was never trained with.
- `variant_common.py` — `GATE_MODES["v7_constant_vector"]` is `"constant"`, not `"unit"`.
- `hrpo_constant_vector.py` — refuses `--latent_source projection` for a training run: it
  resolves to the *same* experiment directory as the constant arm (§7.10).
- `hrpo_overhead.py` — skips generation timing for `gate_mode="constant"` (the isolated timing
  harness only knows unit/standard); params and FLOPs still recorded.
- `run_ablations.sh`, `cluster/run_variant.slurm` — `v7` → `hrpo_constant_vector.py`.
  Deliberately **not** added to `variants_for_size`, so `--variant all` never sweeps the control
  into the main ablation.
- `cluster/submit_followups.sh` — `tier8` (§6).

---

## 6. Immediate next step

**Tier 8 has run (T1, C3 × 3 seeds; C1 seed 42) — results, 2026-09-13** (1.5B, rp=1.0, GSM8K
1319, per-seed paired McNemar; `results/replicate_analysis.json`):

| comparison | arm A | arm B | Δ (B−A) | seeds p<.05 |
|---|---|---|---|---|
| T1 vs A1 — warm start on v3 | 70.48 | 68.44 | **−2.05** | 2/3 |
| C3 vs v1 — warm start on v1 | 69.72 | 68.51 | **−1.21** | 1/3 |
| T1 vs C3 — latent vs GRPO, both warm | 68.51 | 68.44 | **−0.08** | 0/3 |
| C1 vs A1 — constant u vs projection (seed 42) | 71.65 | 70.28 | −1.36 (p=0.256) | 0/1 |

Latent-off (`results/latent_off_summary.json`): T1 +0.91 / −0.53 / +0.45 (mean +0.28; cold A1
+0.58); C3 identical correctness vectors ×3 (sanity pass); C1 −1.52 (single seed, inside the
±1.8 per-seed spread the cold arms show — not interpretable alone).

**Reading:** the warm start does **not** rescue the latent channel. It costs accuracy on both arms
— consistent with check 4, where the warm starts were already ~2 pts below base before RL — and
the headline null replicates with both arms warm (T1 ≈ C3). Nor did it create dependence on the
latent pathway (T1 latent-off ≥ normal). C1 is uninterpretable at n=1.

**C2 — Dr. GRPO de-bias: run and analysed 2026-09-16 (results below).** `--dr_grpo {off,len,std,both}`
(default `off`) sets trainer attributes via `variant_common.apply_dr_grpo`, called from
`train_with_resume`. They are read by `grpo_compute_loss` (constant `max_completion_length`
normaliser) and `GRPOTrainer._prepare_inputs` (no `/std`). Directory suffix `-dr{mode}`, arm-key
field `dr`; comparisons C2-v1 vs v1, C2-v3 vs A1, and C2-v3 vs C2-v1 (the headline question under
the de-biased objective). Verified by `tests/test_dr_grpo.py` — `off` is bit-identical to commit
`8fae960` for loss, KL, completion length and gradients at 1/2/4 chunks; `both` matches the
Dr. GRPO formula; the switch is on the trainer when `train()` starts; pre-C2 checkpoints resume as
`off` and refuse to resume as `both` — and by a local `PatchFastRL` regeneration of
`unsloth_compiled_cache/UnslothGRPOTrainer.py`, which contains every change and imports.

**C2 results** (1.5B, rp=1.0, seeds 42/43/44; all six logs confirmed
`[dr_grpo] mode=both: loss normaliser=512.0, advantage /std=False`):

| comparison | A | B | Δ (B−A) | seeds p<.05 |
|---|---|---|---|---|
| C2-v1 vs v1 — Dr. GRPO on GRPO | 69.72 | **72.00** | **+2.27** | 2/3 |
| C2-v3 vs A1 — Dr. GRPO on HRPO | 70.48 | **72.00** | **+1.52** | 0/3 (all 3 positive) |
| C2-v3 vs C2-v1 — latent vs GRPO, both de-biased | 72.00 | 72.00 | **+0.00** | 1/3, and the sign flips |

Both C2 arms are the top two arms in the study, above all six settled ones. **Dr. GRPO is the only
intervention in this whole study that moved accuracy at all** — and it moves GRPO and HRPO alike,
so the HRPO≈GRPO null is *not* an artefact of GRPO's length/difficulty biases. C2-v1 seed 43
(74.15) is the single best run recorded.

Supporting diagnostics:
- **Latent-off:** C2-v1 identical correctness vectors ×3 (sanity pass). C2-v3 is −0.23/−0.45/−0.38
  (mean −0.35) — all three NEGATIVE, where cold A1 was +0.58. The only hint that de-biasing gave
  the latent channel any function; magnitude (3–6 questions) is inside noise, so do not lead with it.
- **Completion length went UP, not down:** C2 ends at 267–268 tokens vs 250–256 cold. Removing the
  length bias did not shorten completions here — the opposite of the Stage-0 warm start's effect.
- **Hidden ratio is unchanged:** C2-v3 0.1677 vs A1 0.1678. The de-bias did not make the model use
  the latent channel more.

**Caveat to state with any C2 result:** at an unchanged learning rate both switches also shrink the
update (÷512 instead of the ~250-token mean length; dropping ÷std, where group std ≤ 0.58), so C2
conflates de-biasing with a smaller effective step size. Measured: mean pre-clipping grad norm over
the last 100 steps is 0.39 (cold) vs 0.12 (C2). `max_grad_norm=0.1` clips both, which removes much
of that gap — but C2's steps fall below the clip threshold far more often. Inherent to Dr. GRPO as
published, not a flaw of this implementation.

Run — there is no `--max_steps`, so verify on seed 42 before the rest:
```bash
SEEDS=42 bash cluster/submit_followups.sh tier9
grep -n "\[dr_grpo\] mode=both" slurm_logs/hrpo-1.5b-C2-*-seed42-*.log     # both jobs
grep -c "dr_grpo_len_constant\|dr_grpo_scale_rewards" unsloth_compiled_cache/UnslothGRPOTrainer.py  # expect 4
SEEDS="43 44" bash cluster/submit_followups.sh tier9
```

**Second task — MATH (`tier10`): run and analysed 2026-09-25 — the study's one positive signal.**

**Standard GRPO collapses on MATH.** Completion length drifts up all run, hits the 1024 cap at
step ~752, and a truncated completion carries no `####` — which the reward requires — so it scores
0 however good the reasoning. Reward 0.56 → 0.14, KL 0.09 → 2.2, grad_norm 0.5 → 11 against a 0.1
clip; onset step ~650-700 in BOTH arms, so it is the objective, not the latent channel. Final
accuracy lands *below the base model*: v1 **19.80**, v3 **22.80** against base MATH-500 **35.60**.
It never fired on GSM8K, where completions sat near 250 tokens against a 512 cap.

**Dr. GRPO rescues it** (`--dr_grpo both`; seed 42, the only seed with both conditions completed):
v1 19.80 → **37.40** (+17.60, p≈0), v3 22.80 → **49.80** (+27.00, p≈0). 5 of 6 de-biased runs stay
stable (peak lengths 681-760, grad_norm 0.08-0.26).

**MATH-500 accuracy, de-biased, 3 seeds** (base 35.60):

| arm | seeds 42 / 43 / 44 | mean |
|---|---|---|
| v1 GRPO | 37.40 / 49.20 / 48.60 | 45.07 |
| v3 learned Λ | 49.80 / 51.60 / 52.20 | **51.20** |

Δ = **+6.13** for the latent arm, all three seeds positive (+12.40 p≈0, +2.40 p=0.24, +3.60
p=0.086). **Read this carefully**: the +12.40 is inflated because v1 seed 42 is the one de-biased
run that still drifted (length 765, reward 0.181). On the two clean seeds the gap is +2.40/+3.60,
neither significant. What is striking is the spread — v3 spans 2.4 points across seeds, v1 spans
11.8 — i.e. the latent arm looks *more stable* on the harder task. n=3; a hypothesis, not a claim.

**Runtime**: 55h per run at `per_device 4 x accum 8`, 32h at `8 x 4` (GRPO generates once per
micro-batch, so the first config doubled generate() calls). GSM8K is ~6h.

**Implementation (unchanged below).**
Every result in this study is GSM8K, which is its most obvious limitation. Scope is deliberately
narrow: **v1 vs v3 only**, rp=1.0, 3 seeds (6 runs) — the two arms that differ solely in the latent
pathway. The other four variants are inert on GSM8K and do not justify MATH's compute.

How it works: `--dataset {gsm8k,math}` (default gsm8k, so nothing existing changes) →
`variant_common.build_train_dataset` / `reward_answer_fn`; all seven variant scripts are now
dataset-agnostic (their hardcoded `preprocess_gsm8k` is gone). Directory suffix **`-dsmath`**,
arm-key 7th field `ds`, and `eval_gsm8k.evaluate_model(task=...)` serves both tasks — the two
share SYSTEM_PROMPT, the `####` marker and `extract_from_response`, and differ only in the split,
the gold field, the decode budget (1024 vs 512) and `max_seq_length` (1536 vs 1024).

Data (verified by loading both locally, 2026-09-16): **no local MATH copy exists on the cluster**.
Train comes from `EleutherAI/hendrycks_math` — 7 subject configs, **7500 examples, all with boxed
answers, 234 optimizer steps** at total batch 32 (GSM8K is 233, so the runs are comparable).
Evaluation is **always MATH-500** (`HuggingFaceH4/MATH-500`, 500 problems), never a local MATH test
split, which would be ~5000 problems and silently change the denominator.

Configuration (passed explicitly by tier10 so `run_config.json` records it): completions 1024 per
the HRPO paper's MATH row, `per_device_train_batch_size 4` × `gradient_accumulation_steps 8` to
hold the total train batch at 32 while offsetting the longer completions' memory.

Analysis: MATH arms are excluded from the GSM8K primary table/figure and from
`replicate_summary.png` (one accuracy axis across two different test sets would invite reading
them as comparable), and are compared only within MATH — "latent vs GRPO on MATH (v3 vs v1)".
The base zero-shot reference is evaluated per task into
`results/base_model_evals/1.5b_base_math.json`.

Run (verify one seed before the rest — MATH runs are slower than GSM8K):
```bash
SEEDS=42 bash cluster/submit_followups.sh tier10
grep -n "\[math\]" slurm_logs/hrpo-1.5b-math-*-seed42-*.log     # loader source + any dropped rows
SEEDS="43 44" bash cluster/submit_followups.sh tier10
```

**Still open:** MATH runs (tier10); C1 seeds 43/44 (report C1 as a single-seed pilot otherwise); MATH
v1-vs-v3 extension (scoped, not started); the cluster copies of `analyse_replicates.py` and
`generate_report.py` carry bugs §7.12–13 until the next sync-up.

**Phase 5 is written** — `tier8` in `cluster/submit_followups.sh`, gated on the explicit name.
Dry-run verified on both paths (warm start present → submits; absent → skips with the exact
command to produce it). Resolved directories, all fresh and distinct from the settled cold arms:

| Arm | Seeds | Experiment directory |
|---|---|---|
| T1 | 42/43/44 | `1.5b-v3_learned_lambda-rp1-warmon` (+`-seedN`) |
| C3 | 42/43/44 | `1.5b-v1_grpo_baseline-rp1-warmoff` (+`-seedN`) |
| C1 | 42 | `1.5b-v7_constant_vector-rp1` |
| C2 | — | **not implemented**; port the Dr. GRPO toggles from the fusion repo, verify bit-exact with both toggles off first |

`analyse_replicates.parse_suffix` resolves these to `warm='on'` / `'off'` / `None`, so T1 can
never be pooled with the settled A1.

Rather than chaining with `--dependency=afterok`, tier8 **requires the warm start to already
exist** (it checks for `adapter_model.safetensors`) and skips the arm with instructions if not.
That avoids duplicate Stage-0 jobs when the warm starts were launched by hand.
**Only seed 42's warm starts exist so far** — produce 43/44 first (<20 min each).

Then update `cluster/run_eval_and_analysis.slurm` so `eval_all`, `analyse_replicates` and
`generate_report` pick up the new arms, and add `--latent_off` evals for them.

---

## 7. Hard-won bugs — do not rediscover these

1. **`attention_mask` silently kills Λ's gradient.** `unsloth/models/llama.py:707` detaches
   `inputs_embeds` when ALL of: no `labels` passed, `attention_mask` passed, no
   `past_key_values`, embeddings frozen — then re-marks it `requires_grad_(True)`, creating a
   *fresh leaf*. The blend runs, `a_t`/output require grad, the loss requires grad and falls
   (LoRA is inside the layers, downstream of the new leaf) — but Λ upstream gets **nothing**.
   This cost six cluster rounds. **Stage-0 must never pass `attention_mask` to the model.** The
   RL log-prob call (`rl_replacements.py:409`) omits it for the same reason, which is why v3 RL
   was never affected. `stage0_warmstart.compute_loss` now raises if no gradient reaches Λ.
2. **Warm-started runs collide with their cold counterpart.** Before the fix, a warm v3
   resolved to `experiments/1.5b-v3_learned_lambda-rp1` — the settled A1 result — and
   `analyse_replicates` would have pooled T1 and A1 as two *seeds of one arm*. Fixed via the
   `-warm{on|off}` suffix and the 5-tuple arm key. If you add another axis, do **both**.
3. **`--variant_suffix -rp1` crashes argparse** (leading `-` reads as a flag). Use the `=`
   form: `--variant_suffix=-rp1`.
4. **`torch.autograd.grad()` is unusable under reentrant gradient checkpointing**
   ("incompatible with .grad()"). Use a **backward hook** on the parameter instead — that is
   what `LAMBDA_GRAD` does, and it is a stronger check anyway (proves gradient *flows*, not
   just that a path exists).
5. **dtype**: softmax in fp32 for stability, then cast probs **back to the embedding dtype**
   before the einsum, or you get `expected scalar type BFloat16 but found Float`.
   `hrpo_project_latent` deliberately does not cast internally — the caller owns it.
6. **C1's `u` must be registered BEFORE `get_peft_model`**, or `modules_to_save`'s deep-copy
   won't contain it and it never reaches the adapter. It lives inside `thinking_residual_Lambda`
   so it inherits Λ's 1e-3 optimizer group and gets saved automatically.
7. **Login node ≠ compute node.** A bare `python ...` fails at model load with
   `Connection reset by peer` + an unwritable `~/.cache/huggingface`. Always `sbatch`; every
   slurm here carries the `HF_HOME` + squid-proxy env block.
8. **Never `pip install` transformers/unsloth/trl** — it shadows the vendored forks that carry
   the gate. `run_stage0.slurm` has a preflight that asserts all three resolve inside the repo.
9. **A gate variant needs its eval path built at the same time as its training path.** C1 was
   fully implemented and tested, but `GATE_MODES["v7_constant_vector"]` said `"unit"` and
   `eval_gsm8k` had no constant mode — so C1 would have trained with `u` and been *scored* with
   the Eq.3 projection, producing a meaningless number with nothing to flag it. Two ordering
   traps in the fix: `u` must be registered **before** `load_adapter` (else the checkpoint's key
   has no parameter to land in and is silently dropped), and the eval now **raises** if `u` is
   bit-identical to its init after loading. Whenever you add a variant, check `GATE_MODES`,
   `eval_gsm8k`, `analyse_gate` and `hrpo_overhead` together.
10. **v1's Λ is uninitialised memory and often reads NaN.** v1 never calls
    `reset_lambda_parameters`, and `disable_gate` makes `is_thinking` always False so the blend
    never runs — Λ is inert, which is why every settled v1 run is unaffected. But it silently
    disarmed `load_stage0_init`'s "did Λ actually move?" guard, because `nan == 0.0` is False.
    The loader now keys that check on whether the *file* carries a Λ tensor, and **raises**
    (rather than warning) if it does and Λ did not move — the failure it guards against is a run
    that trains from a cold Λ while every log claims it was warm-started.
11. **`experiment_dir` has no `--latent_source` axis**, so v7 in projection mode resolved to the
    same directory as the constant arm. `hrpo_constant_vector.py` now refuses projection mode
    for a training run. Same family as bug 2: *every* new axis needs a directory suffix, or an
    arm silently overwrites/pools with another.
12. **Sorting 5-field arm keys crashes once warm arms exist — and the pipeline hides it.**
    `sorted(arms.items())` in `analyse_replicates.main` compared `warm='on'` with `warm=None`
    (T1 and A1 tie on the first four fields) → `TypeError`; `plot_arms` also still unpacked keys
    as 4 values. Neither fired until tier 8 produced the first warm arm. Fixed with
    `arm_sort_key`; pinned by `tests/test_replicate_arm_keys.py`. **The cluster job still printed
    "pipeline complete"** — each stage runs regardless of the previous one's exit — and the stale
    August `replicate_analysis.json` was synced back looking current. After every analysis job,
    `grep -n Traceback slurm_logs/hrpo-eval-analysis-*.err` and check output timestamps.
13. **`key_findings` looked arms up with a 4-field key** (`seed_arms.get((size, variant, rp,
    "full"))`) after the key grew a 5th field (warm). Every lookup missed, so the generated
    `key_findings.txt` has said "PENDING — no rp=1.0 evaluation" for RQ1/2/3/8/9 since then,
    although all six arms exist with 3 seeds. No crash, just wrong prose. Fixed to the full
    6-field key; `add_comparison` now pads short keys. **Every time the arm key changes arity,
    grep for `.get((` and literal key tuples, not only for unpacks.** (The hand-written
    RESULTS_SECTION / SUPERVISOR_SUMMARY never used this file.)
14. **Vendored files have mixed line endings.** `unsloth/models/rl_replacements.py` is CRLF;
    `trl/trainer/grpo_trainer.py` is LF. A Python text-mode read/write silently converts CRLF
    → LF, turning a 23-line change into a 1105-line diff that buries the real hunks. Patch
    with the Edit tool or byte-preserving code, and sanity-check `git diff --stat` line counts.
15. **A flag that renames the directory must be proven to reach the trainer.** The first C2
    wiring added `apply_dr_grpo` but never called it from `train_with_resume`: `--dr_grpo both`
    would have produced a `-drboth` directory trained with the ORIGINAL objective. Caught by a
    test; now pinned by a behavioural test that inspects the trainer at the moment `train()`
    starts.
16. **`FULL_TEST_SIZE = 1319` is GSM8K's, and three places gate on it.** `eval_all`
    (is this eval complete?), `analyse_replicates.load_vector` and `generate_report._seed_arms`
    all discard a run whose eval has fewer rows. MATH-500 has 500, so every MATH run would have
    been silently dropped as "partial" — no error, just missing arms. Now `TASK_TEST_SIZE` /
    `TASK_FULL` per task, keyed off the `task` field that `eval_gsm8k` writes into each
    `eval_results.json`.
17. **`extract_boxed_answer` never returns `None`.** It splits on the substring `"boxed"`, so
    prose with no `\boxed` yields an invented answer ("no boxed answer here" →
    `\boxed{answer here}`). Any "is None" check on it is dead code. Use
    `utils.has_boxed_answer`. On real data this is nearly moot: 7,498/7,500 MATH train and
    500/500 MATH-500 solutions have a genuine boxed answer; the other two (train indices 5341,
    5343) contain an empty `\boxed{}`. **The RL set keeps them** — every completed MATH run
    trained on all 7,500, and dropping two rows reshuffles the whole data order. Stage 0 drops
    them.
18. **A bare MATH answer is read as its last number.** `process_math_answer` understands only a
    boxed answer or one in LaTeX delimiters; `#### \frac{\pi}{2}` scores as `2`. A Stage-0
    target in that form scores 1.0 under the RL reward on only 5,080/7,500 problems; the boxed
    form `#### \boxed{X}` on all of them. `utils.math_sft_target` builds the boxed form.
    (Trained models learn their own format — they literally write `#### {answer} \(X\)`, copying
    the system prompt's placeholder — which also parses correctly.)
19. **`eval_all` evaluates cancelled runs.** It scores the newest checkpoint whether or not
    training finished, so the two MATH seed-43 runs stopped at 500/937 entered the results as
    arms — and, snapshotted before the collapse, scored higher than every completed run.
    `variant_common.training_incomplete` now excludes them in `analyse_replicates` and
    `generate_report`, and `eval_all` records `training_complete` per row.

---

## 8. Decisions already taken (don't relitigate)

- `r_min = 0.98`, not the brief's 0.95 — match the existing `-rp1` arms.
- C1 `u` init = **`mean`** (the brief's spec). Measured on Qwen2.5-1.5B: centroid norm 0.3270
  vs mean row norm 1.0126 → ratio **0.3229**, i.e. `u` starts only 3.1× smaller than what it
  replaces. `--constant_init {mean_scaled,random_row}` exist if diagnostics show `u` stalling.
- The constant path **skips** the `/||probs||` normalisation — dividing would rescale `u` per
  token, so it would not be constant.
- Stage-0 latents are **detached** (pass 1 under `no_grad`). Not a shortcut: in RL they come
  from `generate()`, which is itself `no_grad`, so they are already constants in the loss
  forward. This keeps Stage 0 faithful to the RL path.

---

## 9. Commands

```bash
# sync UP (carries the vendored transformers change — required)
rsync -av --exclude '.git' --exclude '.venv' --exclude 'wandb' \
  --exclude '__pycache__' --exclude 'unsloth_compiled_cache' \
  --exclude 'slurm_logs' --exclude 'experiments' --exclude 'results' \
  /home/zacht/Dissertation-HRPO/ \
  ztay632@foscsmlprd01.its.auckland.ac.nz:/data/ztay632/Dissertation-HRPO/

# CPU tests
python -m pytest tests/ -v            # expect 14/14 (excludes debug_blend_gradient)

# warm start (per seed) — seeds 43/44 must exist before tier8 runs T1/C3 there
BLEND=on  SEED=42 sbatch cluster/run_stage0.slurm
BLEND=off SEED=42 sbatch cluster/run_stage0.slurm
SMOKE=1 BLEND=on  sbatch cluster/run_stage0.slurm     # 10-step machinery check

# checks 3 + 4
sbatch cluster/run_stage0_checks.slurm

# is a warm start actually carrying Lambda? (login node, no GPU, seconds)
python3 -c "
from safetensors.torch import load_file
for b in ('on','off'):
    sd=load_file(f'experiments/1.5b-stage0-{b}-rmin0.98-seed42/adapter_model.safetensors')
    lam=[k for k in sd if 'thinking_residual_Lambda' in k]
    print(b, len(sd), 'tensors;', [(k, sd[k].float().mean().item()) for k in lam])"
# blend on -> Lambda mean 6.7641754150390625;  blend off -> NO Lambda keys

# tier8 (Phase 5)
bash cluster/submit_followups.sh tier8                    # T1+C3 x 42/43/44, C1 x 42
SEEDS=42 bash cluster/submit_followups.sh tier8           # single-seed pass
C1_SEEDS="42 43 44" bash cluster/submit_followups.sh tier8  # full C1 set

# gradient-path diagnostic if the blend ever goes quiet again
sbatch cluster/run_blend_debug.slurm

# sync BACK — logs
mkdir -p /home/zacht/Dissertation-HRPO/slurm_logs
rsync -av ztay632@foscsmlprd01.its.auckland.ac.nz:'/data/ztay632/Dissertation-HRPO/slurm_logs/hrpo-*' \
  /home/zacht/Dissertation-HRPO/slurm_logs/

# sync BACK — diagnostics only, never the multi-GB weights
rsync -av --include='*/' \
  --include='stage0_log.jsonl' --include='stage0_summary.json' \
  --include='eval_results_stage0.json' --exclude='*' \
  ztay632@foscsmlprd01.its.auckland.ac.nz:/data/ztay632/Dissertation-HRPO/experiments/ \
  /home/zacht/Dissertation-HRPO/experiments/
```

⚠️ **Never `rsync --delete` into `results/`** without `--exclude '*.md'` — it deletes the
hand-written narrative docs (it already did once; they were recovered from the session
transcript).

---

## 10. Quick state check

```bash
# Stage-0 warm starts present and what Lambda did
python3 -c "
import json;d=json.load(open('experiments/1.5b-stage0-on-rmin0.98-seed42/stage0_summary.json'))
print('final Lambda:', d['final_lambda'])"

# warm arms must never share a key with cold ones
python3 -c "
from analyse_replicates import discover_arms
for k in sorted(discover_arms(), key=str): print(k)"
```
