# HRPO ablation — handover

Context for an agent picking up this project. Everything here is verified against the repo as
of 2026-08-10. Branch: `Ablations`.

---

## 1. What this project is

An MSc dissertation **ablation study of HRPO** (Hybrid Reasoning Policy Optimization; Yue et
al., 2025, arXiv:2505.18454). HRPO makes an LLM reason partly in **continuous latent space**:
it projects a hidden state back onto the embedding manifold (Eq. 3) and blends it into the next
input embedding through an input-dependent **gate** (Eq. 4), trained with an RL objective (a
GRPO variant), no chain-of-thought supervision.

**This repo is the breakdown/ablation only.** It decomposes HRPO into 6 nested variants and
measures each component's contribution. (Separate "improvements" work — context fusion etc. —
lives in a *different* repo and is out of scope here.)

**Headline result (settled):** on GSM8K with Qwen2.5-1.5B, at the paper-faithful setting, **all
six variants tie at ~70%** — no HRPO component (latent projection, input gate, or reward
shaping) beats plain GRPO. An earlier version of the study reported a false "the input gate
hurts" positive, caused by an off-paper `repetition_penalty=1.2` combined with single-seed
evaluation. See §4.

Task convention throughout: **single greedy decode, no test-time scaling** (no majority vote /
best-of-N).

---

## 2. The six variants

A nested chain — each adds exactly one component to the previous. `VARIANT_ID`s and the gate
config are in `variant_common.py` (`VARIANT_CHAIN` L38, `GATE_MODES` L49).

| ID | `VARIANT_ID` | Script | What it changes | Gate handling in the script |
|----|--------------|--------|-----------------|------------------------------|
| v1 | `v1_grpo_baseline` | `hrpo_grpo_baseline.py` | Pure GRPO, no projection, no gate | `disable_gate(model)` |
| v2 | `v2_fixed_blend` | `hrpo_fixed_blend.py` | Eq. 3 projection on; Λ **frozen** at init; `r_t=i_t=1` | `apply_unit_gate_patch` + `freeze_gate_modules` |
| v3 | `v3_learned_lambda` | `hrpo_learned_lambda.py` | Projection on; Λ **trained**; `r_t=i_t=1` (gates removed) | `apply_unit_gate_patch` (Λ trainable) |
| v4 | `v4_full_hrpo_gate` | `hrpo_full_gate.py` | **Standard published HRPO** — `r_t`, `i_t`, Λ all trained; outcome-only reward | standard gate (no patch) |
| v5 | `v5_hrpo_format_reward` | `hrpo_format_reward.py` | v4 + shaped reward `FORMAT_REWARD=0.1`, `REASONING_BONUS=0.05` | standard gate |
| v6 | `v6_hrpo_full` | `hrpo_full.py` | v5 + repetition-loop reward zeroing (`zero_repetition_loops=True`) | standard gate |

Each consecutive step answers one question: v1→v2 does *any* latent blend help? v2→v3 does
*learning* Λ help? v3→v4 does *input-dependent gating* help? v4→v5 does reward shaping help?
v5→v6 does repetition handling help? **All answers at rp=1.0 are "no, within seed noise."**

**Critical implementation detail (v2/v3):** the unit gates (`r_t=i_t=1`) are enforced by
`apply_unit_gate_patch` (`variant_common.py` L274/L284), an instance-level patch that *bypasses*
the gate linear layers. Merely omitting those layers from the optimizer is **not** sufficient —
they still execute with random weights, giving `σ(random)≈0.5` instead of 1. Worth a sentence
in the writeup.

---

## 3. Where the gate (Eq. 4) is actually implemented

**Not in the pip packages.** The repo vendors forks of `transformers`, `unsloth`, and `trl`;
`import transformers` etc. must resolve to these repo directories. The gate lives in:

- **`transformers/models/qwen2/modeling_qwen2.py`** — the blend:
  - L487: `a_t = exp( -c · softplus(-Λ) · r_t )`  (note: codebase uses `softplus(-Λ)`, `c=8`)
  - L517–518: `thinking_residual_gate_r`, `thinking_residual_gate_i` (the `W_a`, `W_x` linears)
  - L531–534: `r_t = σ(gate_r(embeds))`, `i_t = σ(gate_i(embeds))`,
    `return a_t*embeds + sqrt(1 - a_t² + eps)*(i_t*residual), a_t`
  - `sqrt(1 - a_t²)` is the **hidden ratio** — the fraction of latent in the blend, the key
    diagnostic. `Λ` is a per-dimension learnable vector.
- **`unsloth/models/llama.py`** — the analogous gate for the Llama path (not used for the 1.5B/3B
  Qwen runs but present).

⚠️ **Never `pip install unsloth/transformers/trl`** into the env — it silently shadows the
vendored fork with the stock, gate-less package. If `is_bfloat16_supported` import errors appear
on a fresh machine, the fix is to rsync the vendored dirs, not pip install.

---

## 4. The results (settled, 1.5B, rp=1.0, 3 seeds 42/43/44)

Numbers of record: `results/replicate_analysis.json`. Written up in
`results/RESULTS_SECTION.md`. Slides: artifact
`https://claude.ai/code/artifact/c5023c0b-b0dd-4588-beca-41893729b3c7`.

**Accuracy (seed mean [min–max]):**

| Base | v1 | v2 | v3 | v4 | v5 | v6 |
|------|----|----|----|----|----|----|
| 61.11 | 69.72 [68.5,70.5] | 70.20 [69.8,70.5] | 70.48 [68.7,71.7] | 70.05 [67.4,72.0] | 70.58 [70.1,71.1] | 69.29 [68.2,70.1] |

All six span **1.29 pt**; every Wilson CI overlaps. Δ vs v1: +0.48 / +0.76 / +0.33 / +0.86 / −0.43.

**The confound (§4 story).** The dissertation protocol used `repetition_penalty=1.2` (not in
the paper); applied to logits *before* the softmax that feeds both sampling **and** the Eq. 3
projection. Removing it (→1.0) lifts **every** arm, including the non-latent baseline:

| | v1 | v2 | v3 | v4 | v5 | v6 |
|---|----|----|----|----|----|----|
| rp1.2 | 62.47 | 63.38 | 64.39 | 62.29 | 58.98 | 63.23 |
| rp1.0 | 69.72 | 70.20 | 70.48 | 70.05 | 70.58 | 69.29 |
| Δ | +7.25 | +6.82 | +6.09 | +7.76 | +11.60 | +6.06 |

v1 (no latent projection) gains as much as the latent arms → the penalty is a **general**
rollout-quality degradation, not latent-specific. At rp=1.2 + single seed, an earlier analysis
produced a false "remove the gate → +4.5, p=0.0004" headline (a seed-42 draw). **The
methodological finding — off-paper hyperparameter + single-seed eval manufactured a
significant-looking result — is the real contribution.**

**Mechanism — the gate collapses to a constant** (`results/gate_analysis.json`, all rp=1.0):
- per-token gate activation **variance = 0.000** (not input-dependent)
- hidden ratio identical across token types: **0.11867–0.11871** (number/operator/word/punct)
- Λ per-dimension **CV = 0.288** (trained) vs **0.300** (frozen init) → no specialisation
- final hidden ratio **v3 0.167 (gate off) → v4 0.119 (gate on)** → a constant rescale
- Λ gets **4.6× less** gradient in v4 than v3; gate params carry **2.5× Λ's** gradient
- trainable gate params at equal accuracy: **v3 1,536 vs v4 4,723,200**
- reward↔ratio r=0.58 is **within-run only** (both rise with step); cross-variant it inverts
  (v3 has the highest ratio, not the highest accuracy) — do NOT read it as "latent helps"

**Failure modes** (`results/failure_analysis.json`, seed-42 checkpoint, vs v1, n=1319):
- v4: both-correct 797 / v4-only 92 / v1-only 106 / both-wrong 324
- v6: both-correct 793 / v6-only 106 / v1-only 110 / both-wrong 310
- ~95% of HRPO-wrong responses are **wrong-answer-in-correct-format** (reasoning failures);
  repetition loops ~0% → v6's repetition handling targets a near-absent failure mode.

**Overhead** (`results/overhead_matrix.json`, rp-independent): gate ~4% FLOPs, <0.3% gen
wall-clock; v3 matches v4 at 1,536 vs 4.7M trainable gate params.

**Reconciliation with the paper:** paper GRPO 0.711 / HRPO 0.720 (headline greedy). Ours rp=1.0
v1 0.697 / v4 0.701; paper-faithful (attention-only LoRA) v4 0.677 (within ~2 pt of 0.720). The
paper's *own* significance test (their Table 9, sampled) is GRPO 0.710 / HRPO 0.712, **+0.002
n.s.** — we reproduce that non-significance (v1→v4 +0.33 pt, n.s.).

**3B is DEFERRED** — only single-seed rp=1.2 exists (v1 81.88, v2 80.06, v4 82.64, v6 74.91);
no 3B result is reported. It's the one remaining experiment (see §7).

---

## 5. Repo map

**Shared plumbing — `variant_common.py`:**
- `MODEL_NAMES` (L25): `1.5b`→Qwen2.5-1.5B-Instruct, `3b`→Qwen2.5-3B-Instruct.
- `build_arg_parser` (L63): all hyperparameter defaults (HRPO paper Table 4/5). Key flags:
  `--repetition_penalty` (default **1.2**; pass `1.0` for the corrected setting), `--seed`,
  `--lora_targets {full,attention}`, `--model_size`.
- `LORA_TARGETS` (L115): `full`=q,k,v,o,gate,up,down (main ablation); `attention`=q,k,v,o
  (paper-faithful).
- `experiment_dir(size,variant,args)` (L122): builds `experiments/{size}-{variant}` and appends
  suffixes `-seed{N}` (N≠42), `-rp{X}` (X≠1.2), `-lora{targets}` (≠full). So the corrected
  seed-42 v4 run is `experiments/1.5b-v4_full_hrpo_gate-rp1`.
- `parse_experiment_name(name)` (L147): `(size, base_variant, suffix)`. Analysis keys gate mode
  off the BASE variant.
- `experiment_is_complete` (L161): true iff a checkpoint's `global_step==max_steps` (934).
- `train_with_resume` (L205): resume-from-checkpoint with a config-fingerprint guard.
- `apply_unit_gate_patch` (L284), `disable_gate` (L293), `freeze_gate_modules` (L301).
- `GateGradientLogger` (L358): logs pre-clip-reconstructable gradient norms (used by §4 gradient
  flow).

**Training entry points:** the 6 `hrpo_*.py` scripts in §2. Run one variant:
`python hrpo_full_gate.py --model_size 1.5b --repetition_penalty 1.0 --seed 43`.

**Evaluation & analysis:**
- `eval_all.py` — discovers every `experiments/*` checkpoint, evaluates on full GSM8K test
  (n=1319), writes per-dir `eval_results.json` + `results/comparison_table.{json,csv}`. **Skips
  already-evaluated checkpoints** (`eval_is_complete`). `--sizes` (reads `HRPO_SIZES` env,
  default all).
- `analyse_gate.py` — mechanism (Λ CV, hidden ratios, gradient flow, token-category, PCA,
  activation heatmap). `--static` / `--dynamic`. **`--variant_suffix=-rp1`** makes it read the
  rp=1.0 checkpoints (see §6). Writes `results/gate_analysis.json` + `gate_p*` plots.
- `analyse_failures.py` — confusion matrix + error taxonomy vs v1. Takes two checkpoint paths.
- `analyse_replicates.py` — groups runs by (size,variant,rp,lora), per-arm mean/range + per-seed
  paired McNemar → `results/replicate_analysis.json`. **This is the source of truth for
  accuracy.**
- `hrpo_overhead.py --variant_matrix` — param counts, FLOPs, wall-clock → `overhead_matrix.json`.
- `generate_report.py` — the report: `table0_primary` (rp=1.0 ablation) + `ablation_accuracy_by_size.png`
  (primary figure) + LaTeX tables + `key_findings.txt`. `SIZES` reads `HRPO_SIZES`.

**Cluster infra (`cluster/`):**
- `run_variant.slurm` — trains one variant (reads `MODEL_SIZE`, `VARIANT_NUM`, `EXTRA_ARGS`).
- `submit_all.sh` — the main 10-cell matrix.
- `submit_followups.sh` — tiered follow-ups. `tier6` = 1.5B v5/v6 @ rp=1.0 ×3 seeds; `tier7` =
  3B chain @ rp=1.0 (`SEEDS=42` env to throttle). tier6/tier7 gated on explicit name only.
- `run_eval_and_analysis.slurm` — the post-training pipeline (see §6).
- `cleanup.sh` — dry-run disk reclaim (deletes non-final checkpoints of COMPLETE runs).

**Cluster:** `ztay632@foscsmlprd01.its.auckland.ac.nz:/data/ztay632/Dissertation-HRPO`.
SLURM env: `HOME/TMPDIR/HF_HOME=/data/ztay632`, squid proxy, `PYTHONUNBUFFERED=1`.

**Docs (in `results/`):** `RESULTS_SECTION.md` (the writeup), `SUPERVISOR_SUMMARY.md` (slide
deck), `DISSERTATION_WRITING_PROMPT.md` (self-contained prompt to write the chapter). `changes.md`
(repo root) is the append-only change log — read its tail for recent history.

---

## 6. The analysis pipeline & how to run it

The whole study is reported at **rp=1.0** and currently **scoped to 1.5B**. Two env knobs in
`cluster/run_eval_and_analysis.slurm` control this:

- **`HRPO_SIZES`** (default `1.5b`) — which sizes appear in the report. Set `"1.5b 3b"` after the
  3B rp=1.0 runs exist. Threaded to eval_all/analyse_gate/generate_report via their `SIZES`.
- **`ANALYSIS_SUFFIX`** (default `-rp1`) — which checkpoints the gate/failure diagnostics read.
  `analyse_gate.py --variant_suffix=-rp1` remaps `-rp1` dirs onto bare-variant keys so every
  plot reads the corrected checkpoints.

Pipeline order: eval_all → **wipe `results/plots/*.png` + clear gate/failure JSONs** →
analyse_gate (static+dynamic) → analyse_failures → hrpo_overhead → analyse_replicates →
generate_report. To run: `sbatch cluster/run_eval_and_analysis.slurm`. Only `eval_all` skips
(evaluated checkpoints); every analysis stage regenerates.

**Sync (do this carefully — see §8):** results come back with
`rsync -av --delete --exclude '*.md' cluster:.../results/ local/results/`.

---

## 7. Status & open items

- ✅ **1.5B v1→v6 at rp=1.0, 3 seeds — complete and settled.** All docs + slides reflect it.
- ⏸ **3B deferred** — exists only at confounded rp=1.2. To do it: `bash cluster/submit_followups.sh tier7`,
  then `HRPO_SIZES="1.5b 3b" sbatch cluster/run_eval_and_analysis.slurm`, re-sync, un-defer §6.7
  of RESULTS_SECTION.
- 0.5B never run (base model unavailable on the cluster).
- The 3 `results/*.md` docs are **untracked in git** — recommend committing them (they were once
  lost to a bad sync; see §8).

---

## 8. Gotchas that have already bitten (do not repeat)

1. **`rsync --delete` on `results/` destroyed the hand-written `.md` docs** (they're local-only,
   not on the cluster). Always sync results with `--exclude '*.md'`, or git-track the docs.
   Recovery last time was by replaying Write/Edit ops from the session transcript.
2. **`analyse_gate.py --variant_suffix -rp1` (space form) crashes argparse** — the leading `-`
   reads as a flag. Use the **`=` form**: `--variant_suffix=-rp1`.
3. **`generate_report.ensure_static_gate_plots()` used to re-run the static gate analysis at the
   rp=1.2 default and clobber the rp=1.0 Λ figures at report time.** Fixed: it now skips if the
   figures exist and honours `ANALYSIS_SUFFIX`. If Λ figures ever show rp=1.2 checkpoint paths,
   this regressed.
4. **v2/v3 need `apply_unit_gate_patch`** — omitting gate layers from the optimizer is not enough
   (σ(random)≈0.5, not 1).
5. **Never pip-install unsloth/transformers/trl** — shadows the vendored gate (§3).
6. Diagnostics (gate mechanism, failure modes) are **seed-42 only** by design; accuracy and
   significance use all 3 seeds.
7. **`results/key_findings.txt` provenance flags are a VERIFIED FALSE ALARM.** The current
   `gate_analysis.json` predates `analyse_gate`'s `_source_suffix` marker, so generate_report
   emitted "most likely rp=1.2" notes. Ground truth: every checkpoint path in
   `gate_analysis.json` is `-rp1`, and plot1 sample accuracy is 0.705 (rp=1.2 would be ~0.62) —
   the gate mechanism IS on rp=1.0 checkpoints. Also fixed two naive auto-interpretations in
   `key_findings.txt` (RQ4 called a 0.00001 gap "numbers higher"; RQ5 called CV≈init
   "specialisation") — both now state the null correctly in generate_report.py. **Re-run
   `analyse_gate` once** to write `_source_suffix` for all sections and regenerate a clean
   `key_findings.txt`. Authoritative doc = `RESULTS_SECTION.md`, not `key_findings.txt`.

---

## 9. Quick verification an agent can run

```bash
# accuracy of record (all arms, seeds, rp):
python3 -c "import json; d=json.load(open('results/replicate_analysis.json'))['arms']; \
[print(k, v['n_seeds'], round(v['mean']*100,2)) for k,v in sorted(d.items())]"

# confirm gate analysis is rp=1.0 (checkpoint paths must contain -rp1):
python3 -c "import json; print(json.load(open('results/gate_analysis.json'))['plot2_lambda_cv_1.5b']['v4_full_hrpo_gate']['checkpoint'])"
```
