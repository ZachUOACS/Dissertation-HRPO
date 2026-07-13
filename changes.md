# changes.md — HRPO Dissertation Ablation Work Log

Running log of every change, per Section 15 of the task specification. Entries are
append-only; reversals get a new entry.

---

### [2026-07-13 01:10] Section 2 repo mapping (no code changed)

**File(s):** none (read-only survey)

**Change:** Completed the required reading (HRPO paper §3.1/§3.2, `patch.py`, `utils.py`,
`hrpo_gsm8k.py`, `eval_gsm8k.py`, `hrpo_overhead.py`) and the repo mapping. Findings recorded
here verbatim so later entries can reference them.

**Repo mapping results:**

1. **Gate parameter definitions.** NOT under `unsloth/models/` as the spec assumed — they live
   in the *vendored transformers* package:
   - `transformers/models/qwen2/modeling_qwen2.py:517-519` — `Qwen2Model.__init__` defines
     `self.thinking_residual_gate_r = nn.Linear(hidden, hidden)`,
     `self.thinking_residual_gate_i = nn.Linear(hidden, hidden)`,
     `self.thinking_residual_Lambda = ThinkingResidualLambda(config)`.
   - `transformers/models/qwen2/modeling_qwen2.py:469-489` — `ThinkingResidualLambda`
     (`c = 8.0`, `Lambda = nn.Parameter(torch.randn(hidden))`).
   - A parallel copy exists in `transformers/models/llama/modeling_llama.py` (Llama backbones,
     unused for this Qwen-only study).

2. **`reset_lambda_parameters`.** Defined at `modeling_qwen2.py:476-483`; accepts
   `r_min=0.9, r_max=0.999`. It draws `u ~ U[r_min, r_max]` and sets
   `Lambda = -log(u^(-1/c) - 1)`. Call site: `hrpo_gsm8k.py:57-59`
   (`model.model.model.thinking_residual_Lambda.reset_lambda_parameters(...)`), called AFTER
   `get_peft_model`. Verified in peft 0.14 source
   (`peft/utils/other.py`, `ModulesToSaveWrapper.__getattr__`) that this attribute access
   forwards to the **active adapter copy** (`modules_to_save["default"]`), so the *trained*
   Lambda copy is the one being initialised. The `original_module` keeps `randn` init but is
   inactive while the adapter is enabled.

3. **Sign convention differs from the paper.** The forward is
   `a_t = exp(-c * softplus(-Lambda) * r_t)` (`modeling_qwen2.py:485-489`) — note
   `softplus(-Λ)`, not `softplus(Λ)` as in paper Eq. 4. With the init above this still yields
   `a_t(r_t=1) ~ U[r_min, r_max]`. All analysis code (Task 4 Plot 2/3) must therefore use
   `a = exp(-8·softplus(-Λ))`, NOT the spec's literal `exp(-8·softplus(Λ))`.

4. **Equation 4 blend (gate) call sites.**
   - Module method: `Qwen2Model.thinking_residual` at `modeling_qwen2.py:530-534`:
     `r_t = σ(gate_r(e)); i_t = σ(gate_i(e)); a_t = Λ(r_t);
     return a_t*e + sqrt(1-a_t²+eps)*(i_t*h), a_t`.
   - Training forward: `unsloth/models/llama.py:663-669` (`LlamaModel_fast_forward`; blends
     `inputs_embeds[thinking_mask]` with `thinking_embeds[thinking_mask]`).
   - Incremental decode: `unsloth/models/llama.py:940-949`
     (`LlamaModel_fast_forward_inference`; also emits `embeds_ratio = a_t.mean(-1)`, set to 1.0
     for non-think tokens; returned through `hidden_states=[thinking_embeds, is_thinking,
     embeds_ratio]` at `llama.py:1025`).

5. **Equation 3 projection call site.** `transformers/generation/utils.py:3367-3372` inside the
   patched `GenerationMixin._sample`:
   `last_thinking_states = einsum('bv,vd->bd', probs, W_e) / sqrt(sum(probs²))`, with
   `probs = softmax(logits_processor(...))` — the generation temperature IS the Eq. 3 τ.
   Think-span detection: `is_thinking = [model.answer_start not in decoded_completion]`
   (`utils.py:3368`), i.e. the span ends at the first `####`. `is_inference=True` switches
   token *selection* to argmax (`utils.py:3352-3356`) while the projection still uses the
   temperature-softmaxed distribution — this is why `eval_gsm8k.py` passes the training
   temperature parsed from the checkpoint path even for greedy eval.

6. **`modules_to_save` handling.** Passed at `hrpo_gsm8k.py:48-52` with all three gate modules.
   `unsloth/models/llama.py:2336-2350` validates entries (only `lm_head`, `embed_tokens`,
   `thinking_residual*` allowed); `llama.py:2457-2478` then, per module present, casts the
   adapter copy to float32 and sets `requires_grad_(True)` **individually per gate module** —
   so passing only `thinking_residual_Lambda` (variant v3) is supported by the existing code.

7. **v3 feasibility check (spec §6 question).** Omitting `gate_r`/`gate_i` from
   `modules_to_save` is **NOT sufficient** to fix `r_t = i_t = 1`. Those `nn.Linear` layers are
   not in the pretrained Qwen checkpoint, so `from_pretrained` leaves them randomly initialised
   (σ(random·e) ≈ 0.5, not 1) and they still execute inside
   `Qwen2Model.thinking_residual`. Fixing `r_t = i_t = 1` therefore requires bypassing the
   gate computation. Chosen mechanism (implemented later, see subsequent entries): an
   instance-level override of `Qwen2Model.thinking_residual` that computes
   `a_t = Λ(ones)` and blends `a_t·e + sqrt(1-a_t²+eps)·h` — identical maths to the module
   method with `r_t = i_t = 1`. This must be applied at **both training and evaluation** time
   for v2/v3 (the random gate weights are not saved in the adapter, so an unpatched eval would
   use different random gates).

8. **GRPOConfig / repetition_penalty.** `GRPOConfig` (vendored `trl/trainer/grpo_config.py`)
   has NO `repetition_penalty` field, and the trainer's `GenerationConfig`
   (`trl/trainer/grpo_trainer.py:427-432`; compiled twin
   `unsloth_compiled_cache/UnslothGRPOTrainer.py:891-896`) sets only
   `max_new_tokens/do_sample/temperature/pad_token_id`. **repetition_penalty is currently set
   nowhere.** The spec's fixed `repetition_penalty = 1.2` will be applied by assigning
   `trainer.generation_config.repetition_penalty` after trainer construction in the new
   variant scripts.

9. **`FORMAT_REWARD` / `REASONING_BONUS` / `_is_repetition_loop`.** **None of these exist in
   the current `utils.py`.** The reward (`utils.py:39-62`, `get_reward_func`) is pure outcome ×
   format-match: 1.0 iff extracted answer equals gold AND `####` occurs exactly once, else 0.0.
   They must be added (defaults 0.0 / disabled) without changing default behaviour.

10. **`hrpo_analysis.py` is lost.** Only `__pycache__/hrpo_analysis.cpython-312.pyc` remains;
    the source is not in the working tree nor anywhere in git history (repo has 5 commits, none
    include it). Bytecode inspection shows it contained `HRPOAnalyser`
    (`_register_hooks`/`_make_hook` via `register_forward_hook`, `_token_category`, PCA,
    lambda-distribution, reward-ratio plots) and a `HRPOAnalysisCallback`. Because the source
    cannot be extended, `analyse_gate.py` will be written fresh, reusing the same
    forward-hook mechanism design (hooks on `thinking_residual_gate_r/gate_i/Lambda`).

11. **`_experiment_is_complete` does not exist** in `hrpo_gsm8k.py`; the current guard is
    `os.path.exists(exp_name) and len(os.listdir(exp_name)) > 0` (`hrpo_gsm8k.py:25-27`).
    A stricter completeness check (checkpoint with `trainer_state.json` whose
    `global_step == max_steps`) will be implemented in the new infrastructure.

12. **`eval_gsm8k.py` format.** `evaluate_model(model_path, adapter_path, temperature,
    is_inference, batch_size, num_samples, save_results)` → greedy decode (`is_inference=True`)
    with `max_new_tokens=512`, saves `{checkpoint}/eval_results.json` with
    `{'metrics': {accuracy, correct, total, model_path, timestamp}, 'results': [{question,
    true_answer, generated_answer, full_response, correct}, ...]}`.

13. **Existing checkpoints and their exact hyperparameters** (from `training_args.bin` +
    `trainer_state.json` + wandb metadata):
    - `experiments/Qwen2.5-1.5B-Instruct-gsm8k-group8-lora32-rmin0.98-temp0.5/checkpoint-1868`:
      the protected HRPO baseline. group8, per-device bs 8, ga 4, r_min 0.98, temp 0.5,
      seed 42. Full 1319-question eval exists (acc 0.658). Does NOT match the ablation
      hyperparameters (group 4 / bs 4) → it is preserved untouched and is NOT the v4 run.
    - `experiments/Qwen2.5-0.5B-Instruct-...-rmin0.98-temp0.5/checkpoint-1868`: same, 0.5B
      (acc 0.4215, full eval).
    - `experiments/1.5b-v1_grpo_baseline/checkpoint-1868`: trained 2026-05-23 by a previous
      (now lost) modified training script with args
      `--variant_id v1_grpo_baseline --no_gate --gen_rep_penalty 1.0 --format_reward 0.0
      --reasoning_bonus 0.0`, group 4, bs 4, ga 4, seed 42 — matches spec §5 exactly EXCEPT
      repetition_penalty (1.0, i.e. none). `adapter_config.json` shows `modules_to_save: []`
      (LoRA only). Its `eval_results.json` covers only 100 samples → must be re-evaluated on
      the full test set.
    - wandb runs of 2026-05-24/25 show two aborted `v2_hrpo_gate_only` attempts (an older
      variant scheme); no checkpoints exist for them. The current 6-variant spec supersedes
      that scheme.

14. **`hidden_ratio` is already logged per training step.** The compiled trainer
    (`UnslothGRPOTrainer.py:1187-1202`) appends `embeds_ratio`, `hidden_ratio`,
    `completion_length`, `kl` to `_metrics`, which land in each checkpoint's
    `trainer_state.json` `log_history` alongside `reward`. Task 4 Plots 4/5/6 will read these
    directly from `trainer_state.json` — no wandb dependency.
    Note artifact: the v1 run logs `hidden_ratio ≈ 1.0` because its (bypassed) random gates
    still reported `a_t ≈ 0` for think tokens; v1 hidden ratio is 0 **by definition** and will
    be treated as such in analysis.

15. **`patch.py` weight decay on Lambda.** Spec said Λ weight decay was "already fixed to 0.0 —
    verify". **It is NOT**: `patch.py:39-45` gives the `thinking_residual_Lambda` group
    `weight_decay = self.args.weight_decay` (= 0.1). Per the HRPO paper (App. A) the 0.1 decay
    on Λ is *intentional* (pushes Λ down → hidden ratio up), and both protected checkpoints
    were trained with it. It is left at 0.1 for all variants to preserve comparability with
    the published method; no change made.

16. **`hrpo_overhead.py`** already provides: component-wise `count_parameters`,
    `estimate_gate_flops` (MAC convention), `measure_wall_clock`, `measure_memory_overhead`
    (with `torch.cuda.empty_cache()`), isolated-subprocess timing (`_run_isolated_timing`),
    `run_all_scales`, `_discover_best_checkpoint`. Task 5 will extend it with a variant
    dimension rather than duplicating it.

**Reason:** Section 0 rule 1 and Section 2 of the task spec require this mapping before any
code.

**Assumption (if any):**
- (a) The existing `1.5b-v1_grpo_baseline/checkpoint-1868` is accepted as the ablation's 1.5B
  v1 run (hyperparameters verified identical to §5 except `repetition_penalty`), per §7 rule 4
  (skip runs with a valid checkpoint). The `repetition_penalty` mismatch (1.0 vs the spec's
  1.2 for new runs) is a known confound and is flagged in the final report; retraining v1 with
  1.2 only requires deleting/renaming the directory and re-running `run_ablations.sh`.
- (b) New variant scripts implement "no gate" (v1) by setting `model.answer_start = ""` so
  `is_thinking` is always False: the blend is skipped for every token in both rollout and loss
  forward, giving numerically pure GRPO. This was chosen over weight surgery because it is
  minimal, reversible, and verifiable (embeds_ratio ≡ 1).
- (c) Evaluation must mirror the training-time gate configuration (v1: gate disabled;
  v2/v3: unit-gate patch; v4-v6: stock path). The stock `eval_gsm8k.py` is kept untouched for
  the protected checkpoints; `eval_all.py` implements variant-aware loading with the same
  decode/scoring code path.

**Verification:** all line numbers quoted above were read directly from the files in this
working tree; `training_args.bin` values dumped with torch.load for all three existing
checkpoint-1868 dirs; peft `ModulesToSaveWrapper.__getattr__` source inspected in
`.venv/lib/python3.12/site-packages/peft/utils/other.py:243-266`.

---

### [2026-07-13 01:25] Add FORMAT_REWARD / REASONING_BONUS / _is_repetition_loop to utils.py

**File(s):** utils.py

**Change:** Added module constants `FORMAT_REWARD = 0.0`, `REASONING_BONUS = 0.0`, a
character-level repetition-loop detector `_is_repetition_loop` (regex backreference over the
final 600 chars: any 8-120 char substring repeated >= 4x back-to-back), and extended
`get_reward_func(process_answer_func, format_reward=None, reasoning_bonus=None,
zero_repetition_loops=False)`. With defaults the reward is bit-identical to the previous
implementation (1.0 iff correct AND exactly one `####`, else 0.0). Shaping semantics:
`+format_reward` when `####` occurs exactly once; `+reasoning_bonus` additionally when >= 10
chars of reasoning precede `####`; `zero_repetition_loops=True` zeroes the total reward when a
loop is detected (v6).

**Reason:** Spec Section 2 found these do not exist; v5/v6 (Section 6) require them. Additive
change keeps rule 3 (baseline preservation) intact because default behaviour is unchanged.

**Assumption (if any):** The spec does not define the loop detector or bonus semantics.
Character-level detection was chosen because the paper's observed loops include CJK text
without whitespace (paper Fig. 25); thresholds (8-120 chars, >= 4 repeats, 600-char tail) flag
paper-style loops but not legitimate structured reasoning ("Step 1: ... Step 2: ...").
REASONING_BONUS rewards a non-trivial (>= 10 char) reasoning span before `####`, per the
paper's discussion of premature answer emission.

**Verification:** Unit test executed with .venv python: default rewards on 5 crafted
completions equal original semantics ([1,0,0,0,1]); v5 shaping gives [1.15,0.15,0,0,1.15];
v6 zeroing gives [...,0.0] on the loop case; detector flags "能量大"x40 and a repeated CJK
phrase, and does NOT flag normal arithmetic reasoning or enumerated steps. All assertions
passed ("ALL PASS").

---

### [2026-07-13 01:55] Task 1 — variant training infrastructure (variant_common.py + six variant scripts)

**File(s):** variant_common.py (new), hrpo_grpo_baseline.py (new), hrpo_fixed_blend.py (new),
hrpo_learned_lambda.py (new), hrpo_full_gate.py (new), hrpo_format_reward.py (new),
hrpo_full.py (new)

**Change:** Added `variant_common.py` with variant-neutral helpers (Section 5 defaults arg
parser; `experiments/{size}-{variant_id}` naming; strict `experiment_is_complete`
[checkpoint trainer_state.json with global_step == max_steps]; `get_inner_model`;
`apply_unit_gate_patch` [instance-level Eq.4 with r_t=i_t=1]; `disable_gate`
[answer_start=""]; `freeze_gate_modules`; `assert_optimizer_gate_params`;
`set_rollout_repetition_penalty`; `GateGradientLogger` [per-step L2 grad norms of
Lambda/gate_r/gate_i/LoRA to gradient_log.jsonl via on_pre_optimizer_step]; `RunLogger` +
`merge_training_log` for train_summary.json / training_log.json]. Added six standalone variant
scripts, each a full copy of the hrpo_gsm8k.py flow differing only in its Section 6 knobs:
v1 no gate (modules_to_save=None, no reset, gate disabled, no patch_trainer_optimizer);
v2 all three gate modules saved then frozen + unit gates; v3 Lambda-only saved/trained + unit
gates; v4 standard HRPO; v5 = v4 + format_reward 0.1 / reasoning_bonus 0.05;
v6 = v5 + repetition-loop zeroing. hrpo_gsm8k.py, patch.py untouched.

**Reason:** Spec Sections 6 and 7 (Task 1): independently runnable, auditable variant scripts.

**Assumption (if any):**
- v1's gate disable uses `answer_start = ""` (is_thinking always False), chosen per mapping #7
  and logged assumption (b) in the first entry.
- Discovered during smoke testing: peft's `ModulesToSaveWrapper.enable_adapters` (fired when
  the trainer's `disable_adapter()` context for reference log-probs exits) re-sets
  `requires_grad=True` on adapter copies each step. v2's freeze is nevertheless effective
  because the optimizer param groups are constructed while frozen, so the gate params are
  never registered with the optimizer. Documented in `freeze_gate_modules`; verified below.
- The final Trainer checkpoint is saved at train end even when max_steps < save_steps, so
  every completed run exposes a `checkpoint-{max_steps}` directory for evaluation.

**Verification:** End-to-end smoke test (scratchpad/smoke_one.py) ran each of the six variants
on Qwen2.5-0.5B with 4 GSM8K prompts (bs 2, group 2, 96 completion tokens) through real
rollouts, reward computation, backward and optimizer steps. Assertions all passed:
- v1: modules_to_save=[], hidden_ratio NaN at every step (blend never fired), coherent
  pure-token rollouts, no unit-gate patch.
- v2: hidden_ratio constant 0.1456 across steps (expected ~0.12 for U[0.98,0.999] init);
  Lambda tensor bit-identical after training; no thinking_residual param in any optimizer
  group.
- v3: only Lambda trainable (in optimizer), unit-gate patch active, gradient_log.jsonl has
  Lambda norm > 0 and gate_r/gate_i null.
- v4/v5/v6: stock gate, all three modules trainable, gradient log has non-null Lambda AND
  gate norms; initial hidden_ratio 0.1033 (< v2/v3, consistent with random r_t≈0.5 raising
  a_t = u^r_t).
- All variants: rollout GenerationConfig repetition_penalty == 1.2;
  train_summary.json written with peak GPU memory; optimizer param-group assertions passed
  (v1/v2 no gate params; v3 Lambda only; v4-v6 Lambda + gates).

### [2026-07-13 02:00] Task 1 — run_ablations.sh

**File(s):** run_ablations.sh (new)

**Change:** Driver accepting `--model_size [0.5b|1.5b|3b|all]` and `--variant [v1..v6|all]`
(extra args after `--` pass through to the variant scripts). Encodes the Section 6 matrix
(all six variants at 1.5b; v1/v2/v4/v6 at 0.5b and 3b), skips runs already complete via
`variant_common.experiment_is_complete`, continues past failed runs, and merges each run's
start/end time, status and peak GPU memory (from train_summary.json) into the shared
training_log.json via `variant_common.merge_training_log`.

**Reason:** Spec Section 7 (Task 1) items 1-5.

**Assumption (if any):** `_experiment_is_complete` "already present in hrpo_gsm8k.py" (spec
§7.4) does not exist there (mapping #11); the stricter check in variant_common is used
instead. The pre-existing 1.5b-v1 checkpoint is detected as complete and will be skipped
rather than retrained (see mapping assumption (a) re: its repetition_penalty=1.0).

**Verification:** `bash -n` syntax check passed; `experiment_is_complete` returns True for
`experiments/1.5b-v1_grpo_baseline` and the protected HRPO dir, False for a nonexistent dir;
arg-parser defaults verified (rp 1.2, r_min 0.98, bs 4, group 4).

---

### [2026-07-13 02:35] Task 2 — eval_gsm8k.py additive gate_mode support + eval_all.py

**File(s):** eval_gsm8k.py (modified, additive), eval_all.py (new)

**Change:** `evaluate_model` gained three optional, default-preserving parameters:
`gate_mode` ("standard" | "unit" | "disabled"; default "standard" = previous behaviour),
`adapter_path=None` support (base zero-shot), and `save_path` override. `gate_mode` mirrors the
training-time gate configuration: "disabled" (answer_start="") for v1 and base models,
"unit" (apply_unit_gate_patch) for v2/v3, "standard" for v4-v6 and the legacy checkpoints.
Metrics now additionally record base_model and gate_mode. New `eval_all.py` implements
Section 8: discovery of final checkpoints under experiments/ (both `{size}-{variant}` and
legacy `Qwen2.5-*` names), base-model zero-shot reference evals to
results/base_model_evals/{size}_base.json, missing-eval driver (an eval counts as complete
only when total >= 1319), Wilson 95% CIs, McNemar tests with continuity correction (vs v1 and
vs the immediately preceding available variant in the Section 6 chain, guarded by a
question-order pairing check), results/comparison_table.{json,csv}, and a printed summary.

**Reason:** Spec Section 8 (Task 2); rule 7 (single greedy decode per question — inherited
from evaluate_model's is_inference=True path).

**Assumption (if any):**
- Base zero-shot and v1 evals run with the gate disabled: without an HRPO-trained adapter the
  gate weights are random and the hybrid path corrupts generation. The pre-existing
  0.5b base eval (0.00 accuracy over 100) is consistent with having been run with random
  gates active; it will be re-evaluated. (New spot check: 0.5B base with gate disabled gets
  1/8 on a smoke subset.)
- "Discovers every checkpoint" is implemented as the final (highest-step) checkpoint per
  experiment; evaluating every intermediate checkpoint would triple GPU cost without adding a
  comparison-table row. Intermediate checkpoints remain evaluable by pointing eval_gsm8k.py
  at them directly.
- Verified empirically that transformers' `load_adapter` correctly loads `modules_to_save`
  gate weights (Lambda matches the adapter file up to bf16 rounding), so no custom loading is
  needed.

**Verification:**
- Unit tests: wilson_ci(868,1319) = (0.632, 0.683); wilson_ci(0,100) = (0, 0.037);
  McNemar b=15/c=5 gives chi2 = 4.05, p = 0.0442 (hand-checked); zero-discordance returns
  (0, 1).
- `eval_all.py --skip_eval` rebuilt the comparison table from existing results: 3 experiments
  discovered; the 1.5b base-vs-v1 McNemar (chi2 0.174, p 0.6767) exactly reproduces the
  values stored in the pre-existing comparison_table.json by the lost pipeline; the 100-sample
  v1 eval is correctly not paired against 1319-sample legacy evals.
- End-to-end smoke: evaluate_model on the v2 smoke checkpoint with gate_mode="unit" (2/8) and
  base zero-shot with gate_mode="disabled" (1/8), both saving well-formed eval_results.json.

---

### [2026-07-13 03:05] Task 3 — stats_utils.py + analyse_failures.py

**File(s):** stats_utils.py (new), analyse_failures.py (new), eval_all.py (imports moved)

**Change:** Moved wilson_ci/mcnemar_test into a lightweight `stats_utils.py` (so analysis
scripts don't import unsloth/torch through eval_all). Added `analyse_failures.py` per Section
9: takes two evaluated checkpoint paths (HRPO variant + v1 GRPO baseline), verifies the evals
are paired (same length + question-order probes), builds the 2x2 confusion matrix
(counts/percentages with Wilson CIs on each quadrant proportion and per-model accuracy),
sub-categorises HRPO-wrong responses (repetition_loop via _is_repetition_loop -> format
failure [`####` absent or >1] -> wrong_answer_correct_format; empty_response checked first),
computes the answer-magnitude difficulty proxy (<=10 / 11-100 / >100 with Wilson CIs per
group per model), writes the first 20 "GRPO only correct" examples as markdown with both full
responses, saves results/failure_analysis.json (keyed by comparison tag so multiple pairs
accumulate) and two plots (error-category bars; difficulty grouped bars with CI whiskers)
using a shared CVD-validated palette (plot_style.py, fixed variant->hue mapping).

**Reason:** Spec Section 9 (Task 3); plots feed Task 6.

**Assumption (if any):** "Per-category accuracy with 95% Wilson CIs" is implemented as Wilson
CIs on (a) each confusion-quadrant proportion, (b) each model's overall accuracy, and (c)
accuracy per difficulty group — covering both plausible readings. Error sub-categorisation is
reported for the primary "GRPO only correct" group and, as context, for all HRPO-wrong
responses. Loop detection takes precedence over format failure since loops usually also break
the format.

**Verification:** Synthetic paired evals (12 questions covering all four quadrants and all
four error kinds) produced the exactly expected confusion matrix (4/2/4/2), expected error
sub-categories (loop 1 / format 1 / wrong-format 2), 4 markdown examples, and both plots
(rendered and visually checked: grouped bars, CI whiskers, correct fixed colors).
categorise_error unit-checked on empty/double-tag/no-tag/loop/wrong-answer inputs;
difficulty_group on 3/50/500/-500.

---

### [2026-07-13 03:50] Task 4 — analyse_gate.py (8 plots + gradient flow + activation variance)

**File(s):** analyse_gate.py (new), plot_style.py (new), variant_common.py (VARIANT_CHAIN /
GATE_MODES moved here so eval_all.py and analyse_gate.py share one registry), eval_all.py
(imports updated)

**Change:** New analyse_gate.py implementing all Section 10 analyses, split into a static
group (no GPU: Plots 2/3/4/5/6 + gradient-flow, computed from adapter_model.safetensors
Lambda vectors, trainer_state.json log_history [which already logs per-step hidden_ratio and
reward — mapping #14] and gradient_log.jsonl) and a dynamic group (GPU: Plots 1/7/8 +
gate-activation variance, regenerating completions with return_thinking_embeds=True and
reading the per-token embeds_ratio the model already emits at
unsloth/models/llama.py:947-1025). Quantities accumulate in results/gate_analysis.json;
figures in results/plots/. plot_style.py fixes a CVD-validated palette and a permanent
variant->hue mapping used by every figure.

**Reason:** Spec Section 10 (Task 4); the original hrpo_analysis.py source is lost (mapping
#10) so this is a fresh implementation reusing the same signal sources.

**Assumption (if any):**
- Plot 2/3 use the codebase Lambda convention a = exp(-8*softplus(-Λ)) (mapping #3), NOT the
  spec's literal exp(-8*softplus(Λ)); verified numerically against the torch module
  (max |diff| < 1e-6).
- LOWESS is a local tricube-weighted linear regression implemented in numpy (statsmodels is
  not installed in .venv); recovers sin(3x) from noisy samples with mean abs error 0.0098.
- Per-token alignment: embeds_ratio[:, i] corresponds to input token input_ids[:, i]
  (derived from the _sample/forward bookkeeping; prompt positions and the final token are
  filled with ratio 1). Think span = completion tokens strictly before the first ANSWER_START
  occurrence (string-accumulated per token), which also covers v1 (whose thinking_mask is
  all-False by construction).
- Plot 7 uses the Eq.3 projected hidden states (thinking_embeds) as "think-span hidden
  states" — these are the latent representations the gate blends, and they are computed for
  all variants including v1.
- Gate-activation variance "across responses to the same question" uses multi-sample
  generation (is_inference=False, 4 samples/question by default). This is analysis
  instrumentation, not evaluation — spec rule 7 (single greedy decode) continues to hold for
  all accuracy numbers.

**Verification:**
- lambda_to_ratio matches the torch reference (exp/softplus/sqrt chain) to 1e-6.
- Static plots run on real artifacts: the protected 1.5B HRPO checkpoint yields Λ CV 0.277
  (0.5B: 0.278), final hidden ratio 0.137, and reward-vs-ratio Pearson r=0.200 (p=2.4e-18),
  Spearman r=0.081 (p=4.6e-4) over 1868 logged steps; gradient-flow plot renders from the v3
  smoke log (Lambda norms present, gate norms null as expected).
- Dynamic analyses ran end-to-end on the 0.5B smoke checkpoints: Plot 1 (6 examples;
  near-uniform ratios ~0.1033 across categories, correct for a 4-step model), Plot 8
  (v2 rows std_within == 0.0 exactly — frozen Λ + unit gates — vs v4 std_within > 0),
  activation variance (~0 for near-untrained v4), Plot 7 PCA over v1/v2/v4 (EVR 28%/12%).
  All four figures rendered and visually checked; heatmap title spacing fixed after review.

---

### [2026-07-13 12:20] Task 5 — hrpo_overhead.py variant-matrix extension

**File(s):** hrpo_overhead.py (extended; existing functions untouched)

**Change:** Added a `--variant_matrix` mode (plus `--matrix_sizes/--matrix_variants/
--gen_prompt_len/--gen_tokens/--gen_batch/--n_gen_runs/--matrix_output`) implementing spec
Section 11 over the ablation checkpoints under experiments/{size}-{variant}/:
- `variant_param_breakdown` — analytic gate parameter counts (gate_r/gate_i: h(h+1) each,
  Lambda: h) reported as total AND trainable per variant (v1: 0/0; v2: total saved but 0
  trainable; v3: Lambda only; v4-v6: all), cross-checked against the parameters actually
  saved in each adapter_model.safetensors.
- `estimate_gate_flops_variant` — the existing MAC-convention derivation (Kaplan et al. 2020,
  baseline 6·N·D) split into projection / gate-linear / Lambda+blend components and switched
  per variant (v1: zero; v2/v3: projection + Lambda/blend only, since the unit-gate patch
  bypasses the gate linears entirely; v4-v6: all).
- `_run_isolated_generation_timing` — one fresh subprocess per (size, variant) that times
  real `generate()` calls with the hybrid path ON (answer_start = U+FFF0, never decoded) vs
  OFF (answer_start = ""), decode length pinned via min_new_tokens == max_new_tokens so both
  configs execute identical step counts; peak memory recorded per run after
  torch.cuda.empty_cache() + reset_peak_memory_stats (the existing memory methodology),
  averaged over n runs; plus a microbenchmark of the Eq.3 projection einsum.
- `run_variant_matrix` — drives the loop, prints the sizes x variants wall-clock table, the
  FLOPs table and the params total/trainable breakdown, and saves
  results/overhead_matrix.json.

**Reason:** Spec Section 11 (Task 5).

**Assumption (if any):**
- Discovered while extending: the pre-existing wall-clock methodology times bare
  `model(input_ids=...)` forward passes, which never receive `is_thinking`/
  `last_thinking_states` — the Eq.4 blend NEVER executes on that path and the Eq.3 projection
  (inside the generation loop) is not exercised either, so it measured LoRA overhead only.
  The new matrix therefore times generation with the hybrid path toggled via `answer_start`,
  which is the mechanism the codebase itself uses to delimit the think span. The existing
  functions/flags are left untouched for backwards compatibility.
- The Eq.3 projection einsum runs unconditionally in the patched `_sample` for every variant
  (including v1), so the ON-vs-OFF wall-clock difference isolates the Eq.4 blend; the
  projection's own cost is reported separately via the microbenchmark
  (production runs under unsloth's bf16 autocast, so the benchmark casts to the embedding
  dtype after an initial dtype-mismatch failure was fixed).
- v1's wall-clock overhead is 0 by construction (gate disabled) and its subprocess timing is
  skipped.

**Verification:** Ran the matrix at 0.5b against the protected legacy HRPO checkpoint
(stand-in for v2/v4 rows; the session's smoke checkpoints were ephemeral): parameter table
gives total 1,608,320 = 2·(896²+896)+896 with trainable 0/0/1,608,320 for v1/v2/v4; FLOPs
overhead 0% / 7.33% / 7.41% (projection-dominated at 0.5B, consistent with the expectation
that FLOPs%% falls with model size); generation timing subprocesses completed for both gate
modes (ON/OFF deltas within noise at 3 runs x 32 tokens — real runs should use the defaults,
5 runs x 128 tokens), projection einsum measured at 0.687 ms/step; JSON saved and tables
printed.

---

### [2026-07-13 12:45] Task 6 — generate_report.py

**File(s):** generate_report.py (new)

**Change:** Report generator reading results/{comparison_table,failure_analysis,
gate_analysis,overhead_matrix}.json and producing: five LaTeX tables in
results/latex_tables/ (Table 1 ablation accuracy with Wilson CIs incl. base zero-shot and
legacy reference rows; Table 2 McNemar p-values for consecutive-chain pairs and each variant
vs v1, starred at p<0.05; Table 3 failure-mode breakdown at 1.5B; Table 4 overhead
[gate params total/trainable, FLOPs, wall-clock, memory]; Table 5 Lambda heterogeneity CVs);
two summary figures in results/plots/ (ablation accuracy bar chart with CI whiskers, one
panel per size; scale-vs-accuracy line plot per variant); an idempotent regeneration pass of
the static gate figures via analyse_gate.run_static plus a pending-list for the GPU-bound
ones; and results/key_findings.txt with one entry per Section 13 research question, each
citing the specific supporting numbers, or an explicit PENDING line naming the exact command
that will produce the missing data.

**Reason:** Spec Section 12 (Task 6).

**Assumption (if any):** Since only the 1.5B v1 checkpoint and the two legacy HRPO
checkpoints currently exist (14 ablation training runs are still to be executed on the
training machine), key_findings.txt is designed to be regenerated after every batch of runs:
statements switch automatically from PENDING to numeric findings once eval_all.py /
analyse_gate.py / analyse_failures.py / hrpo_overhead.py --variant_matrix outputs appear.
Non-ASCII Λ in variant labels is converted to $\Lambda$ for LaTeX.

**Verification:** Ran twice on current results/: all five .tex files written (booktabs
tabulars; spot-checked Table 1 shows base 0.5b 0.00 [0.0,3.7], base 1.5b 64.00 [54.2,72.7],
v1 67.00 [57.3,75.4], legacy rows 42.15/65.81 with correct CIs; no raw unicode Λ remains in
any .tex), both figures rendered and visually checked, key_findings.txt has all 12 RQ
sections with accurate PENDING markers, static gate regeneration ran (plot 5 refreshed;
plots 2/3/4/6 correctly reported pending because no v2-v6 checkpoints exist yet).

---

### [2026-07-13 12:50] Final state summary (no code change)

**File(s):** none

**Change:** All six tasks delivered and smoke-tested. Deliverables per spec Section 14:
hrpo_grpo_baseline.py, hrpo_fixed_blend.py, hrpo_learned_lambda.py, hrpo_full_gate.py,
hrpo_format_reward.py, hrpo_full.py, variant_common.py, run_ablations.sh, eval_all.py,
analyse_failures.py, analyse_gate.py, generate_report.py, stats_utils.py, plot_style.py,
changes.md; utils.py and eval_gsm8k.py extended additively (defaults preserve prior
behaviour); hrpo_overhead.py extended with --variant_matrix; hrpo_gsm8k.py and patch.py
untouched. All Python files pass py_compile; run_ablations.sh passes bash -n.

**Reason:** Spec Section 15 requires the log to close out each task.

**Assumption (if any):** none.

**Verification:** Remaining work is compute, not code: (1) `./run_ablations.sh --model_size
all --variant all` (13 runs; 1.5b-v1 skips as complete), (2) `python eval_all.py`
(re-evaluates the 100-sample v1 and stale base evals to the full 1319 set),
(3) `python analyse_gate.py --static` and `--dynamic --model_size <size>` per size,
(4) `python analyse_failures.py experiments/{size}-v4_full_hrpo_gate/checkpoint-1868
experiments/{size}-v1_grpo_baseline/checkpoint-1868` (and other pairs of interest),
(5) `python hrpo_overhead.py --variant_matrix`, (6) `python generate_report.py`.

---

### [2026-07-13 13:15] Cluster job scripts + concurrent-safe training log

**File(s):** cluster/run_variant.slurm (new), cluster/submit_all.sh (new),
cluster/run_eval_and_analysis.slurm (new), variant_common.py (merge_training_log now
flock-guarded)

**Change:** Added SLURM scripts following the Auckland cluster guidelines (single GPU,
job-name/time/output via sbatch flags, environment block with HOME/TMPDIR/HF_HOME/proxy/
TORCHDYNAMO_DISABLE, cd /data/ztay632/Dissertation-HRPO + .venv activation).
`run_variant.slurm` takes MODEL_SIZE/VARIANT_NUM via --export and delegates to
run_ablations.sh (skip logic + training_log.json). `submit_all.sh` submits the Section 6
matrix (1.5b: v1-v6; 0.5b/3b: v1,v2,v4,v6 = 14 jobs) with wall times 24h/30h/48h for
0.5b/1.5b/3b, optionally filtered to one size/variant. `run_eval_and_analysis.slurm` runs the
whole post-training pipeline (eval_all -> analyse_gate static+dynamic -> analyse_failures for
v4/v6 vs v1 per size -> hrpo_overhead --variant_matrix -> generate_report).
`merge_training_log` now takes an exclusive fcntl flock on training_log.json.lock so
concurrent jobs finishing together cannot clobber each other's entries.

**Reason:** The 14 training runs execute on the SLURM cluster; concurrent jobs share
training_log.json.

**Assumption (if any):** Wall times sized from the measured 18.5 h wall clock of the
completed 1.5B v1 run (wandb _runtime) on the local RTX 3060 Ti; cluster GPUs finish earlier
and jobs simply end. WANDB_MODE defaults to offline on the cluster (all analyses read
trainer_state.json, so nothing depends on wandb being online); override by exporting
WANDB_MODE=online before submitting.

**Verification:** bash -n passes on all three scripts; flock-guarded merge_training_log
exercised against a temp log (entry written, lock file created/cleaned).

---

### [2026-07-13 13:25] Cluster wandb mode: online by default

**File(s):** cluster/run_variant.slurm, cluster/run_eval_and_analysis.slurm

**Change:** Removed the WANDB_MODE=offline default from both SLURM scripts — the cluster
account is logged in to wandb and the proxy variables are exported, so training jobs log
online to the "latent-reasoning" project as before. Offline mode remains available by
exporting WANDB_MODE=offline before submitting.

**Reason:** User confirmed the cluster is authenticated with wandb.

**Assumption (if any):** none.

**Verification:** bash -n passes on both scripts.
