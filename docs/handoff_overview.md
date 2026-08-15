# HRPO × (Context-Prediction Fusion + Dr. GRPO Debias) — Full Handoff Overview

**Purpose:** complete context transfer for an agent joining this work to help with planning.
Assumes no prior knowledge of the repo or the papers. Everything below is verified against the
current working tree (uncommitted changes on `main`) unless explicitly marked as a hypothesis.

**Repo:** `/home/zacht/dissertation` (local dev/analysis) ↔ cluster `/data/$UPI/dissertation` (all compute).

---

## 0. TL;DR — the 60-second version

A dissertation extension that takes **HRPO** (Hybrid Reasoning Policy Optimization, a latent-reasoning
RL method) and adds two independent, flag-gated changes, evaluated as a **2×2 factorial**:

- **Change A — "Fusion":** add a *contextual hidden state* channel to HRPO's latent-token gate, via a
  learned per-dimension gate `β`. A generalization of **LT-Tuning**'s fixed-scalar fusion.
- **Change B — "Debias":** remove the two normalization biases GRPO has, per **Dr. GRPO** —
  (1) per-response `1/|y|` length normalization → constant `C`; (2) `÷std` of group rewards → drop it.

Status after 3 seeds on GSM8k / Qwen2.5-1.5B-Instruct:

| Effect | Verdict |
|---|---|
| **Debias** | ✅ **+2.16 ± 1.01pp, 3/3 seeds positive, pooled McNemar p = 1.9e-05.** A real claim. |
| **Fusion** | ➖ Trains correctly (β opens 0.27→0.48), but **no measurable accuracy effect** (−1.02 ± 1.96pp, sign flips). |
| **Interaction** | ❌ Not robust (−1.44 ± 3.53pp, sign flips across seeds). |
| **Debias's *mechanism*** | ⚠️ **Does NOT reproduce Dr. GRPO's token-efficiency claim — opposite sign.** Accuracy is bought *with* tokens. Robust 3/3 seeds. This is the most interesting finding. |

The implementation is **verified bit-exact** against pre-change HRPO, and the baseline reproduces the
HRPO paper (71.6% vs paper's 72.0%).

---

## 1. Project framing

### 1.1 The base method: HRPO
HRPO (Yue et al. 2025, arXiv:2505.18454) does RL (GRPO-style) on a model that reasons partly in
**latent space** rather than purely in discrete tokens. At each "thinking" position the input embedding
for the next step is not just the sampled token's embedding — it's a **gated blend** of the sampled
token and a continuous latent quantity.

**The HRPO gate (its Eq. 4)** — verified at
[modeling_qwen2.py:594](../transformers/models/qwen2/modeling_qwen2.py#L594):

```
r_t = sigmoid(gate_r(e_hat));  i_t = sigmoid(gate_i(e_hat))
a_t = exp(-c · softplus(-Λ) · r_t)          # c = 8.0, Λ ∈ ℝ^H learned      [Λ site :469]
e_{t+1} = a_t · e_hat + sqrt(1 - a_t² + ε) · (i_t · residual)   if t ∈ think-span
e_{t+1} = e_hat                                                  otherwise
```

- `e_hat` = embedding of the **sampled discrete token**.
- `residual` = **`e_pred`**, HRPO's Eq. 3 *predictive vocabulary interpolation*
  `e_pred = Wᵉᵀ · p/‖p‖` (p = the output prob distribution). Computed **only at rollout**
  ([generation/utils.py:3387](../transformers/generation/utils.py#L3387)).
- **`hidden_ratio = sqrt(1 - a_t²)`** — the fraction of the next input coming from the latent
  channel. This is the key quantity: it measures *how latent* the model actually is.
- **`r_min`** (`--residual_r_min`) initializes `Λ` and therefore controls `hidden_ratio`.
  `r_min=0.99` → `hidden_ratio ≈ 0.09` (the paper's GSM8k-1.5B setting, **minimal latent**);
  `r_min=0.95` → `hidden_ratio ≈ 0.20` (~2.3× the latent leverage). **This turned out to matter a lot.**

⚠️ **Sign gotcha:** it is `softplus(−Λ)`, not `softplus(+Λ)`. Any analysis/plotting code that
re-derives `a_t` from `Λ` must use the negated form.

**Critical structural fact:** `e_pred` is a position-`t` *output* that feeds the position-`t+1`
*input*. That circular dependency can't be reconstructed in a single teacher-forced log-prob pass, so
HRPO **stores `e_pred` in the rollout buffer** (`thinking_embeds`) and **replays it as a detached
constant** during the loss forward. Gradient therefore never flows *through* `e_pred` into the model
body — only into the gate params stacked on top (`gate_r`, `gate_i`, `Λ`). **This precedent drives the
single most important design decision in Change A (§3.3).**

### 1.2 What this repo is
A **vendored** HRPO checkout: local copies of `transformers/`, `trl/`, `unsloth/` are *edited in place*
(the method lives inside the library code, not on top of it), plus top-level training scripts
`hrpo_gsm8k.py` / `hrpo_math.py` / `hrpo_rag.py` / `hrpo_mmlu.py`, `patch.py` (optimizer groups),
`utils.py` (prompts, answer extraction, rewards).

Note `experiments/` holds runs from **earlier, unrelated dissertation directions** (a projection +
self-distillation extension, `proj*/kd*` dirs). **This extension's runs are all in `runs/`.** Don't
confuse them.

---

## 2. The two changes at a glance

| Arm | `fusion_enabled` | `length_norm` | `advantage_std_norm` | Name |
|---|---|---|---|---|
| 1 | false | `response` | true | vanilla HRPO (baseline) |
| 2 | **true** | `response` | true | +fusion |
| 3 | false | **`constant`** | **false** | +debias |
| 4 | **true** | **`constant`** | **false** | +fusion+debias |

**Design invariant (and it holds, tested):** all defaults reproduce vanilla HRPO **bit-for-bit**, and
the two changes **share no code path**. Change A touches only the *input-side latent construction*;
Change B touches only the *advantage* and the *loss denominator*. That's what makes arm 4 legitimately
decomposable as arm2-effect + arm3-effect + interaction.

---

## 3. Change A — Context-Prediction Fusion (input side)

### 3.1 The gap it targets
HRPO's gate blends `e_hat` (sampled token) with `e_pred` (predictive vocab interpolation). It has **no
raw contextual hidden-state channel**.

📌 **Notation trap that wastes hours:** the HRPO paper calls the Eq. 3 quantity `h_{t+1}`, which *looks*
like a hidden state but **is not one** — it's a vocabulary-space interpolation, exactly what LT-Tuning
calls `e_pred`. The thing HRPO actually lacks is LT-Tuning's *other* ingredient: the raw hidden state
`h_I` from an intermediate transformer layer.

### 3.2 The math
```
h_bar_I = h_I / ‖h_I‖                             # if fusion_normalize_h (default on)
β_t     = sigmoid(W_β · gate_input + b_β)         # learned, PER-DIMENSION
h_tilde = (1 − β_t) · e_pred + β_t · h_bar_I      # the fused latent
```
Then HRPO's Eq. 4 runs **unchanged** with one substitution: `residual := h_tilde` (instead of `e_pred`).

- `gate_input` = `h_I` (default, `--fusion_gate_input hidden`) or `e_hat` (`token`).
- **`β` plays LT-Tuning's `α`** — the weight **ON the hidden state**, *not* `1−α`. Easy to invert; don't.
- `b_β` init `-4` ⇒ `β ≈ 0.018` ⇒ starts ≈ baseline. **`b_β → −∞` ⇒ `h_tilde == e_pred` exactly**
  ⇒ fusion is a **strict generalization** of baseline (this is an actual asserted test, §5).

**Relation to LT-Tuning:** LT-Tuning uses `e_fusion = α·h + (1−α)·e_pred` with a **fixed scalar** `α`.
We generalize to a **learned, per-dimension, input-conditioned** gate. We deliberately do **not** import
LT-Tuning's training pipeline — its 3-stage curriculum and Stage-1 CoT SFT are incompatible with HRPO's
no-CoT RL design. **Only the fusion formula transfers.**

### 3.3 ⭐ KEY DESIGN DECISION — how `h_I` reaches the loss forward ("Option A")
`h_I` has the *exact same* circular dependency as `e_pred` (position-`t` output → position-`t+1` input).
Two options existed:

- **Option A (CHOSEN, user decision 2026-07-25):** capture `h_I` at layer `I` during the rollout, store
  it **detached** in the rollout buffer, replay it in the log-prob forward — **symmetric with how HRPO
  already treats `e_pred`**.
- Option B (rejected): a two-pass recompute, ~2× training cost.

**Consequences you must know:**
- Gradient **does not flow into layer `I`**. All fusion learning happens in the trainable `W_β`/`b_β`
  gate stacked on top — exactly as HRPO already learns `gate_r`/`gate_i`/`Λ` on top of a detached `e_pred`.
- It is **no less on-policy** than HRPO already is (single-use trajectories, no importance sampling).
- Any spec text saying "recompute, don't store" is **superseded** by this decision.

### 3.4 Tied embeddings ⇒ no adapter
Qwen2.5 ties input/output embeddings, so no adapter is needed (matching LT-Tuning's own 1B/3B/Qwen
runs). `hrpo_gsm8k.py` **asserts `tie_word_embeddings`** at load. For untied backbones (e.g. Llama-3.1),
`--fusion_use_adapter` enables LT-Tuning's Eq. 20 adapter `g(h) = W_up · GELU(W_down · h)`
(bottleneck 1024) applied to `h_I` before normalization. **Fusion is implemented on `Qwen2Model` only**;
Llama backbones are guarded by `getattr(...)` so Llama *baseline* is safe, but Llama *fusion* would need
the modeling edit mirrored (out of scope).

### 3.5 Implementation — the `h_I` data flow end-to-end

```
ROLLOUT (generation, transformers/generation/utils.py::_sample)
  └─ per decode step, unsloth/models/llama.py inference fwd:
     ├─ capture h_I at layer I           [llama.py:995-1053]  → outputs.hidden_states[3]
     ├─ fuse using PREVIOUS step's h_I    [llama.py:952-960]  (last_hidden_I carry)
     └─ append h_I to the thinking_hidden buffer  [utils.py:3392-3412]
  └─ _sample returns a 5-TUPLE: (ids, thinking_embeds, thinking_mask, embeds_ratio, thinking_hidden)
                                                                        [utils.py:3457-3467]
TRAINER  (trl/trainer/grpo_trainer.py)
  ├─ unpack the 5th element                 [:565]
  └─ put "thinking_hidden" in the rollout dict [:711]
LOSS FORWARD (unsloth/models/rl_replacements.py)
  ├─ read inputs["thinking_hidden"]         [:475]
  ├─ thread into grpo_accumulated_loss      [:399, :512]
  └─ pass into the policy forward            [:429-430]
     └─ unsloth/models/llama.py training fwd applies fusion  [llama.py:667-674]
```

Fusion math itself lives once, in `hrpo_ext.py::context_prediction_fusion`, called from
[modeling_qwen2.py:574](../transformers/models/qwen2/modeling_qwen2.py#L574) `fuse_context_prediction()`
via a **call-time import** (see §9.1 for why that matters).

---

## 4. Change B — Dr. GRPO debias (gradient side)

HRPO's objective inherits **both** normalization biases Liu et al. identify in GRPO. HRPO already
dropped PPO ratio/clipping, so only these two are in scope. Both fixes are a **direct port**.

### 4.1 Bias 1 — response-level length bias
Vanilla: `loss_i = Σ_t(loss · mask) / |y_i|`. Dividing by the response's *own* token count reweights
rollouts by length. Dr. GRPO's argument: this **dilutes the per-token penalty on long wrong answers**,
so incorrect responses balloon in length.

**Fix:** divide by a **constant `C`** so every token contributes equally.
- Site: [rl_replacements.py:266-274](../unsloth/models/rl_replacements.py#L266) inside `grpo_compute_loss`.
- **`C` defaults to `max_completion_length` = 1024** — the generation budget, which is **Dr. GRPO's own
  released choice** (`MAX_TOKENS`). ⚠️ **This choice is the mechanism behind our most surprising result — see §7.4.**
- Only the **loss denominator** changes. Logged metrics keep the true `|y_i|`.
- `|y_i|` counts **discrete** completion tokens (to first EOS). Latent/think positions still emit a
  discrete token, so they count. The fix changes the denominator only, **not which positions are masked**.
- `C` is threaded through `grpo_compute_loss` → `UnslothEfficientGRPO` (+1 `None` in `backward`) →
  `grpo_accumulated_loss` — passed explicitly, **no closure/free variables** (source-injection safety, §9.1).

### 4.2 Bias 2 — question-level difficulty bias
Vanilla: `Â_i = (r_i − mean_g) / (std_g + 1e-4)`. Dividing by the **per-question** reward std up-weights
low-variance (too-easy / too-hard) groups.

**Fix:** drop `÷std`, keep mean-centering: `Â_i = r_i − mean_g` (Dr. GRPO Appendix A: unbiased, equals
RLOO up to a constant scale).
- Site: [grpo_trainer.py:646-651](../trl/trainer/grpo_trainer.py#L646) in `_prepare_inputs`, via
  `hrpo_ext.compute_group_advantages`.
- A fully tied group yields **exactly 0** under mean-centering — no NaN. (It does *not* rescue reward
  collapse; out of scope.)

### 4.3 Gradient scale — read before touching the LR
**Both** edits shrink the gradient norm:
- constant length norm: per-token weight goes `1/|y_i|` → `1/1024`, i.e. loss scales by `|y_i|/C`.
- dropping `÷std`: advantages rescale by `std` (∈ (0, 0.5] for binary rewards).

Dr. GRPO: these are **absorbable into the learning rate**. We log `grad_norm` and `reward_std` in every
arm so a mismatch is attributable. **We deliberately do NOT auto-retune the LR or `max_grad_norm` (0.1).**
If you change `C`, expect a gradient-scale shift and decide about the LR *deliberately*.

The KL term and HRPO's token-IDs-only reference log-probs are **untouched**.

---

## 5. Verification — why we trust the implementation

Two independent layers, both **passing**:

1. **16/16 CPU unit tests** — `tests/test_fusion_debias.py`, no GPU/model needed:
   ```
   CUDA_VISIBLE_DEVICES="" venv/bin/python -m pytest tests/test_fusion_debias.py -q
   ```
   Covers: `β→−∞` exact reduction to `e_pred`; shapes/dtype/device; think-mask boundary (only masked
   positions change); `h_bar` unit-scale; `hidden` vs `token` gate inputs differ; adapter path;
   **gradient reaches `W_β`/`b_β` and does NOT leak into the detached `h_I`**; advantages ±std / tied /
   all-correct / all-wrong / multi-group independence; loss reduction uniform-under-constant vs
   length-dependent-under-response; and **both change-independence directions** (flipping debias flags
   doesn't change fusion output, and vice versa).

2. **Numerical bit-exact equivalence on GPU** — `equivalence_harness.py` + `slurm/equivalence.sbatch`,
   run via `./run_ablation.sh equiv`. It dumps a fixed-batch forward+backward summary
   (loss / hidden_sum / logits_sum / shared_grad_norm) from a **clean pre-change `git worktree`** and from
   the current tree, then asserts `new(baseline) == ref` **and** `new(b_β=−∞ fusion) == ref`.
   Per-name-seeded param fill makes shared weights byte-identical across checkouts regardless of
   `gate_beta`'s existence. **Result: PASSED with |Δ| = 0.**

3. **External validity:** baseline reproduces the HRPO paper — **71.6% vs the paper's 72.0%** on GSM8k-1.5B.

---

## 6. How to run it

### 6.1 Flags

**Change A (model-level; CLI → inner `Qwen2Model` attrs)**

| Flag | Default | Meaning |
|---|---|---|
| `--fusion_enabled` | off | add the contextual-hidden channel |
| `--fusion_layer_I` | `-2` | which layer supplies `h_I` (penultimate) |
| `--fusion_gate_input` | `hidden` | gate `β` on `h_I` (`hidden`) or `e_hat` (`token`) |
| `--fusion_beta_init` | `-4.0` | `b_β` init (`-1`≈β0.27, `0`≈β0.5) |
| `--fusion_normalize_h` | on | use `h_I/‖h_I‖` |
| `--fusion_use_adapter` | off | Eq. 20 adapter (untied backbones only) |
| `--lr_residual_gate_beta` | `1e-3` | **dedicated** LR for `β` (vs 1e-4 for gate_r/gate_i) |

**Change B (GRPOConfig)**

| Flag | Default | Meaning |
|---|---|---|
| `--length_norm` | `response` | `response` = ÷`\|y_i\|`; `constant` = ÷`C` |
| `--advantage_std_norm` / `--no-advantage_std_norm` | on | on = ÷std; off = mean-center only |
| `--length_norm_constant` | `None` | `C`; **None ⇒ `max_completion_length` (1024)** |

**Also relevant:** `--residual_r_min` (0.99 default = paper; **0.95 = what the real runs used**),
`--seed`, `--group_size` (4), `--temperature` (0.5), `--lora_rank` (32), `--lr` (5e-6), `--beta` (KL, 0.005),
`--max_completion_length` (1024), `--max_grad_norm` (0.1), `--optimizer` (paged_adamw_8bit).

### 6.2 Commands
```bash
# training — each arm GENERATES a Slurm script and sbatch-submits it (never runs python directly)
./run_ablation.sh baseline|fusion|debias|fusion_debias|all  [--rmin 0.95] [--beta-init -1] \
                                                            [--seed N] [--smoke] [--dry-run]
./run_ablation.sh equiv           # the bit-exact equivalence check
# secondary arms: fusion_token, fusion_adapter, len_only, std_only

./run_eval.sh                     # sbatches one eval per finished run dir → eval_results.json

# analysis (LOCAL, CPU, no deps) — auto-detects multi-seed
python analyze_factorial.py --glob 'runs/*-rmin0.95-b-1-full/eval_results.json'
```
Run dirs: `${PROJECT_ROOT}/runs/<model>-<arm>-s<seed>[-rmin<r>][-b<init>]-<full|smoke>/`.
Jobs are `--requeue` + `--resume`, so **re-submitting the same script continues from the last checkpoint**.

---

## 7. Results

Setup: Qwen2.5-1.5B-Instruct, GSM8k (test N=1319), LoRA r=32, group size 4, temp 0.5, 934 steps.
All comparisons are **paired exact McNemar on qid-aligned** question sets (qid = GSM8k test row index,
stored per-question by `eval_ablation.py`).

### 7.1 Run 1 — `r_min = 0.99` (seed 42) — ⚠️ FUSION ARMS INVALID
| Arm | Accuracy | vs baseline | p |
|---|---|---|---|
| baseline | 71.57% (944) | — | — |
| debias | 72.10% (951) | +0.53pp | 0.68 n.s. |
| fusion | 69.22% (913) | −2.35pp | 0.034 *(confounded)* |
| fusion+debias | 54.44% (718) | −17.1pp | 6e-30 *(confounded)* |

Two bugs invalidated the fusion arms (**both since fixed** — see §8): the fusion gate was **frozen at
its init value** for the entire run, and `fusion_debias` **collapsed late** (reward 0.8 → 0.19 by step
850; eval ran on the post-collapse checkpoint). Only **debias-vs-baseline** was a valid read.

### 7.2 Run 2 — `r_min = 0.95`, `β-init −1`, gate fixed (seed 42) — valid
Both problems cleared: **β trained 0.270 → 0.476**, `hidden_ratio` 0.160 → 0.182, **no collapse** (all
four arms stable at ~0.83–0.85 reward; `fusion_debias` late reward 0.839).

| Arm | Accuracy | vs baseline | p |
|---|---|---|---|
| baseline | 71.65% | — | — |
| debias | **74.53%** | **+2.88pp** | **0.015 \*** |
| fusion | 69.98% | −1.67pp | 0.175 n.s. |
| fusion+debias | 71.34% | −0.30pp | 0.834 n.s. |

### 7.3 ⭐ Three-seed verdict (42/43/44, `r_min=0.95`) — **SUPERSEDES the single-seed claims**

Per-arm accuracy (GSM8k test N=1319/seed, %):

| Arm | s42 | s43 | s44 | mean ± SD |
|---|---|---|---|---|
| baseline | 71.65 | 70.96 | 69.90 | **70.84 ± 0.88** |
| debias | 74.53 | 74.60 | 72.02 | **73.72 ± 1.47** |
| fusion | 69.98 | 71.57 | 70.05 | **70.53 ± 0.90** |
| fusion+debias | 71.34 | 70.28 | 74.30 | **71.97 ± 2.08** |

2×2 factorial effects:

| Effect (pp) | s42 | s43 | s44 | mean ± std | Verdict |
|---|---|---|---|---|---|
| **Debias main** | +2.12 | +1.18 | +3.18 | **+2.16 ± 1.01** | ✅ **robust** (3/3 positive) |
| Fusion main | −2.43 | −1.86 | **+1.21** | −1.02 ± 1.96 | ➖ no effect (sign flips) |
| Interaction | −1.52 | −4.93 | **+2.12** | −1.44 ± 3.53 | ❌ not robust (sign flips) |

Pooled (stratified) McNemar — discordant pairs summed across the 3 seeds (`b` = A correct & B wrong,
`c` = A wrong & B correct; Δ̄ = mean per-seed accuracy diff):

| Contrast (A vs B) | Δ̄ (pp) | Σb | Σc | p | |
|---|---|---|---|---|---|
| debias vs baseline | +2.88 | 408 | 294 | **1.9e-05** | \*\*\* |
| fusion vs baseline | −0.30 | 311 | 323 | 0.66 | ns |
| fusion+debias vs baseline | +1.14 | 361 | 316 | 0.091 | ns |
| fusion+debias vs debias | −1.74 | 308 | 377 | 0.0093 | \*\* † |
| fusion vs fusion+debias | −1.44 | 351 | 408 | 0.042 | \* † |

† starred but **sign-inconsistent** across seeds — the pooled p overstates confidence (see caution below).

**What died between run 2 and the 3-seed verdict** (important — don't resurrect these):
- ❌ "Fusion hurts (−1.7pp)" — **did not replicate**; s44 is positive. Fusion is **accuracy-neutral**.
- ❌ "Fusion cancels debias" — the interaction **flips sign**; in s44 `fusion_debias` was the **best**
  arm (74.30%).
- ⚠️ **Statistical caution to preserve:** pooled `fusion_debias vs debias` = −1.74pp, **p = 0.009 \*\***
  *looks* significant, but it averages two negative seeds with one **significantly positive** seed
  (s44 +2.27, p = 0.04). **When the sign flips across strata, the pooled p overstates confidence.**
  Per-seed is the honest read.
- **Meta-finding:** stacking both changes gives the **least stable** arm (widest spread, 70.3–74.3).
  A reason to prefer debias *alone*.

### 7.4 ⭐ Length analysis — the Dr. GRPO efficiency signature is NOT reproduced (opposite sign)

Dr. GRPO's *headline* claim is **token-efficiency**: the `1/|y|` norm lets incorrect responses balloon;
the constant norm should stop that. Δ = growth from early (first 15%) to late (last 15%) of training,
parsed from the training logs in `logs/` (no new runs needed).

| Arm (seed 42) | correct len | incorrect len | correct **grew** | incorrect **grew** |
|---|---|---|---|---|
| baseline | 251 | 337 | **+1** | **+65** |
| debias | 295 | 390 | **+54** | **+124** |
| fusion | 243 | 308 | +8 | +46 |
| fusion+debias | 268 | 343 | +20 | +83 |

- Baseline shows the **textbook pathology**: correct length flat (+1), incorrect balloons (+65).
- **Debias does the opposite of the paper**: it *grows* length in both classes and does **not** remove
  the incorrect-grows-faster asymmetry (+70 vs baseline's +64).
- **Mechanism (confirmed, not speculation):** `C = max_completion_length = 1024` while responses are only
  ~250–390 tokens, so `C ≫ |y_i|` for the whole run. That makes each response's loss scale with
  `|y_i|/C` instead of being length-invariant → a gradient that **rewards longer correct reasoning**.
  Note `C = max budget` is **Dr. GRPO's own released choice** — this is faithful, not a bug.
- **3-seed robust** (direction holds 3/3):

  | Seed | baseline Δincorrect | debias Δincorrect | Δ(debias−base) | baseline Δcorrect | debias Δcorrect |
  |---|---|---|---|---|---|
  | 42 | +65 | +124 | **+59** | +1 | +54 |
  | 43 | +43 | +112 | **+69** | +8 | +52 |
  | 44 | +57 | +73 | **+16** | +9 | +21 |

⇒ **Reportable:** in HRPO-latent training at `r_min=0.95`, debias buys **accuracy by spending tokens** —
the opposite direction to Dr. GRPO's efficiency framing. This is arguably the thesis's most novel result.

### 7.5 Secondary hypothesis (single seed, unverified)
**Debias's benefit may be latent-channel-dependent:** neutral at `r_min=0.99` (+0.5pp n.s.), significant
at `r_min=0.95` (+2.9pp). Same seed, near-identical baseline (71.57 → 71.65); **the only change was
`r_min`**. But the `r_min=0.99` side rests on **one seed** — it's a hypothesis, not a claim.

---

## 8. Bugs found and fixed (institutional knowledge — read before debugging anything)

1. **⭐ Frozen fusion gate.** `β` sat *exactly* at `sigmoid(−4) = 0.01794` for every step of run 1.
   **Root cause:** in unsloth's `get_peft_model`, *after* PEFT wraps the `modules_to_save` gates,
   unsloth **manually re-enables `requires_grad`/dtype for each gate via a hard-coded list** — which
   only covered `gate_r`/`gate_i`/`Lambda`
   ([llama.py:2512-2523](../unsloth/models/llama.py#L2512)). `thinking_residual_gate_beta` had **no
   branch** → stayed `requires_grad=False` → filtered out by `patch.py` (which only takes
   `requires_grad` params) → **never in the optimizer**. Fixed by adding `gate_beta` + adapter branches.
   *Lesson: adding a new trainable module to this stack requires touching that hard-coded loop.*
2. **Startup guard** (`hrpo_gsm8k.py:161-178`): for fusion arms, builds the optimizer early and
   **asserts `gate_beta` is trainable AND in an optimizer param group**. Fails loudly — this class of
   bug cannot silently recur.
3. **Learnability fix.** Even once trainable, `b_β = −4` with the shared 1e-4 gate LR couldn't move `β`
   in ~934 steps (cumulative Adam movement ≈ 0.09 ≪ the +4 needed to open). Fix: `gate_beta` got its
   **own optimizer group** (`--lr_residual_gate_beta`, default **1e-3**) and runs use `--beta-init -1`.
4. **⭐ Eval-time fusion silently OFF.** In `_sample`, the `last_hidden_I` carry was inside
   `if return_thinking_embeds`. Eval doesn't request those buffers → **fusion arms evaluated as
   baseline**. Fixed: `h_I` is carried whenever `fusion_on`; buffer accumulation stays guarded.
   **Training was unaffected** (rollouts always request the buffers) → **trained checkpoints are valid,
   no retraining was needed**.
5. **Seed only affected LoRA init.** `args.seed` went to `random_state` but not `GRPOConfig.seed`, so
   seeds didn't vary data order or rollout sampling. Fixed (`hrpo_gsm8k.py:137`). Seed-42 is
   bit-unchanged (HF's default seed was already 42). *Verified live for 43/44:* at the first rollout
   LoRA `B=0` ⇒ the model is seed-independent, so a differing first-rollout reward can only come from
   data/gen RNG — s44's first reward 0.031 ≠ s42's 0.0 ⇒ the fix was active.
6. **Offline model loading.** unsloth's loader does a network `HfFileSystem.glob` unless `model_name` is
   a local dir → broke `HF_HUB_OFFLINE` on the air-gapped compute node. Fixed via
   `utils.resolve_model_path` (repo id → local cache snapshot) + `use_exact_model_name=True`.
7. **`sync_from_cluster.sh` aborted under `set -e`** when a directory was missing — silently returned
   only `logs/`. Now fault-tolerant per-directory.

---

## 9. Gotchas for anyone editing this code

### 9.1 ⭐ `PatchFastRL` rebuilds the trainer from source strings
unsloth **regenerates `UnslothGRPOTrainer` by source injection at import time**. This is the single
biggest footgun in the repo:
- `_prepare_inputs` **survives** (only 2 string swaps) → the Bias-2 edit there is safe.
- `compute_loss` is **fully replaced** by `rl_replacements.py:446` → don't edit it in `grpo_trainer.py`.
- New `GRPOConfig` fields **must be scalar** (bool/int/float/str/None) to survive the rebuild.
- Helper values must be **passed explicitly through the call chain, never captured as free variables**
  (this is why `C` is threaded through 3 functions rather than closed over).
- `hrpo_ext.py` helpers are imported at **call time**, not module top-level, so the rebuild can't
  stale-bind them. (Repo root is on `sys.path` when launching `python hrpo_*.py`.)

### 9.2 Other traps
- **Two forward paths exist** and both need fusion applied: training/log-prob
  ([llama.py:667](../unsloth/models/llama.py#L667)) and inference/rollout
  ([llama.py:952](../unsloth/models/llama.py#L952)). Bug #4 above was exactly this asymmetry.
- The inference forward packs `outputs.hidden_states` as a **list**
  `[thinking_embeds, is_thinking, embeds_ratio, h_I]` — index 3 is `h_I`, present **only** when fusion
  is on. Real per-layer states are *not* exposed during generation.
- **Fusion config is runtime state, not saved in the adapter.** Any eval script MUST re-enable it
  (`eval_ablation.py::_set_fusion_config` parses the arm from the run-dir name). It also has to walk to
  the `Qwen2Model` because **nesting differs between `get_peft_model` (train) and `load_adapter` (eval)**.
- **HRPO never calls `save_model()`** — only `checkpoint-<step>/` dirs exist. Eval must resolve the
  latest checkpoint. With `save_total_limit=2`, only the last ~2 survive — *this is why run 1's
  post-collapse eval couldn't fall back to a healthy earlier checkpoint.*
- **Debias arms eval with fusion OFF** — debias only changes the training loss, not inference.
- **McNemar pairing must key on `qid`** (stable GSM8k test row index), never enumeration order.

---

## 10. Claim status board (what you may assert)

| Statement | Status |
|---|---|
| Implementation is correct; defaults reproduce HRPO bit-exactly | ✅ **Verified** (equivalence + 16 tests) |
| Baseline reproduces the paper (71.6% vs 72.0%) | ✅ Verified |
| **Debias = +2.16 ± 1.01pp on GSM8k-1.5B, pooled p = 1.9e-05** | ✅ **Claim (3 seeds)** |
| **Debias lengthens responses (opposite of Dr. GRPO efficiency); mechanism = `C ≫ \|y\|`** | ✅ **Claim (3 seeds, direction 3/3)** |
| Fusion trains correctly (β 0.27→0.48) but is accuracy-neutral | ✅ Claim (3 seeds) |
| Fusion × debias interaction | ❌ Not robust — sign flips; do not claim |
| "Fusion hurts" / "fusion cancels debias" | ❌ **Retracted** — seed-42 artifacts |
| Debias's benefit is latent-channel-dependent (`r_min` 0.99 vs 0.95) | ⚠️ **Hypothesis** — 1 seed at 0.99 |
| Anything about Llama backbones + fusion | 🚫 Not implemented |

---

## 11. Open work — the planning surface

Roughly in order of value-per-GPU-hour:

1. **Close the `r_min` hypothesis (cheapest real win).** Run baseline + debias at **`r_min=0.99`, seeds
   43/44** — 4 jobs. Reuse run 1's s42 (baseline 71.57 / debias 72.10). This upgrades
   "debias is latent-channel-dependent" from hypothesis to a 3-seed claim, which is a genuinely
   *mechanistic* result (debias's value scales with how latent the model is).
   ```bash
   ./run_ablation.sh baseline debias --seed 43     # no --rmin ⇒ default 0.99, dirs *-s4N-full
   ./run_ablation.sh baseline debias --seed 44
   python analyze_factorial.py --glob 'runs/*-s[0-9][0-9]-full/eval_results.json'
   ```
2. **`C` sweep for debias.** Directly probes the §7.4 mechanism: replace `C = 1024` with something near
   the actual response length (e.g. a running mean, or a fixed 256/384). Prediction: a smaller `C`
   recovers Dr. GRPO's efficiency effect and *may* cost the accuracy gain — that trade-off **is** the
   finding. Needs `--length_norm_constant` (already exists) or a small running-mean implementation.
   ⚠️ Changing `C` changes the gradient scale (§4.3) — decide about the LR deliberately.
3. **Dataset expansion.** Findings need **long, variable-length CoT** to be meaningful.
   - **📌 Correction to earlier project notes:** they claim "no `train_rag.py` exists" and scope a
     port-from-scratch. **In fact `hrpo_rag.py`, `hrpo_math.py`, and `hrpo_mmlu.py` all already exist**
     (upstream HRPO scripts, complete with loaders and rewards — RAG uses `Dataset.load_from_disk` +
     `reward_func_rag` EM in `utils.py`). What they lack is **zero fusion/debias flags** (verified:
     0 occurrences in all three). So the real work is **porting the ~40-line flag/config block from
     `hrpo_gsm8k.py`** (CLI args → `modules_to_save` → inner-model attrs → `GRPOConfig` fields →
     startup guard), not writing a trainer. Much cheaper than previously scoped.
   - **Best target = the HRPO paper's own RAG datasets** (NQ, TriviaQA, HotpotQA, 2WikiMQA, Bamboogle).
     They are **retrieve-then-read single-turn** — contexts are *pre-retrieved into the dataset*, so
     there's **no live retrieval / FAISS / search server / multi-turn** at train or eval time ⇒ infra
     cost ≈ 0. Reward = EM.
   - **Thesis angle:** the §7.4 length mechanism *predicts* debias helps **more on multi-hop**
     (long reasoning: HotpotQA / 2Wiki / Bamboogle) than **single-hop** (short: NQ / TQ). That's a
     testable mechanistic story, not just breadth.
   - **🚩 GATING QUESTION (unresolved):** do `RAG_Train_Merged` and `RAG_Eval/{NQ,TQ,2Wiki,HotpotQA,Bamboogle}_Eval`
     actually exist on the cluster (`/data/$UPI/RAG_Eval/`)? **They are not present locally.** If not,
     they need offline building. Also: `eval_rag.py` is **not fusion-aware** — it needs
     `eval_ablation.py`'s `_set_fusion_config` merged in.
   - Tier 2: MATH / GSM-Hard (drop-in, same numeric reward). **AVOID MMLU/ARC/multiple-choice** —
     ~1-token answers kill the length mechanism entirely.
4. **Sub-arm attribution** (`len_only` / `std_only`, already wired). Which of the two Dr. GRPO fixes
   drives the +2.16pp? Cheap and it sharpens the claim considerably.
5. **Fusion's negative result — decide whether to invest.** It's a legitimate publishable null
   (implementation verified, gate demonstrably learns, 3 seeds). Options if pushing further: higher
   `r_min` leverage, `--fusion_gate_input token`, a different `fusion_layer_I`, or Option-B two-pass so
   gradient actually reaches layer `I`. **Recommendation: report the null and spend the GPU hours on
   #1/#2/#3** — the debias mechanism story is where the novelty is.
6. **Latent-space qualitative analysis** — `dump_latents.py` exists for PCA feature-collapse plots
   (LT-Tuning Fig. 5 style). Unused so far.

---

## 12. Operating protocol (hard constraints)

1. **Never run training, eval, or any GPU/model forward locally.** Local = code editing + CPU tests +
   analysis of synced results. All compute goes through `sbatch` on the cluster.
2. **No absolute cluster paths in committed code** — they live in `cluster_env.sh` only.
3. Save **LoRA adapters only**, never merged models; small `save_total_limit`; prune optimizer state.
4. Every run writes a config at startup (variant, git commit, GPU, torch version).
5. Sync: `sync_to_cluster.sh` (push code) → `sbatch` → `sync_from_cluster.sh` (pull `runs/`, `logs/`,
   `equiv/`, `experiments/`).
6. Model constraint: **Qwen2.5-1.5B-Instruct and 3B only** — 0.5B collapses under HRPO.
7. Wall time is set from measured smoke throughput; **under-asking kills the job** (jobs are
   `--requeue`/`--resume`, so re-submitting continues from the last checkpoint).

## 13. Where everything lives

| What | Path |
|---|---|
| Shared, unit-tested math (fusion + both debias ops) | `hrpo_ext.py` |
| Design rationale + provenance | `docs/fusion_debias_design.md` |
| Original task spec (⚠️ partly superseded — see §3.3) | `docs/hrpo_fusion_debias_task.md` |
| Slide-ready findings, all 3 runs | `docs/findings.md` |
| Change log + file:line map | `CHANGES.md` |
| Published results slide | https://claude.ai/code/artifact/1aace900-0cee-4daf-a9d3-8a4511c1c470 |
| Tests | `tests/test_fusion_debias.py` (+ `conftest.py`) |
| Equivalence harness | `equivalence_harness.py`, `slurm/equivalence.sbatch` |
| Runners | `run_ablation.sh`, `run_eval.sh`, `sync_{to,from}_cluster.sh`, `cluster_env.sh` |
| Eval + analysis | `eval_ablation.py`, `analyze_factorial.py` |
| **This extension's runs** (eval_results.json synced locally) | `runs/` |
| Training logs (source of the length analysis) | `logs/hrpo_<arm>_full-*.out` |
| Older, unrelated extension's runs | `experiments/` |
