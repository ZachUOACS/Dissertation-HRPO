# Design note: HRPO + Context-Prediction Fusion (Change A) + Dr. GRPO debias (Change B)

Two **independent, orthogonal** changes to HRPO, behind flags that compose into a 2×2 factorial
(fusion{off,on} × debias{off,on}). Every default reproduces vanilla HRPO **bit-for-bit**; the
sampled-token branch, token sampling, log-prob computation, the reward, and the token-ID-only KL to
`π_ref` are all untouched.

| Arm | `fusion_enabled` | `length_norm` / `advantage_std_norm` | Name |
|---|---|---|---|
| 1 | false | `response` / `true` | vanilla HRPO |
| 2 | true  | `response` / `true` | +fusion |
| 3 | false | `constant` / `false`| +debias |
| 4 | true  | `constant` / `false`| +fusion+debias |

---

## Change A — Context-Prediction Fusion (input-side latent construction)

**What HRPO does today.** At each latent step, HRPO's gate (its Eq. 4) blends the sampled-token
embedding `e_hat` with a predictive vocabulary interpolation `e_pred = Wᵉᵀ·p/‖p‖` (its Eq. 3).
HRPO's gate has **no contextual-hidden-state channel**. (Note on notation: HRPO calls the Eq. 3
quantity `h_{t+1}`, but it is *not* a raw hidden state — it is exactly what LT-Tuning calls
`e_pred`. The channel HRPO lacks is LT-Tuning's *other* ingredient, the raw contextual hidden
state.)

**What we add.** A third channel — the raw contextual hidden state `h_I` at an intermediate layer
`I` — fused with `e_pred` before the gate:

```
h_bar_I   = h_I / ||h_I||                            # unit-normalize (fusion_normalize_h)
beta_t    = sigmoid(W_beta · gate_input + b_beta)     # learned, per-dimension
h_tilde   = (1 - beta_t) * e_pred + beta_t * h_bar_I  # the fused latent
```

Then HRPO's Eq. 4 is applied **with one substitution** (`e_pred → h_tilde`), everything else
identical:

```
e_{t+1} = a_t * e_hat + sqrt(1 - a_t^2) * (i_t * h_tilde)   if t in think
e_{t+1} = e_hat                                              otherwise
```

`gate_input` is `h_I` (default, `fusion_gate_input="hidden"`) or `e_hat` (`"token"`).

**This is a generalization of LT-Tuning, not a reimplementation and not equivalent to it.**
LT-Tuning (its Eq. 7) uses `e_fusion = α·h + (1-α)·e_pred` with a **fixed scalar** `α`. We replace
that fixed scalar with a **learned, per-dimension, input-conditioned gate** `beta_t`. The
correspondence is `beta_t ↔ α` (the weight **on the hidden state**, *not* `1-α`). We deliberately do
**not** import LT-Tuning's training pipeline: its three-stage curriculum and Stage-1 CoT SFT are
incompatible with HRPO's no-CoT RL design. Only the fusion *formula* transfers. `b_beta` is
initialized strongly negative (default `-4`) so `beta_t ≈ 0` and training starts ≈ baseline; at
`b_beta → -inf`, `beta_t = 0` and `h_tilde = e_pred` **exactly** — fusion is a strict generalization.

**Tied embeddings ⇒ no adapter.** Qwen2.5 ties input/output embeddings, so (per LT-Tuning, whose
1B/3B and Qwen runs use none) no adapter is needed. We assert `tie_word_embeddings` at load. For
untied backbones (e.g. Llama-3.1-8B), `fusion_use_adapter` enables LT-Tuning's Eq. 20 adapter
`g(h) = W_up · GELU(W_down · h)` (bottleneck ~1024), applied to `h_I` before normalization.

**How `h_I` is obtained — Option A (store detached from the rollout).** `h_I` has the *same*
circular dependency as `e_pred`: it is a position-`t` output feeding the position-`t+1` input, which
is impossible to reconstruct inside a single teacher-forced log-prob pass. HRPO already resolves
this for `e_pred` by **storing it in the rollout buffer and replaying it as a detached constant**.
We do exactly the same for `h_I` (captured at layer `I` during the rollout's decode forward, stored
alongside `e_pred`, replayed in the log-prob forward). Consequences: (a) it is *no less* on-policy
than HRPO already is — single-use trajectories, no importance sampling; (b) **gradient does not flow
into layer `I`** (matching how `e_pred` is already a detached constant); all fusion learning is in
the trainable `W_beta`/`b_beta` gate (and the optional adapter), exactly as HRPO learns
`gate_r`/`gate_i`/`Λ` on top of a detached `e_pred`. This was a deliberate choice over a ~2× two-pass
recompute; the `W_beta`/`b_beta` gate is what makes the context channel learnable.

`W_beta`/`b_beta` live in the module `thinking_residual_gate_beta`, named so they share HRPO's
existing gate optimizer group / learning rate (`patch.py`), and are LoRA-compatible via
`modules_to_save` (added only when `fusion_enabled`).

---

## Change B — Dr. GRPO debias (gradient aggregation)

HRPO's objective (its Eq. 6) inherits **both** normalization biases that Liu et al. identify in GRPO
and remove in Dr. GRPO. HRPO already dropped the PPO ratio/clipping, so only these two are in scope.

**Bias 1 — response-level length bias.** The `1/|y_i|` per-response factor reweights rollouts by
their own token count. Fix (Dr. GRPO §3.2): divide the loss by a **constant** `C` instead, so every
token contributes equally. `C` defaults to `max_completion_length` (the generation budget =
Dr. GRPO's `MAX_TOKENS`). Only the **loss** denominator changes; logged metrics keep the true
per-response length. `|y_i|` counts *discrete* completion tokens (latent/think positions still emit a
discrete token, so they count); the fix changes the denominator only, not which positions are masked.

**Bias 2 — question-level difficulty bias.** Dividing the centered reward by the **per-question**
reward std up-weights low-variance (too-easy/too-hard) groups. Fix: drop `÷std`, keep mean-centering
`Â_i = r_i − mean(r)` (Dr. GRPO Appendix A: unbiased, equals RLOO up to a constant scale). A fully
tied group still yields exactly 0 under mean-centering (no NaN; this does not rescue reward collapse,
which is out of scope).

Both edits are a **direct port of Dr. GRPO's two fixes** to HRPO's Eq. 6. The KL term and HRPO's
token-IDs-only reference log-probs are left exactly as-is.

**Gradient scale.** Both edits change the gradient norm (Dr. GRPO: "other constants also work with
differences in gradient norm"). Removing `÷std` rescales advantages by a factor absorbable into the
learning rate. We log `grad_norm` (HF Trainer default) and `reward_std` in all arms so any
LR/grad-clip mismatch under the new scale is attributable; we do **not** silently retune the LR.

---

## Why this is a factorial (interactions)

Removing `÷std` rescales advantages, changing the learning pressure on **all** trainable params —
including the new gate `W_beta`/`b_beta`. So we log **mean `beta_t` per arm** (arm 2 vs arm 4): if
`beta_t` grows faster in the debiased arm, that is a real, reportable interaction. Arm 4 should read
as (arm 2 effect) + (arm 3 effect) + interaction; per-arm `beta_t`, `grad_norm`, `reward_std`, and
the correct/incorrect length split are what make that decomposition possible.

The two changes never share a code path: Change A touches only the input-side latent
(`context_prediction_fusion`); Change B touches only the advantage (`compute_group_advantages`) and
the loss denominator (`grpo_compute_loss`). See `hrpo_ext.py` for the shared, unit-tested math and
`CHANGES.md` for exact file/line sites.
