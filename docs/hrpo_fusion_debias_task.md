# Task: Add LT-Tuning Context-Prediction Fusion to HRPO + Remove the two GRPO Normalization Biases (2×2 Factorial)

**Attachments required:** the HRPO paper, the "Latent Thoughts Tuning" (LT-Tuning) paper, and
the Dr. GRPO paper.

This task contains **two independent, orthogonal changes** to HRPO, implemented behind flags so
they compose into a **2×2 factorial ablation**:

| Arm | `fusion_enabled` | `length_norm` / `advantage_std_norm` | Name |
|---|---|---|---|
| 1 | `false` | `"response"` / `true` | **vanilla HRPO** (baseline) |
| 2 | `true` | `"response"` / `true` | **+fusion** |
| 3 | `false` | `"constant"` / `false` | **+debias** |
| 4 | `true` | `"constant"` / `false` | **+fusion+debias** |

The two changes hit different pipeline stages — **Change A** edits the *input-side latent
construction* (HRPO Eq. 3/4), **Change B** edits the *gradient aggregation* (HRPO Eq. 6) — so
they must not share code paths, and each must be independently toggleable with defaults that
reproduce today's behaviour bit-for-bit.

Read the relevant sections of all three papers before writing code — do not rely on this brief
alone for the derivations:

- **HRPO** (Yue et al., NeurIPS 2025, arXiv:2505.18454) — §3.2, Eq. 3 (the projection),
  Eq. 4 (the gating mechanism), Eq. 6 (the objective).
- **LT-Tuning** (Liu et al.) — §4.3 + Eq. 7 (Context-Prediction Fusion), Appendix B (theory),
  Appendix G.3 / Eq. 20 (adapter).
- **Dr. GRPO** (Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective*,
  COLM 2025) — §3.1 (the two biases), §3.2 + Listing 1 (the fix and its implementation diff),
  Appendix A (proof both terms should be absent; RLOO equivalence), Appendix C / Fig. 8 (the
  four-way ablation grid this factorial mirrors).

**Framing note so you don't misread HRPO's notation:** HRPO's Eq. 3 quantity (which the paper
calls `h_{t+1}`) is *not* a raw hidden state — it is a probability-weighted vocabulary
interpolation, i.e. the same object LT-Tuning calls `e_pred`. The channel HRPO is missing is
LT-Tuning's *other* fusion ingredient: the raw contextual hidden state.

---

## Cluster environment (READ BEFORE ANY SETUP OR RUNS)

All work happens on an ML cluster with the following non-negotiable rules. Violating the
storage rule means **files are deleted without notice**, so get this right first.

### Storage

- **Everything** — environments, datasets, checkpoints, logs, pip/conda caches, HF cache —
  lives under `/data/ztay632/`. **Never** install software or store anything in `/home`.
- Disk quota is **300 GB**. Checkpointing four factorial arms on 1.5B/3B backbones adds up:
  keep only the latest + best checkpoint per arm (rolling delete), and log a `du -sh` of the
  project directory at the start of each job so quota pressure is visible before it kills a
  run.
- Redirect temp and home at the top of **every** shell and every Slurm script:

```bash
export TMPDIR=/data/ztay632/
export HOME=/data/ztay632/
```

- Also pin the caches explicitly so nothing silently writes to the real home:

```bash
export HF_HOME=/data/ztay632/.cache/huggingface
export PIP_CACHE_DIR=/data/ztay632/.cache/pip
export TRITON_CACHE_DIR=/data/ztay632/.cache/triton
```

### Internet access (proxy)

The cluster has **no direct outbound internet**. All downloads (pip, conda, HF model/dataset
pulls) go through the proxy `squid.auckland.ac.nz:3128`:

```bash
export http_proxy=http://squid.auckland.ac.nz:3128
export https_proxy=http://squid.auckland.ac.nz:3128
```

For pip specifically, `pip install --proxy http://squid.auckland.ac.nz:3128 <pkg>` also works;
for conda, set the proxy in `~/.condarc` (which, with `HOME` redirected, lives at
`/data/<UPI>/.condarc`). Prefer downloading the Qwen2.5 checkpoints once into `HF_HOME` and
running everything after that with `HF_HUB_OFFLINE=1`, so jobs never depend on the proxy at
runtime.

### Job execution — Slurm only

**Never run training, evaluation, or notebooks directly in the SSH shell** — they may fail
silently. Every run, including the Phase 3 smoke runs, is submitted via `sbatch`. Compute-node
internet is not guaranteed even via proxy, which is another reason all model/data downloads
happen up front on the login node.

Template every arm's job on this structure:
<UPI> = ztay632
```bash
#!/bin/bash
# ======== SLURM Job Configuration ========
#SBATCH --job-name=hrpo_<arm>             # e.g. hrpo_baseline, hrpo_fusion_debias
#SBATCH --time=HH:MM:SS                   # set per run type; smoke runs short, full runs long
#SBATCH --open-mode=append
#SBATCH --output=/data/<UPI>/hrpo/logs/%x-%j.out
#SBATCH --error=/data/<UPI>/hrpo/logs/%x-%j.err
#SBATCH --gres=gpu:1                      # >1 GPU requires approval via the Slack channel bot

# ======== Environment ========
export TMPDIR=/data/<UPI>/
export HOME=/data/<UPI>/
export HF_HOME=/data/<UPI>/.cache/huggingface
export HF_HUB_OFFLINE=1

cd /data/<UPI>/<project_path>
source venv/bin/activate

# ======== Run ========
python3 train.py --config configs/<arm>.yaml
```

Submit with `sbatch run_<arm>.sh`, monitor with `squeue -u $USER`. Since each factorial arm is
an independent single-GPU job, submit all four arms as **separate sbatch jobs** and let the
scheduler parallelize them — do not chain them serially inside one job unless wall-time
math forces it.

Implications for this task:

- The **one-command ablation runner** (see below) must **generate and `sbatch`-submit** one
  Slurm script per arm, not launch python directly.
- Each arm writes its checkpoints, logs, and diagnostics to its own subdirectory under
  `/data/<UPI>/hrpo/runs/<arm>/`, so arms never clobber each other and the 2×2 comparison is
  a directory diff.
- Wall time (`--time`) must be set from the measured smoke-run throughput plus margin;
  under-asking kills the job, so make checkpoints resumable and note the resume command in
  `CHANGES.md`.

---

## Phase 1 — Explore first, do NOT edit yet

Locate and report back, with **file paths and line references**, before making any changes.
Ask rather than guess if anything is ambiguous.

**For Change A (fusion):**

1. Where the gating mechanism (HRPO Eq. 4: `r_t`, `i_t`, `a_t`, `Lambda`, `c`) is implemented.
2. Where the Eq. 3 projection (`W_e^T @ p / ||p||`) is computed.
3. Where hidden states are produced during hybrid rollout, and whether intermediate-layer
   hidden states are accessible (`output_hidden_states` / hook / return path).
4. How the gate params (`Lambda`, the linear layers) are registered, initialized, and given
   their separate learning rates, and how LoRA is applied.

**For Change B (debias):**

5. Where the per-token loss and group advantages are computed. The length normalization will
   most likely appear as a `masked_mean`-style helper — a
   `(tensor * mask).sum(dim) / mask.sum(dim)` pattern — matching Listing 1 of the Dr. GRPO
   paper. The std division will be in the advantage computation, near the group `mean`.
6. Be aware the length bias may be present in **two** places: an explicit `1/|y_i|` term and/or
   an implicit one inside a loss-reduction helper. Check both; do not divide by length twice.

**Shared:**

7. Where log-probs, advantages, and the KL-to-reference term are computed (Change A must not
   touch them; Change B touches only the advantage and the loss denominator).

Summarize all of the above and **wait for confirmation before editing anything** — the exact
form these take in this codebase may differ from the papers' listings.

---

## Phase 2A — The fusion change (Change A)

### Background

HRPO's gate currently blends the **sampled token embedding** with a **predictive vocabulary
interpolation**. It has no contextual-hidden-state channel. LT-Tuning shows that fusing a
contextual hidden state with the predictive term improves stability and mitigates feature
collapse. Add that third channel to HRPO's gate.

Target model is **Qwen2.5 with tied input/output embeddings**, so **no adapter is needed**
(LT-Tuning only uses its Eq. 20 adapter for untied-embedding models like Llama-3.1-8B).

Assert the tie at load time and log it:

```python
model.config.tie_word_embeddings  # expect True
```

If `False`, warn loudly and fall back to the adapter path (see Optional below).

### Notation at reasoning step `t`

| Symbol | Meaning | Status |
|---|---|---|
| `e_hat_{t+1}` | embedding of the sampled discrete token | HRPO, unchanged |
| `e_pred_{t+1}` | `W_e^T @ p_{t+1} / ||p_{t+1}||` | HRPO Eq. 3, unchanged |
| `h_I_t` | raw hidden state at intermediate layer `I` | **NEW** (default: penultimate) |
| `h_bar_I_t` | `h_I_t / ||h_I_t||` | **NEW** — normalized the *same way* HRPO normalizes `e_pred` in Eq. 3, to preserve scale/variance |

### New learned fusion gate (per-dimension, sigmoid)

```
beta_t = sigmoid(W_beta @ h_I_t + b_beta)
```

### Fused latent term

```
h_tilde_{t+1} = (1 - beta_t) * e_pred_{t+1} + beta_t * h_bar_I_t
```

### HRPO Eq. 4 with ONE substitution (`e_pred` → `h_tilde`), everything else identical

```
e_{t+1} = a_t * e_hat_{t+1} + sqrt(1 - a_t^2) * (i_t * h_tilde_{t+1})   if t in think
e_{t+1} = e_hat_{t+1}                                                    otherwise
```

### Optional — only if `tie_word_embeddings` is False

Implement LT-Tuning's Eq. 20 adapter behind `fusion_use_adapter`:

```
g(h) = W_up @ GELU(W_down @ h)     # bottleneck dim ~1024
```

and use `g(h_I_t)` in place of `h_bar_I_t` (still normalized).

---

## Phase 2B — The debias change (Change B)

### Background: what is being fixed and why

HRPO's policy gradient (its Eq. 6) is:

```
∇θ J_HRPO(θ) = E[ (1/g) Σ_i  (1/|y_i|)  Σ_{t=1..|y_i|} ∇θ log πθ(y_i,t | x, y_i,<t, H_i,<t) · Â_i,t ]
               − β ∇θ D_KL[πθ ‖ πref]

where   Â_i = ( r_i − mean([r_1..r_g]) ) / std([r_1..r_g])
```

This inherits **both** normalization biases that Liu et al. identify in GRPO and remove in
Dr. GRPO. Note HRPO has already dropped the PPO likelihood ratio and clipping (§3.2: "we omit
the likelihood ratio and directly use raw log probabilities"), so the *clipping*-related biases
discussed elsewhere in the RLVR literature are not in scope here. Only these two are:

**Bias 1 — response-level length bias.** The `1/|y_i|` factor divides rollout *i*'s entire
contribution by its own token count. Because `|y_i|` varies per rollout, this is a
*non-constant per-response reweighting*, not a harmless average. Since HRPO broadcasts a single
response-level `Â_i` across all tokens of response *i*, the effect is directional: for
`Â_i > 0` shorter responses get a larger per-token push (favouring terse correct answers), and
for `Â_i < 0` longer responses are penalised *less* per token (favouring verbose incorrect
answers). Dr. GRPO shows this produces the "response length keeps climbing after reward
plateaus" artifact, concentrated in incorrect responses.

**Bias 2 — question-level difficulty bias.** The `std([r_1..r_g])` denominator inside `Â_i` is
computed **per question** (over the g rollouts for a single input x), not batch-wide. Dividing
by it scales that question's whole gradient contribution by `1/std_x`, so low-variance groups
(nearly all-correct or nearly all-wrong — i.e. too easy or too hard) are up-weighted relative to
mixed, informative groups. This matters especially for HRPO, which attributes GRPO's 4.6%
underperformance vs PPO on the 1.5B backbone to "sparser rewards and limited sampled
trajectories" and shows reward collapse in its Figure 3 — exactly the low-std regime where this
bias is worst.

### The fixes (Dr. GRPO §3.2)

1. Replace `1/|y_i|` with `1/C` for a **constant** C (the generation budget / `MAX_TOKENS`), so
   every token contributes equally regardless of which rollout it came from.
2. Drop the `÷ std` entirely, keeping only mean-centering: `Â_i = r_i − mean([r_1..r_g])`.
   Liu et al. show in Appendix A that this is unbiased and equals RLOO (Ahmadian et al. 2024;
   Kool et al. 2019) up to a constant scale.

### HRPO-specific details a naive port will get wrong

- **What `|y_i|` counts.** HRPO's token sum runs over *discrete* token positions `y_i,t`; the
  hidden representations `H_i,<t` enter only as conditioning inputs to `log πθ`, and rewards
  are computed "solely on the discrete tokens within the answer span." So the length being
  normalized is the discrete-token length. Confirm this against the code and state in a comment
  whether latent steps contribute to the count in the current implementation. Do not silently
  change which positions are counted — that is a separate decision from this fix.
- **Masking.** Whatever mask currently selects trainable positions must be preserved untouched.
  Only the *denominator* changes, from a per-response sum to a constant. Prompt/pad positions
  must still be excluded from the numerator.
- **Zero-std groups.** With `advantage_std_norm: false`, the current `0/0 → 0` guard for
  all-tied groups may become dead code or may need adjusting. A fully tied group still yields a
  zero advantage under mean-centering (this fix does **not** rescue reward collapse — that is a
  reward-densification problem, out of scope here). Make sure the code path is correct and does
  not produce NaNs under either flag setting.
- **Gradient scale.** Both edits alter the gradient norm. Dr. GRPO notes "other constants also
  work with differences in gradient norm," and removing `÷std` changes the advantage scale by a
  factor absorbable into the learning rate. Log the gradient norm before/after and flag clearly
  in your summary if the existing LR or grad-clip threshold looks mismatched under the new
  scale. Do not silently retune the LR.
- **The KL term.** `−β ∇θ D_KL` is unaffected by both fixes. Leave it, and leave HRPO's
  token-IDs-only reference log-prob computation, exactly as-is.

---

## Hard constraints (both changes)

- The sampled-token branch (`a_t * e_hat_{t+1}`) must be **bit-identical** to current
  behaviour. Token sampling and log-prob computation must not change at all. Change A must not
  touch advantage standardization, the reward, the objective, or the token-ID-only KL to
  `pi_ref`; Change B touches **only** the loss denominator and the advantage formula, nothing
  else.
- Initialize `b_beta` strongly negative (e.g. `-4`) so `beta_t ≈ 0` at start. At `beta_t = 0`
  the fusion code must reduce **exactly** to current HRPO. This is a strict generalization, not
  a rewrite.
- Keep everything **strictly on-policy**: no trajectory reuse, no importance sampling, no
  ratio/clipping added back. `h_I_t` must be recomputed in the forward pass HRPO already runs
  for log-probs — do **not** store hidden states in the rollout buffer.
- `W_beta` / `b_beta` go in the same param group and learning rate as HRPO's existing gate
  linear layers, and must be LoRA-compatible / trainable alongside the existing setup.
- **Do not import LT-Tuning's training pipeline.** Its three-stage curriculum and Stage-1 CoT
  SFT are incompatible with HRPO's no-CoT RL design. Only the fusion formula transfers.
- **The two changes must not entangle.** No shared new helper may make one flag's behaviour
  depend on the other's setting. Each of the four factorial arms must be exactly the sum of its
  parts.
- All defaults preserve today's behaviour exactly, so existing runs are unaffected and every
  change is bisectable.

---

## Config flags

These are **ablation arms** — do not hardcode any of them.

```yaml
# Change A — fusion
fusion_enabled:       false        # false must reproduce baseline HRPO exactly
fusion_layer_I:       -2           # which hidden layer supplies context
fusion_gate_input:    "hidden"     # "hidden" (gate on h_I_t) | "token" (gate on e_hat)
fusion_beta_init:     -4.0         # b_beta init
fusion_use_adapter:   false        # optional untied-embedding path
fusion_normalize_h:   true

# Change B — debias
length_norm:          "response"   # "response" (current HRPO) | "constant" (Dr. GRPO fix)
advantage_std_norm:   true         # true = current HRPO; false = mean-centering only
length_norm_constant: null         # C when length_norm == "constant"; default = max gen length
```

---

## Interactions to monitor (why this is a factorial, not two silos)

Removing `÷std` rescales advantages, which changes the effective learning pressure on **all**
trainable params — including the new gate `W_beta`/`b_beta`. So:

- Track the **mean `beta_t` trajectory separately per debias condition** (arm 2 vs arm 4). If
  `beta_t` grows faster in the debiased arm, that is a real, reportable interaction, not noise.
- Log gradient norms in all four arms so any LR/grad-clip mismatch is attributable to the right
  flag.
- Arm 4's result must be interpretable as (arm 2 effect) + (arm 3 effect) + interaction; the
  logging above is what makes that decomposition possible.

---

## Phase 3 — Verification

Write tests that actually run:

1. **Equivalence (arm 1).** With all default flags, the new code must produce numerically
   identical loss and gradients to the pre-change code on a fixed batch and seed. Assert this
   in a test. Additionally: with `fusion_enabled=true` but `b_beta` forced to `-inf`, outputs
   must match the unmodified implementation to within float tolerance on a fixed seed.
2. **Shapes / dtype / device** correctness for `h_I_t` across batch and sequence dims,
   including the think-mask boundary (fusion must apply **only** inside the think span).
3. **Scale check** — log the norms of `e_pred`, `h_bar_I`, and `h_tilde`; they should be
   comparable.
4. **Gradient check** — confirm gradients flow to `W_beta`, `b_beta`, and into layer `I`, and
   confirm **no** new gradient path was introduced into the sampling / log-prob computation.
5. **Advantage unit tests** — correct mean-centering with and without std; correct handling of
   an all-tied group, an all-correct group, and an all-wrong group under both settings.
6. **Loss-reduction unit test** — construct two rollouts of very different lengths with known
   advantages and assert the per-token weighting is uniform under `"constant"` and
   length-dependent under `"response"`.
7. **Independence test** — flipping the Change A flags must not alter the advantage/loss
   denominators, and flipping the Change B flags must not alter `h_tilde` or the gate output,
   on a fixed batch.
8. **Smoke run** — a short training run on the smallest configured backbone under **all four
   factorial arms**, submitted via `sbatch` per the cluster rules (`run_ablation.sh all
   --smoke`), confirming no NaNs and that all new logs populate.

Do not claim either bias is fixed, or that fusion helps, on the basis of tests alone — the
empirical claims require the smoke/full runs and the curves below.

---

## Training-time logging (all arms)

- mean response length of **correct** rollouts
- mean response length of **incorrect** rollouts (the length-bias signature is
  incorrect-response length rising while reward plateaus — invisible without the split;
  Dr. GRPO Fig. 5)
- mean group reward std (to see how much the difficulty reweighting was doing)
- gradient norm
- mean `beta_t` over training (per arm, per the interactions section)
- a utility to dump latent representations for PCA-based feature-collapse analysis
  (LT-Tuning Figure 5 style)

---

## The 2×2 factorial — one-command ablation runner

Provide a single entry point that, per the cluster rules above, **writes a Slurm script for
the named arm and submits it with `sbatch`** (never runs python directly), e.g.:

```
./run_ablation.sh baseline          # arm 1: vanilla HRPO
./run_ablation.sh fusion            # arm 2: +fusion (gated on hidden)
./run_ablation.sh debias            # arm 3: +debias (both Dr. GRPO fixes)
./run_ablation.sh fusion_debias     # arm 4: +fusion+debias
./run_ablation.sh all               # submits all four as independent Slurm jobs
```

The runner should accept a `--smoke` flag that swaps in the short wall time and reduced step
count for the Phase 3 smoke runs, and each submission must route outputs to
`/data/<UPI>/hrpo/runs/<arm>/` as described above.

plus the secondary fusion arms for the gate-input / adapter ablation (run only on top of the
fusion arm, not the full grid):

```
run_ablation.sh fusion_token      # fusion gated on e_hat instead of h_I_t
run_ablation.sh fusion_adapter    # fusion + Eq. 20 adapter
```

Optionally also expose the two Change-B sub-arms (`len_only`, `std_only`) matching Dr. GRPO's
Fig. 8 grid, for diagnosing which bias carries the weight if arm 3 moves the metrics.

---

## Explicitly out of scope

Do not change any of these while doing this task:

- **LED / any change to the rollout decoding policy** — dropped from this round entirely.
- the reward function or reward sparsity (no PRM, no self-consistency shaping)
- the on-policy constraint — trajectories remain single-use
- the KL coefficient β or the reference-model setup
- **any length penalty in the reward.** If an accuracy-conditioned length reward is added
  later, note that Arora & Zanette report advantage normalization collapsing length *and*
  accuracy under a length penalty — which interacts with `advantage_std_norm` here. That
  interaction is a separate experiment; flag it in your summary but do not implement it now.

---

## Deliverables

1. **A short design note (markdown)** stating exactly what changed, what the equations are,
   and which parts came from which paper. Be explicit that the fusion gate is a
   *generalization* of LT-Tuning's fixed scalar `alpha` into a learned per-dimension gate —
   **not** a reimplementation of their method, and **not** equivalent to it — and that the
   debias edits are a direct port of Dr. GRPO's two fixes to HRPO's Eq. 6.
2. **The code changes** — minimal, localized, and independent per the constraints above.
3. **The tests** as described in Phase 3.
4. **The one-command ablation runner** for the four factorial arms (+ secondary fusion arms).
5. **A short `CHANGES.md`** covering: exact file/line locations of the gate, the Eq. 3
   projection, and both bias sites as found in the codebase; what each flag does; the chosen
   constant C and why; any gradient-scale concern; and the command lines for all arms.

Work incrementally. **Show the Phase 1 summary and wait for confirmation before editing
anything.**

---

## Provenance (for the writeup)

| Component | Source |
|---|---|
| Gating structure (`r_t`, `i_t`, `a_t`, `Lambda`, `sqrt(1-a_t^2)` coupling) | HRPO Eq. 4 |
| `e_pred` predictive interpolation + `/ ||·||` normalization | HRPO Eq. 3 |
| Objective / advantage being debiased | HRPO Eq. 6 |
| Fusing a contextual hidden state with the predictive term | LT-Tuning Eq. 7 |
| Adapter `W_up @ GELU(W_down @ h)` | LT-Tuning Eq. 20 (Appendix G.3), untied-embedding models only |
| Skipping the adapter on tied-embedding models | LT-Tuning (1B/3B and Qwen2.5 runs use none) |
| Constant-denominator length fix (`1/C`) | Dr. GRPO §3.2 + Listing 1 |
| Mean-centering-only advantage (no `÷std`) | Dr. GRPO §3.2; unbiasedness + RLOO equivalence in Appendix A (Ahmadian et al. 2024; Kool et al. 2019) |
| Four-way debias ablation grid | Dr. GRPO Appendix C / Fig. 8 |
| Learned **per-dimension** gate `beta_t` replacing fixed scalar `alpha` | **This work** — generalization, not from either paper |
| Conditioning `beta_t` on `h_I_t` | **This work**, motivated by LT-Tuning's `alpha†` depending on off-manifold distance `D_h` |
| The 2×2 fusion × debias factorial and per-arm `beta_t` monitoring | **This work** — experimental design, not from any source paper |

Note on correspondence: `beta_t` plays the role of LT-Tuning's `alpha` (the weight on the
hidden state), **not** `(1 - alpha)`.
