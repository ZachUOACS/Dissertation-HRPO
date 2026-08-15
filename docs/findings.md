# HRPO Fusion × Debias — findings (GSM8k-1.5B, seeds 42/43/44)

*2×2 factorial on HRPO (Qwen2.5-1.5B-Instruct, GSM8k, 934 steps, `r_min=0.95`).*
*Fusion = LT-Tuning context-prediction fusion (Change A); Debias = Dr. GRPO's two fixes (Change B).*
*All numbers below are the **fixed** codebase, three seeds. The earlier bug-confounded run-1 numbers are
retired — see §3.*

**One-line status:** one robust result (debias +2.2pp), one honest null (fusion accuracy-neutral on
GSM8k — but under-tested here, see §6), one mechanistic finding (length/accuracy trade opposite to
Dr. GRPO), and one non-result (the fusion×debias interaction is seed noise).

---

## 1 — What we tested

- **2×2 factorial** on top of HRPO, each change behind a flag (defaults = vanilla HRPO):
  - **Fusion (A):** add a *learned per-dimension gate* `β` that blends a contextual hidden state into
    HRPO's latent "thinking" channel — `h̃ = (1−β)·e_pred + β·h_I` (a learned generalization of
    LT-Tuning's fixed scalar α). This is the **novel architectural mechanism**.
  - **Debias (B):** Dr. GRPO — constant-length loss normalization + drop the reward-std division. A
    direct port of an existing fix.
- Arms: **baseline / +fusion / +debias / +fusion+debias.** The two changes hit different pipeline
  stages and share no code path, so each arm is exactly the sum of its parts.

---

## 2 — Implementation verified + code map

- **Bit-exact equivalence harness:** the new code reproduces pre-change HRPO **bit-for-bit** (`|Δ| = 0`
  on loss and gradients), and `β→−∞` collapses fusion exactly to baseline. 16/16 CPU unit tests pass.
- **Baseline reproduces the paper:** at the paper's own `r_min=0.99`, seed-42 baseline = **71.6%**
  (paper reports 72.0%). At our main `r_min=0.95` setting the 3-seed baseline is **70.8 ± 0.9%**.

**Change A — Fusion** *(input side: how each latent token is built)*

| What | Where |
|---|---|
| Learned per-dim gate `β` + `fuse_context_prediction()` → `h̃ = (1−β)·e_pred + β·h_I` | `transformers/models/qwen2/modeling_qwen2.py` |
| Apply fusion in the two forwards; capture layer-`I` hidden `h_I` during rollout | `unsloth/models/llama.py` |
| Carry `h_I` through generation + rollout buffer into the log-prob forward | `transformers/generation/utils.py`, `trl/trainer/grpo_trainer.py`, `unsloth/models/rl_replacements.py` |
| CLI flags, tied-embedding assert, trainable-gate registration + its optimizer LR group | `hrpo_gsm8k.py`, `patch.py` |

**Change B — Debias** *(gradient side: how the RL loss is aggregated — a port of Dr. GRPO)*

| What | Where |
|---|---|
| Advantage: toggle the ÷std normalization (`advantage_std_norm`) | `trl/trainer/grpo_trainer.py` |
| Loss: replace per-response length ÷\|y\| with a constant C (`length_norm`) | `unsloth/models/rl_replacements.py` |
| New scalar config flags | `trl/trainer/grpo_config.py` |

Shared, unit-tested math lives in one module `hrpo_ext.py`. Design note: `docs/fusion_debias_design.md`.

---

## 3 — Corrections applied (the run-1 story, now closed)

The first run (seed 42, `r_min=0.99`) produced misleading fusion numbers due to three bugs, **all now
fixed and verified**. Its fusion-arm numbers are retired; only its baseline/debias points survive
(they don't touch the buggy paths — see §5).

1. **Frozen fusion gate (the big one).** `thinking_residual_gate_beta` was silently excluded from the
   optimizer (a `requires_grad` omission in unsloth's PEFT re-enable loop, `unsloth/models/llama.py`),
   so `β` sat at its init (`sigmoid(−4) ≈ 0.018`) all run. **Fixed** + a startup assertion that fails
   loudly if the gate is ever un-optimized. Confirmed: `β` now trains (0.270 → 0.476), `hidden_ratio`
   0.160 → 0.182.
2. **Late collapse of fusion+debias.** Run-1 fusion+debias cratered ~step 650→850 (0.8→0.19). It was a
   **one-off instability** — did not recur in any of the three fixed-code seeds (late reward ~0.84).
3. **Eval fusion-off bug.** At inference the `h_I` recurrence carry was gated behind
   `return_thinking_embeds`, so fusion silently ran as baseline at eval. **Fixed.** Training was
   unaffected (rollouts always carry it) → trained checkpoints were valid.
4. **Under-powered latent channel.** Run 1 used `r_min=0.99` (paper's GSM8k setting) → `hidden_ratio`
   ~0.09, so fusion (which only touches the latent channel) had almost no leverage. All results below
   use **`r_min=0.95`** (`hidden_ratio` ~0.18–0.20).
5. **Seed threading.** Added `seed = args.seed` to the GRPOConfig so seeds vary the full pipeline
   (set_seed, data order, rollout sampling), not just LoRA init. Confirmed live for seeds 43/44.

---

## 4 — Headline results (3 seeds, `r_min=0.95`)

**Accuracy (GSM8k test, N=1319/seed, mean ± SD across seeds 42/43/44):**

| Arm | Fusion | Debias | Accuracy | per-seed (42/43/44) |
|---|:---:|:---:|:---:|:---:|
| **baseline** | – | – | **70.84 ± 0.88** | 71.65 / 70.96 / 69.90 |
| **debias** | – | ✓ | **73.72 ± 1.47** | 74.53 / 74.60 / 72.02 |
| **fusion** | ✓ | – | **70.53 ± 0.90** | 69.98 / 71.57 / 70.05 |
| **fusion+debias** | ✓ | ✓ | **71.97 ± 2.08** | 71.34 / 70.28 / 74.30 |

**Pooled McNemar** (exact, two-sided; discordant pairs summed across the 3 seeds):

| Contrast (A vs B) | Δ̄ (pp) | Σb | Σc | p | |
|---|:---:|:---:|:---:|:---:|:---:|
| debias vs baseline | +2.88 | 408 | 294 | **1.9e-05** | \*\*\* |
| fusion vs baseline | −0.30 | 311 | 323 | 0.66 | ns |
| fusion+debias vs baseline | +1.14 | 361 | 316 | 0.091 | ns |
| fusion+debias vs debias | −1.74 | 308 | 377 | 0.0093 | \*\* † |
| fusion vs fusion+debias | −1.44 | 351 | 408 | 0.042 | \* † |

**2×2 factorial effects** (per seed → mean ± SD):

| Effect (pp) | s42 | s43 | s44 | mean ± SD | Verdict |
|---|:---:|:---:|:---:|:---:|:---|
| **Debias main** | +2.12 | +1.18 | +3.18 | **+2.16 ± 1.01** | ✅ robust (3/3 positive) |
| Fusion main | −2.43 | −1.86 | +1.21 | −1.02 ± 1.96 | ➖ no effect (sign flips) |
| Interaction | −1.52 | −4.93 | +2.12 | −1.44 ± 3.53 | ❌ not robust (sign flips) |

† starred but **sign-inconsistent** across seeds — see §7. `b` = A correct & B wrong, `c` = A wrong &
B correct. Reproduce: `python analyze_factorial.py --glob 'runs/*-rmin0.95-b-1-full/eval_results.json'`.

---

## 5 — Finding 1 (robust): Debias is a +2.2pp win, and it is latent-channel-dependent

- **Debias gives a significant accuracy gain: +2.16 ± 1.01pp, positive in every seed,** pooled McNemar
  **p = 1.9e-05**. This is a *claim*, not an observation.
- **Latent-channel dependence (open, single-seed).** At `r_min=0.99` (run 1, seed 42) debias was
  neutral: baseline 71.57 → debias 72.10, **+0.53pp (n.s.)**. At `r_min=0.95` it is +2.2pp and
  significant. Same seed, essentially identical baseline. ⇒ **Working hypothesis: debias's benefit
  appears specifically when the latent channel carries more of the signal.** This currently rests on a
  single seed at `r_min=0.99` — the targeted confirmation (baseline+debias, seeds 43/44 at `r_min=0.99`)
  is the cheapest way to turn it into a claim.
- Caveat on novelty: debias itself is a *port* of Dr. GRPO. The dissertation-relevant part is not "the
  port works" but **why it works in HRPO specifically** — i.e. the latent-channel dependence and the
  length mechanism (§8).

---

## 6 — Finding 2 (null, but under-tested): Fusion trains correctly but is accuracy-neutral on GSM8k

- **No accuracy effect:** fusion vs baseline = −0.30pp, **p = 0.66 (n.s.)**; the per-seed main effect
  flips sign (−2.4 / −1.9 / **+1.2**). The run-1 "−2.4pp fusion hurts" was the frozen-gate confound and
  **did not replicate**.
- **The gate is genuinely working, though:** `β` trained to **0.48** — the model *chose* to use ~half
  contextual-hidden — yet top-line accuracy did not move. That is a result to **explain**, not a dead
  end.
- **Why GSM8k under-tests this mechanism:** fusion's job is to inject the *contextual* hidden state
  `h_I` into the latent channel. GSM8k is self-contained arithmetic — there is almost no external
  context to fuse, so a context-fusion mechanism having ~no effect is close to *expected*. The regime
  where it should matter is **context-rich multi-hop QA** (HotpotQA / 2WikiMQA / Bamboogle), where the
  latent tokens must integrate retrieved passages. **GSM8k is the weakest possible test of fusion.**
- **What "no accuracy change" leaves open (mechanistic questions, not yet answered):** does the learned
  *per-dimension* β specialize (by dim, by reasoning step) in ways a fixed scalar α cannot? Does fusion
  change the latent representation / which questions are solved even at equal accuracy?
  `dump_latents.py` (LT-Tuning Fig-5 feature-collapse) + per-dim β extraction answer these on the
  checkpoints we already have.

---

## 7 — Finding 3 (non-result): the fusion×debias interaction is seed noise

- The interaction is **not robust**: −1.52 / −4.93 / **+2.12** across seeds, SD (3.53) ≫ mean (1.44),
  sign flips. In seed 44, `fusion+debias` was the **best** arm (74.3%).
- ⚠️ **Statistics caution.** The pooled `fusion+debias vs debias` contrast reads p = 0.009 (\*\*), but it
  averages two negative seeds with one significantly **positive** seed (s44 +2.27, p = 0.04). **When the
  effect flips sign across strata, the pooled p overstates confidence** — the per-seed view is the
  honest one. (Good methods point: it shows we're not cherry-picking a pooled number.)
- **Meta-finding:** stacking both changes gives the **least stable** arm (widest cross-seed spread,
  71.97 ± 2.08) — a practical reason to prefer debias *alone* over the combination.

---

## 8 — Finding 4 (robust mechanism): Debias does the *opposite* of Dr. GRPO's token-efficiency

*Dr. GRPO's headline claim is token-efficiency: vanilla GRPO's `1/|y_i|` norm dilutes the per-token
penalty on long wrong answers, so incorrect responses balloon; the constant norm should stop that.*

**Baseline shows the textbook bias; debias makes it worse, not better — in all 3 seeds:**

| Seed | baseline Δincorrect | debias Δincorrect | Δ (debias−baseline) | baseline Δcorrect | debias Δcorrect |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 42 | +65 | +124 | **+59** | +1 | +54 |
| 43 | +43 | +112 | **+69** | +8 | +52 |
| 44 | +57 | +73 | **+16** | +9 | +21 |

*(Δ = mean completion length, last 15% − first 15% of training, in tokens.)*

- **Baseline reproduces the pathology** Dr. GRPO targets: correct length flat (+1..+9), incorrect
  balloons (+43..+65).
- **Debias grows length *more*** (correct and incorrect), and does **not** remove the
  incorrect-grows-faster asymmetry — the reverse of the paper, seed-robust (3/3, magnitude weakest at
  s44).
- **Mechanism (confirmed):** our constant `C = max_completion_length = 1024` (Dr. GRPO's *own* released
  choice) ≫ actual response length (~250–390 tokens) all run, so the "constant" norm makes each
  response's loss scale with length (`loss ∝ |y_i|/C`) instead of length-invariant averaging → a
  gradient that **rewards longer correct reasoning**. Net effect: **accuracy bought by spending tokens**
  — opposite to the efficiency framing.
- **Follow-up:** a **sweep of C** (running-mean response length vs the max budget). If a smaller C
  recovers the efficiency effect, the claim strengthens to "C is the control knob for the
  accuracy/length trade in HRPO." Does not change the accuracy conclusion.

---

## 9 — Status, open questions, next steps

**Solid, reportable now:**
- ✅ Implementation correct (bit-exact equivalence; baseline reproduces the paper).
- ✅ **Debias = +2.2pp on GSM8k-1.5B, 3/3 seeds, pooled p = 1.9e-05.**
- ✅ **Length/accuracy trade opposite to Dr. GRPO efficiency** — 3-seed robust; mechanism (C ≫ |y|)
  confirmed.
- ✅ **Fusion trains correctly (β→0.48) but is accuracy-neutral on GSM8k** — a fair statement of a null,
  with the caveat that GSM8k under-tests the mechanism (§6).
- ❌ Fusion×debias interaction is **not** a real effect (seed noise).

**Open (with the cheapest way to close each):**
- Latent-channel dependence of debias → baseline+debias, seeds 43/44 at `r_min=0.99` (4 cluster jobs).
- What fusion actually *does* (the β=0.48 puzzle) → `dump_latents.py` + per-dim β on existing
  checkpoints (cheap, GPU sbatch).
- Where fusion helps → run fusion on **context-rich multi-hop QA** (HotpotQA/2WikiMQA/Bamboogle), the
  regime it's designed for. This is the real test of the novel mechanism.
- The length/efficiency trade → C sweep.

**Framing for the dissertation:** the object of study is HRPO's **latent reasoning channel** — how much
of it (r_min), *what goes into it* (fusion, the novel mechanism), and how it's optimized (debias / the
C length-trade, supporting characterization). Fusion is the intended spine; the current GSM8k null is
one (weak) data point on it, not a verdict.

*Caveats: single r_min=0.99 seed for the latent-channel comparison; single-model (1.5B); GSM8k only so
far. The multi-dataset (HRPO's QA suite) and scale (3B) axes are the planned breadth.*
