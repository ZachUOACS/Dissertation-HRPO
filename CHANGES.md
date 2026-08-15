# CHANGES — HRPO fusion (Change A) + Dr. GRPO debias (Change B)

A 2×2 factorial (fusion × debias) behind flags. All defaults reproduce vanilla HRPO bit-for-bit.
Full design + provenance: [docs/fusion_debias_design.md](docs/fusion_debias_design.md). Shared,
unit-tested math: [hrpo_ext.py](hrpo_ext.py). Tests: [tests/test_fusion_debias.py](tests/test_fusion_debias.py).

## Where things live in this codebase (as found + as edited)

| Thing | File:line |
|---|---|
| HRPO gate (Eq. 4) `a_t*e_hat + sqrt(1-a_t²)*(i_t*residual)` | [modeling_qwen2.py:594](transformers/models/qwen2/modeling_qwen2.py#L594); `Λ`/`c=8` at [:469](transformers/models/qwen2/modeling_qwen2.py#L469) |
| Eq. 3 projection `Wᵉᵀ·p/‖p‖` (`e_pred`, computed at rollout) | [generation/utils.py:3387](transformers/generation/utils.py#L3387) |
| **NEW** fusion gate `W_beta`/`b_beta` (`thinking_residual_gate_beta`) | [modeling_qwen2.py:528](transformers/models/qwen2/modeling_qwen2.py#L528) |
| **NEW** `fuse_context_prediction` (thin wrapper → `hrpo_ext`) | [modeling_qwen2.py:574](transformers/models/qwen2/modeling_qwen2.py#L574) |
| Fusion applied — training (log-prob) forward | [unsloth/models/llama.py:671](unsloth/models/llama.py#L671) |
| Fusion applied — inference (rollout) forward | [unsloth/models/llama.py:959](unsloth/models/llama.py#L959) |
| `h_I` captured at layer `I` during rollout | [unsloth/models/llama.py:1039](unsloth/models/llama.py#L1039) |
| `h_I` carried + stored per-position in `_sample` | [generation/utils.py](transformers/generation/utils.py) (`thinking_hidden`) |
| `h_I` replayed into policy forward | [rl_replacements.py `grpo_accumulated_loss`](unsloth/models/rl_replacements.py) |
| **Bias 1 site** — length norm `(loss_i*mask).sum(1)/|y_i|` | [rl_replacements.py:272](unsloth/models/rl_replacements.py#L272) (fix at :271–274) |
| **Bias 2 site** — std norm `(r−mean)/(std+1e-4)` | was `grpo_trainer.py:646`; now [grpo_trainer.py:650](trl/trainer/grpo_trainer.py#L650) via `compute_group_advantages` |
| New GRPOConfig scalar fields | [grpo_config.py:219,228,236](trl/trainer/grpo_config.py#L219) |
| Gate/beta optimizer group (name-matched `thinking_residual_gate`) | [patch.py:32](patch.py#L32) (unchanged — `beta` auto-lands here) |

Rebuild note: `PatchFastRL` rebuilds `UnslothGRPOTrainer(_UnslothGRPOTrainer)`. `_prepare_inputs`
survives (2 string swaps only) so the std edit is safe; `compute_loss` is fully replaced by
`rl_replacements.py`; new config fields are scalar so they survive. Production sites call the
`hrpo_ext.*` helpers via **call-time imports** (rebuild-safe; repo root on `sys.path`).

## Flags (all default to baseline HRPO)

**Change A — fusion** (model-level, set via `hrpo_gsm8k.py` CLI → inner `Qwen2Model` attrs)
| Flag | Default | Meaning |
|---|---|---|
| `--fusion_enabled` | off | add the contextual-hidden channel to the gate |
| `--fusion_layer_I` | `-2` | which layer supplies `h_I` (penultimate) |
| `--fusion_gate_input` | `hidden` | gate `beta` on `h_I` (`hidden`) or `e_hat` (`token`) |
| `--fusion_beta_init` | `-4.0` | `b_beta` init; `beta≈0` ⇒ start ≈ baseline |
| `--fusion_normalize_h` | on | use `h_I/‖h_I‖` |
| `--fusion_use_adapter` | off | LT-Tuning Eq. 20 adapter (untied backbones only) |

**Change B — debias** (GRPOConfig)
| Flag | Default | Meaning |
|---|---|---|
| `--length_norm` | `response` | `response` = ÷`|y_i|` (baseline); `constant` = ÷`C` (Dr. GRPO) |
| `--advantage_std_norm` / `--no-advantage_std_norm` | on | on = ÷std (baseline); off = mean-center only (Dr. GRPO) |
| `--length_norm_constant` | `None` | `C`; **None ⇒ `max_completion_length`** (=1024, the generation budget = Dr. GRPO's `MAX_TOKENS`) |

**Chosen constant `C` = `max_completion_length` (1024)** — this is HRPO's generation budget, a fixed
constant across training, exactly Dr. GRPO's `MAX_TOKENS`. Any other constant only rescales the
gradient (absorbable into the LR).

## Gradient-scale concern (read before tuning)

Both Change-B edits change the gradient norm:
- `constant` length norm: for the average correct response (`|y_i|≈167` on GSM8k-8B per LT-Tuning
  Table 6, less at 1.5B) the per-token loss weight changes from `1/|y_i|` to `1/1024`, i.e. the loss
  (and gradient) shrinks by roughly `|y_i|/C`. Expect a smaller `grad_norm`.
- dropping `÷std`: advantages rescale by `std` (∈(0,0.5] for binary rewards), so the gradient shrinks.
These are **absorbable into the learning rate** (Dr. GRPO). We log `grad_norm` (HF default) and
`reward_std` per arm. **We do not retune the LR or `max_grad_norm` (=0.1) automatically** — if the
smoke-run `grad_norm` looks mismatched under the new scale, flag it and adjust deliberately.

## Logging added (all arms)

`completion_length_correct` / `completion_length_incorrect` (the length-bias signature; Dr. GRPO
Fig. 5), `reward_std` (existing), `grad_norm` (HF default), `fusion_beta` (mean `beta_t`, fusion
arms only). Latent dump for PCA feature-collapse (LT-Tuning Fig. 5): `dump_latents.py`.

## Command lines

Tests (CPU, no GPU/model forward):
```
CUDA_VISIBLE_DEVICES="" venv/bin/python -m pytest tests/test_fusion_debias.py -q
```

Ablation arms (each **generates a Slurm script and `sbatch`-submits it**; never runs python directly):
```
./run_ablation.sh baseline            # arm 1: --  (vanilla HRPO)
./run_ablation.sh fusion              # arm 2: --fusion_enabled
./run_ablation.sh debias              # arm 3: --length_norm constant --no-advantage_std_norm
./run_ablation.sh fusion_debias       # arm 4: --fusion_enabled --length_norm constant --no-advantage_std_norm
./run_ablation.sh all                 # arms 1–4 as independent single-GPU jobs
./run_ablation.sh all --smoke         # Phase 3 smoke: short walltime + --max_steps 20, all four arms
# secondary / diagnostic:
./run_ablation.sh fusion_token        # fusion gated on e_hat
./run_ablation.sh fusion_adapter      # fusion + Eq. 20 adapter
./run_ablation.sh len_only            # Change B: constant length norm only  (Dr. GRPO Fig. 8 grid)
./run_ablation.sh std_only            # Change B: drop std norm only
# options: --model <hf_name>  --seed <n>  --dry-run
```
Everything lives under `PROJECT_ROOT` (`/data/$UPI/dissertation`): runs →
`${PROJECT_ROOT}/runs/<model>-<arm>-s<seed>-<full|smoke>/`, Slurm logs → `${PROJECT_ROOT}/logs`,
generated sbatch scripts → `${PROJECT_ROOT}/slurm_gen`, equivalence dumps → `${PROJECT_ROOT}/equiv`
(the equivalence git worktree is the one exception — it must sit outside the repo, so it goes to a
temp sibling under `$TMPDIR`). Jobs are `--requeue` and `--resume`, so **re-submitting the same
script continues from the latest checkpoint** (wall time is set from measured smoke throughput;
under-asking kills the job).

Latent dump (via sbatch on the cluster):
```
python3 dump_latents.py --adapter_path <run_dir>/checkpoint-XXX --fusion_enabled --out_path latents.npz
```

## Verification status

- Phase 3 pure-function tests: **16/16 pass on CPU** (equivalence incl. `b_beta=-inf`, shapes/dtype/
  device + think-mask boundary, scale check, gradient to `W_beta`/`b_beta` & none into detached `h_I`
  or the sampling path, advantage ±std / tied / all-correct / all-wrong, loss reduction uniform vs
  length-dependent, and both independence directions).
- Numerical arm-1 equivalence (test #1) — dumps a fixed-batch forward+backward summary from a clean
  pre-change `git worktree` (HEAD) and from the current tree, then asserts `new(baseline) == ref` and
  `new(b_beta=-inf fusion) == ref` within tolerance. One command on the cluster:
  `./run_ablation.sh equiv` (or `sbatch slurm/equivalence.sbatch`). Harness: `equivalence_harness.py`
  (the `--compare` logic is CPU-verified; the forward needs a GPU). rtol=1e-2 tolerates bf16
  cross-process kernel jitter while a real regression shifts the summary by >>1%.
- The 4-arm smoke (no NaN, all new logs populate): **`./run_ablation.sh all --smoke`** — not claimed here.

---

## Post-run fixes (after the first seed-42 factorial)

The first factorial ran and evaluated (equivalence check **passed bit-exactly**; baseline reproduced
the paper's HRPO GSM8k-1.5B = 0.716). Analysis (`analyze_factorial.py`) surfaced bugs; findings +
plots for a writeup are in [docs/findings.md](docs/findings.md). Fixes:

1. **Frozen fusion gate (root cause + fix).** In unsloth's `get_peft_model`, after PEFT wraps the
   `modules_to_save` gates, unsloth manually re-enables `requires_grad`/dtype for each
   `thinking_residual` gate — but the hard-coded list only had `gate_r`/`gate_i`/`Lambda`
   ([unsloth/models/llama.py:2497-2511](unsloth/models/llama.py#L2497-L2511)). `thinking_residual_gate_beta`
   had **no branch** → stayed `requires_grad=False` → filtered out by `patch.py` → never optimized →
   `β` frozen at init the entire run. **Fix:** added `gate_beta` (+ adapter) branches to that loop.
2. **Startup guard** ([hrpo_gsm8k.py](hrpo_gsm8k.py)): for fusion arms, builds the optimizer early and
   **asserts `gate_beta` is trainable and in an optimizer param group** — fails loudly instead of
   silently training a frozen gate.
3. **Dedicated gate LR** ([patch.py](patch.py)): `gate_beta` now has its own optimizer group,
   `--lr_residual_gate_beta` (default **1e-3**, vs 1e-4 for `gate_r`/`gate_i`), so `β` can actually
   move from a near-closed init within a ~934-step run.
4. **Eval fusion-at-inference fix** ([generation/utils.py](transformers/generation/utils.py)): the
   `h_I` recurrence carry was gated behind `return_thinking_embeds`, so at eval (which doesn't request
   the buffers) fusion silently ran as baseline. Now `h_I` is carried whenever fusion is on. **Training
   was unaffected** (rollouts always request the buffers) — checkpoints are valid.
5. **Offline model loading** ([utils.py](utils.py) `resolve_model_path`): unsloth's loader does a
   network `HfFileSystem.glob` unless `model_name` is a local dir, which broke under `HF_HUB_OFFLINE`
   on the air-gapped node. Now the repo id is resolved to its local cache snapshot (+ `use_exact_model_name`).
6. **Diagnostics:** `h_I_absmean` logged per step (rule out a zero context capture); `fusion_beta`
   already logged per arm.
7. **New runner knobs** ([run_ablation.sh](run_ablation.sh)): `--rmin` (residual_r_min; `0.99`=paper
   GSM8k config with minimal latent, `0.95`=hidden_ratio≈0.20 so fusion has leverage) and `--beta-init`
   (fusion `b_beta` init; `-1`≈β0.27, `0`≈β0.5). Both tag the run dir; the eval parser handles the tags.
8. **`sync_from_cluster.sh`**: each dir is pulled fault-tolerantly (a missing `experiments/` no longer
   aborts the whole sync under `set -e`, which is why the first sync only returned `logs/`).
9. **`eval_ablation.py`** (fusion-aware eval, per-question `qid` for McNemar) + **`analyze_factorial.py`**
   (McNemar + 2×2 decomposition).

**Recommended re-run** (fusion now learnable + given latent leverage):
```
./run_ablation.sh fusion --smoke --rmin 0.95 --beta-init -1     # confirm the fix (assert passes, beta moves)
./run_ablation.sh all --rmin 0.95 --beta-init -1                # real factorial
python analyze_factorial.py --glob 'runs/*-rmin0.95-b-1-full/eval_results.json'
```

## Second factorial run — results (seed 42, r_min=0.95, β-init −1)

Re-run completed and evaluated. **Both run-1 problems are cleared:** the fusion gate trained
(`β` 0.270 → 0.476; was frozen at 0.018), and no arm collapsed (`fusion_debias` late reward 0.839 —
the run-1 0.8→0.19 collapse did not recur). Full slide-ready writeup:
[docs/findings.md](docs/findings.md) (§6–8, formerly Slides 7–12).

Accuracy (N=1319, qid-aligned McNemar):

| Arm | Accuracy | vs baseline | p |
|---|:---:|:---:|:---:|
| baseline | 71.65% | — | — |
| debias | **74.53%** | **+2.88pp** | **0.015 \*** |
| fusion | 69.98% | −1.67pp | 0.175 n.s. |
| fusion+debias | 71.34% | −0.30pp | 0.834 n.s. |

Factorial: Debias **+2.12pp**, Fusion −2.43pp, interaction −1.52pp. `fusion_debias` vs `debias` =
−3.18pp (p=0.005 **) — fusion significantly cancels debias's gain.

Findings: (1) **Debias is a real +2.9pp win**, and it's **latent-channel-dependent** (neutral at
`r_min=0.99` in run 1, significant at `r_min=0.95`). (2) **Fusion trains correctly but gives no benefit**
and cancels debias. (3) **Debias does NOT reproduce Dr. GRPO's token-efficiency** — it *lengthens*
responses (correct grew +54 vs baseline +1; incorrect +124 vs +65). Mechanism (confirmed): the
constant `C = max_completion_length = 1024` (Dr. GRPO's own choice) ≫ actual response length ~250–390,
so `loss ∝ |y_i|/C` rewards longer correct reasoning — accuracy is bought with tokens, the opposite of
the efficiency framing. Follow-up: **seeds 43/44** + a **C sweep** (running-mean length vs max budget).

## Three-seed verdict (seeds 42/43/44, r_min=0.95) — revises the single-seed claims above

Seeds 43/44 completed + evaluated. `analyze_factorial.py` now auto-detects multiple seeds → per-seed
factorial + cross-seed mean±std + a **pooled (stratified) McNemar** (discordants summed across seeds).
Added `seed = args.seed` to the GRPOConfig ([hrpo_gsm8k.py](hrpo_gsm8k.py)) so seeds vary the *full*
pipeline (set_seed, data order, rollout sampling), not just LoRA init; seed-42 unchanged (HF default
was 42). Full writeup: [docs/findings.md](docs/findings.md) §4–7 (3-seed verdict).

| Effect (pp) | s42 | s43 | s44 | mean±std | verdict |
|---|:---:|:---:|:---:|:---:|---|
| Debias main | +2.12 | +1.18 | +3.18 | **+2.16±1.01** | ✅ robust (3/3 +, pooled p=1.9e-05) |
| Fusion main | −2.43 | −1.86 | +1.21 | −1.02±1.96 | ➖ no effect (pooled p=0.66 ns) |
| Interaction | −1.52 | −4.93 | +2.12 | −1.44±3.53 | ❌ not robust (sign flips) |

**Only the debias win survives.** "Fusion hurts" (s42) did not replicate → fusion is accuracy-neutral.
"Fusion cancels debias" did not hold → the interaction flips sign (s44's `fusion_debias` was the best
arm, 74.3%). Note: pooled `fusion_debias vs debias` = p=0.009 ** but averages two negative seeds with
one significantly **positive** seed — pooling overstates confidence when the sign flips; per-seed is the
honest read. `fusion_debias` is the widest-spread arm (71.97±2.08) → stacking both is least stable.
