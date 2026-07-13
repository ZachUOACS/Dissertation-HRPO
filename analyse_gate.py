"""
analyse_gate.py — Task 4 gate behaviour analysis (spec Section 10).

Recreates (fresh — the original hrpo_analysis.py source is lost, see
changes.md mapping #10) the interpretability tooling: eight plots plus the
gradient-flow and gate-activation-variance analyses. All quantitative outputs
accumulate in results/gate_analysis.json; figures go to results/plots/.

Two groups of analyses:

STATIC (no GPU; from saved adapters / trainer_state.json / gradient logs):
  Plot 2 — Lambda distribution heterogeneity across variants (per size)
  Plot 3 — Lambda heterogeneity (CV) vs model size (per variant)
  Plot 4 — hidden-ratio trajectory per variant (per size)
  Plot 5 — final hidden ratio vs model size (per variant)
  Plot 6 — reward vs hidden ratio (scatter + LOWESS, Pearson/Spearman)
  Gradient-flow analysis — |grad Lambda| vs |grad gate_r/gate_i| per step

DYNAMIC (GPU; regenerates completions on trained checkpoints):
  Plot 1 — hidden ratio by token category (correct vs incorrect)
  Plot 7 — PCA of think-span projected hidden states across variants
  Plot 8 — per-token hidden-ratio heatmaps, v2 vs v4, same questions
  Gate-activation variance — a_t variance within/across responses (v4)

Usage:
  python analyse_gate.py --static
  python analyse_gate.py --dynamic --model_size 1.5b --n_examples 200
  python analyse_gate.py --static --dynamic --model_size 1.5b
"""

import argparse
import glob
import json
import os
import re

import numpy as np
from safetensors.torch import load_file

from stats_utils import wilson_ci  # noqa: F401 (re-exported for report use)
from plot_style import apply_style, variant_color, variant_label
from variant_common import MODEL_NAMES, VARIANT_CHAIN, GATE_MODES

plt = apply_style()

SIZES = ["0.5b", "1.5b", "3b"]
RESULTS_JSON = "results/gate_analysis.json"
PLOTS_DIR = "results/plots"

LAMBDA_KEY = "base_model.model.model.thinking_residual_Lambda.Lambda"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def lambda_to_ratio(lam: np.ndarray) -> np.ndarray:
    """Codebase convention (changes.md mapping #3): a = exp(-8*softplus(-L)),
    hidden ratio per dimension = sqrt(1 - a^2), evaluated at r_t = 1."""
    softplus_neg = np.logaddexp(0.0, -lam)          # log(1 + exp(-L)), stable
    a = np.exp(-8.0 * softplus_neg)
    return np.sqrt(np.clip(1.0 - a ** 2, 0.0, 1.0))


def find_experiments():
    """{(size, variant_id): exp_dir} for spec-named experiment dirs."""
    out = {}
    for exp_dir in sorted(glob.glob("experiments/*")):
        m = re.match(r"^(0\.5b|1\.5b|3b)-(v\d_[a-z_]+)$", os.path.basename(exp_dir))
        if m:
            out[(m.group(1), m.group(2))] = exp_dir
    return out


def final_checkpoint(exp_dir):
    ckpts = []
    for d in glob.glob(os.path.join(exp_dir, "checkpoint-*")):
        m = re.match(r"checkpoint-(\d+)$", os.path.basename(d))
        if m and os.path.exists(os.path.join(d, "adapter_model.safetensors")):
            ckpts.append((int(m.group(1)), d))
    return max(ckpts)[1] if ckpts else None


def load_lambda(checkpoint) -> np.ndarray:
    sd = load_file(os.path.join(checkpoint, "adapter_model.safetensors"))
    if LAMBDA_KEY not in sd:
        raise KeyError(f"{LAMBDA_KEY} not in {checkpoint} (v1 has no gate)")
    return sd[LAMBDA_KEY].float().numpy()


def load_history(exp_dir):
    """(steps, hidden_ratio, reward) arrays from the final checkpoint's
    trainer_state.json log_history."""
    ckpt = final_checkpoint(exp_dir)
    with open(os.path.join(ckpt, "trainer_state.json")) as f:
        history = json.load(f)["log_history"]
    steps, ratios, rewards = [], [], []
    for h in history:
        if "hidden_ratio" in h and "reward" in h:
            steps.append(h["step"])
            ratios.append(h["hidden_ratio"])
            rewards.append(h["reward"])
    return np.array(steps), np.array(ratios, dtype=float), np.array(rewards, dtype=float)


def ema(x, alpha=0.02):
    out = np.empty_like(x, dtype=float)
    m = x[0]
    for i, v in enumerate(x):
        m = alpha * v + (1 - alpha) * m
        out[i] = m
    return out


def lowess(x, y, frac=0.3):
    """Tricube-weighted local linear regression (statsmodels not installed)."""
    order = np.argsort(x)
    xs, ys = np.asarray(x)[order], np.asarray(y)[order]
    n = len(xs)
    k = max(3, int(frac * n))
    out = np.empty(n)
    for i in range(n):
        d = np.abs(xs - xs[i])
        idx = np.argsort(d)[:k]
        dmax = d[idx].max()
        w = np.ones(k) if dmax == 0 else (1 - (d[idx] / dmax) ** 3) ** 3
        A = np.stack([np.ones(len(idx)), xs[idx]], axis=1) * w[:, None]
        try:
            beta, *_ = np.linalg.lstsq(A, ys[idx] * w, rcond=None)
            out[i] = beta[0] + beta[1] * xs[i]
        except np.linalg.LinAlgError:
            out[i] = ys[idx].mean()
    return xs, out


def update_results(section: str, payload):
    os.makedirs("results", exist_ok=True)
    data = {}
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    data[section] = payload
    with open(RESULTS_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[gate_analysis.json] updated section '{section}'")


# ---------------------------------------------------------------------------
# STATIC PLOTS
# ---------------------------------------------------------------------------

def plot2_lambda_distribution(experiments, size="1.5b"):
    """Sorted per-dimension hidden-ratio curves + CV per variant with a gate."""
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    cvs = {}
    plotted = False
    for variant in VARIANT_CHAIN[1:]:   # v2..v6 (v1 has no Lambda)
        exp_dir = experiments.get((size, variant))
        if not exp_dir:
            continue
        ckpt = final_checkpoint(exp_dir)
        if not ckpt:
            continue
        ratio = lambda_to_ratio(load_lambda(ckpt))
        cv = float(np.std(ratio) / np.mean(ratio)) if np.mean(ratio) > 0 else None
        cvs[variant] = {"cv": round(cv, 5) if cv is not None else None,
                        "mean_ratio": round(float(np.mean(ratio)), 5),
                        "checkpoint": ckpt}
        ax.plot(np.sort(ratio), color=variant_color(variant),
                linestyle="--" if variant == "v2_fixed_blend" else "-",
                label=f"{variant_label(variant)} (CV={cv:.3f})")
        plotted = True
    if not plotted:
        print(f"[plot2] no gate variants with checkpoints at {size}; skipped")
        return None
    ax.set_xlabel("hidden dimension (sorted by ratio)")
    ax.set_ylabel("hidden ratio  $\\sqrt{1-a^2}$  at $r_t=1$")
    ax.set_title(f"Λ heterogeneity across variants ({size})")
    ax.legend(fontsize=7.5)
    path = os.path.join(PLOTS_DIR, f"gate_p2_lambda_distribution_{size}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot2] {path}")
    return cvs


def plot3_lambda_cv_vs_size(experiments, variant="v4_full_hrpo_gate"):
    sizes, cvs = [], []
    for size in SIZES:
        exp_dir = experiments.get((size, variant))
        if not exp_dir:
            continue
        ckpt = final_checkpoint(exp_dir)
        if not ckpt:
            continue
        ratio = lambda_to_ratio(load_lambda(ckpt))
        sizes.append(size)
        cvs.append(float(np.std(ratio) / np.mean(ratio)))
    if not sizes:
        print(f"[plot3] no {variant} checkpoints; skipped")
        return None
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(range(len(sizes)), cvs, marker="o", markersize=6,
            color=variant_color(variant))
    for i, cv in enumerate(cvs):
        ax.annotate(f"{cv:.3f}", (i, cv), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_xlabel("model size")
    ax.set_ylabel("Λ hidden-ratio CV (std/mean)")
    ax.set_title(f"Λ heterogeneity vs model size ({variant_label(variant)})")
    path = os.path.join(PLOTS_DIR, f"gate_p3_lambda_cv_vs_size_{variant}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot3] {path}")
    return dict(zip(sizes, [round(c, 5) for c in cvs]))


def plot4_hidden_ratio_trajectory(experiments, size="1.5b"):
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    finals = {}
    plotted = False
    for variant in VARIANT_CHAIN[1:]:
        exp_dir = experiments.get((size, variant))
        if not exp_dir:
            continue
        try:
            steps, ratios, _ = load_history(exp_dir)
        except (FileNotFoundError, KeyError):
            continue
        good = np.isfinite(ratios)
        if good.sum() < 10:
            continue
        c = variant_color(variant)
        ax.plot(steps[good], ratios[good], color=c, alpha=0.15, linewidth=0.7)
        ax.plot(steps[good], ema(ratios[good]), color=c,
                label=variant_label(variant))
        finals[variant] = round(float(np.mean(ratios[good][-50:])), 5)
        plotted = True
    if not plotted:
        print(f"[plot4] no trajectories at {size}; skipped")
        return None
    ax.set_xlabel("training step")
    ax.set_ylabel("mean hidden ratio")
    ax.set_title(f"Hidden-ratio trajectory during training ({size})")
    ax.legend(fontsize=7.5)
    path = os.path.join(PLOTS_DIR, f"gate_p4_hidden_ratio_trajectory_{size}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot4] {path}")
    return finals


def plot5_final_ratio_vs_size(experiments):
    variants = ["v1_grpo_baseline", "v2_fixed_blend",
                "v4_full_hrpo_gate", "v6_hrpo_full"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    payload = {}
    plotted = False
    for variant in variants:
        xs, ys = [], []
        for i, size in enumerate(SIZES):
            if variant == "v1_grpo_baseline":
                # ratio = 0 by definition (no blending) — spec Section 10.
                if (size, variant) in experiments:
                    xs.append(i)
                    ys.append(0.0)
                continue
            exp_dir = experiments.get((size, variant))
            if not exp_dir:
                continue
            try:
                _, ratios, _ = load_history(exp_dir)
            except (FileNotFoundError, KeyError):
                continue
            good = np.isfinite(ratios)
            if good.sum() < 10:
                continue
            xs.append(i)
            ys.append(float(np.mean(ratios[good][-50:])))
        if xs:
            ax.plot(xs, ys, marker="o", markersize=5,
                    color=variant_color(variant), label=variant_label(variant))
            payload[variant] = {SIZES[i]: round(y, 5) for i, y in zip(xs, ys)}
            plotted = True
    if not plotted:
        print("[plot5] nothing to plot; skipped")
        return None
    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels(SIZES)
    ax.set_xlabel("model size")
    ax.set_ylabel("final mean hidden ratio")
    ax.set_title("Hidden ratio at convergence vs model size")
    ax.legend(fontsize=7.5)
    path = os.path.join(PLOTS_DIR, "gate_p5_final_hidden_ratio_vs_size.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot5] {path}")
    return payload


def plot6_reward_vs_ratio(experiments, size="1.5b", variant="v4_full_hrpo_gate"):
    from scipy import stats as sstats
    exp_dir = experiments.get((size, variant))
    if not exp_dir:
        print(f"[plot6] {size}-{variant} not found; skipped")
        return None
    steps, ratios, rewards = load_history(exp_dir)
    good = np.isfinite(ratios) & np.isfinite(rewards)
    ratios, rewards, steps = ratios[good], rewards[good], steps[good]
    if len(ratios) < 20:
        print("[plot6] too few points; skipped")
        return None
    pearson_r, pearson_p = sstats.pearsonr(ratios, rewards)
    spearman_r, spearman_p = sstats.spearmanr(ratios, rewards)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    sc = ax.scatter(ratios, rewards, s=7, c=steps, cmap="Blues",
                    alpha=0.6, linewidths=0)
    xs, smooth = lowess(ratios, rewards, frac=0.3)
    ax.plot(xs, smooth, color="#e34948", label="LOWESS")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("training step", fontsize=8)
    ax.set_xlabel("mean hidden ratio")
    ax.set_ylabel("mean reward")
    ax.set_title(f"Reward vs hidden ratio ({size} {variant_label(variant)})\n"
                 f"Pearson r={pearson_r:.3f} (p={pearson_p:.1e}), "
                 f"Spearman r={spearman_r:.3f} (p={spearman_p:.1e})", fontsize=9)
    ax.legend(fontsize=8)
    path = os.path.join(PLOTS_DIR, f"gate_p6_reward_vs_ratio_{size}-{variant}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot6] {path}")
    return {"pearson_r": round(float(pearson_r), 4),
            "pearson_p": float(pearson_p),
            "spearman_r": round(float(spearman_r), 4),
            "spearman_p": float(spearman_p),
            "n_steps": int(len(ratios))}


def gradient_flow(experiments, size="1.5b"):
    """|grad Lambda| vs |grad gate_r/gate_i| from gradient_log.jsonl (v3/v4)."""
    payload = {}
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    plotted = False
    for ax, variant in zip(axes, ["v3_learned_lambda", "v4_full_hrpo_gate"]):
        exp_dir = experiments.get((size, variant))
        log_path = os.path.join(exp_dir, "gradient_log.jsonl") if exp_dir else None
        ax.set_title(variant_label(variant), fontsize=9)
        ax.set_xlabel("training step")
        if not log_path or not os.path.exists(log_path):
            ax.text(0.5, 0.5, "no gradient log", transform=ax.transAxes,
                    ha="center", color="#8a8a85")
            continue
        records = [json.loads(l) for l in open(log_path)]
        steps = [r["step"] for r in records]
        summary = {}
        for key, color in (("Lambda", variant_color("v3_learned_lambda")),
                           ("gate_r", variant_color("v4_full_hrpo_gate")),
                           ("gate_i", variant_color("v6_hrpo_full"))):
            vals = np.array([r[key] if r[key] is not None else np.nan
                             for r in records], dtype=float)
            if np.isfinite(vals).any():
                ax.semilogy(steps, ema(np.nan_to_num(vals, nan=0.0), 0.05),
                            color=color, label=f"|grad {key}|")
                summary[key] = {"mean": float(np.nanmean(vals)),
                                "median": float(np.nanmedian(vals))}
            else:
                summary[key] = None
        if summary.get("Lambda") and summary.get("gate_r"):
            summary["gate_to_lambda_ratio"] = (
                (summary["gate_r"]["mean"] + (summary["gate_i"]["mean"]
                 if summary.get("gate_i") else 0.0))
                / max(summary["Lambda"]["mean"], 1e-12))
        payload[variant] = summary
        ax.legend(fontsize=7.5)
        plotted = True
    axes[0].set_ylabel("gradient L2 norm (log)")
    fig.suptitle(f"Gradient flow: Λ vs input gates ({size})", fontsize=10)
    if plotted:
        path = os.path.join(PLOTS_DIR, f"gate_gradient_flow_{size}.png")
        fig.savefig(path)
        print(f"[gradient_flow] {path}")
    plt.close(fig)
    return payload or None


# ---------------------------------------------------------------------------
# DYNAMIC ANALYSES (GPU)
# ---------------------------------------------------------------------------

def token_category(token_str: str) -> str:
    t = token_str.strip()
    if not t:
        return "other"
    if any(ch.isdigit() for ch in t):
        return "number"
    if re.fullmatch(r"[+\-*/=×÷^%<>]+", t):
        return "operator"
    if re.fullmatch(r"[A-Za-z']+", t):
        return "word"
    if re.fullmatch(r"[^\w\s]+", t):
        return "punctuation"
    return "other"


TOKEN_CATEGORIES = ["number", "operator", "word", "punctuation", "other"]


def load_model_for_analysis(size, checkpoint, gate_mode):
    """Mirror eval_gsm8k.evaluate_model's model setup (mapping #7)."""
    from unsloth import FastLanguageModel
    from utils import ANSWER_START
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAMES[size], max_seq_length=1024,
        load_in_4bit=False, fast_inference=False)
    model.answer_start = ANSWER_START
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    if checkpoint is not None:
        model.load_adapter(checkpoint)
    if gate_mode == "disabled":
        model.answer_start = ""
    elif gate_mode == "unit":
        from variant_common import apply_unit_gate_patch
        apply_unit_gate_patch(model)
    model = FastLanguageModel.for_inference(model)
    return model, tokenizer


def collect_generation_data(size, checkpoint, gate_mode, questions, answers,
                            temperature=0.5, max_new_tokens=512, batch_size=8,
                            greedy=True, keep_embeds=False):
    """Generate completions with return_thinking_embeds=True and return, per
    example: token ids, per-token hidden ratio (sqrt(1-embeds_ratio^2)),
    think mask (True before the first ANSWER_START token), correctness, and
    optionally the projected hidden states (thinking_embeds, cpu fp16)."""
    import torch
    from transformers import GenerationConfig
    from utils import (ANSWER_START, SYSTEM_PROMPT, extract_from_response,
                       process_gsm8k_answer)

    model, tokenizer = load_model_for_analysis(size, checkpoint, gate_mode)
    records = []
    for start in range(0, len(questions), batch_size):
        qs = questions[start:start + batch_size]
        golds = answers[start:start + batch_size]
        prompts = [tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": q.strip()}],
            tokenize=False, add_generation_prompt=True) for q in qs]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        padding_side="left", add_special_tokens=False)
        ids = enc["input_ids"].to(model.device)
        mask = enc["attention_mask"].to(model.device)
        p_len = ids.size(1)
        out_ids, thinking_embeds, thinking_mask, embeds_ratio = model.generate(
            ids, attention_mask=mask,
            generation_config=GenerationConfig(
                do_sample=True, temperature=temperature,
                max_new_tokens=max_new_tokens),
            processing_class=tokenizer,
            is_inference=greedy,
            return_thinking_embeds=True,
        )
        for j in range(out_ids.size(0)):
            comp_ids = out_ids[j, p_len:]
            text = tokenizer.decode(comp_ids)
            text = text.split(tokenizer.special_tokens_map["eos_token"])[0]
            pred = process_gsm8k_answer(extract_from_response(text))
            gold = process_gsm8k_answer(golds[j])
            ratio_row = embeds_ratio[j, p_len:].float().cpu().numpy()
            hidden_row = np.sqrt(np.clip(1.0 - ratio_row ** 2, 0.0, 1.0))
            # Think span: tokens strictly before the first ANSWER_START token,
            # bounded by the effective (pre-eos) completion length.
            toks = tokenizer.convert_ids_to_tokens(comp_ids.tolist())
            n_eff = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            think_end = n_eff
            running = ""
            for k, tid in enumerate(comp_ids.tolist()[:n_eff]):
                running += tokenizer.decode([tid])
                if ANSWER_START in running:
                    think_end = k
                    break
            rec = {
                "question": qs[j],
                "correct": pred == gold and pred != "",
                "token_ids": comp_ids.tolist()[:n_eff],
                "token_strs": [tokenizer.decode([t]) for t in comp_ids.tolist()[:n_eff]],
                "hidden_ratio": hidden_row[:n_eff],
                "think_end": think_end,
            }
            if keep_embeds:
                rec["think_embeds"] = (
                    thinking_embeds[j, p_len:p_len + think_end]
                    .float().cpu().numpy().astype(np.float16))
            records.append(rec)
        del out_ids, thinking_embeds, thinking_mask, embeds_ratio
        torch.cuda.empty_cache()
        print(f"  generated {min(start + batch_size, len(questions))}/{len(questions)}")
    del model
    torch.cuda.empty_cache()
    return records


def get_questions(n, seed=42):
    from datasets import load_dataset
    from utils import extract_hash_answer
    ds = load_dataset("openai/gsm8k", "main")["test"]
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return list(ds["question"]), [extract_hash_answer(a) for a in ds["answer"]]


def plot1_ratio_by_token_category(size, variant, n_examples, batch_size):
    experiments = find_experiments()
    exp_dir = experiments.get((size, variant))
    if not exp_dir:
        print(f"[plot1] {size}-{variant} missing; skipped")
        return None
    ckpt = final_checkpoint(exp_dir)
    questions, answers = get_questions(n_examples)
    print(f"[plot1] generating on {len(questions)} questions with {ckpt}")
    records = collect_generation_data(size, ckpt, GATE_MODES[variant],
                                      questions, answers,
                                      batch_size=batch_size)
    buckets = {(cat, corr): [] for cat in TOKEN_CATEGORIES for corr in (True, False)}
    for rec in records:
        for k in range(rec["think_end"]):
            cat = token_category(rec["token_strs"][k])
            buckets[(cat, rec["correct"])].append(float(rec["hidden_ratio"][k]))
    payload = {}
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    width = 0.38
    x = np.arange(len(TOKEN_CATEGORIES))
    from plot_style import CORRECT_COLOR, INCORRECT_COLOR
    for off, corr, color, lab in ((-width / 2, True, CORRECT_COLOR, "correct"),
                                  (width / 2, False, INCORRECT_COLOR, "incorrect")):
        means, errs, ns = [], [], []
        for cat in TOKEN_CATEGORIES:
            vals = buckets[(cat, corr)]
            means.append(np.mean(vals) if vals else 0.0)
            errs.append((1.96 * np.std(vals) / max(1, len(vals)) ** 0.5) if vals else 0.0)
            ns.append(len(vals))
            payload[f"{cat}_{'correct' if corr else 'incorrect'}"] = {
                "mean": round(float(means[-1]), 5), "n_tokens": len(vals)}
        ax.bar(x + off, means, width - 0.03, yerr=errs, capsize=2,
               color=color, label=lab,
               error_kw={"linewidth": 0.9, "ecolor": "#3a3a37"})
    ax.set_xticks(x)
    ax.set_xticklabels(TOKEN_CATEGORIES)
    ax.set_ylabel("mean hidden ratio")
    ax.set_title(f"Hidden ratio by token category "
                 f"({size} {variant_label(variant)}, think span, "
                 f"n={len(records)} examples)", pad=22, fontsize=9)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, fontsize=8)
    path = os.path.join(PLOTS_DIR, f"gate_p1_ratio_by_token_category_{size}-{variant}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot1] {path}")
    payload["n_examples"] = len(records)
    payload["accuracy_on_sample"] = round(
        float(np.mean([r["correct"] for r in records])), 4)
    return payload


def plot7_pca_think_states(size, n_examples, n_states, batch_size):
    from sklearn.decomposition import PCA
    experiments = find_experiments()
    variants = [v for v in ("v1_grpo_baseline", "v2_fixed_blend", "v4_full_hrpo_gate")
                if (size, v) in experiments and final_checkpoint(experiments[(size, v)])]
    if len(variants) < 2:
        print(f"[plot7] need >=2 of v1/v2/v4 at {size}; skipped")
        return None
    questions, answers = get_questions(n_examples)
    rng = np.random.default_rng(42)
    all_states, meta = [], []
    for variant in variants:
        ckpt = final_checkpoint(experiments[(size, variant)])
        print(f"[plot7] generating for {variant} ({ckpt})")
        records = collect_generation_data(size, ckpt, GATE_MODES[variant],
                                          questions, answers,
                                          batch_size=batch_size, keep_embeds=True)
        pool, pool_meta = [], []
        for rec in records:
            emb = rec.get("think_embeds")
            if emb is None or len(emb) == 0:
                continue
            for k in range(len(emb)):
                pool.append(emb[k])
                pool_meta.append((variant, rec["correct"]))
        if not pool:
            continue
        idx = rng.choice(len(pool), size=min(n_states, len(pool)), replace=False)
        for i in idx:
            all_states.append(pool[i])
            meta.append(pool_meta[i])
    if not all_states:
        print("[plot7] no think states collected; skipped")
        return None
    X = np.stack(all_states).astype(np.float32)
    pca = PCA(n_components=2, random_state=42)
    Y = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for variant in variants:
        for corr, marker in ((True, "o"), (False, "x")):
            sel = [i for i, (v, c) in enumerate(meta) if v == variant and c == corr]
            if not sel:
                continue
            ax.scatter(Y[sel, 0], Y[sel, 1], s=12, marker=marker,
                       color=variant_color(variant), alpha=0.55,
                       linewidths=1.0 if marker == "x" else 0,
                       label=f"{variant_label(variant)} "
                             f"({'correct' if corr else 'incorrect'})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(f"PCA of think-span projected hidden states ({size})")
    ax.legend(fontsize=7, markerscale=1.2)
    path = os.path.join(PLOTS_DIR, f"gate_p7_pca_think_states_{size}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot7] {path}")
    return {"variants": variants,
            "n_states_per_variant": n_states,
            "explained_variance_ratio": [round(float(v), 4)
                                         for v in pca.explained_variance_ratio_]}


def plot8_heatmaps(size, n_questions, batch_size):
    experiments = find_experiments()
    pair = [(v, final_checkpoint(experiments[(size, v)]))
            for v in ("v2_fixed_blend", "v4_full_hrpo_gate")
            if (size, v) in experiments and final_checkpoint(experiments[(size, v)])]
    if len(pair) < 2:
        print(f"[plot8] need both v2 and v4 at {size}; skipped")
        return None
    questions, answers = get_questions(n_questions)
    per_variant = {}
    for variant, ckpt in pair:
        print(f"[plot8] generating for {variant} ({ckpt})")
        per_variant[variant] = collect_generation_data(
            size, ckpt, GATE_MODES[variant], questions, answers,
            batch_size=batch_size)
    fig, axes = plt.subplots(n_questions, 1, figsize=(8.0, 1.35 * n_questions))
    if n_questions == 1:
        axes = [axes]
    vmax = 0.0
    rows_per_q = []
    payload = {}
    for qi in range(n_questions):
        rows = []
        for variant, _ in pair:
            rec = per_variant[variant][qi]
            rows.append(rec["hidden_ratio"][:rec["think_end"]])
        width = max((len(r) for r in rows), default=1)
        mat = np.full((2, max(width, 1)), np.nan)
        for ri, r in enumerate(rows):
            mat[ri, :len(r)] = r
        vmax = max(vmax, np.nanmax(mat) if np.isfinite(mat).any() else 0.0)
        rows_per_q.append(mat)
        payload[f"q{qi}"] = {
            variant: {"std_within": round(float(np.std(r)), 5) if len(r) else None,
                      "mean": round(float(np.mean(r)), 5) if len(r) else None,
                      "n_tokens": int(len(r))}
            for (variant, _), r in zip(pair, rows)}
    for qi, (ax, mat) in enumerate(zip(axes, rows_per_q)):
        im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0.0,
                       vmax=max(vmax, 1e-6), interpolation="nearest")
        ax.set_yticks([0, 1])
        ax.set_yticklabels([variant_label(v) for v, _ in pair], fontsize=7)
        ax.set_xticks([])
        ax.grid(False)
        ax.set_title(f"Q{qi + 1}: {questions[qi][:88]}...", fontsize=7, loc="left")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("per-token hidden ratio", fontsize=8)
    fig.suptitle(f"Per-token gate activation: v2 (frozen Λ) vs v4 (full gate), {size}",
                 fontsize=10, y=1.02)
    path = os.path.join(PLOTS_DIR, f"gate_p8_activation_heatmap_{size}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot8] {path}")
    return payload


def gate_activation_variance(size, variant, n_questions, n_samples, batch_size):
    """v4: variance of a_t within a response (across tokens) and across
    sampled responses to the same question."""
    experiments = find_experiments()
    exp_dir = experiments.get((size, variant))
    ckpt = final_checkpoint(exp_dir) if exp_dir else None
    if not ckpt:
        print(f"[variance] {size}-{variant} missing; skipped")
        return None
    questions, answers = get_questions(n_questions)
    rep_q = [q for q in questions for _ in range(n_samples)]
    rep_a = [a for a in answers for _ in range(n_samples)]
    print(f"[variance] sampling {n_samples} responses x {n_questions} questions")
    records = collect_generation_data(size, ckpt, GATE_MODES[variant],
                                      rep_q, rep_a, batch_size=batch_size,
                                      greedy=False)
    within, per_response_means = [], {}
    for i, rec in enumerate(records):
        r = rec["hidden_ratio"][:rec["think_end"]]
        if len(r) < 2:
            continue
        a_t = np.sqrt(np.clip(1.0 - np.asarray(r) ** 2, 0.0, 1.0))  # back to a_t
        within.append(float(np.var(a_t)))
        per_response_means.setdefault(i // n_samples, []).append(float(np.mean(a_t)))
    across = [float(np.var(v)) for v in per_response_means.values() if len(v) >= 2]
    payload = {
        "variant": variant, "size": size,
        "n_questions": n_questions, "n_samples_per_question": n_samples,
        "within_response_variance_mean": round(float(np.mean(within)), 6) if within else None,
        "within_response_variance_median": round(float(np.median(within)), 6) if within else None,
        "across_response_variance_mean": round(float(np.mean(across)), 6) if across else None,
    }
    print(f"[variance] {payload}")
    return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_static(args):
    experiments = find_experiments()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for size in args.sizes:
        cvs = plot2_lambda_distribution(experiments, size)
        if cvs:
            update_results(f"plot2_lambda_cv_{size}", cvs)
        finals = plot4_hidden_ratio_trajectory(experiments, size)
        if finals:
            update_results(f"plot4_final_ratios_{size}", finals)
        p6 = plot6_reward_vs_ratio(experiments, size)
        if p6:
            update_results(f"plot6_reward_ratio_{size}", p6)
        gf = gradient_flow(experiments, size)
        if gf:
            update_results(f"gradient_flow_{size}", gf)
    p3 = plot3_lambda_cv_vs_size(experiments)
    if p3:
        update_results("plot3_lambda_cv_vs_size_v4", p3)
    p5 = plot5_final_ratio_vs_size(experiments)
    if p5:
        update_results("plot5_final_ratio_vs_size", p5)


def run_dynamic(args):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    size = args.model_size
    p1 = plot1_ratio_by_token_category(size, args.plot1_variant,
                                       args.n_examples, args.batch_size)
    if p1:
        update_results(f"plot1_token_category_{size}-{args.plot1_variant}", p1)
    p7 = plot7_pca_think_states(size, min(args.n_examples, 60),
                                args.pca_states, args.batch_size)
    if p7:
        update_results(f"plot7_pca_{size}", p7)
    p8 = plot8_heatmaps(size, args.heatmap_questions, args.batch_size)
    if p8:
        update_results(f"plot8_heatmap_{size}", p8)
    var = gate_activation_variance(size, args.plot1_variant,
                                   args.variance_questions,
                                   args.variance_samples, args.batch_size)
    if var:
        update_results(f"gate_activation_variance_{size}", var)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--model_size", type=str, default="1.5b",
                        choices=SIZES, help="size for dynamic analyses")
    parser.add_argument("--sizes", nargs="+", default=SIZES,
                        help="sizes for static analyses")
    parser.add_argument("--plot1_variant", type=str, default="v4_full_hrpo_gate")
    parser.add_argument("--n_examples", type=int, default=200)
    parser.add_argument("--pca_states", type=int, default=300)
    parser.add_argument("--heatmap_questions", type=int, default=5)
    parser.add_argument("--variance_questions", type=int, default=20)
    parser.add_argument("--variance_samples", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    if not args.static and not args.dynamic:
        args.static = True
    if args.static:
        run_static(args)
    if args.dynamic:
        run_dynamic(args)


if __name__ == "__main__":
    main()
