"""
analyse_failures.py — Task 3 failure mode analysis (spec Section 9).

Takes two evaluated checkpoint paths — an HRPO variant and the v1 GRPO
baseline at the same model size — and compares them question-by-question on
their saved eval_results.json (single greedy decode each; no re-generation).

Outputs:
  results/failure_analysis.json      — confusion matrix, error sub-categories,
                                       difficulty-proxy accuracies (Wilson CIs)
  results/failure_examples/*.md      — first 20 "GRPO only correct" examples
  results/plots/failure_categories_<tag>.png
  results/plots/failure_difficulty_<tag>.png

Usage:
  python analyse_failures.py experiments/1.5b-v4_full_hrpo_gate/checkpoint-1868 \
                             experiments/1.5b-v1_grpo_baseline/checkpoint-1868
"""

import argparse
import json
import math
import os
import re

from utils import ANSWER_START, _is_repetition_loop
from stats_utils import wilson_ci

import numpy as np
from plot_style import apply_style, variant_color, variant_label, PALETTE

plt = apply_style()


# ---------------------------------------------------------------------------
# Loading / pairing
# ---------------------------------------------------------------------------

def load_eval(checkpoint: str):
    path = os.path.join(checkpoint, "eval_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run eval_all.py (or eval_gsm8k.py) first.")
    with open(path) as f:
        data = json.load(f)
    return data["metrics"], data["results"]


def infer_tag(checkpoint: str) -> str:
    """experiments/1.5b-v4_full_hrpo_gate/checkpoint-1868 -> 1.5b-v4_full_hrpo_gate"""
    parts = os.path.normpath(checkpoint).split(os.sep)
    for p in reversed(parts):
        if not p.startswith("checkpoint-"):
            return p
    return "unknown"


def infer_variant(tag: str) -> str:
    m = re.match(r"(0\.5b|1\.5b|3b)-(v\d_[a-z_]+)$", tag)
    return m.group(2) if m else tag


# ---------------------------------------------------------------------------
# Error taxonomy (spec Section 9)
# ---------------------------------------------------------------------------

def categorise_error(full_response: str) -> str:
    """Order matters: an empty response is not a loop; a loop usually also
    breaks the format, so the loop label takes precedence over format."""
    if not full_response or not full_response.strip():
        return "empty_response"
    if _is_repetition_loop(full_response):
        return "repetition_loop"
    n_tags = full_response.count(ANSWER_START)
    if n_tags == 0 or n_tags > 1:
        return "format_failure"
    return "wrong_answer_correct_format"


ERROR_CATEGORIES = ["repetition_loop", "format_failure",
                    "wrong_answer_correct_format", "empty_response"]


# ---------------------------------------------------------------------------
# Difficulty proxy (answer magnitude)
# ---------------------------------------------------------------------------

def difficulty_group(true_answer: str) -> str:
    try:
        mag = abs(float(str(true_answer).replace(",", "")))
    except (TypeError, ValueError):
        return "non_numeric"
    if mag <= 10:
        return "small (<=10)"
    if mag <= 100:
        return "medium (11-100)"
    return "large (>100)"


DIFFICULTY_ORDER = ["small (<=10)", "medium (11-100)", "large (>100)"]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hrpo_checkpoint", help="evaluated HRPO-variant checkpoint")
    parser.add_argument("grpo_checkpoint", help="evaluated v1 GRPO baseline checkpoint")
    parser.add_argument("--n_examples", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    hrpo_metrics, hrpo_results = load_eval(args.hrpo_checkpoint)
    grpo_metrics, grpo_results = load_eval(args.grpo_checkpoint)
    hrpo_tag = infer_tag(args.hrpo_checkpoint)
    grpo_tag = infer_tag(args.grpo_checkpoint)
    tag = f"{hrpo_tag}_vs_{grpo_tag}"

    if len(hrpo_results) != len(grpo_results):
        raise ValueError(f"Unpaired evals: {len(hrpo_results)} vs {len(grpo_results)} questions")
    for i in (0, len(hrpo_results) // 2, len(hrpo_results) - 1):
        if hrpo_results[i]["question"] != grpo_results[i]["question"]:
            raise ValueError(f"Question order mismatch at index {i} — evals are not paired")

    n = len(hrpo_results)

    # ---- 2x2 confusion matrix -------------------------------------------
    groups = {"both_correct": [], "both_wrong": [],
              "hrpo_only_correct": [], "grpo_only_correct": []}
    for i, (h, g) in enumerate(zip(hrpo_results, grpo_results)):
        if h["correct"] and g["correct"]:
            groups["both_correct"].append(i)
        elif not h["correct"] and not g["correct"]:
            groups["both_wrong"].append(i)
        elif h["correct"]:
            groups["hrpo_only_correct"].append(i)
        else:
            groups["grpo_only_correct"].append(i)

    confusion = {}
    for key, idxs in groups.items():
        lo, hi = wilson_ci(len(idxs), n)
        confusion[key] = {"count": len(idxs),
                          "pct": round(100 * len(idxs) / n, 2),
                          "proportion_ci95": [round(lo, 4), round(hi, 4)]}

    # ---- error sub-categorisation ---------------------------------------
    def subcategorise(idxs):
        counts = {c: 0 for c in ERROR_CATEGORIES}
        for i in idxs:
            counts[categorise_error(hrpo_results[i]["full_response"])] += 1
        total = max(1, len(idxs))
        return {c: {"count": v, "pct": round(100 * v / total, 2)}
                for c, v in counts.items()}

    hrpo_wrong_all = groups["both_wrong"] + groups["grpo_only_correct"]
    subcats = {
        # primary (spec): where the gate actively hurts
        "grpo_only_correct": subcategorise(groups["grpo_only_correct"]),
        # secondary context: every HRPO-wrong response
        "all_hrpo_wrong": subcategorise(hrpo_wrong_all),
    }

    # ---- per-model accuracy with Wilson CI -------------------------------
    def acc_entry(results):
        correct = sum(1 for r in results if r["correct"])
        lo, hi = wilson_ci(correct, n)
        return {"accuracy": round(correct / n, 4), "correct": correct,
                "total": n, "ci95": [round(lo, 4), round(hi, 4)]}

    accuracies = {hrpo_tag: acc_entry(hrpo_results),
                  grpo_tag: acc_entry(grpo_results)}

    # ---- difficulty proxy -------------------------------------------------
    difficulty = {}
    for grp in DIFFICULTY_ORDER + ["non_numeric"]:
        idxs = [i for i, r in enumerate(hrpo_results)
                if difficulty_group(r["true_answer"]) == grp]
        if not idxs:
            continue
        entry = {"n": len(idxs)}
        for label, results in (("hrpo", hrpo_results), ("grpo", grpo_results)):
            correct = sum(1 for i in idxs if results[i]["correct"])
            lo, hi = wilson_ci(correct, len(idxs))
            entry[label] = {"accuracy": round(correct / len(idxs), 4),
                            "correct": correct,
                            "ci95": [round(lo, 4), round(hi, 4)]}
        difficulty[grp] = entry

    # ---- qualitative examples: GRPO only correct --------------------------
    examples_dir = os.path.join(args.out_dir, "failure_examples")
    os.makedirs(examples_dir, exist_ok=True)
    for rank, i in enumerate(groups["grpo_only_correct"][:args.n_examples]):
        h, g = hrpo_results[i], grpo_results[i]
        md = (
            f"# GRPO-only-correct example {rank + 1} (question index {i})\n\n"
            f"**HRPO checkpoint:** `{args.hrpo_checkpoint}`\n\n"
            f"**GRPO checkpoint:** `{args.grpo_checkpoint}`\n\n"
            f"**HRPO error category:** {categorise_error(h['full_response'])}\n\n"
            f"## Question\n\n{h['question']}\n\n"
            f"**True answer:** {h['true_answer']}\n\n"
            f"## HRPO response (WRONG — extracted: `{h['generated_answer']}`)\n\n"
            f"```\n{h['full_response']}\n```\n\n"
            f"## GRPO response (CORRECT — extracted: `{g['generated_answer']}`)\n\n"
            f"```\n{g['full_response']}\n```\n"
        )
        with open(os.path.join(examples_dir, f"{rank + 1:02d}_idx{i}.md"), "w") as f:
            f.write(md)
    print(f"Wrote {min(args.n_examples, len(groups['grpo_only_correct']))} "
          f"examples to {examples_dir}/")

    # ---- plots -------------------------------------------------------------
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    hrpo_variant = infer_variant(hrpo_tag)
    grpo_variant = infer_variant(grpo_tag)

    # Failure-category bar chart (counts within grpo_only_correct).
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    cats = ERROR_CATEGORIES
    vals = [subcats["grpo_only_correct"][c]["count"] for c in cats]
    bars = ax.bar(range(len(cats)), vals, width=0.62,
                  color=variant_color(hrpo_variant))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v),
                ha="center", va="bottom", fontsize=8, color="#3a3a37")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=8)
    ax.set_ylabel("questions")
    ax.set_title(f"HRPO error categories where GRPO is correct\n"
                 f"({variant_label(hrpo_variant)} wrong, "
                 f"{variant_label(grpo_variant)} right; n={len(groups['grpo_only_correct'])})")
    cat_plot = os.path.join(plots_dir, f"failure_categories_{tag}.png")
    fig.savefig(cat_plot)
    plt.close(fig)

    # Difficulty-proxy grouped bars with Wilson CI error bars.
    grps = [g for g in DIFFICULTY_ORDER if g in difficulty]
    x = np.arange(len(grps))
    width = 0.36
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for off, (label, variant) in ((-width / 2, ("hrpo", hrpo_variant)),
                                  (width / 2, ("grpo", grpo_variant))):
        accs = [difficulty[g][label]["accuracy"] for g in grps]
        los = [difficulty[g][label]["accuracy"] - difficulty[g][label]["ci95"][0] for g in grps]
        his = [difficulty[g][label]["ci95"][1] - difficulty[g][label]["accuracy"] for g in grps]
        ax.bar(x + off, accs, width - 0.03, yerr=[los, his], capsize=2.5,
               color=variant_color(variant), label=variant_label(variant),
               error_kw={"linewidth": 1.0, "ecolor": "#3a3a37"})
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n={difficulty[g]['n']})" for g in grps], fontsize=8)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy by answer-magnitude difficulty proxy", pad=24)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, fontsize=8)
    diff_plot = os.path.join(plots_dir, f"failure_difficulty_{tag}.png")
    fig.savefig(diff_plot)
    plt.close(fig)
    print(f"Wrote {cat_plot} and {diff_plot}")

    # ---- JSON --------------------------------------------------------------
    analysis = {
        "hrpo_checkpoint": args.hrpo_checkpoint,
        "grpo_checkpoint": args.grpo_checkpoint,
        "hrpo_tag": hrpo_tag,
        "grpo_tag": grpo_tag,
        "n_questions": n,
        "accuracies": accuracies,
        "confusion_matrix": confusion,
        "hrpo_error_subcategories": subcats,
        "difficulty_proxy": difficulty,
        "example_indices_grpo_only_correct": groups["grpo_only_correct"][:args.n_examples],
    }
    out_json = os.path.join(args.out_dir, "failure_analysis.json")
    existing = {}
    if os.path.exists(out_json):
        try:
            with open(out_json) as f:
                existing = json.load(f)
            if "confusion_matrix" in existing:  # old single-pair format
                existing = {existing.get("hrpo_tag", "previous") + "_vs_" +
                            existing.get("grpo_tag", "previous"): existing}
        except json.JSONDecodeError:
            existing = {}
    existing[tag] = analysis
    with open(out_json, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Saved analysis under key '{tag}' in {out_json}")

    # ---- stdout summary ------------------------------------------------------
    print("\n2x2 confusion matrix:")
    for key in ("both_correct", "hrpo_only_correct", "grpo_only_correct", "both_wrong"):
        c = confusion[key]
        print(f"  {key:<20} {c['count']:>5}  ({c['pct']}%)  "
              f"CI95 [{c['proportion_ci95'][0]}, {c['proportion_ci95'][1]}]")
    print("\nHRPO error categories (where GRPO correct):")
    for cat in ERROR_CATEGORIES:
        s = subcats["grpo_only_correct"][cat]
        print(f"  {cat:<28} {s['count']:>5}  ({s['pct']}%)")


if __name__ == "__main__":
    main()
