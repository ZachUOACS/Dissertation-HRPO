"""
generate_report.py — Task 6 statistical summary report (spec Section 12).

Reads everything under results/ (comparison_table.json, failure_analysis.json,
gate_analysis.json, overhead_matrix.json) and produces:

  results/latex_tables/table1_ablation_results.tex
  results/latex_tables/table2_mcnemar_pvalues.tex
  results/latex_tables/table3_failure_modes.tex
  results/latex_tables/table4_overhead.tex
  results/latex_tables/table5_lambda_heterogeneity.tex
  results/plots/ablation_accuracy_by_size.png
  results/plots/scale_vs_accuracy.png
  results/key_findings.txt

The eight gate-behaviour figures are produced by analyse_gate.py; the static
subset (Plots 2-6, gradient flow) is regenerated here if missing. Dynamic
figures (Plots 1/7/8) and the failure plots require GPU/eval runs and are
listed as pending when absent. No GPU is required by this script itself.

Usage: python generate_report.py
"""

import glob
import json
import os
import re

import numpy as np

from plot_style import apply_style, variant_color, variant_label
from variant_common import VARIANT_CHAIN

plt = apply_style()

SIZES = ["0.5b", "1.5b", "3b"]
RESULTS = "results"
LATEX_DIR = os.path.join(RESULTS, "latex_tables")
PLOTS_DIR = os.path.join(RESULTS, "plots")

ROW_ORDER = ["base_zero_shot"] + VARIANT_CHAIN


def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def latex_escape(s: str) -> str:
    s = str(s)
    if not (s.startswith("$") or "\\" in s):  # already-LaTeX cells pass through
        s = s.replace("_", r"\_").replace("%", r"\%").replace("#", r"\#")
    return s.replace("Λ", r"$\Lambda$")


def write_table(name: str, caption: str, label: str, header: list, rows: list,
                col_spec: str = None):
    os.makedirs(LATEX_DIR, exist_ok=True)
    col_spec = col_spec or ("l" + "r" * (len(header) - 1))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(latex_escape(h) for h in header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(c) for c in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path = os.path.join(LATEX_DIR, name)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[latex] {path}")
    return path


def fmt_p(p):
    if p is None:
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def sig_mark(p):
    return r"$^{*}$" if (p is not None and p < 0.05) else ""


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table1_ablation(comparison):
    by_key = {(r["model_size"], r["variant_id"]): r for r in comparison}
    rows = []
    for variant in ROW_ORDER:
        row = [variant_label(variant) if variant != "base_zero_shot" else "base (zero-shot)"]
        for size in SIZES:
            r = by_key.get((size, variant))
            if r:
                row.append(f"{r['accuracy']*100:.2f} "
                           f"[{r['ci_lower']*100:.1f}, {r['ci_upper']*100:.1f}]")
            else:
                row.append("--")
        rows.append(row)
    # legacy references
    for (size, variant), r in sorted(by_key.items()):
        if variant.startswith("legacy_"):
            row = [f"legacy HRPO ({size}, group8)"] + [
                (f"{r['accuracy']*100:.2f} [{r['ci_lower']*100:.1f}, "
                 f"{r['ci_upper']*100:.1f}]" if s == size else "--") for s in SIZES]
            rows.append(row)
    return write_table(
        "table1_ablation_results.tex",
        "GSM8K accuracy (\\%) with Wilson 95\\% confidence intervals for all "
        "ablation variants and model sizes (single greedy decode per question, "
        "$n$=1319 unless noted).",
        "tab:ablation-results",
        ["Variant"] + [f"Qwen2.5-{s}" for s in SIZES], rows)


def table2_mcnemar(comparison):
    by_key = {(r["model_size"], r["variant_id"]): r for r in comparison}
    rows = []
    for size in SIZES:
        present = [v for v in VARIANT_CHAIN if (size, v) in by_key]
        for prev, cur in zip(present, present[1:]):
            r = by_key[(size, cur)]
            p = r.get("vs_prev_mcnemar_p")
            rows.append([size, f"{variant_label(prev)} $\\rightarrow$ {variant_label(cur)}",
                         fmt_p(p) + sig_mark(p)])
        # vs v1 for the chain tail (headline comparisons)
        for v in present[1:]:
            r = by_key[(size, v)]
            p = r.get("vs_v1_mcnemar_p")
            chi2 = r.get("vs_v1_mcnemar_chi2")
            rows.append([size, f"{variant_label(v)} vs {variant_label('v1_grpo_baseline')}",
                         (f"$\\chi^2$={chi2:.2f}, p=" if chi2 is not None else "")
                         + fmt_p(p) + sig_mark(p)])
    if not rows:
        rows = [["--", "no paired full evaluations available yet", "--"]]
    return write_table(
        "table2_mcnemar_pvalues.tex",
        "McNemar's test (continuity-corrected) between consecutive variants in "
        "the causal chain, and each variant vs the v1 GRPO baseline, per model "
        "size. $^{*}$: $p<0.05$.",
        "tab:mcnemar", ["Size", "Comparison", "p-value"], rows, col_spec="llr")


def table3_failure_modes(failure):
    rows = []
    if failure:
        for tag, analysis in sorted(failure.items()):
            if not isinstance(analysis, dict) or "confusion_matrix" not in analysis:
                continue
            if not analysis.get("hrpo_tag", "").startswith("1.5b"):
                continue
            cm = analysis["confusion_matrix"]
            sub = analysis["hrpo_error_subcategories"]["grpo_only_correct"]
            variant = re.sub(r"^1\.5b-", "", analysis["hrpo_tag"])
            rows.append([
                variant_label(variant),
                f"{cm['both_correct']['count']} ({cm['both_correct']['pct']}\\%)",
                f"{cm['hrpo_only_correct']['count']} ({cm['hrpo_only_correct']['pct']}\\%)",
                f"{cm['grpo_only_correct']['count']} ({cm['grpo_only_correct']['pct']}\\%)",
                f"{cm['both_wrong']['count']} ({cm['both_wrong']['pct']}\\%)",
                f"{sub['repetition_loop']['count']}/"
                f"{sub['format_failure']['count']}/"
                f"{sub['wrong_answer_correct_format']['count']}/"
                f"{sub['empty_response']['count']}",
            ])
    if not rows:
        rows = [["--", "--", "--", "--", "--", "pending: run analyse\\_failures.py"]]
    return write_table(
        "table3_failure_modes.tex",
        "Failure-mode breakdown vs the v1 GRPO baseline at 1.5B. Error split "
        "is within `GRPO only correct': repetition loop / format failure / "
        "wrong answer with correct format / empty.",
        "tab:failure-modes",
        ["Variant", "Both correct", "HRPO only", "GRPO only", "Both wrong",
         "Error split (l/f/w/e)"], rows, col_spec="lrrrrr")


def table4_overhead(matrix):
    rows = []
    if matrix:
        for key, cell in sorted(matrix.get("cells", {}).items()):
            size, variant = key.split("-", 1)
            p = cell["params"]
            t = cell.get("timing", {})
            wc = t.get("wall_clock_overhead_pct")
            rows.append([
                size, variant_label(variant),
                f"{p['gate_params_total']:,}",
                f"{p['gate_params_trainable']:,}",
                f"{cell['flops']['gate_overhead_percent']:.3f}\\%",
                (f"{wc:.2f}\\%" if wc is not None else "--"),
                (f"{t['memory_overhead_mb']:.0f} MB"
                 if "memory_overhead_mb" in t else "--"),
            ])
    if not rows:
        rows = [["--"] * 7]
        rows[0][-1] = "pending: run hrpo\\_overhead.py --variant\\_matrix"
    return write_table(
        "table4_overhead.tex",
        "Computational overhead per model size and variant: gate parameters "
        "(total vs trainable), analytical FLOPs overhead (MAC convention, "
        "baseline $6ND$), generation wall-clock overhead (hybrid path on vs "
        "off, fixed decode length) and peak-memory overhead.",
        "tab:overhead",
        ["Size", "Variant", "Gate params", "Trainable", "FLOPs", "Wall-clock",
         "Memory"], rows, col_spec="llrrrrr")


def table5_lambda(gate):
    rows = []
    if gate:
        for size in SIZES:
            cvs = gate.get(f"plot2_lambda_cv_{size}")
            if not cvs:
                continue
            for variant, entry in cvs.items():
                if entry.get("cv") is not None:
                    rows.append([size, variant_label(variant),
                                 f"{entry['cv']:.4f}",
                                 f"{entry['mean_ratio']:.4f}"])
    if not rows:
        rows = [["--", "pending: run analyse\\_gate.py --static", "--", "--"]]
    return write_table(
        "table5_lambda_heterogeneity.tex",
        "$\\Lambda$ heterogeneity: coefficient of variation (std/mean) of the "
        "per-dimension hidden ratio $\\sqrt{1-a^2}$ at $r_t{=}1$, per variant "
        "and model size.",
        "tab:lambda-heterogeneity",
        ["Size", "Variant", "CV", "Mean hidden ratio"], rows, col_spec="llrr")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_ablation_bars(comparison):
    by_key = {(r["model_size"], r["variant_id"]): r for r in comparison}
    sizes = [s for s in SIZES if any(k[0] == s for k in by_key)]
    if not sizes:
        print("[fig] no comparison data; ablation bar chart skipped")
        return
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.6 * len(sizes), 3.4),
                             sharey=True)
    if len(sizes) == 1:
        axes = [axes]
    for ax, size in zip(axes, sizes):
        variants = [v for v in ROW_ORDER if (size, v) in by_key]
        xs = np.arange(len(variants))
        for i, v in enumerate(variants):
            r = by_key[(size, v)]
            err = [[r["accuracy"] - r["ci_lower"]], [r["ci_upper"] - r["accuracy"]]]
            ax.bar(i, r["accuracy"], 0.62, yerr=err, capsize=3,
                   color=variant_color(v),
                   error_kw={"linewidth": 1.0, "ecolor": "#3a3a37"})
        ax.set_xticks(xs)
        ax.set_xticklabels([variant_label(v) for v in variants],
                           rotation=40, ha="right", fontsize=7)
        ax.set_title(size)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("GSM8K accuracy")
    fig.suptitle("Ablation accuracy with Wilson 95% CIs", fontsize=10)
    path = os.path.join(PLOTS_DIR, "ablation_accuracy_by_size.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] {path}")


def fig_scale_lines(comparison):
    by_key = {(r["model_size"], r["variant_id"]): r for r in comparison}
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    plotted = False
    for variant in ROW_ORDER:
        xs, ys, los, his = [], [], [], []
        for i, size in enumerate(SIZES):
            r = by_key.get((size, variant))
            if r:
                xs.append(i)
                ys.append(r["accuracy"])
                los.append(r["accuracy"] - r["ci_lower"])
                his.append(r["ci_upper"] - r["accuracy"])
        if len(xs) >= 1:
            ax.errorbar(xs, ys, yerr=[los, his], marker="o", markersize=4.5,
                        capsize=2.5, color=variant_color(variant),
                        label=variant_label(variant), linewidth=1.8,
                        elinewidth=0.9)
            plotted = True
    if not plotted:
        print("[fig] no comparison data; scale line plot skipped")
        plt.close(fig)
        return
    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels(SIZES)
    ax.set_xlabel("model size")
    ax.set_ylabel("GSM8K accuracy")
    ax.set_title("Accuracy vs model scale per variant")
    ax.legend(fontsize=7)
    path = os.path.join(PLOTS_DIR, "scale_vs_accuracy.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] {path}")


def ensure_static_gate_plots():
    """Regenerate the static gate figures (Plots 2-6, gradient flow) if any
    are missing; dynamic figures (1/7/8) need GPU and are only reported."""
    import argparse as ap
    import analyse_gate
    # Idempotent: regenerates from whatever checkpoints/logs exist, skips rest.
    analyse_gate.run_static(ap.Namespace(sizes=SIZES))
    missing = []
    for pattern, desc in [
        ("gate_p1_ratio_by_token_category_*.png", "Plot 1 (token category — GPU)"),
        ("gate_p2_lambda_distribution_*.png", "Plot 2 (Λ distribution)"),
        ("gate_p3_lambda_cv_vs_size_*.png", "Plot 3 (Λ CV vs size)"),
        ("gate_p4_hidden_ratio_trajectory_*.png", "Plot 4 (ratio trajectory)"),
        ("gate_p5_final_hidden_ratio_vs_size.png", "Plot 5 (final ratio vs size)"),
        ("gate_p6_reward_vs_ratio_*.png", "Plot 6 (reward vs ratio)"),
        ("gate_p7_pca_think_states_*.png", "Plot 7 (PCA — GPU)"),
        ("gate_p8_activation_heatmap_*.png", "Plot 8 (heatmaps — GPU)"),
    ]:
        if not glob.glob(os.path.join(PLOTS_DIR, pattern)):
            missing.append(desc)
    if missing:
        print("[fig] gate plots still pending: " + "; ".join(missing))
    return missing


# ---------------------------------------------------------------------------
# Key findings (Section 13 research questions)
# ---------------------------------------------------------------------------

def acc_str(r):
    return (f"{r['accuracy']*100:.2f}% "
            f"(95% CI [{r['ci_lower']*100:.1f}, {r['ci_upper']*100:.1f}], "
            f"{r['correct']}/{r['total']})")


def key_findings(comparison, failure, gate, matrix):
    by_key = {(r["model_size"], r["variant_id"]): r for r in (comparison or [])}

    def row(size, variant):
        return by_key.get((size, variant))

    def pair_stmt(size, a, b, rq):
        ra, rb = row(size, a), row(size, b)
        if not ra or not rb or ra["total"] != rb["total"] or ra["total"] < 1319:
            return (f"PENDING — requires full paired evaluations of {a} and {b} "
                    f"at {size} (run run_ablations.sh then eval_all.py).")
        p = rb.get("vs_prev_mcnemar_p") if VARIANT_CHAIN.index(b) - 1 >= 0 else None
        # fall back to vs_v1 if the pair is (v1, b)
        if a == "v1_grpo_baseline" and rb.get("vs_v1_mcnemar_p") is not None:
            p = rb["vs_v1_mcnemar_p"]
        verdict = ("a statistically significant difference (p={:.4f} < 0.05)"
                   if (p is not None and p < 0.05) else
                   "no statistically significant difference (p={:.4f})").format(
                       p if p is not None else float("nan"))
        return (f"At {size}, {variant_label(a)} scores {acc_str(ra)} vs "
                f"{variant_label(b)} {acc_str(rb)}; McNemar's test shows {verdict}.")

    lines = []
    add = lines.append
    add("KEY FINDINGS — one paragraph per research question (spec Section 13).")
    add("Generated by generate_report.py from results/*.json; regenerate after")
    add("new training/evaluation runs to refresh the numbers.")
    add("")

    add("RQ1 (does any latent blending help — v2 vs v1, per scale):")
    for size in SIZES:
        add(f"  [{size}] " + pair_stmt(size, "v1_grpo_baseline", "v2_fixed_blend", 1))
    add("")

    add("RQ2 (does learning Λ help — v3 vs v2, 1.5b):")
    add("  " + pair_stmt("1.5b", "v2_fixed_blend", "v3_learned_lambda", 2))
    add("")

    add("RQ3 (does input-dependent gating help — v4 vs v3, 1.5b):")
    add("  " + pair_stmt("1.5b", "v3_learned_lambda", "v4_full_hrpo_gate", 3))
    add("")

    add("RQ4 (does the gate open more for semantically important tokens):")
    p1 = None
    if gate:
        for k, v in gate.items():
            if k.startswith("plot1_token_category_"):
                p1 = (k, v)
    if p1:
        k, v = p1
        num = v.get("number_correct", {}).get("mean")
        word = v.get("word_correct", {}).get("mean")
        add(f"  From {k}: mean hidden ratio on number tokens = {num}, on word "
            f"tokens = {word} (correct responses, think span, "
            f"n={v.get('n_examples')} examples). "
            + ("Numbers receive a higher ratio than filler words."
               if (num or 0) > (word or 0) else
               "No preferential opening for numbers over words observed."))
    else:
        add("  PENDING — run analyse_gate.py --dynamic (Plot 1).")
    add("")

    add("RQ5 (is the Λ distribution heterogeneous across dimensions):")
    if gate and any(k.startswith("plot2_lambda_cv_") for k in gate):
        for size in SIZES:
            cvs = gate.get(f"plot2_lambda_cv_{size}")
            if cvs:
                stmt = ", ".join(f"{variant_label(v)} CV={e['cv']}"
                                 for v, e in cvs.items() if e.get("cv") is not None)
                add(f"  [{size}] {stmt}. CV >> 0 indicates per-dimension "
                    f"specialisation; compare v2 (frozen init) as the null reference.")
    else:
        add("  PENDING — run analyse_gate.py --static (Plot 2).")
    add("")

    add("RQ6 (do correct/incorrect responses separate in PCA space):")
    p7 = next(((k, v) for k, v in (gate or {}).items()
               if k.startswith("plot7_pca_")), None)
    if p7:
        k, v = p7
        add(f"  From {k}: joint PCA over {v['variants']} explains "
            f"{v['explained_variance_ratio']} of variance in 2D; see "
            f"gate_p7_pca_think_states_*.png for the separation structure.")
    else:
        add("  PENDING — run analyse_gate.py --dynamic (Plot 7).")
    add("")

    add("RQ7 (does higher hidden ratio correlate with higher reward):")
    p6 = next(((k, v) for k, v in (gate or {}).items()
               if k.startswith("plot6_reward_ratio_")), None)
    if p6:
        k, v = p6
        add(f"  From {k}: Pearson r={v['pearson_r']} (p={v['pearson_p']:.2e}), "
            f"Spearman r={v['spearman_r']} (p={v['spearman_p']:.2e}) over "
            f"{v['n_steps']} training steps.")
    else:
        add("  PENDING — run analyse_gate.py --static (Plot 6).")
    add("")

    add("RQ8 (at what scale does HRPO stop helping over GRPO):")
    stmts = []
    for size in SIZES:
        r4, r1 = row(size, "v4_full_hrpo_gate"), row(size, "v1_grpo_baseline")
        if r4 and r1 and r4.get("vs_v1_mcnemar_p") is not None:
            stmts.append(f"[{size}] v4 vs v1 p={r4['vs_v1_mcnemar_p']:.4f}"
                         f"{' (significant)' if r4['vs_v1_mcnemar_p'] < 0.05 else ''}")
    add("  " + ("; ".join(stmts) if stmts
                else "PENDING — requires v4 and v1 full evaluations at every size."))
    add("")

    add("RQ9 (do reward shaping v5 and repetition handling v6 help over v4):")
    add("  " + pair_stmt("1.5b", "v4_full_hrpo_gate", "v5_hrpo_format_reward", 9))
    add("  " + pair_stmt("1.5b", "v5_hrpo_format_reward", "v6_hrpo_full", 9))
    add("")

    add("RQ10 (is the wall-clock overhead justified by the accuracy gain):")
    if matrix and comparison:
        done = False
        for key, cell in matrix.get("cells", {}).items():
            size, variant = key.split("-", 1)
            wc = cell.get("timing", {}).get("wall_clock_overhead_pct")
            r_v, r_1 = row(size, variant), row(size, "v1_grpo_baseline")
            if wc is not None and wc > 0 and r_v and r_1 and r_v["total"] >= 1319 <= r_1["total"]:
                gain = (r_v["accuracy"] - r_1["accuracy"]) * 100
                add(f"  [{key}] accuracy gain {gain:+.2f} points at {wc:.2f}% "
                    f"generation overhead = {gain / wc:+.3f} points per overhead %.")
                done = True
        if not done:
            add("  PENDING — requires overhead_matrix.json cells plus full paired "
                "evaluations (positive measured overhead).")
    else:
        add("  PENDING — run hrpo_overhead.py --variant_matrix and eval_all.py.")
    add("")

    add("RQ11 (do r_t/i_t produce meaningful per-token variation):")
    p8 = next(((k, v) for k, v in (gate or {}).items()
               if k.startswith("plot8_heatmap_")), None)
    var = next(((k, v) for k, v in (gate or {}).items()
                if k.startswith("gate_activation_variance_")), None)
    if p8 or var:
        if p8:
            k, v = p8
            stds = [q["v4_full_hrpo_gate"]["std_within"] for q in v.values()
                    if "v4_full_hrpo_gate" in q and q["v4_full_hrpo_gate"]["std_within"] is not None]
            v2_stds = [q["v2_fixed_blend"]["std_within"] for q in v.values()
                       if "v2_fixed_blend" in q and q["v2_fixed_blend"]["std_within"] is not None]
            add(f"  From {k}: v4 within-response per-token ratio std = "
                f"{stds} vs v2 (frozen, unit gates) = {v2_stds} — v2 is uniform "
                f"by construction; v4's std quantifies what r_t/i_t add.")
        if var:
            k, v = var
            add(f"  From {k}: a_t variance within responses = "
                f"{v['within_response_variance_mean']}, across sampled responses "
                f"to the same question = {v['across_response_variance_mean']}.")
    else:
        add("  PENDING — run analyse_gate.py --dynamic (Plot 8 + variance).")
    add("")

    add("RQ12 (most common failure mode at each scale):")
    if failure:
        for tag, analysis in sorted(failure.items()):
            if not isinstance(analysis, dict) or "hrpo_error_subcategories" not in analysis:
                continue
            sub = analysis["hrpo_error_subcategories"]["all_hrpo_wrong"]
            top = max(sub.items(), key=lambda kv: kv[1]["count"])
            add(f"  [{analysis['hrpo_tag']}] most common HRPO error: {top[0]} "
                f"({top[1]['count']} cases, {top[1]['pct']}% of HRPO-wrong).")
    else:
        add("  PENDING — run analyse_failures.py per size.")
    add("")

    path = os.path.join(RESULTS, "key_findings.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[report] {path}")


def main():
    os.makedirs(LATEX_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    comparison = load_json(os.path.join(RESULTS, "comparison_table.json")) or []
    failure = load_json(os.path.join(RESULTS, "failure_analysis.json"))
    gate = load_json(os.path.join(RESULTS, "gate_analysis.json"))
    matrix = load_json(os.path.join(RESULTS, "overhead_matrix.json"))

    table1_ablation(comparison)
    table2_mcnemar(comparison)
    table3_failure_modes(failure)
    table4_overhead(matrix)
    table5_lambda(gate)

    fig_ablation_bars(comparison)
    fig_scale_lines(comparison)
    ensure_static_gate_plots()

    # failure plots are produced by analyse_failures.py; report their state
    fplots = glob.glob(os.path.join(PLOTS_DIR, "failure_*.png"))
    if fplots:
        print(f"[fig] failure plots present: {len(fplots)}")
    else:
        print("[fig] failure plots pending: run analyse_failures.py")

    key_findings(comparison, failure, gate, matrix)
    print("\nReport artefacts written under results/.")


if __name__ == "__main__":
    main()
