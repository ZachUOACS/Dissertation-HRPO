"""
generate_report.py — Task 6 statistical summary report (spec Section 12).

Reads everything under results/ (comparison_table.json, failure_analysis.json,
gate_analysis.json, overhead_matrix.json) and produces:

  results/latex_tables/table0_primary_results.tex
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

Accuracy is reported at PRIMARY_RP (rp=1.0), seed-averaged: the rp=1.2 runs
are confounded by the repetition penalty and are excluded from the primary
table, the figures and key_findings.txt.

Usage: python generate_report.py
"""

import glob
import json
import os
import re

import numpy as np

from plot_style import apply_style, variant_color, variant_label
from variant_common import VARIANT_CHAIN, CONTROL_VARIANTS

plt = apply_style()

SIZES = os.environ.get("HRPO_SIZES", "0.5b 1.5b 3b").split()
FULL_TEST_SIZE = 1319
# Per-task sizes: FULL_TEST_SIZE is GSM8K's. A MATH row (MATH-500) would
# otherwise be discarded as a partial evaluation.
TASK_TEST_SIZE = {"gsm8k": FULL_TEST_SIZE, "math": 500}
# The setting everything is reported at. rp=1.2 was an artefact of the study
# protocol (it cost every arm ~7 points, baseline included) and is excluded
# from the report; the checkpoints stay on disk as the data of record.
PRIMARY_RP = 1.0
RESULTS = "results"
LATEX_DIR = os.path.join(RESULTS, "latex_tables")
PLOTS_DIR = os.path.join(RESULTS, "plots")

ROW_ORDER = ["base_zero_shot"] + VARIANT_CHAIN


def css_line():
    return "#c9ccd1"


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
        "Ablation decomposition at the study protocol's \\textbf{rp=1.2} "
        "setting (single seed, single greedy decode, $n$=1319). This setting is "
        "now known to be confounded by the repetition penalty (Table~\\ref{tab:primary}); "
        "it is retained as the component decomposition, not the headline result. "
        "Accuracy (\\%) with Wilson 95\\% CIs.",
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

# ---------------------------------------------------------------------------
# PRIMARY results: repetition-penalty corrected, seed-averaged (Table 0 + figure)
# ---------------------------------------------------------------------------

def _seed_arms():
    """{(size, variant, rp, lora, warm, dr): {seed: (accuracy, outcome_vector | None)}}.

    Two sources, merged. The local experiment directories give accuracy *and*
    the per-question outcome vector that paired McNemar needs; comparison_table
    .json gives accuracy alone. The table is the authority on which runs exist
    — some arms' checkpoints live only on the cluster (v5/v6 at rp=1.0 here) —
    so a row with no local directory still contributes its accuracy, just with
    a None vector, and the statements that need a vector say so instead of
    silently dropping the arm."""
    arms = {}
    try:
        from analyse_replicates import discover_arms, parse_suffix
    except Exception as exc:                       # noqa: BLE001
        print(f"[warn] seed-arm view unavailable ({exc})")
        return arms
    for key, seed_map in discover_arms().items():
        arms.setdefault(key, {}).update(seed_map)
    for r in load_json(os.path.join(RESULTS, "comparison_table.json")) or []:
        if r.get("total", 0) < TASK_TEST_SIZE.get(r.get("task", "gsm8k"), FULL_TEST_SIZE):
            continue
        # a cancelled run's newest checkpoint is a mid-run snapshot, not an arm
        if r.get("training_complete") is False:
            continue
        variant, _, suffix = r["variant_id"].partition("-")
        # base_zero_shot, legacy_* are dropped; controls (C1) are kept as arms,
        # and _primary_arms still keeps them out of the ablation table/figure.
        if variant not in VARIANT_CHAIN and variant not in CONTROL_VARIANTS:
            continue
        seed, rp, lora, warm, dr, ds = parse_suffix(f"-{suffix}" if suffix else "")
        arms.setdefault((r["model_size"], variant, rp, lora, warm, dr, ds), {}).setdefault(
            seed, (r["accuracy"], None))
    return arms


def _primary_arms(seed_arms=None):
    """Group every run into (size, base variant, rp), restricted to the
    standard 'full' LoRA set. Returns {(size, variant): {1.2: (mean, lo, hi,
    n), 1.0: (...)}} — the before/after the confound was removed."""
    seed_arms = _seed_arms() if seed_arms is None else seed_arms
    out = {}
    for (size, variant, rp, lora, warm, dr, ds), seed_map in seed_arms.items():
        # Warm-started arms (Stage-0 T1/C3), Dr. GRPO arms (C2) and second-task
        # arms (MATH) are separate experiments from the cold GSM8K ablation
        # chain and must never be folded into its cells.
        if (lora != "full" or warm is not None or dr is not None or ds is not None
                or variant not in VARIANT_CHAIN):
            continue
        accs = [v[0] for v in seed_map.values()]
        out.setdefault((size, variant), {})[rp] = (
            float(np.mean(accs)), float(min(accs)), float(max(accs)), len(accs))
    return out


def _fmt_cell(entry):
    if entry is None:
        return "--"
    mean, lo, hi, n = entry
    rng = f" [{lo*100:.1f}, {hi*100:.1f}]" if n > 1 else ""
    tag = f" (n={n})" if n > 1 else ""
    return f"{mean*100:.2f}{rng}{tag}"


def table0_primary(arms):
    """Primary results table — the ablation chain at the corrected rp=1.0
    setting only, seed-averaged with min--max range. rp=1.2 is not tabulated
    here; the penalty's effect is stated once, in the caption and the text."""
    rows = []
    for size in SIZES:
        base = arms.get((size, "v1_grpo_baseline"), {}).get(1.0)
        for variant in VARIANT_CHAIN:
            e = arms.get((size, variant), {}).get(1.0)
            if not e:
                continue
            delta = f"{(e[0]-base[0])*100:+.1f}" if base else "--"
            rows.append([f"Qwen2.5-{size}", variant_label(variant),
                         _fmt_cell(e), delta])
    if not rows:
        rows = [["--", "pending (run rp=1.0 chain)", "--", "--"]]
    return write_table(
        "table0_primary_results.tex",
        "Primary results — GSM8K accuracy (\\%) at the corrected rp=1.0 setting, "
        "seed-averaged with min--max range (n = seeds). $\\Delta$ is versus the "
        "v1 GRPO baseline at the same scale. Removing the study protocol's "
        "off-paper repetition\\_penalty = 1.2 raised every arm by $\\approx$7 "
        "points (including the non-latent baseline), so all results are reported "
        "at rp = 1.0.",
        "tab:primary",
        ["Model", "Variant", "Accuracy (rp=1.0)", "$\\Delta$ vs GRPO"],
        rows, col_spec="llrr")


def fig_ablation_rp1(arms):
    """PRIMARY accuracy figure — the ablation chain at the corrected rp=1.0
    setting, per size. Dot = seed mean, whiskers = min--max across seeds,
    dotted line = GRPO (v1) baseline. Contains no rp=1.2 data. Written as the
    canonical `ablation_accuracy_by_size.png`."""
    sizes = [s for s in SIZES
             if any((s, v) in arms and 1.0 in arms[(s, v)] for v in VARIANT_CHAIN)]
    if not sizes:
        print("[fig] no rp=1.0 arms; ablation figure skipped")
        return
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.8 * len(sizes), 3.8),
                             sharey=True)
    if len(sizes) == 1:
        axes = [axes]
    pts = [arms[(s, v)][1.0] for s in sizes for v in VARIANT_CHAIN
           if (s, v) in arms and 1.0 in arms[(s, v)]]
    lo = min(p[1] for p in pts)
    hi = max(p[2] for p in pts)
    pad = 0.03 * (hi - lo) if hi > lo else 0.02
    for ax, size in zip(axes, sizes):
        variants = [v for v in VARIANT_CHAIN
                    if (size, v) in arms and 1.0 in arms[(size, v)]]
        for i, v in enumerate(variants):
            mean, mn, mx, n = arms[(size, v)][1.0]
            col = variant_color(v)
            if mx > mn:
                ax.plot([i, i], [mn, mx], color=col, lw=1.6, alpha=.5, zorder=2)
            ax.plot(i, mean, "o", ms=8, color=col, markeredgecolor=col, zorder=3)
        base = arms.get((size, "v1_grpo_baseline"), {}).get(1.0)
        if base:
            ax.axhline(base[0], color="#8a8a85", lw=1, ls=":", zorder=1)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([variant_label(v) for v in variants],
                           rotation=40, ha="right", fontsize=7)
        ax.set_xlim(-0.6, len(variants) - 0.4)
        ax.set_title(size)
        ax.set_ylim(lo - pad, hi + pad)
    axes[0].set_ylabel("GSM8K accuracy")
    fig.suptitle("Ablation decomposition (rp = 1.0, corrected setting)\n"
                 "dot = seed mean, whiskers = min–max across seeds; "
                 "dotted = GRPO baseline",
                 fontsize=9, y=1.06)
    path = os.path.join(PLOTS_DIR, "ablation_accuracy_by_size.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}")


def fig_scale_rp1(arms):
    """Accuracy vs model scale at rp=1.0 — drawn only when >=2 sizes have
    rp=1.0 data (i.e. after the 3B rp=1.0 chain is in)."""
    sizes = [s for s in SIZES
             if any((s, v) in arms and 1.0 in arms[(s, v)] for v in VARIANT_CHAIN)]
    if len(sizes) < 2:
        print("[fig] <2 sizes with rp=1.0 data; scale figure skipped")
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for v in VARIANT_CHAIN:
        xs, ys, los, his = [], [], [], []
        for i, s in enumerate(sizes):
            e = arms.get((s, v), {}).get(1.0)
            if e:
                mean, mn, mx, n = e
                xs.append(i)
                ys.append(mean)
                los.append(mean - mn)
                his.append(mx - mean)
        if xs:
            ax.errorbar(xs, ys, yerr=[los, his], marker="o", markersize=4.5,
                        capsize=2.5, color=variant_color(v),
                        label=variant_label(v), linewidth=1.8, elinewidth=0.9)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_xlabel("model size")
    ax.set_ylabel("GSM8K accuracy")
    ax.set_title("Accuracy vs scale per variant (rp = 1.0)")
    ax.legend(fontsize=7)
    path = os.path.join(PLOTS_DIR, "scale_vs_accuracy.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] {path}")


def ensure_static_gate_plots():
    """Regenerate the static gate figures (Plots 2-6, gradient flow) ONLY if
    they are missing; dynamic figures (1/7/8) need GPU and are only reported.

    Critically, this must NOT clobber the setting the pipeline already produced:
    the pipeline runs `analyse_gate --static --variant_suffix=-rp1` (rp=1.0), and
    if this fallback re-ran run_static with the default (empty) suffix it would
    overwrite those rp=1.0 Λ figures with the rp=1.2 main cells. So: skip when the
    figures are already present, and if regenerating, honour ANALYSIS_SUFFIX
    (default -rp1) rather than the rp=1.2 default."""
    sentinel = [os.path.join(PLOTS_DIR, f"gate_p2_lambda_distribution_{s}.png")
                for s in SIZES]
    if sentinel and all(os.path.exists(p) for p in sentinel):
        return  # already produced (at its correct setting) by the pipeline
    import argparse as ap
    import analyse_gate
    analyse_gate.TARGET_SUFFIX = os.environ.get("ANALYSIS_SUFFIX", "-rp1")
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


def _acc_summary(seed_map):
    """'70.48% (mean of 3 seeds, range [68.69, 71.65])'."""
    accs = sorted(v[0] for v in seed_map.values())
    mean = float(np.mean(accs)) * 100
    if len(accs) == 1:
        return f"{mean:.2f}% (1 seed)"
    return (f"{mean:.2f}% (mean of {len(accs)} seeds, "
            f"range [{accs[0]*100:.2f}, {accs[-1]*100:.2f}])")


def _paired_stats(seed_a, seed_b):
    """Per-seed paired McNemar over the seeds both arms share and for which a
    local outcome vector exists. Returns [(seed, delta, p), ...]."""
    from stats_utils import mcnemar_test
    out = []
    for s in sorted(set(seed_a) & set(seed_b)):
        (acc_a, vec_a), (acc_b, vec_b) = seed_a[s], seed_b[s]
        if not vec_a or not vec_b:
            continue
        _, p = mcnemar_test(vec_b, vec_a)          # B vs A, question-aligned
        out.append((s, acc_b - acc_a, p))
    return out


def _paired_verdict(per_seed):
    """Read a per-seed McNemar sweep as one verdict. A real effect is
    significant in the same direction in every seed; anything else is reported
    as what it is — a seed-dependent difference, not a finding."""
    detail = ", ".join(f"seed {s} p={p:.4f} ({d*100:+.2f} pts)"
                       for s, d, p in per_seed)
    n = len(per_seed)
    n_sig = sum(1 for _, _, p in per_seed if p < 0.05)
    one_way = len({d > 0 for _, d, _ in per_seed}) == 1
    if n_sig == n and one_way:
        verdict = (f"significant in all {n} seeds and in the same direction — "
                   f"a consistent difference")
    elif n_sig == 0:
        verdict = f"significant in none of the {n} seeds — no difference"
    else:
        verdict = (f"significant in {n_sig}/{n} seeds"
                   + ("" if one_way else " and the sign flips across seeds")
                   + " — not a consistent effect")
    return f"Per-seed paired McNemar: {detail} — {verdict}."


def _stale_note(source_suffix):
    """Label a mechanism/failure statement whose numbers did not come from the
    rp=1.0 checkpoints. Both analyse_gate.py and analyse_failures.py strip the
    run suffix from their keys, so the setting cannot be read off a key name —
    it comes from gate_analysis.json's recorded _source_suffix, or from the
    checkpoint paths that analyse_failures.py stores. `None` means the section
    pre-dates that record, which is the case worth flagging loudest: it is
    almost certainly a run made before the --variant_suffix argparse fix, i.e.
    rp=1.2."""
    if source_suffix == "-rp1":
        return ""
    if source_suffix is None:
        return ("  [!] SOURCE: not recorded — written before the provenance "
                "marker, so most likely the rp=1.2 checkpoints (the "
                "--variant_suffix argparse bug). Re-run analyse_gate.py at "
                "--variant_suffix=-rp1 before citing this.")
    return (f"  [!] SOURCE: '{source_suffix or 'rp=1.2 main cells'}' checkpoints "
            f"— re-run over the rp=1.0 runs before citing this alongside the "
            f"accuracy numbers above.")


def key_findings(seed_arms, failure, gate, matrix):
    def arm(size, variant, rp=PRIMARY_RP):
        # The COLD, standard-LoRA, original-objective arm. The key must carry
        # every field of the arm key (size, variant, rp, lora, warm, dr): a
        # shorter tuple never matches and every statement silently goes empty.
        return seed_arms.get((size, variant, rp, "full", None, None, None))

    def pair_stmt(size, a, b):
        sa, sb = arm(size, a), arm(size, b)
        if not sa or not sb:
            missing = ", ".join(v for v, s in ((a, sa), (b, sb)) if not s)
            return (f"PENDING — no rp={PRIMARY_RP:.1f} evaluation of {missing} at "
                    f"{size} (run run_ablations.sh -- --repetition_penalty 1.0, "
                    f"then eval_all.py).")
        delta = (np.mean([v[0] for v in sb.values()])
                 - np.mean([v[0] for v in sa.values()])) * 100
        head = (f"At {size} (rp={PRIMARY_RP:.1f}), {variant_label(a)} scores "
                f"{_acc_summary(sa)} vs {variant_label(b)} {_acc_summary(sb)}; "
                f"Δ = {delta:+.2f} points.")
        per_seed = _paired_stats(sa, sb)
        if per_seed:
            return f"{head} {_paired_verdict(per_seed)}"
        return (f"{head} Per-seed paired McNemar unavailable here — the "
                f"per-question outcome vectors for these arms are not on this "
                f"machine (checkpoints live on the cluster); Δ is the "
                f"seed-mean difference only.")

    lines = []
    add = lines.append

    gate_sources = (gate or {}).get("_source_suffix", {})

    def _suffix_from_checkpoints(payload):
        """Infer a section's rp setting from any checkpoint paths it recorded,
        for JSONs written before the _source_suffix marker existed. A dir named
        '...-rp1' is the corrected setting; a bare '...' dir is the rp=1.2 main
        cell. Returns '-rp1', '' , 'mixed', or None (no paths recorded)."""
        if payload is None:
            return None
        dirs = re.findall(r"experiments/([0-9a-z._+-]+)/checkpoint-\d+",
                          json.dumps(payload))
        if not dirs:
            return None
        if all("-rp1" in d for d in dirs):
            return "-rp1"
        if all("-rp1" not in d for d in dirs):
            return ""
        return "mixed"

    def add_stale(suffix):
        note = _stale_note(suffix)
        if note:
            add(note)

    def add_stale_section(section):
        """Flag a gate_analysis.json section not computed at rp=1.0. Prefer the
        recorded _source_suffix; fall back to the section's checkpoint paths so an
        older JSON (pre-marker) is still read correctly rather than false-flagged."""
        suffix = gate_sources.get(section)
        if suffix is None:
            suffix = _suffix_from_checkpoints((gate or {}).get(section))
        add_stale(suffix)

    add("KEY FINDINGS — one paragraph per research question (spec Section 13).")
    add("Generated by generate_report.py from results/; regenerate after new")
    add("training/evaluation runs to refresh the numbers.")
    add("")
    add(f"ALL accuracies below are the corrected rp={PRIMARY_RP:.1f} setting, averaged over")
    add("seed replicates, with per-seed paired McNemar rather than a single-seed")
    add("test. The rp=1.2 runs are excluded: that off-paper repetition_penalty")
    add("cost every arm ~7 points INCLUDING the non-latent GRPO baseline, so any")
    add("comparison drawn at rp=1.2 is confounded, not a property of the method.")
    add("")

    add("RQ1 (does any latent blending help — v2 vs v1, per scale):")
    for size in SIZES:
        add(f"  [{size}] " + pair_stmt(size, "v1_grpo_baseline", "v2_fixed_blend"))
    add("")

    add("RQ2 (does learning Λ help — v3 vs v2, 1.5b):")
    add("  " + pair_stmt("1.5b", "v2_fixed_blend", "v3_learned_lambda"))
    add("")

    add("RQ3 (does input-dependent gating help — v4 vs v3, 1.5b):")
    add("  " + pair_stmt("1.5b", "v3_learned_lambda", "v4_full_hrpo_gate"))
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
        d = abs((num or 0) - (word or 0))
        # A bare num>word comparison flags a 5th-decimal noise gap as a real
        # effect. Require a meaningful gap (>=1e-3 ~ 1% of the ~0.12 ratio)
        # before claiming content sensitivity; below it, the gate is constant.
        if d < 1e-3:
            interp = (f"Effectively identical (|Δ|={d:.5f}) — no preferential opening "
                      f"for numbers over words; the gate is not content-sensitive.")
        elif (num or 0) > (word or 0):
            interp = "Numbers receive a higher ratio than filler words."
        else:
            interp = "Words receive a higher ratio than numbers."
        add(f"  From {k}: mean hidden ratio on number tokens = {num}, on word "
            f"tokens = {word} (correct responses, think span, "
            f"n={v.get('n_examples')} examples). " + interp)
        add_stale_section(k)
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
                # The frozen v2 is the null reference (Λ never trained). Learned
                # specialisation means trained CV EXCEEDS it; here trained CV
                # matches or undercuts it, so the spread is initialisation, not
                # learning. Decide from the data, not a hardcoded "CV >> 0".
                v2cv = (cvs.get("v2_fixed_blend") or {}).get("cv")
                trained = [e["cv"] for vn, e in cvs.items()
                           if vn != "v2_fixed_blend" and e.get("cv") is not None]
                if v2cv is not None and trained and max(trained) <= v2cv + 1e-3:
                    interp = (f"Every trained variant's CV is at or below the frozen-init "
                              f"reference v2 ({v2cv}) — the per-dimension spread is "
                              f"initialisation, NOT learned specialisation.")
                elif v2cv is not None and trained:
                    interp = (f"Trained CV exceeds the frozen-init reference v2 ({v2cv}) "
                              f"— evidence of per-dimension specialisation.")
                else:
                    interp = "Compare v2 (frozen init) as the null reference."
                add(f"  [{size}] {stmt}. {interp}")
                add_stale_section(f"plot2_lambda_cv_{size}")
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
        add_stale_section(k)
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
        add_stale_section(k)
    else:
        add("  PENDING — run analyse_gate.py --static (Plot 6).")
    add("")

    add("RQ8 (at what scale does HRPO stop helping over GRPO):")
    for size in SIZES:
        add(f"  [{size}] " + pair_stmt(size, "v1_grpo_baseline", "v4_full_hrpo_gate"))
    add("")

    add("RQ9 (do reward shaping v5 and repetition handling v6 help over v4):")
    add("  " + pair_stmt("1.5b", "v4_full_hrpo_gate", "v5_hrpo_format_reward"))
    add("  " + pair_stmt("1.5b", "v5_hrpo_format_reward", "v6_hrpo_full"))
    add("")

    add("RQ10 (is the wall-clock overhead justified by the accuracy gain):")
    # Timing is a property of the architecture (gate on/off), so the rp=1.2
    # overhead cells stand; the accuracy gain they are weighed against is
    # recomputed at rp=1.0, where it is a fraction of what rp=1.2 implied.
    if matrix and seed_arms:
        done = False
        for key, cell in matrix.get("cells", {}).items():
            size, variant = key.split("-", 1)
            wc = cell.get("timing", {}).get("wall_clock_overhead_pct")
            s_v, s_1 = arm(size, variant), arm(size, "v1_grpo_baseline")
            if wc is not None and wc > 0 and s_v and s_1:
                gain = (np.mean([v[0] for v in s_v.values()])
                        - np.mean([v[0] for v in s_1.values()])) * 100
                add(f"  [{key} @ rp={PRIMARY_RP:.1f}] accuracy gain {gain:+.2f} points at "
                    f"{wc:.2f}% generation overhead = {gain / wc:+.3f} points "
                    f"per overhead %.")
                done = True
        if not done:
            add(f"  PENDING — requires overhead_matrix.json cells plus rp="
                f"{PRIMARY_RP:g} evaluations (positive measured overhead).")
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
        add_stale_section((p8 or var)[0])
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
            # analyse_failures.py strips the suffix from its tag but keeps the
            # checkpoint paths, so provenance is read from those.
            paths = f"{analysis['hrpo_checkpoint']} {analysis['grpo_checkpoint']}"
            add_stale("-rp1" if "-rp1/" in paths else "")
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
    # Drop rows that are not full-test-set evaluations: a partial eval (e.g. a
    # stale 100-sample run) is not comparable to the 1319-question rows and
    # would silently corrupt Table 1 and the accuracy figures.
    full = [r for r in comparison if r.get("total", 0) >= FULL_TEST_SIZE]
    for r in comparison:
        if r not in full:
            print(f"[warn] excluding partial eval from report: "
                  f"{r['model_size']}-{r['variant_id']} (n={r.get('total')} "
                  f"< {FULL_TEST_SIZE})")
    comparison = full
    failure = load_json(os.path.join(RESULTS, "failure_analysis.json"))
    gate = load_json(os.path.join(RESULTS, "gate_analysis.json"))
    matrix = load_json(os.path.join(RESULTS, "overhead_matrix.json"))

    seed_arms = _seed_arms()
    primary_arms = _primary_arms(seed_arms)
    table0_primary(primary_arms)       # PRIMARY: rp=1.0 ablation chain
    fig_ablation_rp1(primary_arms)     # PRIMARY figure — rp=1.0 only
    fig_scale_rp1(primary_arms)        # rp=1.0 scale (once >=2 sizes present)
    table1_ablation(comparison)        # secondary reference (single-run cells)
    table2_mcnemar(comparison)
    table3_failure_modes(failure)
    table4_overhead(matrix)
    table5_lambda(gate)

    ensure_static_gate_plots()

    # Retire figures that mixed in the confounded rp=1.2 setting; the report is
    # now reported entirely at rp=1.0. (The rp=1.2 checkpoints remain on disk as
    # the data of record; they are simply no longer plotted.)
    # (replicate_summary.png is no longer listed here: analyse_replicates now
    # draws it at rp=1.0 only, so it carries no confounded cell and is the
    # figure of record for the controls.)
    for stale in ("accuracy_primary_rp.png",):
        p = os.path.join(PLOTS_DIR, stale)
        if os.path.exists(p):
            os.remove(p)
            print(f"[cleanup] removed stale rp=1.2-bearing figure: {p}")

    # failure plots are produced by analyse_failures.py; report their state
    fplots = glob.glob(os.path.join(PLOTS_DIR, "failure_*.png"))
    if fplots:
        print(f"[fig] failure plots present: {len(fplots)}")
    else:
        print("[fig] failure plots pending: run analyse_failures.py")

    key_findings(seed_arms, failure, gate, matrix)
    print("\nReport artefacts written under results/.")


if __name__ == "__main__":
    main()
