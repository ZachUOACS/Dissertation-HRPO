"""
analyse_replicates.py — seed-replicate & factor-aware summary (follow-up runs).

The main comparison table (eval_all.py) treats every experiment directory as a
standalone row, so seed replicates and the repetition-penalty / LoRA controls
appear as separate, unlinked rows. This script groups runs into *arms* — one
per (size, base variant, repetition penalty, LoRA target) — collects their seed
replicates, and reports:

  * per-arm accuracy: mean, min–max range, and each seed;
  * the level-field gate test  v3 vs v4  at each repetition-penalty setting
    (per-seed paired McNemar where seeds match, plus the mean difference);
  * the penalty effect  rp=1.2 vs rp=1.0  within each variant, matched by seed.

Outputs results/replicate_analysis.json and results/plots/replicate_summary.png,
and prints a summary. No GPU or torch — reads saved eval_results.json only.

Usage: python analyse_replicates.py
"""

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from stats_utils import wilson_ci, mcnemar_test
from variant_common import parse_experiment_name, training_incomplete
from plot_style import apply_style, variant_color, variant_label

plt = apply_style()

RESULTS = "results"
PLOTS_DIR = os.path.join(RESULTS, "plots")
FULL = 1319
# MATH is reported on MATH-500, so a MATH arm never reaches FULL and would
# be dropped as a partial evaluation.
TASK_FULL = {"gsm8k": FULL, "math": 500}


# ---------------------------------------------------------------------------
# Parse the experiment-dir suffix into its factors
# ---------------------------------------------------------------------------

def parse_suffix(suffix: str):
    """'-seed43-rp1-loraattention' -> {'seed':43,'rp':1.0,'lora':'attention'}.
    Defaults match the main ablation: seed 42, rp 1.2, LoRA 'full'. The rp
    suffix is written with %g, so '-rp1' means 1.0."""
    seed = 42
    rp = 1.2
    lora = "full"
    warm = None
    m = re.search(r"-seed(\d+)", suffix)
    if m:
        seed = int(m.group(1))
    m = re.search(r"-rp([\d.]+)", suffix)
    if m:
        rp = float(m.group(1))
    m = re.search(r"-lora([a-z]+)", suffix)
    if m:
        lora = m.group(1)
    # A Stage-0 warm start is a separate arm: pooling it with the cold run would
    # average a treatment with its own control under one label.
    m = re.search(r"-warm([a-z0-9]+)", suffix)
    if m:
        warm = m.group(1)
    # Dr. GRPO de-bias (control C2) changes the objective itself, so it is a
    # separate arm for the same reason: -dr{len,std,both}; absent = original.
    dr = None
    m = re.search(r"-dr([a-z]+)", suffix)
    if m:
        dr = m.group(1)
    # Second task. A MATH arm is not a replicate of a GSM8K arm under any
    # circumstances, so it gets its own field rather than being pooled.
    ds = None
    m = re.search(r"-ds([a-z0-9]+)", suffix)
    if m:
        ds = m.group(1)
    return seed, rp, lora, warm, dr, ds


def load_vector(exp_dir: str):
    """(accuracy, binary outcome vector) from the final full eval, or None."""
    best = None
    for d in glob.glob(os.path.join(exp_dir, "checkpoint-*")):
        m = re.match(r"checkpoint-(\d+)$", os.path.basename(d))
        p = os.path.join(d, "eval_results.json")
        if m and os.path.exists(p):
            step = int(m.group(1))
            if best is None or step > best[0]:
                best = (step, p)
    if not best:
        return None
    with open(best[1]) as f:
        data = json.load(f)
    task = data["metrics"].get("task", "gsm8k")
    if data["metrics"]["total"] < TASK_FULL.get(task, FULL):
        return None
    vec = [1 if r["correct"] else 0 for r in data["results"]]
    return sum(vec) / len(vec), vec


def arm_label(size, variant, rp, lora, warm=None, dr=None, ds=None):
    # Dataset first: it is the most significant axis, and a label that buries it
    # invites reading a MATH number as a GSM8K one.
    tag = f"{size} {'GSM8K' if ds is None else ds.upper()} {variant_label(variant)}"
    if rp != 1.2:
        tag += f" · rp={rp:g}"
    if lora != "full":
        tag += f" · {lora} LoRA"
    if warm:
        tag += f" · warm-start {warm}"
    if dr:
        tag += f" · Dr. GRPO ({dr})"
    return tag


# ---------------------------------------------------------------------------
# Discover and group
# ---------------------------------------------------------------------------

def discover_arms():
    """{(size, variant, rp, lora, warm, dr, ds): {seed: (acc, vec)}}."""
    arms = defaultdict(dict)
    for exp_dir in sorted(glob.glob("experiments/*")):
        parsed = parse_experiment_name(os.path.basename(exp_dir))
        if not parsed:
            continue
        size, variant, suffix = parsed
        seed, rp, lora, warm, dr, ds = parse_suffix(suffix)
        if training_incomplete(exp_dir):
            print(f"[skip] {os.path.basename(exp_dir)}: training did not finish — "
                  f"its newest checkpoint is a mid-run snapshot, not an arm")
            continue
        loaded = load_vector(exp_dir)
        if loaded is None:
            continue
        arms[(size, variant, rp, lora, warm, dr, ds)][seed] = loaded
    return arms


def summarise_arm(seed_map):
    accs = {s: v[0] for s, v in seed_map.items()}
    vals = list(accs.values())
    lo, hi = wilson_ci(round(np.mean(vals) * FULL), FULL)  # CI of the mean acc
    return {
        "seeds": {str(s): round(a, 4) for s, a in sorted(accs.items())},
        "n_seeds": len(vals),
        "mean": round(float(np.mean(vals)), 4),
        "min": round(float(min(vals)), 4),
        "max": round(float(max(vals)), 4),
        "range": round(float(max(vals) - min(vals)), 4),
        "mean_wilson_ci": [round(lo, 4), round(hi, 4)],
    }


def paired_by_seed(arm_a, arm_b):
    """Per-seed paired McNemar between two arms on the shared seeds, plus the
    across-seed mean difference (B − A)."""
    shared = sorted(set(arm_a) & set(arm_b))
    per_seed = {}
    for s in shared:
        chi2, p = mcnemar_test(arm_b[s][1], arm_a[s][1])   # B vs A, aligned
        per_seed[str(s)] = {"a_acc": round(arm_a[s][0], 4),
                            "b_acc": round(arm_b[s][0], 4),
                            "delta": round(arm_b[s][0] - arm_a[s][0], 4),
                            "mcnemar_chi2": round(chi2, 3),
                            "mcnemar_p": round(p, 5)}
    mean_a = np.mean([arm_a[s][0] for s in arm_a])
    mean_b = np.mean([arm_b[s][0] for s in arm_b])
    # Paired mean over the seeds both arms share (rigorous when n differs);
    # equals the arm-mean difference once both arms have the same seed set.
    paired_delta = ([arm_b[s][0] - arm_a[s][0] for s in shared] or [float("nan")])
    return {
        "shared_seeds": shared,
        "n_a": len(arm_a), "n_b": len(arm_b),
        "mean_delta_arm_means": round(float(mean_b - mean_a), 4),
        "mean_delta_paired_shared": round(float(np.mean(paired_delta)), 4),
        "per_seed": per_seed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def arm_sort_key(key):
    """Deterministic, total ordering for (size, variant, rp, lora, warm, dr, ds).

    Plain tuple sorting breaks once optional fields are set: T1 and A1 tie on
    the first four fields, so Python compares warm='on' with warm=None and
    raises TypeError (the same holds for dr and ds). Unset (None) sorts first,
    and dataset leads so MATH arms group together rather than interleaving."""
    size, variant, rp, lora, warm, dr, ds = key
    return (ds or "", size, variant, rp, lora, warm or "", dr or "")


def main():
    arms = discover_arms()
    if not arms:
        print("No evaluated experiments found.")
        return

    summary = {arm_label(*k): summarise_arm(v)
               for k, v in sorted(arms.items(), key=lambda kv: arm_sort_key(kv[0]))}

    # ---- headline comparisons (only emitted when both arms exist) ----------
    comps = {}

    def add_comparison(name, key_a, key_b):
        # Keys written before an optional field existed omit its trailing
        # None; pad to full arity, or they silently never match (the bug that
        # left key_findings "PENDING" for arms that existed).
        key_a = tuple(key_a) + (None,) * (7 - len(key_a))
        key_b = tuple(key_b) + (None,) * (7 - len(key_b))
        if key_a in arms and key_b in arms and (set(arms[key_a]) & set(arms[key_b])):
            comps[name] = {"a": arm_label(*key_a), "b": arm_label(*key_b),
                           **paired_by_seed(arms[key_a], arms[key_b])}

    for size in ("1.5b", "3b"):
        # the level-field gate test, at each penalty setting
        add_comparison(f"{size}: gate test @ rp1.2 (v4 vs v3)",
                       (size, "v3_learned_lambda", 1.2, "full", None),
                       (size, "v4_full_hrpo_gate", 1.2, "full", None))
        add_comparison(f"{size}: gate test @ rp1.0 (v4 vs v3)",
                       (size, "v3_learned_lambda", 1.0, "full", None),
                       (size, "v4_full_hrpo_gate", 1.0, "full", None))
        # Stage-0 warm start vs its cold counterpart (brief arms T1 vs A1, C3 vs v1)
        add_comparison(f"{size}: warm start on v3 @ rp1.0 (T1 vs A1)",
                       (size, "v3_learned_lambda", 1.0, "full", None),
                       (size, "v3_learned_lambda", 1.0, "full", "on"))
        add_comparison(f"{size}: warm start on v1 @ rp1.0 (C3 vs v1)",
                       (size, "v1_grpo_baseline", 1.0, "full", None),
                       (size, "v1_grpo_baseline", 1.0, "full", "off"))
        # The headline question (does the latent channel beat plain GRPO?)
        # asked again with BOTH arms warm-started — T1 vs C3 differ only in the
        # latent pathway, exactly as A1 vs v1 do in the cold chain.
        add_comparison(f"{size}: latent vs GRPO, both warm @ rp1.0 (T1 vs C3)",
                       (size, "v1_grpo_baseline", 1.0, "full", "off"),
                       (size, "v3_learned_lambda", 1.0, "full", "on"))
        # Control C1: does the latent need to be input-dependent at all? v3 with
        # the Eq.3 projection replaced by one learned vector u.
        add_comparison(f"{size}: constant vector vs projection @ rp1.0 (C1 vs A1)",
                       (size, "v3_learned_lambda", 1.0, "full", None),
                       (size, "v7_constant_vector", 1.0, "full", None))
        # Control C2: Dr. GRPO de-bias (constant length normaliser, no /std).
        add_comparison(f"{size}: Dr. GRPO on v1 @ rp1.0 (C2-v1 vs v1)",
                       (size, "v1_grpo_baseline", 1.0, "full", None, None),
                       (size, "v1_grpo_baseline", 1.0, "full", None, "both"))
        add_comparison(f"{size}: Dr. GRPO on v3 @ rp1.0 (C2-v3 vs A1)",
                       (size, "v3_learned_lambda", 1.0, "full", None, None),
                       (size, "v3_learned_lambda", 1.0, "full", None, "both"))
        # The headline question once more, under the de-biased objective: if
        # GRPO's length/difficulty biases are why HRPO ties GRPO, the tie
        # should break here.
        add_comparison(f"{size}: latent vs GRPO, both Dr. GRPO @ rp1.0 (C2-v3 vs C2-v1)",
                       (size, "v1_grpo_baseline", 1.0, "full", None, "both"),
                       (size, "v3_learned_lambda", 1.0, "full", None, "both"))
        # Interaction (tier11): the warm start was only ever measured under the
        # standard objective, whose length bias acts on exactly what the warm
        # start changes most — completion length.
        add_comparison(f"{size}: warm start under Dr. GRPO @ rp1.0 (T1+C2 vs C2-v3)",
                       (size, "v3_learned_lambda", 1.0, "full", None, "both"),
                       (size, "v3_learned_lambda", 1.0, "full", "on", "both"))
        add_comparison(f"{size}: warm start under Dr. GRPO, v1 @ rp1.0 (C3+C2 vs C2-v1)",
                       (size, "v1_grpo_baseline", 1.0, "full", None, "both"),
                       (size, "v1_grpo_baseline", 1.0, "full", "off", "both"))
        add_comparison(f"{size}: Dr. GRPO on the warm start @ rp1.0 (T1+C2 vs T1)",
                       (size, "v3_learned_lambda", 1.0, "full", "on", None),
                       (size, "v3_learned_lambda", 1.0, "full", "on", "both"))
        add_comparison(f"{size}: latent vs GRPO, both warm + Dr. GRPO @ rp1.0 (T1+C2 vs C3+C2)",
                       (size, "v1_grpo_baseline", 1.0, "full", "off", "both"),
                       (size, "v3_learned_lambda", 1.0, "full", "on", "both"))
        # Second task: the same latent-vs-GRPO question on MATH. Compared
        # only within MATH — the two tasks share no questions and no scale.
        add_comparison(f"{size}: latent vs GRPO on MATH @ rp1.0 (v3 vs v1)",
                       (size, "v1_grpo_baseline", 1.0, "full", None, None, "math"),
                       (size, "v3_learned_lambda", 1.0, "full", None, None, "math"))
        # MATH under the de-biased objective (the standard-objective MATH runs
        # collapse — see the tier10 comment in submit_followups.sh).
        add_comparison(f"{size}: latent vs GRPO on MATH, de-biased @ rp1.0 (v3 vs v1)",
                       (size, "v1_grpo_baseline", 1.0, "full", None, "both", "math"),
                       (size, "v3_learned_lambda", 1.0, "full", None, "both", "math"))
        add_comparison(f"{size}: Dr. GRPO rescues MATH? v3 standard vs de-biased",
                       (size, "v3_learned_lambda", 1.0, "full", None, None, "math"),
                       (size, "v3_learned_lambda", 1.0, "full", None, "both", "math"))
        add_comparison(f"{size}: Dr. GRPO rescues MATH? v1 standard vs de-biased",
                       (size, "v1_grpo_baseline", 1.0, "full", None, None, "math"),
                       (size, "v1_grpo_baseline", 1.0, "full", None, "both", "math"))
        # MATH warm start (tier12). Dr. GRPO throughout: the standard objective
        # collapses on MATH, so the only usable cold baseline is the de-biased one.
        add_comparison(f"{size}: warm start on MATH, v3 @ rp1.0 (warm vs cold, Dr. GRPO)",
                       (size, "v3_learned_lambda", 1.0, "full", None, "both", "math"),
                       (size, "v3_learned_lambda", 1.0, "full", "on", "both", "math"))
        add_comparison(f"{size}: warm start on MATH, v1 @ rp1.0 (warm vs cold, Dr. GRPO)",
                       (size, "v1_grpo_baseline", 1.0, "full", None, "both", "math"),
                       (size, "v1_grpo_baseline", 1.0, "full", "off", "both", "math"))
        add_comparison(f"{size}: latent vs GRPO on MATH, both warm + Dr. GRPO (v3 vs v1)",
                       (size, "v1_grpo_baseline", 1.0, "full", "off", "both", "math"),
                       (size, "v3_learned_lambda", 1.0, "full", "on", "both", "math"))
        # penalty effect within each variant
        add_comparison(f"{size}: penalty effect on v4 (rp1.2 -> rp1.0)",
                       (size, "v4_full_hrpo_gate", 1.2, "full", None),
                       (size, "v4_full_hrpo_gate", 1.0, "full", None))
        add_comparison(f"{size}: penalty effect on v3 (rp1.2 -> rp1.0)",
                       (size, "v3_learned_lambda", 1.2, "full", None),
                       (size, "v3_learned_lambda", 1.0, "full", None))

    payload = {"arms": summary, "comparisons": comps}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "replicate_analysis.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # ---- figure: mean + range per arm, grouped by penalty ------------------
    plot_arms(arms)

    # ---- stdout ------------------------------------------------------------
    print("\n=== Arms (accuracy: mean [min, max] over seeds) ===")
    for lab, s in summary.items():
        seeds = " ".join(f"{k}:{v*100:.2f}" for k, v in s["seeds"].items())
        print(f"  {lab:<38} {s['mean']*100:5.2f}  [{s['min']*100:.2f}, {s['max']*100:.2f}]"
              f"  n={s['n_seeds']}   ({seeds})")
    print("\n=== Key comparisons (per-seed paired McNemar; Δ = B − A) ===")
    for name, c in comps.items():
        print(f"  {name}   (n_a={c['n_a']}, n_b={c['n_b']})")
        print(f"     Δ arm-means = {c['mean_delta_arm_means']*100:+.2f} pts   "
              f"Δ paired(shared seeds) = {c['mean_delta_paired_shared']*100:+.2f} pts")
        for seed, d in c["per_seed"].items():
            sig = "*" if d["mcnemar_p"] < 0.05 else " "
            print(f"     seed {seed}: {d['a_acc']*100:.2f} -> {d['b_acc']*100:.2f} "
                  f"({d['delta']*100:+.2f})  p={d['mcnemar_p']:.4g}{sig}")
    print(f"\nSaved results/replicate_analysis.json and "
          f"results/plots/replicate_summary.png")


def plot_arms(arms):
    """Mean accuracy with min–max range bars, one row per arm, grouped by rp."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    # rp = 1.2 is the confounded setting and appears in no reported figure —
    # drawing it here is exactly why generate_report used to delete this file.
    # Scoped to rp = 1.0, this is the one figure showing the controls (C1, C2,
    # T1, C3) beside the settled chain. Grouped by size; arm_sort_key breaks the
    # remaining ties, so T1 sits directly under A1 instead of crashing the sort.
    order = sorted((k for k in arms if k[2] == 1.0 and k[6] is None),
                   key=lambda k: (k[0],) + arm_sort_key(k))
    rows = []
    for k in order:
        size, variant, rp, lora, warm, dr, ds = k
        accs = [v[0] for v in arms[k].values()]
        rows.append((arm_label(*k), variant, np.mean(accs), min(accs), max(accs),
                     len(accs)))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.6, max(2.4, 0.46 * len(rows) + 1)))
    for i, (lab, variant, mean, lo, hi, n) in enumerate(rows):
        y = len(rows) - 1 - i
        col = variant_color(variant)
        if hi > lo:
            ax.plot([lo, hi], [y, y], color=col, linewidth=2, alpha=.55,
                    solid_capstyle="round", zorder=2)
            for x in (lo, hi):
                ax.plot([x, x], [y - .16, y + .16], color=col, linewidth=1.4, zorder=2)
        ax.plot(mean, y, "o", color=col, markersize=8,
                markeredgecolor=plt.rcParams["axes.facecolor"], markeredgewidth=1.5,
                zorder=3)
        ax.text(hi + .004, y, f"{mean*100:.1f}" + (f"  (n={n})" if n > 1 else ""),
                va="center", ha="left", fontsize=8.5,
                color="#3a3a37", fontfamily="monospace")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=8.5)
    ax.set_xlabel("GSM8K accuracy")
    ax.set_xlim(0.55, 0.90)
    ax.set_title("GSM8K seed replicates & controls at rp = 1.0 — mean with min–max range",
                 fontsize=10, pad=10)
    ax.grid(axis="y", visible=False)
    fig.savefig(os.path.join(PLOTS_DIR, "replicate_summary.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
