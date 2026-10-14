"""
eval_all.py — Task 2 evaluation pipeline (spec Section 8).

1. Discovers checkpoints under experiments/ without a (full) eval_results.json.
2. Evaluates each base model zero-shot as reference rows.
3. Runs GSM8K evaluation via eval_gsm8k.evaluate_model with the gate mode
   matching each variant's training configuration (v1/base: gate disabled;
   v2/v3: unit gates; v4-v6 and legacy checkpoints: stock gate).
4. Builds the comparison table with Wilson 95% CIs and McNemar tests
   (vs v1 and vs the immediately preceding variant in the causal chain).
5. Saves results/comparison_table.json + .csv and prints a summary table.

Usage:
    python eval_all.py                     # evaluate whatever is missing, build table
    python eval_all.py --skip_eval        # only rebuild the table from saved results
    python eval_all.py --num_samples 50   # quick smoke run (table marked partial)
"""

import unsloth  # noqa: F401  (must be imported before transformers/trl pieces)

import argparse
import csv
import gc
import glob
import json
import math
import os
import re

import torch

from eval_gsm8k import evaluate_model
from stats_utils import wilson_ci, mcnemar_test
from variant_common import (MODEL_NAMES, VARIANT_CHAIN, GATE_MODES, training_incomplete,
                            parse_experiment_name)

FULL_TEST_SIZE = 1319

# Scope the study to a subset of sizes with HRPO_SIZES (space-separated), e.g.
# HRPO_SIZES=1.5b to report 1.5B only. Defaults to every size.
SIZES = os.environ.get("HRPO_SIZES", "0.5b 1.5b 3b").split()

LEGACY_DIR_RE = re.compile(r"^Qwen2\.5-(0\.5B|1\.5B|3B)-Instruct-gsm8k-.*$")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def final_checkpoint(exp_dir: str):
    """Highest-step checkpoint holding either trainable weights (needed to run
    an evaluation) or an existing eval_results.json (enough to rebuild the
    comparison table). Accepting the latter lets `--skip_eval` regenerate the
    table on a machine that has only the small analysis files synced, instead
    of silently dropping every run whose adapter is not present locally."""
    ckpts = []
    for d in glob.glob(os.path.join(exp_dir, "checkpoint-*")):
        m = re.match(r"checkpoint-(\d+)$", os.path.basename(d))
        if m and (os.path.exists(os.path.join(d, "adapter_model.safetensors"))
                  or os.path.exists(os.path.join(d, "eval_results.json"))):
            ckpts.append((int(m.group(1)), d))
    return max(ckpts)[1] if ckpts else None


def get_run_temperature(exp_dir: str, default: float = 0.5) -> float:
    """Eq.3 uses the training temperature even for greedy eval (mapping #5)."""
    summary_path = os.path.join(exp_dir, "train_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                return float(json.load(f)["args"]["temperature"])
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    m = re.search(r"-temp([\d.]+)", exp_dir)
    if m:
        return float(m.group(1).rstrip("."))
    return default


def discover_experiments():
    """Yield dicts: {model_size, variant_id, exp_dir, checkpoint, gate_mode,
    temperature, in_chain}."""
    found = []
    for exp_dir in sorted(glob.glob("experiments/*")):
        name = os.path.basename(exp_dir)
        ckpt = final_checkpoint(exp_dir)
        if ckpt is None:
            continue
        parsed = parse_experiment_name(name)
        if parsed:
            size, base_variant, suffix = parsed
            found.append({
                "model_size": size,
                # Reported id keeps the suffix so replicates/controls are
                # distinguishable rows ...
                "variant_id": base_variant + suffix,
                "base_variant": base_variant,
                "exp_dir": exp_dir,
                "checkpoint": ckpt,
                # ... but the gate configuration MUST come from the base
                # variant, or a follow-up run is evaluated with the wrong
                # gate mode and its accuracy is meaningless.
                "gate_mode": GATE_MODES.get(base_variant, "standard"),
                "temperature": get_run_temperature(exp_dir),
                # Which dataset this run was trained on, from its -ds suffix.
                "task": "math" if "-dsmath" in suffix else "gsm8k",
                "training_complete": not training_incomplete(exp_dir),
                "in_chain": base_variant in VARIANT_CHAIN and not suffix,
            })
            continue
        m = LEGACY_DIR_RE.match(name)
        if m:
            size = m.group(1).lower()
            found.append({
                "model_size": size,
                "variant_id": "legacy_" + name,
                "exp_dir": exp_dir,
                "checkpoint": ckpt,
                "gate_mode": "standard",
                "temperature": get_run_temperature(exp_dir),
                "task": "gsm8k",
                "training_complete": not training_incomplete(exp_dir),
                "in_chain": False,
            })
    return found


# "Latent off" evaluations go to their OWN file so they can never be confused
# with, or overwrite, the headline eval_results.json. eval_is_complete is keyed
# on the path, so the two are never interchangeable.
NORMAL_RESULTS = "eval_results.json"
LATENT_OFF_RESULTS = "eval_results_latent_off.json"


def eval_is_complete(path: str, expected_total: int) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            data = json.load(f)
        return data["metrics"]["total"] >= expected_total
    except (json.JSONDecodeError, KeyError, OSError):
        return False


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

# Reported test-set size per task. FULL_TEST_SIZE (1319) is GSM8K's; a MATH
# run evaluated against it would look permanently incomplete and be skipped.
TASK_TEST_SIZE = {"gsm8k": FULL_TEST_SIZE, "math": 500}

def run_missing_evals(runs, batch_size, num_samples, sizes_present,
                      latent_off=False):
    """Evaluate everything still missing. Each model is evaluated inside its
    own try/except: a failure (e.g. a base model that will not download) is
    recorded and the remaining evaluations still run, so one bad model cannot
    cost a whole GPU allocation. Failures are re-reported at the end.

    latent_off=True re-evaluates each TRAINED checkpoint with the Eq.4 blend
    switched off (gate_mode="disabled" -> disable_gate sets answer_start="", so
    is_thinking is False for every token and the model runs on pure token
    embeddings). Weights are untouched; only the decode path changes. Results go
    to eval_results_latent_off.json. Base models are skipped: they have no latent
    channel and their normal eval already runs with the gate disabled."""
    def expected_for(task):
        return num_samples if num_samples else TASK_TEST_SIZE.get(task, FULL_TEST_SIZE)
    results_filename = LATENT_OFF_RESULTS if latent_off else NORMAL_RESULTS
    failures = []

    def attempt(label, **kwargs):
        try:
            evaluate_model(**kwargs)
            return True
        except Exception as exc:                      # noqa: BLE001
            print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
            failures.append((label, f"{type(exc).__name__}: {exc}"))
            return False
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Base models zero-shot (reference rows — not part of the ablation chain,
    # so a failure here must never block the trained-checkpoint evals).
    os.makedirs("results/base_model_evals", exist_ok=True)
    tasks_present = sorted({r.get("task", "gsm8k") for r in runs}) or ["gsm8k"]
    for size in (() if latent_off else sizes_present):
      for task in tasks_present:
        # GSM8K keeps the original filename so existing base evals still count.
        suffix = "" if task == "gsm8k" else f"_{task}"
        save_path = f"results/base_model_evals/{size}_base{suffix}.json"
        if eval_is_complete(save_path, expected_for(task)):
            print(f"[skip] base {size} {task} ({save_path} complete)")
            continue
        print(f"[eval] base zero-shot {size} {task} (gate disabled)")
        attempt(
            f"base {size} {task}",
            model_path=MODEL_NAMES[size],
            adapter_path=None,
            temperature=0.5,
            is_inference=True,
            batch_size=batch_size,
            num_samples=num_samples,
            save_results=True,
            gate_mode="disabled",
            save_path=save_path,
            task=task,
        )

    # Trained checkpoints (the dissertation results).
    for run in runs:
        save_path = os.path.join(run["checkpoint"], results_filename)
        if eval_is_complete(save_path, expected_for(run.get("task", "gsm8k"))):
            print(f"[skip] {run['model_size']}-{run['variant_id']} ({save_path} complete)")
            continue
        gate_mode = "disabled" if latent_off else run["gate_mode"]
        print(f"[eval{'/latent-off' if latent_off else ''}] "
              f"{run['model_size']}-{run['variant_id']} "
              f"({run['checkpoint']}, gate_mode={gate_mode}, "
              f"temp={run['temperature']})")
        attempt(
            f"{run['model_size']}-{run['variant_id']}"
            f"{' [latent-off]' if latent_off else ''}",
            model_path=MODEL_NAMES[run["model_size"]],
            adapter_path=run["checkpoint"],
            temperature=run["temperature"],
            is_inference=True,
            batch_size=batch_size,
            num_samples=num_samples,
            save_results=True,
            gate_mode=gate_mode,
            save_path=save_path,
            task=run.get("task", "gsm8k"),
            # Mixture temperature the run trained with (eval_d2 runs only; legacy runs
            # carry no key -> None). Meaningless with the blend off, so not passed then.
            latent_temperature=None if latent_off else run.get("latent_temperature"),
            blend_gate=None if latent_off else run.get("blend_gate"),
        )

    if failures:
        print(f"\n{len(failures)} evaluation(s) FAILED (the rest completed; "
              f"re-run eval_all.py to retry just these):")
        for label, err in failures:
            print(f"  - {label}: {err}")
    return failures


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def load_outcomes(path: str):
    """Return (metrics, per-question binary vector, questions)."""
    with open(path) as f:
        data = json.load(f)
    vec = [1 if r["correct"] else 0 for r in data["results"]]
    questions = [r["question"] for r in data["results"]]
    return data["metrics"], vec, questions


def paired(y_a, q_a, y_b, q_b):
    """Verify the two evals cover the same questions in the same order."""
    if len(y_a) != len(y_b):
        return False
    probe = [0, len(q_a) // 2, len(q_a) - 1] if q_a else []
    return all(q_a[i] == q_b[i] for i in probe)


def build_table(runs, num_samples):
    rows = []
    outcomes = {}   # (size, variant_id) -> (vec, questions)

    sizes_present = sorted({r["model_size"] for r in runs} |
                           {s for s in SIZES
                            if os.path.exists(f"results/base_model_evals/{s}_base.json")},
                           key=SIZES.index)

    for size in sizes_present:
        base_path = f"results/base_model_evals/{size}_base.json"
        if os.path.exists(base_path):
            metrics, vec, qs = load_outcomes(base_path)
            outcomes[(size, "base_zero_shot")] = (vec, qs)
            rows.append({"variant_id": "base_zero_shot", "model_size": size,
                         "metrics": metrics})

    for run in sorted(runs, key=lambda r: (SIZES.index(r["model_size"]),
                                           VARIANT_CHAIN.index(r["variant_id"])
                                           if r["variant_id"] in VARIANT_CHAIN else 99)):
        path = os.path.join(run["checkpoint"], "eval_results.json")
        if not os.path.exists(path):
            print(f"[warn] no eval results for {run['exp_dir']}; skipping row")
            continue
        metrics, vec, qs = load_outcomes(path)
        outcomes[(run["model_size"], run["variant_id"])] = (vec, qs)
        rows.append({"variant_id": run["variant_id"],
                     "model_size": run["model_size"], "metrics": metrics})

    run_complete = {(r["model_size"], r["variant_id"]): r.get("training_complete", True)
                    for r in runs}
    table = []
    for row in rows:
        size, variant = row["model_size"], row["variant_id"]
        m = row["metrics"]
        ci_lo, ci_hi = wilson_ci(m["correct"], m["total"])
        entry = {
            "variant_id": variant,
            "model_size": size,
            "task": m.get("task", "gsm8k"),
            "training_complete": run_complete.get((size, variant), True),
            "accuracy": round(m["accuracy"], 4),
            "correct": m["correct"],
            "total": m["total"],
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "vs_v1_mcnemar_chi2": None,
            "vs_v1_mcnemar_p": None,
            "vs_prev_mcnemar_p": None,
        }

        vec, qs = outcomes[(size, variant)]

        # vs v1 (same size), for every non-v1 row where pairing is valid.
        v1 = outcomes.get((size, "v1_grpo_baseline"))
        if variant != "v1_grpo_baseline" and v1 and paired(vec, qs, v1[0], v1[1]):
            chi2, p = mcnemar_test(vec, v1[0])
            entry["vs_v1_mcnemar_chi2"] = round(chi2, 4)
            entry["vs_v1_mcnemar_p"] = round(p, 6)

        # vs the immediately preceding variant present in the causal chain.
        if variant in VARIANT_CHAIN:
            idx = VARIANT_CHAIN.index(variant)
            prev = None
            for cand in reversed(VARIANT_CHAIN[:idx]):
                if (size, cand) in outcomes:
                    prev = outcomes[(size, cand)]
                    break
            if prev and paired(vec, qs, prev[0], prev[1]):
                _, p = mcnemar_test(vec, prev[0])
                entry["vs_prev_mcnemar_p"] = round(p, 6)

        table.append(entry)

    return table


# Every key build_table puts in a row must be listed here: csv.DictWriter raises
# on an unknown key, and because the JSON is written first, that failure leaves a
# correct comparison_table.json beside a stale .csv with only a traceback to show.
COLUMNS = ["variant_id", "model_size", "task", "accuracy", "correct", "total",
           "ci_lower", "ci_upper", "vs_v1_mcnemar_chi2", "vs_v1_mcnemar_p",
           "vs_prev_mcnemar_p", "training_complete"]


def save_and_print(table):
    os.makedirs("results", exist_ok=True)
    with open("results/comparison_table.json", "w") as f:
        json.dump(table, f, indent=2)
    with open("results/comparison_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in table:
            writer.writerow(row)
    print("\nSaved results/comparison_table.json and results/comparison_table.csv\n")

    fmt = "{:<28} {:<6} {:>8} {:>7} {:>6} {:>8} {:>8} {:>10} {:>10} {:>10}"
    print(fmt.format("variant", "size", "acc", "corr", "n", "ci_lo", "ci_hi",
                     "chi2_v1", "p_v1", "p_prev"))
    print("-" * 110)
    for row in table:
        print(fmt.format(
            row["variant_id"][:28], row["model_size"],
            f"{row['accuracy']:.4f}", row["correct"], row["total"],
            f"{row['ci_lower']:.4f}", f"{row['ci_upper']:.4f}",
            "-" if row["vs_v1_mcnemar_chi2"] is None else f"{row['vs_v1_mcnemar_chi2']:.3f}",
            "-" if row["vs_v1_mcnemar_p"] is None else f"{row['vs_v1_mcnemar_p']:.4f}",
            "-" if row["vs_prev_mcnemar_p"] is None else f"{row['vs_prev_mcnemar_p']:.4f}",
        ))


def write_latent_off_summary(runs, out_path="results/latent_off_summary.json"):
    """Pair each run's normal eval with its latent-off eval and report how much
    accuracy depends on the latent channel. Also checks whether the two
    per-question correctness vectors are identical — for v1 (gate already
    disabled) they MUST be, which is the sanity check for the whole mode."""
    rows = []
    for run in runs:
        normal_p = os.path.join(run["checkpoint"], NORMAL_RESULTS)
        off_p = os.path.join(run["checkpoint"], LATENT_OFF_RESULTS)
        if not (os.path.exists(normal_p) and os.path.exists(off_p)):
            continue
        try:
            with open(normal_p) as f:
                normal = json.load(f)
            with open(off_p) as f:
                off = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        nv = [bool(r["correct"]) for r in normal.get("results", [])]
        ov = [bool(r["correct"]) for r in off.get("results", [])]
        n_acc = normal.get("metrics", {}).get("accuracy")
        o_acc = off.get("metrics", {}).get("accuracy")
        rows.append({
            "model_size": run["model_size"],
            "variant_id": run["variant_id"],
            "normal_accuracy": n_acc,
            "latent_off_accuracy": o_acc,
            "delta_points": (round((o_acc - n_acc) * 100, 2)
                             if (n_acc is not None and o_acc is not None) else None),
            "identical_correctness_vector": (nv == ov and len(nv) > 0),
            "n_questions": len(nv),
        })
    rows.sort(key=lambda r: (r["model_size"], r["variant_id"]))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"rows": rows}, f, indent=2)

    print(f"\n=== LATENT-OFF SUMMARY ({len(rows)} paired runs) -> {out_path} ===")
    print(f"{'size':>5} {'variant':<42} {'normal':>8} {'off':>8} {'Δpts':>7} {'identical':>10}")
    for r in rows:
        na = f"{r['normal_accuracy']*100:.2f}" if r["normal_accuracy"] is not None else "--"
        oa = f"{r['latent_off_accuracy']*100:.2f}" if r["latent_off_accuracy"] is not None else "--"
        dp = f"{r['delta_points']:+.2f}" if r["delta_points"] is not None else "--"
        print(f"{r['model_size']:>5} {r['variant_id']:<42} {na:>8} {oa:>8} {dp:>7} "
              f"{str(r['identical_correctness_vector']):>10}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Evaluate a subset (smoke runs only; full = 1319)")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Only rebuild the comparison table")
    parser.add_argument("--sizes", nargs="+", default=None, choices=SIZES,
                        help="Restrict to these model sizes (default: all "
                             "discovered). Use to skip a size whose base model "
                             "is unavailable, e.g. --sizes 1.5b 3b")
    parser.add_argument("--skip_base", action="store_true",
                        help="Skip the base-model zero-shot reference evals")
    parser.add_argument("--latent_off", action="store_true",
                        help="Re-evaluate trained checkpoints with the Eq.4 blend "
                             "switched off (pure token embeddings; weights "
                             "untouched). Writes eval_results_latent_off.json "
                             "alongside — never overwrites eval_results.json — "
                             "then prints/writes results/latent_off_summary.json.")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Only evaluate runs whose variant_id contains one of "
                             "these substrings, e.g. --variants v1_grpo v3_learned "
                             "v4_full")
    parser.add_argument("--rp", type=float, default=None,
                        help="Only evaluate runs with this rollout repetition "
                             "penalty, read from the dir suffix (-rpX; absent = "
                             "1.2, the main-cell default). Use --rp 1.0 to target "
                             "the corrected arms.")
    args = parser.parse_args()

    runs = discover_experiments()
    if args.sizes:
        runs = [r for r in runs if r["model_size"] in args.sizes]
    if args.variants:
        runs = [r for r in runs
                if any(v in r["variant_id"] for v in args.variants)]
    if args.rp is not None:
        def _run_rp(variant_id):
            m = re.search(r"-rp([\d.]+)", variant_id)
            return float(m.group(1)) if m else 1.2
        runs = [r for r in runs if _run_rp(r["variant_id"]) == args.rp]
    print(f"Discovered {len(runs)} evaluable experiments:")
    for r in runs:
        print(f"  {r['model_size']:>5} {r['variant_id']:<40} {r['checkpoint']}")

    sizes_present = sorted({r["model_size"] for r in runs}, key=SIZES.index)

    if args.latent_off:
        # Separate mode: never touches eval_results.json or the comparison table.
        if not args.skip_eval:
            run_missing_evals(runs, args.batch_size, args.num_samples,
                              [], latent_off=True)
        write_latent_off_summary(runs)
        return

    if not args.skip_eval:
        run_missing_evals(runs, args.batch_size, args.num_samples,
                          [] if args.skip_base else sizes_present)

    table = build_table(runs, args.num_samples)
    save_and_print(table)


if __name__ == "__main__":
    main()
