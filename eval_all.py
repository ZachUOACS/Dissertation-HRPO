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
from variant_common import MODEL_NAMES, VARIANT_CHAIN, GATE_MODES

FULL_TEST_SIZE = 1319

SIZES = ["0.5b", "1.5b", "3b"]

VARIANT_DIR_RE = re.compile(r"^(0\.5b|1\.5b|3b)-(v\d_[a-z_]+)$")
LEGACY_DIR_RE = re.compile(r"^Qwen2\.5-(0\.5B|1\.5B|3B)-Instruct-gsm8k-.*$")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def final_checkpoint(exp_dir: str):
    ckpts = []
    for d in glob.glob(os.path.join(exp_dir, "checkpoint-*")):
        m = re.match(r"checkpoint-(\d+)$", os.path.basename(d))
        if m and os.path.exists(os.path.join(d, "adapter_model.safetensors")):
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
        m = VARIANT_DIR_RE.match(name)
        if m:
            size, variant = m.group(1), m.group(2)
            found.append({
                "model_size": size,
                "variant_id": variant,
                "exp_dir": exp_dir,
                "checkpoint": ckpt,
                "gate_mode": GATE_MODES.get(variant, "standard"),
                "temperature": get_run_temperature(exp_dir),
                "in_chain": variant in VARIANT_CHAIN,
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
                "in_chain": False,
            })
    return found


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

def run_missing_evals(runs, batch_size, num_samples, sizes_present):
    expected_total = num_samples if num_samples else FULL_TEST_SIZE

    # Base models zero-shot (reference rows).
    os.makedirs("results/base_model_evals", exist_ok=True)
    for size in sizes_present:
        save_path = f"results/base_model_evals/{size}_base.json"
        if eval_is_complete(save_path, expected_total):
            print(f"[skip] base {size} ({save_path} complete)")
            continue
        print(f"[eval] base zero-shot {size} (gate disabled)")
        evaluate_model(
            model_path=MODEL_NAMES[size],
            adapter_path=None,
            temperature=0.5,
            is_inference=True,
            batch_size=batch_size,
            num_samples=num_samples,
            save_results=True,
            gate_mode="disabled",
            save_path=save_path,
        )
        gc.collect()
        torch.cuda.empty_cache()

    # Trained checkpoints.
    for run in runs:
        save_path = os.path.join(run["checkpoint"], "eval_results.json")
        if eval_is_complete(save_path, expected_total):
            print(f"[skip] {run['model_size']}-{run['variant_id']} ({save_path} complete)")
            continue
        print(f"[eval] {run['model_size']}-{run['variant_id']} "
              f"({run['checkpoint']}, gate_mode={run['gate_mode']}, "
              f"temp={run['temperature']})")
        evaluate_model(
            model_path=MODEL_NAMES[run["model_size"]],
            adapter_path=run["checkpoint"],
            temperature=run["temperature"],
            is_inference=True,
            batch_size=batch_size,
            num_samples=num_samples,
            save_results=True,
            gate_mode=run["gate_mode"],
            save_path=save_path,
        )
        gc.collect()
        torch.cuda.empty_cache()


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

    table = []
    for row in rows:
        size, variant = row["model_size"], row["variant_id"]
        m = row["metrics"]
        ci_lo, ci_hi = wilson_ci(m["correct"], m["total"])
        entry = {
            "variant_id": variant,
            "model_size": size,
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


COLUMNS = ["variant_id", "model_size", "accuracy", "correct", "total",
           "ci_lower", "ci_upper", "vs_v1_mcnemar_chi2", "vs_v1_mcnemar_p",
           "vs_prev_mcnemar_p"]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Evaluate a subset (smoke runs only; full = 1319)")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Only rebuild the comparison table")
    args = parser.parse_args()

    runs = discover_experiments()
    print(f"Discovered {len(runs)} evaluable experiments:")
    for r in runs:
        print(f"  {r['model_size']:>5} {r['variant_id']:<40} {r['checkpoint']}")

    sizes_present = sorted({r["model_size"] for r in runs}, key=SIZES.index)
    if not args.skip_eval:
        run_missing_evals(runs, args.batch_size, args.num_samples, sizes_present)

    table = build_table(runs, args.num_samples)
    save_and_print(table)


if __name__ == "__main__":
    main()
