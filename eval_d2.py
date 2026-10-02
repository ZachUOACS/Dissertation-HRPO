"""
eval_d2.py - evaluate the D2-programme arms (experiments/new-*) with exactly the
protocol eval_all.py uses for the settled arms.

Why a separate driver: variant_common.experiment_dir files every arm that moves
the rollout temperature, the learning rate or the exploration noise under a
"new-" prefix, which the legacy EXPERIMENT_NAME_RE deliberately cannot match (so
eval_all / analyse_replicates / generate_report never pool a treatment with its
control). This script owns those directories and reuses eval_all's evaluation
driver unchanged: same evaluate_model call, same gate mode (from the BASE variant),
same Eq.3 temperature (the run's own training temperature, read from
train_summary.json), greedy single decode, full test set.

Noise is only ever applied inside Trainer.training_step, so a saved checkpoint
holds clean weights and evaluation needs no noise handling.

Usage (from the repo root, via cluster/run_eval_d2.slurm):
    python eval_d2.py                    # evaluate whatever is missing
    python eval_d2.py --latent_off       # also score each checkpoint with the blend off
    python eval_d2.py --variants v1_grpo # restrict by substring
"""

import unsloth  # noqa: F401  (must be imported before transformers/trl pieces)

import argparse
import glob
import os
import re

import gc
import json

import torch

from eval_all import (TASK_TEST_SIZE, discover_experiments, final_checkpoint,
                      get_run_temperature, run_missing_evals)
from variant_common import GATE_MODES, MODEL_NAMES, training_incomplete

NEW_RE = re.compile(
    r"^new-(?P<size>0\.5b|1\.5b|3b)-(?P<variant>v\d_[a-z_]+?)"
    r"(?P<suffix>-(?:seed|rp|lora|warm|dr|ds|t|lr|noise).*)?$"
)


def discover_new_runs():
    runs = []
    for exp_dir in sorted(glob.glob("experiments/new-*")):
        m = NEW_RE.match(os.path.basename(exp_dir))
        if not m:
            print(f"[skip] {exp_dir}: name does not parse as a new-arm directory")
            continue
        ckpt = final_checkpoint(exp_dir)
        if ckpt is None:
            continue
        if training_incomplete(exp_dir):
            print(f"[skip] {exp_dir}: training did not finish (newest checkpoint is a "
                  f"mid-run snapshot, not an arm)")
            continue
        variant, suffix = m.group("variant"), m.group("suffix") or ""
        runs.append({
            "model_size": m.group("size"),
            "variant_id": variant + suffix,
            "base_variant": variant,
            "exp_dir": exp_dir,
            "checkpoint": ckpt,
            "gate_mode": GATE_MODES.get(variant, "standard"),
            "temperature": get_run_temperature(exp_dir),
            "task": "math" if "-dsmath" in suffix else "gsm8k",
            "training_complete": True,
        })
    return runs


def discover_baselines():
    """The settled cold Dr. GRPO arms (GSM8K and MATH): the controls every D2 treatment
    is compared with. Their greedy results already exist; they are listed here only so
    the sampled pass@k evaluation covers control and treatment with one protocol."""
    runs = []
    for run in discover_experiments():
        if not run.get("training_complete", True) or "base_variant" not in run:
            continue
        suffix = run["variant_id"][len(run["base_variant"]):]
        tag = re.sub(r"-seed\d+", "", suffix)
        if tag in ("-rp1-drboth", "-rp1-drboth-dsmath"):
            runs.append(run)
    return runs


def passk_is_complete(path, expected_total, k):
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            metrics = json.load(f)["metrics"]
        return metrics["total"] >= expected_total and metrics.get("k") == k
    except (json.JSONDecodeError, KeyError, OSError):
        return False


def run_missing_passk(runs, k, batch_size, num_samples, sample_seed):
    """Sampled evaluation for every run still missing it. One failure never costs the
    rest of the allocation (same policy as eval_all.run_missing_evals)."""
    from eval_passk import evaluate_model_sampled
    failures = []
    for run in runs:
        task = run.get("task", "gsm8k")
        expected = num_samples if num_samples else TASK_TEST_SIZE.get(task, 1319)
        save_path = os.path.join(run["checkpoint"], f"eval_results_pass{k}.json")
        label = f"{run['model_size']}-{run['variant_id']}"
        if passk_is_complete(save_path, expected, k):
            print(f"[skip] {label} ({save_path} complete)")
            continue
        print(f"[pass@{k}] {label} ({run['checkpoint']}, gate_mode={run['gate_mode']}, "
              f"temp={run['temperature']})")
        try:
            evaluate_model_sampled(
                model_path=MODEL_NAMES[run["model_size"]],
                adapter_path=run["checkpoint"],
                temperature=run["temperature"],
                k=k,
                batch_size=batch_size,
                num_samples=num_samples,
                gate_mode=run["gate_mode"],
                save_path=save_path,
                task=task,
                sample_seed=sample_seed,
            )
        except Exception as exc:                      # noqa: BLE001
            print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
            failures.append((label, f"{type(exc).__name__}: {exc}"))
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if failures:
        print(f"\n{len(failures)} pass@{k} evaluation(s) FAILED (re-run to retry just these):")
        for label, err in failures:
            print(f"  - {label}: {err}")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--passk", type=int, default=0,
                        help="k > 0: sampled evaluation (avg@k / pass@k / maj@k / tokens) of "
                             "the new arms AND their Dr. GRPO baselines, written to "
                             "eval_results_pass{k}.json. Never touches eval_results.json.")
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Smoke runs only; full = 1319 (GSM8K) / 500 (MATH)")
    parser.add_argument("--latent_off", action="store_true",
                        help="Score each checkpoint with the Eq.4 blend switched off; "
                             "writes eval_results_latent_off.json beside the normal one")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Only runs whose id contains one of these substrings")
    args = parser.parse_args()

    runs = discover_new_runs()
    if args.variants:
        runs = [r for r in runs if any(v in r["variant_id"] for v in args.variants)]
    print(f"Discovered {len(runs)} evaluable new-arm experiments:")
    for r in runs:
        print(f"  {r['model_size']:>5} {r['variant_id']:<58} temp={r['temperature']}")

    if args.passk > 0:
        baselines = discover_baselines()
        if args.variants:
            baselines = [r for r in baselines
                         if any(v in r["variant_id"] for v in args.variants)]
        print(f"plus {len(baselines)} Dr. GRPO baseline(s) for the sampled evaluation:")
        for r in baselines:
            print(f"  {r['model_size']:>5} {r['variant_id']:<58} temp={r['temperature']}")
        # batch_size counts questions here; the model sees batch_size * k sequences.
        failures = run_missing_passk(baselines + runs, args.passk,
                                     max(1, args.batch_size // 2), args.num_samples,
                                     args.sample_seed)
        raise SystemExit(1 if failures else 0)

    failures = run_missing_evals(runs, args.batch_size, args.num_samples, [],
                                 latent_off=args.latent_off)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
