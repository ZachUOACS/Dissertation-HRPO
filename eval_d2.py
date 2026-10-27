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
from model_registry import SIZE_ALT, TASK_SUFFIX, baseline_tag, task_from_name
from variant_common import GATE_MODES, MODEL_NAMES, training_incomplete

NEW_RE = re.compile(
    rf"^new-(?P<size>{SIZE_ALT})-(?P<variant>v\d_[a-z_]+?)"
    r"(?P<suffix>-(?:seed|rp|lora|warm|dr|ds|t|lr|noise|lt|sl|dg|lp).*)?$"
)


def get_run_blend_gate(exp_dir):
    """The D3 blend settings a run trained with, as the keyword dict set_blend_gate takes, or
    None. train_summary.json first; the "-dg{max}d{d0}to{d1}" directory token is the fallback,
    so a gated run is never evaluated with the ungated blend."""
    summary_path = os.path.join(exp_dir, "train_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                a = json.load(f)["args"]
            if a.get("blend_gate", "off") != "off":
                return {"blend_gate": a["blend_gate"], "blend_max": float(a["blend_max"]),
                        "blend_d0": float(a["blend_d0"]), "blend_d1": float(a["blend_d1"])}
            if "blend_gate" in a:
                return None
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            pass
    num = r"(\d+(?:\.\d+)?(?:e-?\d+)?)"
    m = re.search(rf"-dg{num}d{num}to{num}", os.path.basename(exp_dir))
    if m:
        return {"blend_gate": "divergence", "blend_max": float(m.group(1)),
                "blend_d0": float(m.group(2)), "blend_d1": float(m.group(3))}
    return None


def get_run_latent_temperature(exp_dir):
    """The Eq.3 mixture temperature a run trained with (--latent_temperature), or None.

    Read from train_summary.json like the sampling temperature; the "-lt{value}"
    directory token is the fallback, so a run whose summary is missing is still
    evaluated with the mixture it was trained with, never silently at the default."""
    summary_path = os.path.join(exp_dir, "train_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                value = json.load(f)["args"].get("latent_temperature")
            if value is not None:
                return float(value)
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    m = re.search(r"-lt([\d.]+)", os.path.basename(exp_dir))
    return float(m.group(1).rstrip(".")) if m else None


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
            "latent_temperature": get_run_latent_temperature(exp_dir),
            "blend_gate": get_run_blend_gate(exp_dir),
            "task": task_from_name(suffix),
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
        if tag in tuple("-rp1-" + baseline_tag(t) for t in TASK_SUFFIX):
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
                latent_temperature=run.get("latent_temperature"),
                blend_gate=run.get("blend_gate"),
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
    # 32 is not a tuning choice: every settled greedy result was produced by
    # `eval_all.py --batch_size 32` (cluster/run_eval_and_analysis.slurm), and in bf16 the
    # greedy decode depends on how prompts are padded together. At 16 the same checkpoint
    # (1.5b-v3_learned_lambda-rp1-drboth) scored 959/1319 against its settled 968/1319,
    # with 149 questions changing correctness. New arms must be scored like their baselines.
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Smoke runs only; full = 1319 (GSM8K) / 500 (MATH)")
    parser.add_argument("--latent_off", action="store_true",
                        help="Score each checkpoint with the Eq.4 blend switched off; "
                             "writes eval_results_latent_off.json beside the normal one")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Only runs whose id contains one of these substrings")
    parser.add_argument("--exact", action="store_true",
                        help="with --variants: match run ids exactly instead of by substring, so a "
                             "baseline id does not also select every new arm built on it")
    # Run ids carry no model size (a 3B seed-42 baseline has the 1.5B one's id), so a 3B run is
    # selected by --sizes, never by id alone.
    parser.add_argument("--sizes", nargs="+", default=None,
                        help="only runs of these model sizes (default: all)")
    parser.add_argument("--baselines", action="store_true",
                        help="greedy / latent-off mode: also score the finished cold Dr. GRPO "
                             "baselines selected by --variants/--sizes (how the 3B scale-check "
                             "runs are scored; the 1.5B ones are all complete and are skipped)")
    args = parser.parse_args()

    def selected(variant_id):
        if args.exact:
            return variant_id in args.variants
        return any(v in variant_id for v in args.variants)

    def sized(run_list):
        return [r for r in run_list if not args.sizes or r["model_size"] in args.sizes]

    runs = sized(discover_new_runs())
    if args.variants:
        runs = [r for r in runs if selected(r["variant_id"])]
    if args.baselines and args.passk == 0:
        baselines = sized(discover_baselines())
        if args.variants:
            baselines = [r for r in baselines if selected(r["variant_id"])]
        print(f"Discovered {len(baselines)} finished Dr. GRPO baseline(s) to score:")
        for r in baselines:
            print(f"  {r['model_size']:>5} {r['variant_id']:<58} temp={r['temperature']}")
        runs = baselines + runs
    print(f"Discovered {len(runs)} evaluable experiments:")
    for r in runs:
        print(f"  {r['model_size']:>5} {r['variant_id']:<58} temp={r['temperature']}"
              f" latent_temp={r.get('latent_temperature')} blend_gate={r.get('blend_gate')}")

    if args.passk > 0:
        baselines = sized(discover_baselines())
        if args.variants:
            baselines = [r for r in baselines if selected(r["variant_id"])]
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
