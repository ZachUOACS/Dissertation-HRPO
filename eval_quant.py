"""
eval_quant.py - weight-only quantisation study: greedy evaluation of trained checkpoints with
their weights quantised (weight_quant.py), with the latent blend on and, for latent arms, off.

Each (checkpoint, scheme) writes its own file beside the normal results, so nothing here can
overwrite or be mistaken for eval_results.json:
    eval_results_wq-{scheme}.json             blend as trained
    eval_results_wq-{scheme}_latent_off.json  blend switched off (latent arms only)
The bf16 scheme (merge only, no quantisation) is the reference every other scheme is compared
with, so it is always included. Protocol otherwise identical to eval_all/eval_d2: same
evaluate_model, gate mode from the base variant, the run's own Eq.3 temperature, greedy, batch 32.

Default selection: the settled cold Dr. GRPO arms (eval_d2.discover_baselines) of --task.

Usage (from the repo root, via QUANT=1 cluster/run_eval_d2.slurm):
    python eval_quant.py                                  # MATH, v1 and v3, all seeds
    python eval_quant.py --schemes bf16 nf4 --variants v3_learned_lambda-seed43-rp1-drboth-dsmath
    python eval_quant.py --no_latent_off --task gsm8k
Then: python analyse_quant.py
"""

import unsloth  # noqa: F401  (must be imported before transformers/trl pieces)

import argparse
import gc
import os

import torch

from eval_all import TASK_TEST_SIZE, eval_is_complete
from eval_d2 import discover_baselines, discover_new_runs
from eval_gsm8k import evaluate_model
from variant_common import MODEL_NAMES
from weight_quant import SCHEMES


def result_path(run, scheme, latent_off):
    name = f"eval_results_wq-{scheme}{'_latent_off' if latent_off else ''}.json"
    return os.path.join(run["checkpoint"], name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schemes", nargs="+", default=["bf16", "int8", "int4", "nf4"],
                        choices=SCHEMES)
    parser.add_argument("--task", default="math", choices=["math", "gsm8k"])
    parser.add_argument("--variants", nargs="+", default=None,
                        help="exact run ids (e.g. v3_learned_lambda-seed43-rp1-drboth-dsmath); "
                             "searched among the Dr. GRPO baselines and the new-* arms")
    parser.add_argument("--sizes", nargs="+", default=["1.5b", "3b"])
    parser.add_argument("--no_latent_off", action="store_true",
                        help="skip the blend-off evaluations of the latent arms")
    # 32, as for every settled greedy result (see eval_d2.py on why the batch size matters).
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=None, help="smoke runs only")
    args = parser.parse_args()

    schemes = ["bf16"] + [s for s in args.schemes if s != "bf16"]
    if args.variants:
        pool = discover_baselines() + discover_new_runs()
        runs = [r for r in pool if r["variant_id"] in args.variants]
        missing = set(args.variants) - {r["variant_id"] for r in runs}
        if missing:
            raise SystemExit(f"no finished run with id(s): {sorted(missing)}")
    else:
        runs = [r for r in discover_baselines() if r.get("task", "gsm8k") == args.task]
    runs = [r for r in runs if r["model_size"] in args.sizes]
    if not runs:
        raise SystemExit("nothing selected")

    jobs = []
    for run in runs:
        latent = run["gate_mode"] != "disabled"
        for scheme in schemes:
            jobs.append((run, scheme, False))
            if latent and not args.no_latent_off:
                jobs.append((run, scheme, True))
    print(f"{len(runs)} run(s) x schemes {schemes}: {len(jobs)} evaluation(s)")
    for r in runs:
        print(f"  {r['model_size']:>5} {r['variant_id']:<52} gate={r['gate_mode']} "
              f"temp={r['temperature']}")

    failures = []
    for run, scheme, latent_off in jobs:
        task = run.get("task", "gsm8k")
        save_path = result_path(run, scheme, latent_off)
        label = f"{run['model_size']}-{run['variant_id']} [{scheme}{', latent-off' if latent_off else ''}]"
        if eval_is_complete(save_path, args.num_samples or TASK_TEST_SIZE[task]):
            print(f"[skip] {label} ({save_path} complete)")
            continue
        print(f"[eval] {label}")
        try:
            evaluate_model(
                model_path=MODEL_NAMES[run["model_size"]],
                adapter_path=run["checkpoint"],
                temperature=run["temperature"],
                is_inference=True,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                save_results=True,
                gate_mode="disabled" if latent_off else run["gate_mode"],
                save_path=save_path,
                task=task,
                latent_temperature=None if latent_off else run.get("latent_temperature"),
                blend_gate=None if latent_off else run.get("blend_gate"),
                weight_quant=scheme,
            )
        except Exception as exc:                      # noqa: BLE001
            print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
            failures.append((label, f"{type(exc).__name__}: {exc}"))
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if failures:
        print(f"\n{len(failures)} evaluation(s) FAILED (re-run to retry just these):")
        for label, err in failures:
            print(f"  - {label}: {err}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
