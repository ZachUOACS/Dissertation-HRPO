"""
check_eval_refactor.py - regression check for the eval_gsm8k.py refactor (GPU job).

eval_gsm8k.evaluate_model had its model-loading and task-spec blocks moved into
load_model_for_eval / get_task_spec so the sampled pass@k evaluation shares them. The
move is meant to change nothing. This re-scores ONE settled checkpoint greedily, into a
scratch file, and compares it question by question with the eval_results.json already on
disk. Greedy decoding is deterministic, so with the same batch size the two must agree.

    python check_eval_refactor.py experiments/1.5b-v3_learned_lambda-rp1-drboth
    (batch size 16 = eval_all's default, which produced the settled results)

Exit code 0 = identical. A handful of flipped questions would point at batch-size or
library drift rather than the refactor; a large difference means the refactor is wrong
and nothing new should be evaluated until it is fixed.
"""

import unsloth  # noqa: F401

import argparse
import json
import os
import sys

from eval_all import final_checkpoint, get_run_temperature
from eval_gsm8k import evaluate_model
from variant_common import GATE_MODES, MODEL_NAMES, parse_experiment_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    exp_dir = args.exp_dir.rstrip("/")
    parsed = parse_experiment_name(os.path.basename(exp_dir))
    if not parsed:
        sys.exit(f"{exp_dir} is not a settled ablation directory")
    size, variant, suffix = parsed
    ckpt = final_checkpoint(exp_dir)
    old_path = os.path.join(ckpt, "eval_results.json")
    if not os.path.exists(old_path):
        sys.exit(f"no existing {old_path} to compare with")
    with open(old_path) as f:
        old = json.load(f)

    scratch = os.path.join(ckpt, "eval_results_refactor_check.json")
    evaluate_model(
        model_path=MODEL_NAMES[size],
        adapter_path=ckpt,
        temperature=get_run_temperature(exp_dir),
        is_inference=True,
        batch_size=args.batch_size,
        num_samples=None,
        save_results=True,
        gate_mode=GATE_MODES.get(variant, "standard"),
        save_path=scratch,
        task="math" if "-dsmath" in suffix else "gsm8k",
    )
    with open(scratch) as f:
        new = json.load(f)

    old_vec = [bool(r["correct"]) for r in old["results"]]
    new_vec = [bool(r["correct"]) for r in new["results"]]
    flipped = sum(a != b for a, b in zip(old_vec, new_vec))
    text_diff = sum(a["generated_answer"] != b["generated_answer"]
                    for a, b in zip(old["results"], new["results"]))
    print(f"\nsettled : {old['metrics']['correct']}/{old['metrics']['total']}")
    print(f"refactor: {new['metrics']['correct']}/{new['metrics']['total']}")
    print(f"questions whose correctness changed: {flipped}; whose extracted answer changed: {text_diff}")
    os.remove(scratch)
    if len(old_vec) != len(new_vec) or flipped:
        sys.exit("MISMATCH: do not trust new evaluations until this is explained")
    print("IDENTICAL: the refactor did not change the greedy evaluation")


if __name__ == "__main__":
    main()
