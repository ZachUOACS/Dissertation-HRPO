"""
smoke_family.py - verdict for cluster/run_smoke_family.slurm (cross-model study, Phase 1).

For each arm the smoke job trained (experiments/smoke-..., --max_steps N), checks the run and
scores its step-N checkpoint on 32 questions:
  * trained: trainer_state.json reached global_step N and the loss is finite;
  * latent active: hidden_ratio was logged and is > 0 for every latent arm (v3, v4, c1, lt), and
    absent or NaN for GRPO (v1), which has no blend;
  * scored: greedy evaluation ran with the latent on and, for latent arms, off;
  * format: share of answers containing "####" (the answer marker that also ends the latent span).
A few steps cannot show learning, so accuracy here is a sanity check, not a result. Prints one table
and PASS or FAIL, plus the start of two answers to read by eye.

    python smoke_family.py --size llama1b --steps 4
"""

import unsloth  # noqa: F401  (must be imported before transformers/trl pieces)

import argparse
import gc
import json
import math
import os

import torch

from eval_gsm8k import evaluate_model
from variant_common import GATE_MODES, MODEL_NAMES, build_arg_parser, experiment_dir

GSM8K_ARGS = ["--repetition_penalty", "1.0", "--dr_grpo", "both"]
MATH_ARGS = ["--dataset", "math", "--repetition_penalty", "1.0", "--max_completion_length", "1024",
             "--per_device_train_batch_size", "8", "--gradient_accumulation_steps", "4",
             "--dr_grpo", "both"]
MMLU_ARGS = ["--dataset", "mmlu", "--max_prompt_length", "1024", "--repetition_penalty", "1.0",
             "--dr_grpo", "both"]
# arm -> (variant id, task, extra args, latent temperature at evaluation); must match the ARMS list
# in cluster/run_smoke_family.slurm.
ARMS = {
    "v1": ("v1_grpo_baseline", "gsm8k", [], None),
    "v3": ("v3_learned_lambda", "gsm8k", [], None),
    "v4": ("v4_full_hrpo_gate", "gsm8k", [], None),
    "c1": ("v3_learned_lambda", "gsm8k", ["--stochastic_latent", "0.5"], None),
    "lt": ("v3_learned_lambda", "gsm8k", ["--latent_temperature", "1.0"], 1.0),
    "v3math": ("v3_learned_lambda", "math", [], None),
    "v1mmlu": ("v1_grpo_baseline", "mmlu", [], None),
    "v3mmlu": ("v3_learned_lambda", "mmlu", [], None),
}
DEFAULT_ARMS = ["v1", "v3", "v4", "c1", "lt", "v3math"]
N_QUESTIONS = 32


def run_dir(size, steps, variant, task, extra):
    base = {"math": MATH_ARGS, "mmlu": MMLU_ARGS}.get(task, GSM8K_ARGS)
    args = build_arg_parser(variant).parse_args(
        ["--model_size", size, "--max_steps", str(steps)] + base + extra)
    return experiment_dir(size, variant, args)


def check_training(ckpt, steps, latent):
    path = os.path.join(ckpt, "trainer_state.json")
    if not os.path.exists(path):
        return {"ok": False, "why": "no trainer_state.json"}
    with open(path) as f:
        state = json.load(f)
    logs = [h for h in state.get("log_history", []) if "loss" in h]
    hr = [h["hidden_ratio"] for h in logs if "hidden_ratio" in h
          and h["hidden_ratio"] is not None and not math.isnan(h["hidden_ratio"])]
    out = {"step": state.get("global_step"), "loss": logs[-1]["loss"] if logs else None,
           "kl": logs[-1].get("kl") if logs else None,
           "hidden_ratio": sum(hr) / len(hr) if hr else None}
    why = []
    if out["step"] != steps:
        why.append(f"global_step {out['step']} != {steps}")
    if out["loss"] is None or not math.isfinite(out["loss"]):
        why.append("loss missing or not finite")
    if latent and not (out["hidden_ratio"] and out["hidden_ratio"] > 0):
        why.append("no positive hidden_ratio logged for a latent arm")
    if not latent and out["hidden_ratio"]:
        why.append("hidden_ratio logged for GRPO, which has no blend")
    out["ok"], out["why"] = not why, "; ".join(why)
    return out


def score(size, ckpt, variant, task, latent_temperature, latent_off):
    name = f"eval_smoke{'_latent_off' if latent_off else ''}.json"
    save_path = os.path.join(ckpt, name)
    evaluate_model(model_path=MODEL_NAMES[size], adapter_path=ckpt, temperature=0.5,
                   is_inference=True, batch_size=32, num_samples=N_QUESTIONS, save_results=True,
                   gate_mode="disabled" if latent_off else GATE_MODES[variant], save_path=save_path,
                   task=task, latent_temperature=None if latent_off else latent_temperature)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with open(save_path) as f:
        res = json.load(f)["results"]
    return {"acc": 100 * sum(r["correct"] for r in res) / len(res),
            "marker": 100 * sum("####" in r["full_response"] for r in res) / len(res),
            "sample": res[0]["full_response"][:300]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="llama1b", choices=sorted(MODEL_NAMES))
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=sorted(ARMS))
    a = p.parse_args()

    rows, samples, failed = [], {}, []
    for arm in a.arms:
        variant, task, extra, lt = ARMS[arm]
        latent = GATE_MODES[variant] != "disabled"
        ckpt = os.path.join(run_dir(a.size, a.steps, variant, task, extra), f"checkpoint-{a.steps}")
        tr = check_training(ckpt, a.steps, latent)
        row = {"arm": arm, "task": task, **tr, "acc_on": None, "acc_off": None, "marker": None}
        if os.path.isdir(ckpt):
            try:
                on = score(a.size, ckpt, variant, task, lt, latent_off=False)
                row["acc_on"], row["marker"] = on["acc"], on["marker"]
                samples.setdefault(arm, on["sample"])
                if latent:
                    row["acc_off"] = score(a.size, ckpt, variant, task, lt, latent_off=True)["acc"]
            except Exception as exc:  # noqa: BLE001
                row["ok"], row["why"] = False, f"evaluation failed: {type(exc).__name__}: {exc}"
        if not row["ok"]:
            failed.append(arm)
        rows.append(row)

    fmt = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
    print(f"\n=== Smoke test: {a.size} ({MODEL_NAMES[a.size]}), {a.steps} steps, "
          f"{N_QUESTIONS} questions (a code check, not a result) ===")
    print(f"{'arm':<7}{'task':<7}{'step':>5}{'loss':>9}{'kl':>9}{'hidden':>8}"
          f"{'acc on':>8}{'acc off':>8}{'####%':>7}  status")
    for r in rows:
        print(f"{r['arm']:<7}{r['task']:<7}{fmt(r['step'], 0) if r['step'] is not None else '-':>5}"
              f"{fmt(r['loss']):>9}{fmt(r['kl']):>9}{fmt(r['hidden_ratio']):>8}"
              f"{fmt(r['acc_on'], 1):>8}{fmt(r['acc_off'], 1):>8}{fmt(r['marker'], 0):>7}  "
              f"{'ok' if r['ok'] else 'FAIL: ' + r['why']}")
    for arm in ("v1", "v3", "v3mmlu"):
        if arm in samples:
            print(f"\n--- {arm} answer, first 300 characters ---\n{samples[arm]}")
    print(f"\n{'PASS' if not failed else 'FAIL (' + ', '.join(failed) + ')'}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
