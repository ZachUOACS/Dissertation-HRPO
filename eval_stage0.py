"""Zero-shot GSM8K evaluation of the Stage-0 warm starts (brief check 4).

Stage-0 directories are deliberately named so they do NOT match
EXPERIMENT_NAME_RE — eval_all.py must never mistake a warm start for an ablation
result — so they need their own entry point. eval_gsm8k.py's __main__ also can't
be used directly: it infers the base model and temperature from the checkpoint
path (a `-temp` suffix), which these directories do not have.

Gate mode mirrors how each warm start was trained:
  --blend on  -> "unit"     (v3 configuration: unit gates, Lambda active)
  --blend off -> "disabled" (plain LoRA SFT, no gate code path)

Results go to eval_results_stage0.json inside each Stage-0 directory, never to
eval_results.json, so nothing can leak into the ablation comparison table.

    python eval_stage0.py --sizes 1.5b --seed 42
"""
import argparse
import json
import os

from eval_gsm8k import evaluate_model
from variant_common import MODEL_NAMES

RESULTS_NAME = "eval_results_stage0.json"


def stage0_dir(size, blend, r_min, seed):
    return f"./experiments/{size}-stage0-{blend}-rmin{r_min:g}-seed{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_size", default="1.5b", choices=sorted(MODEL_NAMES))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--r_min", type=float, default=0.98)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--num_samples", type=int, default=None)
    args = ap.parse_args()

    rows = []
    for blend, gate_mode in (("on", "unit"), ("off", "disabled")):
        d = stage0_dir(args.model_size, blend, args.r_min, args.seed)
        if not os.path.isdir(d):
            print(f"[skip] {d} does not exist")
            continue
        save_path = os.path.join(d, RESULTS_NAME)
        if os.path.exists(save_path):
            with open(save_path) as f:
                acc = json.load(f)["metrics"]["accuracy"]
            print(f"[skip] {d} already evaluated: {acc*100:.2f}%")
            rows.append((blend, acc))
            continue
        print(f"[eval] {d} (gate_mode={gate_mode})")
        m = evaluate_model(
            model_path=MODEL_NAMES[args.model_size],
            adapter_path=d,
            temperature=args.temperature,
            is_inference=True,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            save_results=True,
            gate_mode=gate_mode,
            save_path=save_path,
        )
        rows.append((blend, m["accuracy"]))

    print("\n=== STAGE-0 ZERO-SHOT GSM8K (check 4) ===")
    print(f"  {'arm':<26} {'accuracy':>9}   vs base 61.11")
    print(f"  {'base (zero-shot, ref)':<26} {61.11:>8.2f}%        —")
    for blend, acc in rows:
        label = f"stage0 --blend {blend}"
        print(f"  {label:<26} {acc*100:>8.2f}%   {acc*100-61.11:+8.2f}")
    print("\n  (blend on feeds arm T1/v3; blend off feeds arm C3/v1)")


if __name__ == "__main__":
    main()
