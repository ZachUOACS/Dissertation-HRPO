"""
latent_off_pairs.py - per-seed McNemar for every latent-off evaluation (CPU only, no GPU job).

results/latent_off_summary.json records only the two accuracies and whether the correctness
vectors are identical. The paired test needs the discordant counts, so this re-reads, for every
checkpoint that has both files, eval_results.json (blend on) and eval_results_latent_off.json
(blend off, same weights) and reports, question by question:
    b = correct with the blend on only, c = correct with the blend off only,
    McNemar with continuity correction, p from the chi-square with one degree of freedom.

Pure standard library (p = erfc(sqrt(chi2 / 2)) is the df = 1 survival function), so it runs on
the login node or locally on synced files:

    python latent_off_pairs.py                 # every pair under experiments/
    python latent_off_pairs.py --filter dsmath # MATH only
Writes results/latent_off_pairs.json.
"""

import argparse
import glob
import json
import math
import os


def mcnemar(on, off):
    b = sum(1 for x, y in zip(on, off) if x and not y)
    c = sum(1 for x, y in zip(on, off) if y and not x)
    if b + c == 0:
        return b, c, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, math.erfc(math.sqrt(chi2 / 2))


def load_vector(path):
    with open(path) as f:
        data = json.load(f)
    return [bool(r["correct"]) for r in data["results"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--filter", default="", help="only directories containing this substring")
    parser.add_argument("--out", default="results/latent_off_pairs.json")
    args = parser.parse_args()

    rows = []
    for off_path in sorted(glob.glob(os.path.join(args.root, "*", "checkpoint-*",
                                                  "eval_results_latent_off.json"))):
        ckpt = os.path.dirname(off_path)
        run = os.path.basename(os.path.dirname(ckpt))
        on_path = os.path.join(ckpt, "eval_results.json")
        if args.filter not in run or not os.path.exists(on_path):
            continue
        on, off = load_vector(on_path), load_vector(off_path)
        if len(on) != len(off):
            print(f"[skip] {run}: {len(on)} vs {len(off)} questions")
            continue
        b, c, p = mcnemar(on, off)
        n = len(on)
        rows.append({"run": run, "checkpoint": os.path.basename(ckpt), "n": n,
                     "on": round(100 * sum(on) / n, 2), "off": round(100 * sum(off) / n, 2),
                     "delta_off_minus_on": round(100 * (sum(off) - sum(on)) / n, 2),
                     "on_only": b, "off_only": c, "p_mcnemar": p})

    print(f"{'run':58s} {'n':>5s} {'on':>6s} {'off':>6s} {'delta':>6s} {'on only':>7s} "
          f"{'off only':>8s} {'p':>7s}")
    for r in rows:
        print(f"{r['run']:58s} {r['n']:5d} {r['on']:6.2f} {r['off']:6.2f} "
              f"{r['delta_off_minus_on']:+6.2f} {r['on_only']:7d} {r['off_only']:8d} "
              f"{r['p_mcnemar']:7.4f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n{len(rows)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
