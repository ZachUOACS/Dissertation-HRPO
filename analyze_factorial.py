"""Paired analysis of the fusion x debias factorial from per-question eval_results.json.

McNemar (exact, two-sided) on the qid-aligned correct/incorrect flags for each contrast, plus the
2x2 factorial decomposition (main effects + interaction) on accuracy. No external deps.

Single seed  -> the original per-contrast + factorial report.
Multiple seeds (glob spans -s42/-s43/-s44/...) -> per-seed factorial, then a CROSS-SEED aggregate:
  per-arm accuracy mean +/- std, main-effect/interaction mean +/- std with sign-consistency, and a
  POOLED (stratified) McNemar per contrast (discordant pairs summed across seeds -> one exact test).

Usage:
  python analyze_factorial.py --glob 'runs/*-s42-rmin0.95-b-1-full/eval_results.json'   # single seed
  python analyze_factorial.py --glob 'runs/*-rmin0.95-b-1-full/eval_results.json'        # all seeds
"""
import argparse
import glob
import json
import math
import re

ARM_ORDER = ["baseline", "debias", "fusion", "fusion_debias",
             "fusion_token", "fusion_adapter", "len_only", "std_only"]


def binom_two_sided(b, c):
    """Exact two-sided McNemar p-value: P(Binom(b+c, 0.5) <= min(b,c)) * 2."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2 * p)


def sig_mark(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    """Sample std (ddof=1); 0 for a single value."""
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def load(pattern):
    """-> {(seed, arm): {qid: bool}}. Seed from metrics, falling back to '-s<NN>' in the path."""
    out = {}
    for f in glob.glob(pattern):
        d = json.load(open(f))
        m = d["metrics"]
        arm = m["arm"]
        seed = m.get("seed")
        if seed is None:
            mo = re.search(r"-s(\d+)", f)
            seed = int(mo.group(1)) if mo else 0
        out[(int(seed), arm)] = {r["qid"]: bool(r["correct"]) for r in d["results"]}
    return out


# ---- per-(single-seed) computations on an {arm: {qid: bool}} dict --------------------------------
CONTRASTS = [("debias", "baseline"), ("fusion", "baseline"), ("fusion_debias", "baseline"),
             ("fusion_debias", "debias"), ("fusion", "fusion_debias")]


def mcnemar(arms, qids, A, B):
    b = sum(arms[A][q] and not arms[B][q] for q in qids)
    c = sum((not arms[A][q]) and arms[B][q] for q in qids)
    return b, c, binom_two_sided(b, c)


def accuracies(arms, qids):
    N = len(qids)
    return {a: sum(arms[a][q] for q in qids) / N for a in arms}


def factorial_effects(acc):
    """(fusion_main, debias_main, interaction, additive_pred) in pp, or None if arms missing."""
    if not all(a in acc for a in ["baseline", "fusion", "debias", "fusion_debias"]):
        return None
    b_, f_, d_, fd_ = (acc[a] * 100 for a in ["baseline", "fusion", "debias", "fusion_debias"])
    fusion_main = ((f_ - b_) + (fd_ - d_)) / 2
    debias_main = ((d_ - b_) + (fd_ - f_)) / 2
    additive = f_ + d_ - b_
    interaction = fd_ - additive
    return fusion_main, debias_main, interaction, additive


def report_single(arms, qids, header=True):
    N = len(qids)
    acc = accuracies(arms, qids)
    order = [a for a in ARM_ORDER if a in arms]
    if header:
        print(f"N={N} questions (qid-aligned).")
    print("\nAccuracies:")
    for a in order:
        print(f"  {a:15s} {acc[a]*100:6.2f}%  ({sum(arms[a][q] for q in qids)}/{N})")

    print("\nMcNemar (exact, two-sided):")
    for A, B in CONTRASTS:
        if A not in arms or B not in arms:
            continue
        b, c, p = mcnemar(arms, qids, A, B)
        d = (acc[A] - acc[B]) * 100
        print(f"  {A:14s} vs {B:14s}  d={d:+7.2f}pp   {A[:4]}+{B[:4]}-={b:4d}  "
              f"{A[:4]}-{B[:4]}+={c:4d}  p={p:.2e} {sig_mark(p)}")

    eff = factorial_effects(acc)
    if eff:
        fusion_main, debias_main, interaction, additive = eff
        print("\n2x2 factorial (accuracy, pp):")
        print(f"  main effect FUSION : {fusion_main:+.2f}")
        print(f"  main effect DEBIAS : {debias_main:+.2f}")
        print(f"  INTERACTION        : {interaction:+.2f}   (additive prediction {additive:.2f})")
    return acc


# ---- cross-seed aggregate -----------------------------------------------------------------------
def report_multiseed(loaded, seeds, qids):
    N = len(qids)
    per_seed_arms = {s: {a: loaded[(s, a)] for (ss, a) in loaded if ss == s} for s in seeds}
    per_seed_acc = {s: accuracies(per_seed_arms[s], qids) for s in seeds}

    print(f"Seeds: {', '.join(map(str, seeds))}  (K={len(seeds)})   N={N} questions/seed (qid-aligned).")

    # --- per-seed factorial (compact) ---
    print("\n" + "=" * 70)
    print("PER-SEED reports")
    print("=" * 70)
    for s in seeds:
        print(f"\n----- seed {s} -----")
        report_single(per_seed_arms[s], qids, header=False)

    # --- aggregate ---
    print("\n" + "=" * 70)
    print("CROSS-SEED AGGREGATE")
    print("=" * 70)

    arms_present = [a for a in ARM_ORDER if all((s, a) in loaded for s in seeds)]
    print(f"\nPer-arm accuracy (mean +/- std across {len(seeds)} seeds, pp):")
    for a in arms_present:
        vals = [per_seed_acc[s][a] * 100 for s in seeds]
        per = "  ".join(f"s{s}={v:.2f}" for s, v in zip(seeds, vals))
        print(f"  {a:15s} {mean(vals):6.2f} +/- {std(vals):.2f}   [{per}]")

    # factorial effects per seed -> aggregate
    effs = {s: factorial_effects(per_seed_acc[s]) for s in seeds}
    if all(effs[s] for s in seeds):
        labels = ["FUSION main", "DEBIAS main", "INTERACTION"]
        print(f"\n2x2 factorial effects (mean +/- std across seeds, pp):")
        for i, lab in enumerate(labels):
            vals = [effs[s][i] for s in seeds]
            m = mean(vals)
            same = sum(1 for v in vals if (v > 0) == (m > 0) and v != 0)
            per = "  ".join(f"s{s}={v:+.2f}" for s, v in zip(seeds, vals))
            print(f"  {lab:13s} {m:+6.2f} +/- {std(vals):.2f}   sign {same}/{len(seeds)} agree   [{per}]")

    # pooled (stratified) McNemar per contrast: sum discordants across seeds -> one exact test
    print(f"\nContrasts: per-seed d/p, then POOLED McNemar (discordants summed across seeds):")
    for A, B in CONTRASTS:
        if not all((s, A) in loaded and (s, B) in loaded for s in seeds):
            continue
        print(f"\n  {A} vs {B}")
        deltas, B_pool, C_pool = [], 0, 0
        for s in seeds:
            b, c, p = mcnemar(per_seed_arms[s], qids, A, B)
            d = (per_seed_acc[s][A] - per_seed_acc[s][B]) * 100
            deltas.append(d)
            B_pool += b
            C_pool += c
            print(f"    s{s}:  d={d:+6.2f}pp  {A[:4]}+{B[:4]}-={b:4d}  {A[:4]}-{B[:4]}+={c:4d}  "
                  f"p={p:.2e} {sig_mark(p)}")
        p_pool = binom_two_sided(B_pool, C_pool)
        same = sum(1 for d in deltas if (d > 0) == (mean(deltas) > 0))
        print(f"    POOL: d_mean={mean(deltas):+6.2f}pp (sign {same}/{len(seeds)})  "
              f"sumB={B_pool}  sumC={C_pool}  p={p_pool:.2e} {sig_mark(p_pool)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/*-s42-full/eval_results.json")
    args = ap.parse_args()

    loaded = load(args.glob)
    if not loaded:
        print(f"No eval_results.json matched {args.glob}"); return

    # qid alignment across every (seed, arm) loaded
    qsets = {k: set(v) for k, v in loaded.items()}
    common = set.intersection(*qsets.values())
    if any(qsets[k] != common for k in loaded):
        print("WARNING: qid sets differ across arms/seeds; using the intersection.")
    qids = sorted(common)

    seeds = sorted({s for (s, _a) in loaded})
    if len(seeds) == 1:
        arms = {a: loaded[(seeds[0], a)] for (s, a) in loaded}
        report_single(arms, qids)
    else:
        report_multiseed(loaded, seeds, qids)


if __name__ == "__main__":
    main()
