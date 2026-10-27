"""
analyse_quant.py - summarise the weight-only quantisation study (eval_quant.py). CPU only,
standard library only, so it runs on the login node or locally on synced files.

Per checkpoint and scheme, against the same checkpoint's bf16 (merge-only) reference:
  * accuracy change, McNemar (continuity-corrected) on the paired questions;
  * mean generated tokens and the token inflation ratio, mean tokens(scheme) / mean tokens(bf16),
    after the CoT Token Inflation Ratio of arXiv 2606.25519; also on the questions both
    evaluations answer correctly, so the ratio is not driven by which questions are solved;
  * for latent arms, the latent-off gap (blend on minus blend off) at that scheme;
  * deployed weight memory from the storage format (weight_quant.effective_bits).
Per arm: the per-seed values and their mean, with sign counts. v3 against v1: the per-seed
difference in quantisation cost. No pooled p-values (one checkpoint per seed).
Also reports how many answers merging alone changed relative to eval_results.json.

    python analyse_quant.py            # writes results/quant_analysis.json
"""

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

from model_registry import SIZE_ALT  # noqa: E402  (standard library only)

DIR_RE = re.compile(rf"^(?:new-)?(?P<size>{SIZE_ALT})-(?P<variant>v\d_[a-z_]+?)(?P<suffix>-.*)?$")
FILE_RE = re.compile(r"eval_results_wq-(?P<scheme>[a-z0-9]+)(?P<off>_latent_off)?\.json$")
ORDER = ["bf16", "int8", "nf4", "int4", "int3"]


def mcnemar(a, b):
    """b = correct under a only, c = correct under b only; p from chi-square, df 1."""
    n_b = sum(1 for x, y in zip(a, b) if x and not y)
    n_c = sum(1 for x, y in zip(a, b) if y and not x)
    if n_b + n_c == 0:
        return n_b, n_c, 1.0
    chi2 = (abs(n_b - n_c) - 1) ** 2 / (n_b + n_c)
    return n_b, n_c, math.erfc(math.sqrt(chi2 / 2))


ANSWER_START = "####"                          # utils.ANSWER_START: the latent is used until it appears
TOKEN_CAP = {"math": 1024, "gsm8k": 512}       # eval_gsm8k.get_task_spec max_new


def load(path):
    with open(path) as f:
        data = json.load(f)
    res = data["results"]
    return {"metrics": data["metrics"],
            "correct": [bool(r["correct"]) for r in res],
            "tokens": [r.get("num_tokens") for r in res],
            "marker": [ANSWER_START in r.get("full_response", "") for r in res],
            "questions": [r["question"] for r in res]}


def format_rates(ev, task):
    """Share of answers that contain the "####" marker, and share cut at the token cap. The scorer
    does not need the marker (without it, extract_from_response passes the whole response on and
    process_math_answer takes the boxed value, else the LaTeX value, else the last number), so these
    say how much of an accuracy figure rests on answers that are not well formed."""
    cap = TOKEN_CAP.get(task)
    toks = [t for t in ev["tokens"] if t is not None]
    return {"marker_rate": rnd(mean([float(m) for m in ev["marker"]]), 4),
            "cap_rate": rnd(sum(t >= cap for t in toks) / len(toks), 4) if toks and cap else None}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def rnd(x, n=2):
    return None if x is None else round(x, n)


def checkpoint_rows(root):
    rows = []
    for ref_path in sorted(glob.glob(os.path.join(root, "*", "checkpoint-*", "eval_results_wq-bf16.json"))):
        ckpt = os.path.dirname(ref_path)
        run = os.path.basename(os.path.dirname(ckpt))
        m = DIR_RE.match(run)
        if not m:
            print(f"[skip] {run}: name does not parse")
            continue
        suffix = m.group("suffix") or ""
        seed_m = re.search(r"-seed(\d+)", suffix)
        ref = load(ref_path)
        row = {"run": run, "size": m.group("size"), "variant": m.group("variant"),
               "seed": int(seed_m.group(1)) if seed_m else 42,
               "tag": re.sub(r"-seed\d+", "", suffix),
               "task": ref["metrics"].get("task", "gsm8k"), "schemes": {}}
        orig_path = os.path.join(ckpt, "eval_results.json")
        if os.path.exists(orig_path):
            orig = load(orig_path)
            n = len(ref["questions"])
            # a --num_samples run scores the first n questions, so compare on that prefix
            if orig["questions"][:n] == ref["questions"]:
                orig_correct = orig["correct"][:n]
                changed = sum(1 for x, y in zip(orig_correct, ref["correct"]) if x != y)
                row["merge_check"] = {
                    "n": n,
                    "acc_unmerged": rnd(100 * mean(orig_correct)),
                    "acc_merged_bf16": rnd(100 * mean(ref["correct"])),
                    "answers_changed_correctness": changed}
        files = {}
        for path in glob.glob(os.path.join(ckpt, "eval_results_wq-*.json")):
            fm = FILE_RE.search(path)
            if fm:
                files[(fm.group("scheme"), bool(fm.group("off")))] = path
        for scheme in sorted({s for s, _ in files}, key=lambda s: ORDER.index(s) if s in ORDER else 99):
            on = load(files[(scheme, False)]) if (scheme, False) in files else None
            if on is None or on["questions"] != ref["questions"]:
                print(f"[skip] {run} {scheme}: missing or not paired with the bf16 reference")
                continue
            n_b, n_c, p = mcnemar(on["correct"], ref["correct"])
            both = [i for i, (x, y) in enumerate(zip(on["correct"], ref["correct"])) if x and y]
            t_on, t_ref = mean(on["tokens"]), mean(ref["tokens"])
            t_on_both = mean([on["tokens"][i] for i in both])
            t_ref_both = mean([ref["tokens"][i] for i in both])
            wq = on["metrics"].get("weight_quant", {})
            entry = {
                "acc": rnd(100 * mean(on["correct"])),
                "delta_acc_pp": rnd(100 * (mean(on["correct"]) - mean(ref["correct"]))),
                "mcnemar_only_quant": n_b, "mcnemar_only_bf16": n_c, "mcnemar_p": rnd(p, 4),
                "mean_tokens": rnd(t_on, 1),
                "inflation": rnd(t_on / t_ref, 4) if t_on and t_ref else None,
                "inflation_both_correct": rnd(t_on_both / t_ref_both, 4) if t_on_both and t_ref_both else None,
                "weight_gb": wq.get("weight_gb"), "effective_bits": wq.get("effective_bits"),
                "mean_relative_error": wq.get("mean_relative_error"),
                **format_rates(on, row["task"]),
            }
            if (scheme, True) in files:
                off = load(files[(scheme, True)])
                if off["questions"] == on["questions"]:
                    g_b, g_c, g_p = mcnemar(on["correct"], off["correct"])
                    rates_off = format_rates(off, row["task"])
                    entry.update({"acc_latent_off": rnd(100 * mean(off["correct"])),
                                  "latent_gap_pp": rnd(100 * (mean(on["correct"]) - mean(off["correct"]))),
                                  "latent_gap_p": rnd(g_p, 4),
                                  "mean_tokens_latent_off": rnd(mean(off["tokens"]), 1),
                                  "marker_rate_latent_off": rates_off["marker_rate"],
                                  "cap_rate_latent_off": rates_off["cap_rate"]})
            row["schemes"][scheme] = entry
        rows.append(row)
    return rows


def summarise(rows):
    arms = defaultdict(list)
    for r in rows:
        arms[(r["size"], r["variant"], r["tag"], r["task"])].append(r)
    out = []
    for (size, variant, tag, task), rs in sorted(arms.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        schemes = sorted({s for r in rs for s in r["schemes"]},
                         key=lambda s: ORDER.index(s) if s in ORDER else 99)
        per_scheme = {}
        for s in schemes:
            have = [r for r in rs if s in r["schemes"]]
            vals = lambda k: [r["schemes"][s].get(k) for r in have]
            deltas = [d for d in vals("delta_acc_pp") if d is not None]
            gaps = [g for g in vals("latent_gap_pp") if g is not None]
            per_scheme[s] = {
                "seeds": [r["seed"] for r in have],
                "acc": vals("acc"), "mean_acc": rnd(mean(vals("acc"))),
                "delta_acc_pp": deltas, "mean_delta_acc_pp": rnd(mean(deltas)),
                "k_negative": sum(d < 0 for d in deltas),
                "inflation": vals("inflation"), "mean_inflation": rnd(mean(vals("inflation")), 4),
                "mean_inflation_both_correct": rnd(mean(vals("inflation_both_correct")), 4),
                "latent_gap_pp": gaps, "mean_latent_gap_pp": rnd(mean(gaps)),
                "weight_gb": have[0]["schemes"][s].get("weight_gb") if have else None,
                **{f"mean_{k}": rnd(mean(vals(k)), 4)
                   for k in ("marker_rate", "cap_rate", "marker_rate_latent_off", "cap_rate_latent_off")},
            }
        out.append({"size": size, "variant": variant, "tag": tag, "task": task,
                    "n_seeds": len(rs), "schemes": per_scheme})
    return out


def latent_vs_token(rows):
    """Per seed: quantisation cost of the v3 arm minus that of the v1 arm with the same size,
    task, tag and seed (negative = HRPO loses more)."""
    index = {(r["size"], r["variant"], r["tag"], r["task"], r["seed"]): r for r in rows}
    comps = defaultdict(lambda: defaultdict(dict))
    for (size, variant, tag, task, seed), r3 in index.items():
        if not variant.startswith("v3"):
            continue
        r1 = index.get((size, "v1_grpo_baseline", tag, task, seed))
        if r1 is None:
            continue
        for s, e3 in r3["schemes"].items():
            e1 = r1["schemes"].get(s)
            if s == "bf16" or e1 is None:
                continue
            comps[(size, tag, task)][s][seed] = {
                "delta_v3_minus_v1_pp": rnd(e3["delta_acc_pp"] - e1["delta_acc_pp"]),
                "inflation_v3_minus_v1": (rnd(e3["inflation"] - e1["inflation"], 4)
                                          if e3["inflation"] and e1["inflation"] else None)}
    out = []
    for (size, tag, task), by_scheme in sorted(comps.items()):
        for s, per_seed in by_scheme.items():
            d = [v["delta_v3_minus_v1_pp"] for v in per_seed.values()]
            out.append({"size": size, "tag": tag, "task": task, "scheme": s,
                        "per_seed": per_seed, "mean_pp": rnd(mean(d)),
                        "k_v3_better": sum(x > 0 for x in d), "n": len(d)})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--out", default="results/quant_analysis.json")
    args = parser.parse_args()

    rows = checkpoint_rows(args.root)
    if not rows:
        raise SystemExit("no eval_results_wq-bf16.json found")
    arms, comps = summarise(rows), latent_vs_token(rows)

    print("=== Merge check (bf16 merged vs the unmerged eval_results.json) ===")
    for r in rows:
        mc = r.get("merge_check")
        if mc:
            print(f"{r['run']:<60} {mc['acc_unmerged']:.2f} -> {mc['acc_merged_bf16']:.2f} "
                  f"({mc['answers_changed_correctness']} of {mc['n']} answers changed)")
        else:
            print(f"{r['run']:<60} no comparable eval_results.json")
    print("\n=== Per arm (accuracy %, change vs bf16 reference, tokens) ===")
    for a in arms:
        print(f"\n{a['size']} {a['variant']} {a['tag']} [{a['task']}], {a['n_seeds']} seed(s)")
        for s, v in a["schemes"].items():
            gap = f" latent gap {v['mean_latent_gap_pp']:+.2f} {v['latent_gap_pp']}" if v["latent_gap_pp"] else ""
            infl = f" inflation {v['mean_inflation']:.3f} (both-correct {v['mean_inflation_both_correct']:.3f})" \
                if v["mean_inflation"] and v["mean_inflation_both_correct"] else ""
            print(f"  {s:<5} {v['weight_gb'] or 0:6.3f} GB  acc {v['mean_acc']:.2f}  "
                  f"d {v['mean_delta_acc_pp'] if v['mean_delta_acc_pp'] is not None else 0:+.2f} "
                  f"{v['delta_acc_pp']} (neg {v['k_negative']}/{len(v['delta_acc_pp'])}){infl}{gap}")
            pc = lambda x: "-" if x is None else f"{100 * x:.0f}%"
            fmt_off = (f"; latent off: #### {pc(v['mean_marker_rate_latent_off'])}, at cap "
                       f"{pc(v['mean_cap_rate_latent_off'])}" if v["mean_marker_rate_latent_off"] is not None else "")
            print(f"        format: #### {pc(v['mean_marker_rate'])}, at cap {pc(v['mean_cap_rate'])}{fmt_off}")
    if comps:
        print("\n=== HRPO (v3) minus GRPO (v1): quantisation cost, per seed ===")
        for c in comps:
            seeds = " ".join(f"s{sd}:{v['delta_v3_minus_v1_pp']:+.2f}" for sd, v in sorted(c["per_seed"].items()))
            print(f"{c['size']} {c['tag']} [{c['task']}] {c['scheme']:<5} mean {c['mean_pp']:+.2f} "
                  f"(v3 better in {c['k_v3_better']}/{c['n']})  {seeds}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"checkpoints": rows, "arms": arms, "v3_minus_v1": comps}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
