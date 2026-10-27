"""
analyse_d2.py - compare the D2-programme arms with their matched baselines.

Reads saved eval_results.json and trainer_state.json only (no GPU, no torch).

Arms
  * baselines: the settled Dr. GRPO arms, experiments/{size}-{variant}[-seedN]-rp1-drboth
  * new arms : experiments/new-{size}-{variant}-...-rp1-drboth-{t|lr|noise...}[-seedN]
An arm is (size, variant, tag): `tag` is everything after the seed/rp tokens, so
"drboth" is the baseline and e.g. "drboth-noise0.05to0.005" is a treatment.

What it reports (the project's statistics rules)
  * per-seed accuracy, mean, SD, Wilson CI on the pooled-seed mean;
  * for every treatment against the baseline with the same (size, variant):
    per-seed delta, per-seed McNemar p (paired on the same questions), the mean and SD
    of the seed deltas, and k-of-n seeds positive / significant-and-positive;
  * mean completion length over the last 100 training steps (efficiency);
  * per-seed change in mean greedy TEST tokens per answer, when eval_tokens.py has written
    eval_tokens.json for both sides (the length-penalty arms are judged on this).
It never prints a pooled p-value. n = 3 seeds is too few for seed-level tests, so
the seed-level numbers are descriptive.

Seed protocol (state this wherever the numbers are used): in this repository
--seed reaches only get_peft_model(random_state=...), i.e. the LoRA/Lambda
initialisation. GRPOConfig is built without a seed, so data order and the
trainer RNG are the HF default (42) for every seed, baselines and treatments alike.
Same-seed baseline and treatment therefore share init AND data order, which makes
the per-seed comparison a paired one, but the seed SD is an initialisation SD, not
a full training-seed SD.
"""

import glob
import json
import math
import os
import re
from collections import defaultdict

from model_registry import SIZE_ALT, TASK_SUFFIX, TASK_TEST_SIZE, baseline_tag, task_from_name
from passk_utils import paired_t
from stats_utils import mcnemar_test, wilson_ci

TASK_FULL = dict(TASK_TEST_SIZE)
# Cold Dr. GRPO arms of the settled studies: the controls for every D2 treatment.
BASELINE_TAGS = tuple(baseline_tag(t) for t in TASK_SUFFIX)
BASE_RE = re.compile(rf"^(?P<size>{SIZE_ALT})-(?P<variant>v\d_[a-z_]+?)(?P<rest>(?:-.*)?)$")


def split_name(dirname):
    """-> (size, variant, seed, tag, is_new) or None. tag drops the seed token only."""
    is_new = dirname.startswith("new-")
    m = BASE_RE.match(dirname[4:] if is_new else dirname)
    if not m:
        return None
    rest = m.group("rest")
    sm = re.search(r"-seed(\d+)", rest)
    seed = int(sm.group(1)) if sm else 42
    tag = re.sub(r"-seed\d+", "", rest)
    tag = re.sub(r"^-rp1(?=-|$)", "", tag).lstrip("-")   # every reported arm is rp=1.0
    if "-rp" in rest and "-rp1" not in rest:
        return None                                        # any other penalty is not a result
    return m.group("size"), m.group("variant"), seed, tag, is_new


def final_eval(exp_dir):
    best = None
    for d in glob.glob(os.path.join(exp_dir, "checkpoint-*")):
        m = re.match(r"checkpoint-(\d+)$", os.path.basename(d))
        p = os.path.join(d, "eval_results.json")
        if m and os.path.exists(p) and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), d, p)
    return best


def trained_to_end(ckpt_dir):
    try:
        with open(os.path.join(ckpt_dir, "trainer_state.json")) as f:
            st = json.load(f)
        return bool(st.get("max_steps")) and st.get("global_step") == st["max_steps"]
    except (OSError, json.JSONDecodeError):
        return False


def load_test_tokens(ckpt_dir):
    """Mean greedy test tokens per answer from eval_tokens.json (eval_tokens.py), or None.
    Ignored when older than the eval_results.json it was counted from."""
    path, src = os.path.join(ckpt_dir, "eval_tokens.json"), os.path.join(ckpt_dir, "eval_results.json")
    if not os.path.exists(path) or os.path.getmtime(path) < os.path.getmtime(src):
        return None
    try:
        with open(path) as f:
            return json.load(f)["mean"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def tail_completion_length(ckpt_dir, last=100):
    try:
        with open(os.path.join(ckpt_dir, "trainer_state.json")) as f:
            hist = json.load(f).get("log_history", [])
    except (OSError, json.JSONDecodeError):
        return None
    vals = [h["completion_length"] for h in hist if "completion_length" in h]
    vals = vals[-last:]
    return sum(vals) / len(vals) if vals else None


def load_runs():
    """{(size, variant, tag): {seed: dict(acc, vec, questions, length)}}"""
    arms = defaultdict(dict)
    for exp_dir in sorted(glob.glob("experiments/*")):
        parsed = split_name(os.path.basename(exp_dir))
        if not parsed:
            continue
        size, variant, seed, tag, is_new = parsed
        # Legacy directories only count as baselines when they are the cold Dr. GRPO
        # arm (GSM8K or MATH); everything else in experiments/ belongs to the settled
        # studies.
        if not is_new and tag not in BASELINE_TAGS:
            continue
        found = final_eval(exp_dir)
        if not found or not trained_to_end(found[1]):
            continue
        with open(found[2]) as f:
            data = json.load(f)
        if data["metrics"]["total"] < TASK_FULL.get(data["metrics"].get("task", "gsm8k"), 1319):
            continue
        arms[(size, variant, tag)][seed] = {
            "acc": data["metrics"]["accuracy"],
            "correct": data["metrics"]["correct"],
            "total": data["metrics"]["total"],
            "vec": [1 if r["correct"] else 0 for r in data["results"]],
            "questions": [r["question"] for r in data["results"]],
            "length": tail_completion_length(found[1]),
            "test_tokens": load_test_tokens(found[1]),
            "dir": exp_dir,
            "passk": load_passk(found[1]),
        }
    return arms


def load_passk(ckpt_dir):
    """{k: dict(avg, pass, maj, tokens, temperature, avg_vec, pass_vec, questions)} from
    every eval_results_pass{k}.json beside the greedy result (eval_d2.py --passk k)."""
    out = {}
    for path in glob.glob(os.path.join(ckpt_dir, "eval_results_pass*.json")):
        m = re.search(r"eval_results_pass(\d+)\.json$", path)
        if not m:
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        mt, res = data["metrics"], data["results"]
        if mt["total"] < TASK_FULL.get(mt.get("task", "gsm8k"), 1319):
            continue
        out[int(m.group(1))] = {
            "avg": mt["avg_at_k"], "pass": mt["pass_at_k"], "maj": mt["maj_at_k"],
            "tokens": mt.get("mean_tokens"), "temperature": mt.get("temperature"),
            "avg_vec": [r["avg_acc"] for r in res],
            "pass_vec": [1 if r["pass_at_k"] else 0 for r in res],
            "questions": [r["question"] for r in res],
        }
    return out


def summarise_passk(arms):
    """Per arm and k: per-seed avg@k / pass@k / maj@k / tokens and their seed means."""
    out = {}
    for key, seeds in sorted(arms.items()):
        ks = sorted({k for s in seeds.values() for k in s["passk"]})
        for k in ks:
            have = {sd: s["passk"][k] for sd, s in sorted(seeds.items()) if k in s["passk"]}
            entry = {"k": k, "n_seeds": len(have),
                     "temperature": sorted({v["temperature"] for v in have.values()}),
                     "seeds": {str(sd): {"avg": round(v["avg"] * 100, 2),
                                         "pass": round(v["pass"] * 100, 2),
                                         "maj": round(v["maj"] * 100, 2),
                                         "tokens": round(v["tokens"], 1) if v["tokens"] else None}
                               for sd, v in have.items()}}
            for name in ("avg", "pass", "maj"):
                m, sd = mean_sd([v[name] * 100 for v in have.values()])
                entry[f"mean_{name}"], entry[f"sd_{name}"] = round(m, 2), round(sd, 2)
            out[f"{'|'.join(key)}|k={k}"] = entry
    return out


def compare_passk(arms):
    """Treatment minus same-seed baseline on the sampled metrics. avg@k uses a paired t
    over per-question differences (normal approximation for p: n is 500 or 1319);
    pass@k uses McNemar. Both are per seed, i.e. about two checkpoints."""
    rows = []
    for (size, variant, tag), seeds in sorted(arms.items()):
        if tag in BASELINE_TAGS:
            continue
        base_tag = baseline_tag(task_from_name(tag))
        base = arms.get((size, variant, base_tag))
        if not base:
            continue
        ks = sorted({k for s in seeds.values() for k in s["passk"]})
        for k in ks:
            per_seed = {}
            for s in sorted(set(seeds) & set(base)):
                a, b = seeds[s]["passk"].get(k), base[s]["passk"].get(k)
                if not a or not b or a["questions"] != b["questions"]:
                    continue
                diffs = [x - y for x, y in zip(a["avg_vec"], b["avg_vec"])]
                mean, se, t = paired_t(diffs)
                p_t = math.erfc(abs(t) / math.sqrt(2)) if t == t else float("nan")
                _, p_mc = mcnemar_test(a["pass_vec"], b["pass_vec"])
                per_seed[str(s)] = {
                    "delta_avg_pp": round(mean * 100, 2), "se_pp": round(se * 100, 2),
                    "t": round(t, 2), "p_avg": round(p_t, 4),
                    "delta_pass_pp": round((a["pass"] - b["pass"]) * 100, 2),
                    "mcnemar_p_pass": round(p_mc, 4),
                    "delta_tokens": (round(a["tokens"] - b["tokens"], 1)
                                     if a["tokens"] and b["tokens"] else None),
                    "same_temperature": a["temperature"] == b["temperature"],
                }
            if not per_seed:
                continue
            d_avg = [v["delta_avg_pp"] for v in per_seed.values()]
            d_pass = [v["delta_pass_pp"] for v in per_seed.values()]
            m_avg, sd_avg = mean_sd(d_avg)
            m_pass, sd_pass = mean_sd(d_pass)
            rows.append({
                "arm": f"{size} {variant} {tag}", "baseline": f"{size} {variant} {base_tag}",
                "k": k, "per_seed": per_seed, "n": len(per_seed),
                "mean_delta_avg_pp": round(m_avg, 2), "sd_delta_avg_pp": round(sd_avg, 2),
                "k_positive_avg": sum(d > 0 for d in d_avg),
                "k_positive_avg_and_p05": sum(v["delta_avg_pp"] > 0 and v["p_avg"] < 0.05
                                              for v in per_seed.values()),
                "mean_delta_pass_pp": round(m_pass, 2), "sd_delta_pass_pp": round(sd_pass, 2),
                "k_positive_pass": sum(d > 0 for d in d_pass),
                "all_same_temperature": all(v["same_temperature"] for v in per_seed.values()),
            })
    return rows


def mean_sd(xs):
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else float("nan")
    return m, sd


def summarise(arms):
    out = {}
    for key, seeds in sorted(arms.items()):
        accs = [s["acc"] * 100 for _, s in sorted(seeds.items())]
        m, sd = mean_sd(accs)
        lens = [s["length"] for s in seeds.values() if s["length"] is not None]
        corr, tot = sum(s["correct"] for s in seeds.values()), sum(s["total"] for s in seeds.values())
        lo, hi = wilson_ci(corr, tot)
        out["|".join(key)] = {
            "seeds": {str(k): round(v["acc"] * 100, 2) for k, v in sorted(seeds.items())},
            "n_seeds": len(seeds), "mean": round(m, 2), "sd": round(sd, 2),
            "wilson_ci_pooled": [round(lo * 100, 2), round(hi * 100, 2)],
            "completion_length_last100": round(sum(lens) / len(lens), 1) if lens else None,
        }
    return out


def compare(arms):
    rows = []
    for (size, variant, tag), seeds in sorted(arms.items()):
        if tag in BASELINE_TAGS:
            continue
        # A MATH treatment is compared with the MATH baseline, never the GSM8K one.
        base_tag = baseline_tag(task_from_name(tag))
        base = arms.get((size, variant, base_tag))
        if not base:
            continue
        shared = sorted(set(seeds) & set(base))
        per_seed = {}
        for s in shared:
            a, b = seeds[s], base[s]
            if a["questions"] != b["questions"]:
                continue
            chi2, p = mcnemar_test(a["vec"], b["vec"])
            per_seed[str(s)] = {"delta_pp": round((a["acc"] - b["acc"]) * 100, 2),
                                "mcnemar_p": round(p, 4)}
            if a.get("test_tokens") and b.get("test_tokens"):
                per_seed[str(s)]["test_tokens_pct"] = round(
                    100 * (a["test_tokens"] / b["test_tokens"] - 1), 1)
        if not per_seed:
            continue
        deltas = [v["delta_pp"] for v in per_seed.values()]
        m, sd = mean_sd(deltas)
        la = [seeds[s]["length"] for s in shared if seeds[s]["length"] is not None]
        lb = [base[s]["length"] for s in shared if base[s]["length"] is not None]
        rows.append({
            "arm": f"{size} {variant} {tag}", "baseline": f"{size} {variant} {base_tag}",
            "shared_seeds": shared, "per_seed": per_seed,
            "mean_delta_pp": round(m, 2), "sd_delta_pp": round(sd, 2),
            "k_positive": sum(d > 0 for d in deltas), "n": len(deltas),
            "k_positive_and_p05": sum(v["delta_pp"] > 0 and v["mcnemar_p"] < 0.05
                                      for v in per_seed.values()),
            "delta_completion_length": (round(sum(la) / len(la) - sum(lb) / len(lb), 1)
                                        if la and lb else None),
            "test_tokens_pct": [v["test_tokens_pct"] for v in per_seed.values()
                                if "test_tokens_pct" in v],
        })
    return rows


def main():
    arms = load_runs()
    if not arms:
        print("No evaluated arms found. Run eval_d2.py first (and make sure the baselines "
              "have eval_results.json).")
        return
    summary, rows = summarise(arms), compare(arms)
    print("\n=== Arms (greedy accuracy %, seed mean, SD) ===")
    for k, v in summary.items():
        seeds = " ".join(f"{s}:{a:.2f}" for s, a in v["seeds"].items())
        print(f"{k:<62} n={v['n_seeds']} mean={v['mean']:.2f} sd={v['sd']:.2f}  "
              f"len={v['completion_length_last100']}  ({seeds})")
    print("\n=== Treatment minus same-seed baseline (descriptive; no pooled p-values) ===")
    for r in rows:
        per = " ".join(f"s{s}:{d['delta_pp']:+.2f}(p={d['mcnemar_p']:.3f})"
                       for s, d in r["per_seed"].items())
        print(f"{r['arm']:<58} mean={r['mean_delta_pp']:+.2f} sd={r['sd_delta_pp']:.2f} "
              f"positive {r['k_positive']}/{r['n']} (sig+ {r['k_positive_and_p05']}/{r['n']}) "
              f"dLen={r['delta_completion_length']}  {per}"
              + (f"  testTok% {r['test_tokens_pct']}" if r["test_tokens_pct"] else ""))
    pk_summary, pk_rows = summarise_passk(arms), compare_passk(arms)
    if pk_summary:
        print("\n=== Sampled evaluation (avg@k / pass@k / maj@k %, seed means) ===")
        for k, v in pk_summary.items():
            seeds = " ".join(f"{s}:{d['avg']:.2f}/{d['pass']:.2f}" for s, d in v["seeds"].items())
            print(f"{k:<66} n={v['n_seeds']} avg={v['mean_avg']:.2f} pass={v['mean_pass']:.2f} "
                  f"maj={v['mean_maj']:.2f}  T={v['temperature']}  ({seeds})")
        print("\n=== Sampled: treatment minus same-seed baseline (descriptive) ===")
        for r in pk_rows:
            per = " ".join(f"s{s}:avg{d['delta_avg_pp']:+.2f}(t={d['t']:.2f}) pass{d['delta_pass_pp']:+.2f}"
                           for s, d in r["per_seed"].items())
            note = "" if r["all_same_temperature"] else "  [sampled at different temperatures]"
            print(f"{r['arm']:<54} k={r['k']} avg {r['mean_delta_avg_pp']:+.2f} "
                  f"(+ in {r['k_positive_avg']}/{r['n']}, sig+ {r['k_positive_avg_and_p05']}/{r['n']})  "
                  f"pass {r['mean_delta_pass_pp']:+.2f} (+ in {r['k_positive_pass']}/{r['n']})  {per}{note}")
    else:
        print("\n(no sampled results yet: run  python eval_d2.py --passk 8)")

    os.makedirs("results", exist_ok=True)
    with open("results/d2_analysis.json", "w") as f:
        json.dump({"arms": summary, "comparisons": rows,
                   "passk_arms": pk_summary, "passk_comparisons": pk_rows}, f, indent=2)
    print("\nwrote results/d2_analysis.json")


if __name__ == "__main__":
    main()
