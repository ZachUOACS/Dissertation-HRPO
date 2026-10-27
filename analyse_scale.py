"""
analyse_scale.py - the scale check and the cross-model study: cold Dr. GRPO v3 (HRPO) against v1
(GRPO) for every backbone in model_registry.py (Qwen 1.5B and 3B, Llama 3.2 1B and 3B), on
GSM8K and MATH, per seed. Standard library only, so it runs anywhere.

For each (size, task, seed) where both arms have a complete greedy evaluation:
    v1 and v3 accuracy, v3 - v1 with McNemar's test on the paired questions (continuity
    corrected, as stats_utils.mcnemar_test), v3's latent-off accuracy and the cost of switching the
    latent off (with its own McNemar p), and mean greedy test tokens when eval_tokens.json exists.
Per (size, task) it adds the seed mean of v3 - v1 and the sign count. Descriptive: no pooled
p-values, and one seed is not a finding on its own (a single MATH-500 seed needs about 3.5 points
for p < 0.05).

    python analyse_scale.py            # prints the table, writes results/scale_analysis.json
"""

import glob
import json
import math
import os
import re

from model_registry import MODEL_NAMES, SIZE_ALT, TASK_TEST_SIZE, task_from_name

# Rows in the registry's order (Qwen 0.5B, 1.5B, 3B, then the other families).
SIZE_ORDER = list(MODEL_NAMES).index

ARMS = {"v1": "v1_grpo_baseline", "v3": "v3_learned_lambda"}
TEST_SIZE = dict(TASK_TEST_SIZE)
DIR_RE = re.compile(rf"^(?P<size>{SIZE_ALT})-(?P<variant>v1_grpo_baseline|v3_learned_lambda)"
                    r"(?:-seed(?P<seed>\d+))?-rp1-drboth(?P<ds>-ds(?:math|mmlu))?$")


def mcnemar(a, b):
    """(a-only, b-only, p): continuity-corrected McNemar, p = chi2(df=1) survival function."""
    n_a = sum(1 for x, y in zip(a, b) if x and not y)
    n_b = sum(1 for x, y in zip(a, b) if y and not x)
    if n_a + n_b == 0:
        return n_a, n_b, 1.0
    chi2 = (abs(n_a - n_b) - 1) ** 2 / (n_a + n_b)
    return n_a, n_b, math.erfc(math.sqrt(chi2 / 2))


def load(path, task):
    """Per-question correctness, questions and mean tokens of a complete evaluation, or None."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if data["metrics"]["total"] < TEST_SIZE[task]:
            return None
        res = data["results"]
    except (KeyError, ValueError, OSError):
        return None
    out = {"vec": [1 if r["correct"] else 0 for r in res], "questions": [r["question"] for r in res],
           "tokens": None}
    tok_path = os.path.join(os.path.dirname(path), "eval_tokens.json")
    if os.path.basename(path) == "eval_results.json" and os.path.exists(tok_path) \
            and os.path.getmtime(tok_path) >= os.path.getmtime(path):
        with open(tok_path) as f:
            out["tokens"] = json.load(f).get("mean")
    return out


def final_eval_dir(exp_dir):
    ckpts = [(int(m.group(1)), d) for d in glob.glob(os.path.join(exp_dir, "checkpoint-*"))
             if (m := re.search(r"checkpoint-(\d+)$", d))
             and os.path.exists(os.path.join(d, "eval_results.json"))]
    return max(ckpts)[1] if ckpts else None


def discover(root="experiments"):
    """{(size, task, seed): {"v1": ckpt_dir, "v3": ckpt_dir}}"""
    found = {}
    for exp_dir in sorted(glob.glob(os.path.join(root, "*"))):
        m = DIR_RE.match(os.path.basename(exp_dir))
        if not m:
            continue
        ckpt = final_eval_dir(exp_dir)
        if ckpt is None:
            continue
        key = (m.group("size"), task_from_name(m.group("ds") or ""), int(m.group("seed") or 42))
        arm = "v1" if m.group("variant") == ARMS["v1"] else "v3"
        found.setdefault(key, {})[arm] = ckpt
    return found


def pct(vec):
    return 100.0 * sum(vec) / len(vec)


def analyse(root="experiments"):
    rows = []
    for (size, task, seed), arms in sorted(discover(root).items(),
                                           key=lambda kv: (kv[0][1], SIZE_ORDER(kv[0][0]), kv[0][2])):
        if set(arms) != {"v1", "v3"}:
            continue
        v1 = load(os.path.join(arms["v1"], "eval_results.json"), task)
        v3 = load(os.path.join(arms["v3"], "eval_results.json"), task)
        if v1 is None or v3 is None:
            continue
        if v1["questions"] != v3["questions"]:
            raise SystemExit(f"{size} {task} seed {seed}: v1 and v3 were scored on different questions")
        only_v1, only_v3, p = mcnemar(v1["vec"], v3["vec"])
        row = {"size": size, "task": task, "seed": seed, "v1": pct(v1["vec"]), "v3": pct(v3["vec"]),
               "delta_pp": pct(v3["vec"]) - pct(v1["vec"]), "v1_only": only_v1, "v3_only": only_v3,
               "p": p, "v1_tokens": v1["tokens"], "v3_tokens": v3["tokens"],
               "v3_off": None, "off_cost_pp": None, "off_p": None,
               "v1_dir": arms["v1"], "v3_dir": arms["v3"]}
        off = load(os.path.join(arms["v3"], "eval_results_latent_off.json"), task)
        if off is not None and off["questions"] == v3["questions"]:
            _, _, p_off = mcnemar(v3["vec"], off["vec"])
            row.update(v3_off=pct(off["vec"]), off_cost_pp=pct(off["vec"]) - pct(v3["vec"]),
                       off_p=p_off)
        rows.append(row)

    summary = []
    for size, task in sorted({(r["size"], r["task"]) for r in rows},
                             key=lambda st: (st[1], SIZE_ORDER(st[0]))):
        d = [r["delta_pp"] for r in rows if r["size"] == size and r["task"] == task]
        offs = [r["off_cost_pp"] for r in rows
                if r["size"] == size and r["task"] == task and r["off_cost_pp"] is not None]
        summary.append({"size": size, "task": task, "n_seeds": len(d),
                        "mean_delta_pp": sum(d) / len(d), "positive": sum(x > 0 for x in d),
                        "mean_off_cost_pp": sum(offs) / len(offs) if offs else None,
                        "n_off": len(offs)})
    return rows, summary


def fmt(x, spec="+.2f"):
    return "-" if x is None else format(x, spec)


def main():
    rows, summary = analyse()
    print("=== v3 (HRPO) minus v1 (GRPO), cold Dr. GRPO, greedy, per seed ===")
    print(f"{'size':>5} {'task':<6} {'seed':>4} {'v1':>6} {'v3':>6} {'d':>6} {'p':>6} "
          f"{'v3 off':>7} {'off cost':>8} {'off p':>6} {'tok v1':>7} {'tok v3':>7}")
    for r in rows:
        print(f"{r['size']:>5} {r['task']:<6} {r['seed']:>4} {r['v1']:6.2f} {r['v3']:6.2f} "
              f"{r['delta_pp']:+6.2f} {r['p']:6.3f} {fmt(r['v3_off'], '7.2f'):>7} "
              f"{fmt(r['off_cost_pp']):>8} {fmt(r['off_p'], '.3f'):>6} "
              f"{fmt(r['v1_tokens'], '.1f'):>7} {fmt(r['v3_tokens'], '.1f'):>7}")
    print("=== per size and task (descriptive; no pooled p-values) ===")
    for s in summary:
        print(f"{s['size']:>5} {s['task']:<6} seeds={s['n_seeds']} mean d={s['mean_delta_pp']:+.2f} "
              f"positive {s['positive']}/{s['n_seeds']}  latent-off cost "
              f"{fmt(s['mean_off_cost_pp'])} (n={s['n_off']})")
    os.makedirs("results", exist_ok=True)
    with open("results/scale_analysis.json", "w") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=1)
    print("wrote results/scale_analysis.json")


if __name__ == "__main__":
    main()
