"""
check_quant.py - independent check of the quantisation study (stdlib only), beside analyse_quant.py.

  1. merge only: the settled unmerged eval_results.json against the merged bf16 reference of the same
     checkpoint (identical in exact arithmetic, so every change is rounding), McNemar;
  2. each scheme against merged bf16, per seed, McNemar, with answers changed and the token ratio;
  3. v3's quantisation cost minus v1's, per seed, with a per-question sign-flip test
     (d_i = (v3_q - v3_bf16) - (v1_q - v1_bf16), two-sided, 20,000 flips, seed 0);
  4. scoring of answers without "####" at bf16 and int4.

    python check_quant.py                 # from the repo root
    python check_quant.py <repo root>
"""
import json, math, os, random, sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
EXP = os.path.join(root, "experiments")
RUNS = {
    ("v1", 42): "1.5b-v1_grpo_baseline-rp1-drboth-dsmath",
    ("v1", 43): "1.5b-v1_grpo_baseline-seed43-rp1-drboth-dsmath",
    ("v1", 44): "1.5b-v1_grpo_baseline-seed44-rp1-drboth-dsmath",
    ("v3", 42): "1.5b-v3_learned_lambda-rp1-drboth-dsmath",
    ("v3", 43): "1.5b-v3_learned_lambda-seed43-rp1-drboth-dsmath",
    ("v3", 44): "1.5b-v3_learned_lambda-seed44-rp1-drboth-dsmath",
}
SCHEMES = ["bf16", "int8", "nf4", "int4"]


def load(path):
    with open(path) as f:
        return json.load(f)["results"]


def ckpt(run):
    d = os.path.join(EXP, run)
    cks = sorted(x for x in os.listdir(d) if x.startswith("checkpoint-"))
    return os.path.join(d, cks[-1])


def mcnemar(a, b):
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    if n10 + n01 == 0:
        return n10, n01, 1.0
    chi2 = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    return n10, n01, math.erfc(math.sqrt(chi2 / 2))


def vec(res):
    return [1 if r["correct"] else 0 for r in res]


def fmt(res, toks=None):
    toks = toks if toks is not None else [r.get("num_tokens") for r in res]
    mark = sum("####" in r["full_response"] for r in res) / len(res)
    cap = sum(1 for t in toks if t is not None and t >= 1024) / len(res)
    mt = sum(toks) / len(toks)
    return mark, cap, mt


def signflip(d, iters=20000, seed=0):
    obs = sum(d)
    nz = [x for x in d if x]
    rng = random.Random(seed)
    hits = 0
    for _ in range(iters):
        s = sum(x if rng.random() < 0.5 else -x for x in nz)
        if abs(s) >= abs(obs):
            hits += 1
    return (hits + 1) / (iters + 1)


data = {}
for (arm, seed), run in RUNS.items():
    c = ckpt(run)
    orig = load(os.path.join(c, "eval_results.json"))
    tok = json.load(open(os.path.join(c, "eval_tokens.json")))["tokens"]
    off = os.path.join(c, "eval_results_latent_off.json")
    data[(arm, seed, "orig")] = (orig, tok)
    if arm == "v3" and os.path.exists(off):
        data[(arm, seed, "orig_off")] = (load(off), None)
    for s in SCHEMES:
        data[(arm, seed, s)] = (load(os.path.join(c, f"eval_results_wq-{s}.json")), None)
        p = os.path.join(c, f"eval_results_wq-{s}_latent_off.json")
        if os.path.exists(p):
            data[(arm, seed, s + "_off")] = (load(p), None)

# question order identical everywhere?
ref_q = [r["question"] for r in data[("v1", 42, "orig")][0]]
bad = [k for k, (res, _) in data.items() if [r["question"] for r in res] != ref_q]
print("files:", len(data), " question order mismatches:", bad or "none")

print("\n=== 1. Merge only: unmerged eval_results.json vs merged bf16 (same checkpoint) ===")
print("arm seed  unmerged  merged    d    wrong->right right->wrong  p      tok(unm, re-tok) tok(merged)  ####unm/mrg  cap unm/mrg")
for arm in ("v1", "v3"):
    for seed in (42, 43, 44):
        (o, ot), (m, _) = data[(arm, seed, "orig")], data[(arm, seed, "bf16")]
        a, b = vec(o), vec(m)
        n10, n01, p = mcnemar(a, b)
        mo, co, to = fmt(o, ot)
        mm, cm, tm = fmt(m)
        print(f"{arm}  {seed}   {100*sum(a)/500:6.2f}  {100*sum(b)/500:6.2f}  {100*(sum(b)-sum(a))/500:+6.2f}"
              f"   {n01:4d}        {n10:4d}     {p:.4f}   {to:7.1f}        {tm:7.1f}      {100*mo:3.0f}/{100*mm:3.0f}      {100*co:3.1f}/{100*cm:3.1f}")
    if arm == "v3":
        for seed in (42, 43, 44):
            o, m = data[(arm, seed, "orig_off")][0], data[(arm, seed, "bf16_off")][0]
            n10, n01, p = mcnemar(vec(o), vec(m))
            print(f"v3 off {seed} {100*sum(vec(o))/500:6.2f}  {100*sum(vec(m))/500:6.2f}  changed {n10+n01}  p={p:.4f}")

print("\n=== merged v3 - v1 (bf16), per seed; vs unmerged ===")
for seed in (42, 43, 44):
    for tag in ("orig", "bf16"):
        a, b = vec(data[("v1", seed, tag)][0]), vec(data[("v3", seed, tag)][0])
        n10, n01, p = mcnemar(a, b)
        print(f"  s{seed} {tag:5s} v1 {100*sum(a)/500:.2f} v3 {100*sum(b)/500:.2f} d {100*(sum(b)-sum(a))/500:+.2f} p={p:.4f}")

print("\n=== 2. Each scheme vs merged bf16, per seed (McNemar) ===")
for arm in ("v1", "v3"):
    for s in SCHEMES[1:] + [x + "_off" for x in SCHEMES] if arm == "v3" else SCHEMES[1:]:
        ref = "bf16_off" if s.endswith("_off") else "bf16"
        if s == "bf16_off":
            ref = "bf16"
        row = []
        for seed in (42, 43, 44):
            a, b = vec(data[(arm, seed, ref)][0]), vec(data[(arm, seed, s)][0])
            n10, n01, p = mcnemar(a, b)
            _, cap, mt = fmt(data[(arm, seed, s)][0])
            _, capr, mtr = fmt(data[(arm, seed, ref)][0])
            row.append(f"s{seed} {100*(sum(b)-sum(a))/500:+6.2f} (p={p:.3f}, {n10+n01:3d} chg, tok x{mt/mtr:.3f})")
        print(f"{arm} {s:9s} vs {ref:8s} " + " | ".join(row))

print("\n=== 3. v3 - v1 quantisation cost per seed; sign-flip test over questions ===")
for s in SCHEMES[1:]:
    row = []
    for seed in (42, 43, 44):
        v1b, v1q = vec(data[("v1", seed, "bf16")][0]), vec(data[("v1", seed, s)][0])
        v3b, v3q = vec(data[("v3", seed, "bf16")][0]), vec(data[("v3", seed, s)][0])
        d = [(c - b) - (q - a) for a, q, b, c in zip(v1b, v1q, v3b, v3q)]
        row.append(f"s{seed} {100*sum(d)/500:+6.2f} (p={signflip(d):.3f})")
    print(f"{s:5s} " + " | ".join(row))

print("\n=== 4. Answers with no '####': does the scorer still find the right answer? (v1 and v3, int4) ===")
for arm in ("v1", "v3"):
    for s in ("bf16", "int4"):
        tot = nomark = nomark_ok = mark_ok = 0
        for seed in (42, 43, 44):
            for r in data[(arm, seed, s)][0]:
                tot += 1
                if "####" in r["full_response"]:
                    mark_ok += r["correct"]
                else:
                    nomark += 1
                    nomark_ok += r["correct"]
        print(f"{arm} {s:5s} no-#### {nomark}/{tot} ({100*nomark/tot:.0f}%), correct among them "
              f"{100*nomark_ok/max(1,nomark):.1f}%; correct among marked {100*mark_ok/max(1,tot-nomark):.1f}%")
