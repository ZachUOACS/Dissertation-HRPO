"""
probe_latent_temp.py - go/no-go check for the decoupled latent temperature (GPU job, no training).

HRPO's Eq.3 mixture is formed from the sampling distribution (temperature 0.5, top-50 in this
code), so the hypothesis is that at most latent positions the mixture is close to the embedding of
the token that is sampled anyway and carries little extra information. This job measures that on
trained checkpoints before any training run is spent on --latent_temperature.

Part A, measurement. Greedy decodes of trained cold Dr. GRPO v3 checkpoints, recording at every
latent position (still in the reasoning span, sequence not finished) the distribution the rollout
actually mixes over, taken from inside the generation loop (it wraps
transformers.generation.utils.hrpo_latent_probs, so whatever logits processors are really in effect
are included), and the same scores re-tempered to --latent_temperature:
    eff_tokens  exp(entropy): how many tokens the mixture effectively combines
    part_ratio  1 / sum p^2: the square of what Eq.3's L2 normalisation divides by
    p_top1      probability of the most likely token
    cos_top1    cosine between the Eq.3 latent and the embedding of the most likely token
    norm_ratio  ||latent|| / ||embedding of the most likely token||
Part B, re-score. Greedy MATH-500 evaluation of the cold v3 checkpoints with the mixture formed at
--latent_temperature, compared question by question with their settled eval_results.json (same
weights, batch size 32 like the settled results, McNemar per seed). The models were trained with
the sharper mixture, so Part B measures tolerance of the change, not the effect of training with it.

Decision rule, fixed before the run (on the pooled MATH positions of Part A):
    PROCEED  if the sampling-temperature mixture is close to one token
             (median eff_tokens < 1.5 and mean cos_top1 > 0.9)
             and the latent temperature changes it materially
             (mean eff_tokens at the latent temperature >= 1.3 x the sampling-temperature mean)
    STOP     otherwise: the mixture is already spread, or re-tempering barely moves it.
Part B does not enter the rule; it is reported for context.

    python probe_latent_temp.py                       # both parts, defaults below
    python probe_latent_temp.py --skip_rescore        # Part A only (about 20 minutes)
Writes results/latent_temp_probe.json. Run through cluster/run_latent_probe.slurm.
"""

import unsloth  # noqa: F401  (must be imported before transformers pieces)

import argparse
import gc
import json
import math
import os
import statistics as st
from collections import defaultdict

import torch
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

import transformers.generation.utils as gen_utils
from eval_all import final_checkpoint
from eval_gsm8k import evaluate_model, get_task_spec, load_model_for_eval
from utils import ANSWER_START, SYSTEM_PROMPT
from variant_common import GATE_MODES, MODEL_NAMES

MATH_V3 = ["experiments/1.5b-v3_learned_lambda-rp1-drboth-dsmath",
           "experiments/1.5b-v3_learned_lambda-seed43-rp1-drboth-dsmath",
           "experiments/1.5b-v3_learned_lambda-seed44-rp1-drboth-dsmath"]
GSM8K_V3 = ["experiments/1.5b-v3_learned_lambda-rp1-drboth"]
METRICS = ("eff_tokens", "part_ratio", "p_top1", "cos_top1", "norm_ratio")


class Probe:
    """Collects per-position statistics of the Eq.3 mixing distribution.

    Within one decoding step _sample calls the logits processors (MaskRecorder) first and
    hrpo_latent_probs last. The statistics of step t are therefore held as `pending` and
    committed at step t+1, when input_ids include token t and show whether the mixture of
    step t is used (is_thinking is decided from the text up to and including token t, exactly
    as _sample decides it) and whether the sequence had already finished."""

    def __init__(self):
        self.active = False
        self.embed = None
        self.latent_temperature = None
        self.pending = None
        self.values = {"sampling": defaultdict(list), "latent": defaultdict(list)}
        self.config_seen = None

    def reset(self, embed, latent_temperature):
        self.embed = embed
        self.latent_temperature = latent_temperature
        self.pending = None
        self.values = {"sampling": defaultdict(list), "latent": defaultdict(list)}

    @torch.no_grad()
    def stats(self, p):
        p = p.float()
        ent = torch.special.entr(p).sum(-1)
        latent = gen_utils.hrpo_project_latent(p.to(self.embed.dtype), self.embed).float()
        top1 = p.argmax(-1)
        e1 = self.embed[top1].float()
        return {
            "eff_tokens": ent.exp(),
            "part_ratio": 1.0 / (p ** 2).sum(-1),
            "p_top1": p.max(-1).values,
            "cos_top1": torch.nn.functional.cosine_similarity(latent, e1, dim=-1),
            "norm_ratio": latent.norm(dim=-1) / e1.norm(dim=-1),
        }

    def on_step(self, probs, next_token_scores, generation_config):
        if self.config_seen is None:
            self.config_seen = {k: getattr(generation_config, k, None) for k in
                                ("do_sample", "temperature", "top_k", "top_p", "min_p",
                                 "repetition_penalty")}
        latent_p = _ORIGINAL(probs, next_token_scores, generation_config, self.latent_temperature)
        self.pending = {"sampling": self.stats(probs), "latent": self.stats(latent_p)}

    def commit(self, mask):
        if self.pending is None:
            return
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel():
            for which, d in self.pending.items():
                for k, v in d.items():
                    self.values[which][k].extend(v[idx].cpu().tolist())
        self.pending = None


PROBE = Probe()
_ORIGINAL = gen_utils.hrpo_latent_probs


def _wrapped_latent_probs(probs, next_token_scores, generation_config, latent_temperature=None):
    out = _ORIGINAL(probs, next_token_scores, generation_config, latent_temperature)
    if PROBE.active:
        PROBE.on_step(probs, next_token_scores, generation_config)
    return out


# _sample looks the helper up as a module global at every step, so this reaches the loop.
gen_utils.hrpo_latent_probs = _wrapped_latent_probs


class MaskRecorder(LogitsProcessor):
    """Leaves the scores untouched; decides which pending statistics belong to latent positions."""

    def __init__(self, tokenizer, prompt_len, eos_id):
        self.tokenizer, self.prompt_len, self.eos_id = tokenizer, prompt_len, eos_id

    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.prompt_len:]
        if gen.shape[1] > 0 and PROBE.pending is not None:
            strs = self.tokenizer.batch_decode(gen)
            thinking = torch.tensor([ANSWER_START not in s for s in strs], device=gen.device)
            alive = ~(gen == self.eos_id).any(dim=1)
            PROBE.commit(thinking & alive)
        return scores


def summarise(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return {}
    q = lambda f: xs[min(n - 1, int(f * n))]
    return {"mean": round(st.mean(xs), 4), "median": round(q(0.5), 4),
            "p10": round(q(0.1), 4), "p90": round(q(0.9), 4)}


def measure(exp_dir, task, n_questions, batch_size, latent_temperature):
    ckpt = final_checkpoint(exp_dir)
    variant = "v3_learned_lambda"
    model, tokenizer = load_model_for_eval(MODEL_NAMES["1.5b"], ckpt, GATE_MODES[variant], task)
    dataset, q_key, _, max_new, _, _ = get_task_spec(task)
    dataset = dataset.select(range(min(n_questions, len(dataset))))
    PROBE.reset(model.get_input_embeddings().weight.detach(), latent_temperature)
    PROBE.active = True
    try:
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            prompts = [tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q.strip()}],
                tokenize=False, add_generation_prompt=True) for q in batch[q_key]]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left",
                            add_special_tokens=False)
            ids, mask = enc["input_ids"].to(model.device), enc["attention_mask"].to(model.device)
            recorder = MaskRecorder(tokenizer, ids.shape[1], tokenizer.eos_token_id)
            model.generate(ids, attention_mask=mask,
                           generation_config=GenerationConfig(do_sample=True, temperature=0.5,
                                                              max_new_tokens=max_new),
                           logits_processor=LogitsProcessorList([recorder]),
                           processing_class=tokenizer, is_inference=True)
            PROBE.pending = None   # the last step of a batch has no following step to commit it
            print(f"  {exp_dir} {task}: {min(i + batch_size, len(dataset))}/{len(dataset)} questions, "
                  f"{len(PROBE.values['sampling']['eff_tokens'])} latent positions")
    finally:
        PROBE.active = False
    out = {"checkpoint": ckpt, "task": task, "n_questions": len(dataset),
           "n_positions": len(PROBE.values["sampling"]["eff_tokens"]),
           "generation_config_seen": PROBE.config_seen,
           "latent_temperature": latent_temperature}
    raw = {}
    for which in ("sampling", "latent"):
        vals = PROBE.values[which]
        out[which] = {k: summarise(vals[k]) for k in METRICS}
        out[which]["share_eff_tokens_gt_1.5"] = round(
            sum(x > 1.5 for x in vals["eff_tokens"]) / max(1, len(vals["eff_tokens"])), 4)
        out[which]["share_eff_tokens_gt_2"] = round(
            sum(x > 2 for x in vals["eff_tokens"]) / max(1, len(vals["eff_tokens"])), 4)
        raw[which] = {k: list(vals[k]) for k in ("eff_tokens", "cos_top1")}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out, raw


def mcnemar(on, off):
    b = sum(1 for x, y in zip(on, off) if x and not y)
    c = sum(1 for x, y in zip(on, off) if y and not x)
    if b + c == 0:
        return b, c, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, math.erfc(math.sqrt(chi2 / 2))


def rescore(exp_dir, latent_temperature, batch_size):
    ckpt = final_checkpoint(exp_dir)
    settled_path = os.path.join(ckpt, "eval_results.json")
    path = os.path.join(ckpt, f"eval_results_lt{latent_temperature:g}.json")
    done = False
    if os.path.exists(path):
        with open(path) as f:
            done = json.load(f)["metrics"]["total"] >= 500
    if not done:
        evaluate_model(model_path=MODEL_NAMES["1.5b"], adapter_path=ckpt, temperature=0.5,
                       is_inference=True, batch_size=batch_size, save_results=True,
                       gate_mode=GATE_MODES["v3_learned_lambda"], save_path=path, task="math",
                       latent_temperature=latent_temperature)
        gc.collect()
        torch.cuda.empty_cache()
    with open(settled_path) as f:
        settled = json.load(f)
    with open(path) as f:
        new = json.load(f)
    a = [bool(r["correct"]) for r in settled["results"]]
    b_ = [bool(r["correct"]) for r in new["results"]]
    b, c, p = mcnemar(a, b_)
    return {"checkpoint": ckpt, "settled": round(100 * sum(a) / len(a), 2),
            "latent_temperature": latent_temperature,
            "rescored": round(100 * sum(b_) / len(b_), 2),
            "delta": round(100 * (sum(b_) - sum(a)) / len(a), 2),
            "settled_only": b, "rescored_only": c, "p_mcnemar": round(p, 4)}


def decide(raws):
    s_eff = [x for r in raws for x in r["sampling"]["eff_tokens"]]
    s_cos = [x for r in raws for x in r["sampling"]["cos_top1"]]
    l_eff = [x for r in raws for x in r["latent"]["eff_tokens"]]
    if not s_eff:
        return {"verdict": "NO DATA"}
    med = sorted(s_eff)[len(s_eff) // 2]
    near_one = med < 1.5 and st.mean(s_cos) > 0.9
    material = st.mean(l_eff) >= 1.3 * st.mean(s_eff)
    return {"median_eff_tokens_sampling": round(med, 4),
            "mean_cos_top1_sampling": round(st.mean(s_cos), 4),
            "mean_eff_tokens_sampling": round(st.mean(s_eff), 4),
            "mean_eff_tokens_latent": round(st.mean(l_eff), 4),
            "near_one_token": near_one, "material_change": material,
            "verdict": "PROCEED" if (near_one and material) else "STOP"}


def _ranks(xs):
    """Average ranks (ties share their mean rank), for a Spearman correlation without scipy."""
    order = sorted(range(len(xs)), key=xs.__getitem__)
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(xs, ys):
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = st.mean(rx), st.mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / math.sqrt(vx * vy) if vx and vy else float("nan")


def calibrate(raws, latent_temperature):
    """D3 gate calibration from Part A positions. d = 1 - cos(latent, embedding of the token that
    accompanies it); under greedy decoding that token is the argmax, so d = 1 - cos_top1. Reports
    its quantiles, how well it tracks uncertainty (Spearman with the effective number of tokens),
    and the suggested ramp: d0 = 75th percentile (about a quarter of latent positions blended at
    all), d1 = 95th percentile (the top 5% at full weight)."""
    out = {}
    for which, label in (("sampling", "tau 0.5"), ("latent", f"tau {latent_temperature:g}")):
        d = [1.0 - c for r in raws for c in r[which]["cos_top1"]]
        eff = [x for r in raws for x in r[which]["eff_tokens"]]
        if not d:
            continue
        ds = sorted(d)
        q = lambda f: ds[min(len(ds) - 1, int(f * len(ds)))]
        quant = {f"p{int(f * 100)}": q(f) for f in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)}
        # Spearman on a fixed-stride subsample keeps the pure-Python rank step quick
        stride = max(1, len(d) // 20000)
        rho = spearman(d[::stride], eff[::stride])
        out[which] = {"label": label, "n_positions": len(d),
                      "quantiles": {k: float(f"{v:.6g}") for k, v in quant.items()},
                      "spearman_d_vs_eff_tokens": round(rho, 4),
                      "suggested_d0": float(f"{quant['p75']:.3g}"),
                      "suggested_d1": float(f"{quant['p95']:.3g}")}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_temperature", type=float, default=1.0)
    parser.add_argument("--measure_n", type=int, default=100, help="questions per checkpoint")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--skip_rescore", action="store_true")
    parser.add_argument("--calibrate", action="store_true",
                        help="D3: Part A on the MATH checkpoints only, then the gate calibration "
                             "(written to results/d3_gate_calibration.json); no re-scoring")
    parser.add_argument("--out", default="results/latent_temp_probe.json")
    args = parser.parse_args()

    if args.calibrate:
        raws = []
        for exp_dir in MATH_V3:
            if final_checkpoint(exp_dir) is None:
                print(f"[skip] {exp_dir}: no checkpoint")
                continue
            print(f"[calibrate] {exp_dir}")
            _, raw = measure(exp_dir, "math", args.measure_n, args.batch_size,
                             args.latent_temperature)
            raws.append(raw)
        cal = calibrate(raws, args.latent_temperature)
        out = "results/d3_gate_calibration.json" if args.out == parser.get_default("out") else args.out
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(cal, f, indent=2)
        print("\n=== D3 gate calibration: d = 1 - cos(latent, token) at latent positions (MATH) ===")
        for which, c in cal.items():
            print(f"{c['label']}: n={c['n_positions']}  Spearman(d, eff_tokens)="
                  f"{c['spearman_d_vs_eff_tokens']}  suggested d0={c['suggested_d0']} "
                  f"d1={c['suggested_d1']}")
            print("   " + "  ".join(f"{k}={v}" for k, v in c["quantiles"].items()))
        print(f"\n-> {out}")
        return

    report = {"measurement": [], "rescore": []}
    math_raws = []
    for exp_dir, task in [(d, "math") for d in MATH_V3] + [(d, "gsm8k") for d in GSM8K_V3]:
        if final_checkpoint(exp_dir) is None:
            print(f"[skip] {exp_dir}: no checkpoint")
            continue
        print(f"[measure] {exp_dir} ({task})")
        out, raw = measure(exp_dir, task, args.measure_n, args.batch_size, args.latent_temperature)
        report["measurement"].append(out)
        if task == "math":
            math_raws.append(raw)
    report["decision_math"] = decide(math_raws)

    if not args.skip_rescore:
        for exp_dir in MATH_V3:
            if final_checkpoint(exp_dir) is None:
                continue
            print(f"[rescore] {exp_dir} at latent temperature {args.latent_temperature:g}")
            report["rescore"].append(rescore(exp_dir, args.latent_temperature, args.batch_size))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Part A: the Eq.3 mixing distribution at latent positions ===")
    for m in report["measurement"]:
        print(f"{m['checkpoint']} ({m['task']}), {m['n_positions']} positions, "
              f"generation config seen: {m['generation_config_seen']}")
        for which in ("sampling", "latent"):
            s = m[which]
            label = "tau 0.5 (as trained)" if which == "sampling" else \
                f"tau {args.latent_temperature:g} (re-tempered)"
            print(f"  {label:22s} eff_tokens mean {s['eff_tokens']['mean']:.3f} median "
                  f"{s['eff_tokens']['median']:.3f} | p_top1 mean {s['p_top1']['mean']:.3f} | "
                  f"cos_top1 mean {s['cos_top1']['mean']:.4f} | norm_ratio mean "
                  f"{s['norm_ratio']['mean']:.3f} | share eff>1.5 {s['share_eff_tokens_gt_1.5']:.3f}")
    print(f"\nDecision (MATH, rule fixed in the docstring): {report['decision_math']}")
    if report["rescore"]:
        print("\n=== Part B: MATH-500 re-scored with the re-tempered mixture (no training) ===")
        for r in report["rescore"]:
            print(f"{r['checkpoint']}: settled {r['settled']:.2f} -> {r['rescored']:.2f} "
                  f"({r['delta']:+.2f}; settled only {r['settled_only']}, re-scored only "
                  f"{r['rescored_only']}, p {r['p_mcnemar']:.4f})")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
