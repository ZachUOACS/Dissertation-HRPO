"""Fusion-aware GSM8k evaluation for the fusion x debias factorial.

Why a dedicated script (not eval_gsm8k.py):
  1. The fusion arms MUST have fusion re-enabled at inference (fusion_enabled etc. are runtime
     attrs, not saved in the adapter). Without this they evaluate as baseline. The arm is parsed
     from the run-dir name (`<model>-<arm>-s<seed>-<tag>`, written by run_ablation.sh).
  2. HRPO training has no final save_model(); only checkpoint-<step>/ dirs exist -> resolve the
     latest one.
  3. Every result row stores a stable `qid` (GSM8k test row index) so the four arms can be paired
     for McNemar analysis offline.
  4. Offline-safe model loading via resolve_model_path (air-gapped compute node).

Debias arms (debias/len_only/std_only) change only the training loss, not inference, so they eval
with fusion OFF, exactly like baseline.

Standalone:  python eval_ablation.py --run_dir <run_dir> [--batch_size 32] [--sample]
Batch (all): run_eval.sh   (generates + sbatches one eval job per finished run dir)
"""
import unsloth
from unsloth import FastLanguageModel

import os
import re
import json
import argparse
from datetime import datetime

from datasets import load_dataset
from transformers import GenerationConfig
from tqdm import tqdm

from utils import (
    ANSWER_START, SYSTEM_PROMPT, extract_from_response, extract_hash_answer,
    process_gsm8k_answer, resolve_model_path,
)

BASE_MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]

# arm -> fusion config at inference. Order matters for parsing (longest match first).
ARM_FUSION = {
    "fusion_debias":  dict(fusion_enabled=True,  fusion_gate_input="hidden", fusion_use_adapter=False),
    "fusion_token":   dict(fusion_enabled=True,  fusion_gate_input="token",  fusion_use_adapter=False),
    "fusion_adapter": dict(fusion_enabled=True,  fusion_gate_input="hidden", fusion_use_adapter=True),
    "fusion":         dict(fusion_enabled=True,  fusion_gate_input="hidden", fusion_use_adapter=False),
    "baseline":       dict(fusion_enabled=False),
    "debias":         dict(fusion_enabled=False),
    "len_only":       dict(fusion_enabled=False),
    "std_only":       dict(fusion_enabled=False),
}


def parse_arm(run_dir):
    name = os.path.basename(run_dir.rstrip("/"))
    m = re.search(r"-(fusion_debias|fusion_token|fusion_adapter|fusion|baseline|debias|len_only|std_only)-s\d+", name)
    if not m:
        raise ValueError(f"Could not parse arm from run-dir name '{name}'. "
                         f"Expected '<model>-<arm>-s<seed>-<tag>'.")
    return m.group(1)


def resolve_checkpoint(run_dir):
    """Latest checkpoint-<step> dir (HRPO saves no root adapter); else run_dir if it holds one."""
    if os.path.exists(os.path.join(run_dir, "adapter_config.json")):
        return run_dir
    ckpts = [d for d in os.listdir(run_dir) if d.startswith("checkpoint-")
             and os.path.isdir(os.path.join(run_dir, d))]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint-* dir under {run_dir}")
    latest = max(ckpts, key=lambda d: int(d.split("-")[-1]))
    return os.path.join(run_dir, latest)


def _set_fusion_config(model, cfg):
    """Locate the inner Qwen2Model (has fuse_context_prediction) and set fusion attrs on it.
    Robust to the differing PEFT nesting between get_peft_model (train) and load_adapter (eval)."""
    m = model
    for _ in range(6):
        if hasattr(m, "fuse_context_prediction"):
            m.fusion_enabled     = cfg.get("fusion_enabled", False)
            m.fusion_layer_I     = cfg.get("fusion_layer_I", -2)
            m.fusion_gate_input  = cfg.get("fusion_gate_input", "hidden")
            m.fusion_normalize_h = cfg.get("fusion_normalize_h", True)
            m.fusion_use_adapter = cfg.get("fusion_use_adapter", False)
            return m
        m = getattr(m, "model", None)
        if m is None:
            break
    raise RuntimeError("Could not locate the inner model with fuse_context_prediction().")


def evaluate_run(run_dir, temperature, greedy, batch_size=32, num_samples=None):
    arm = parse_arm(run_dir)
    fusion_cfg = ARM_FUSION[arm]
    checkpoint = resolve_checkpoint(run_dir)

    base_model = next((b for b in BASE_MODELS if b.split("/")[-1] in run_dir), BASE_MODELS[0])
    print(f"[eval] run_dir={run_dir}\n[eval] arm={arm}  fusion={fusion_cfg}\n"
          f"[eval] base={base_model}  checkpoint={checkpoint}")

    model_path, use_exact = resolve_model_path(base_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path, max_seq_length=2048, load_in_4bit=False, fast_inference=False,
        use_exact_model_name=use_exact,
    )
    model.answer_start = ANSWER_START
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model.load_adapter(checkpoint)
    model = FastLanguageModel.for_inference(model)
    _set_fusion_config(model, fusion_cfg)   # AFTER load + for_inference

    dataset = load_dataset("openai/gsm8k", "main")["test"]
    if num_samples and len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))  # keep row order -> qid stable
    total = len(dataset)

    results, correct = [], 0
    bar = tqdm(total=total, desc=f"eval[{arm}]", dynamic_ncols=True)
    for i in range(0, total, batch_size):
        batch = dataset[i:i + batch_size]
        prompts = [tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q.strip()}],
            tokenize=False, add_generation_prompt=True) for q in batch["question"]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left",
                        add_special_tokens=False)
        prompt_ids = enc["input_ids"].to(model.device)
        prompt_mask = enc["attention_mask"].to(model.device)
        plen = prompt_ids.size(1)

        outputs = model.generate(
            prompt_ids, attention_mask=prompt_mask,
            generation_config=GenerationConfig(do_sample=not greedy, temperature=temperature,
                                               max_new_tokens=1024),
            processing_class=tokenizer, is_inference=greedy,
        )
        for j, out in enumerate(outputs):
            qid = i + j
            resp = tokenizer.decode(out[plen:]).split(tokenizer.special_tokens_map["eos_token"])[0]
            pred = process_gsm8k_answer(extract_from_response(resp))
            gold = process_gsm8k_answer(extract_hash_answer(batch["answer"][j]))
            is_ok = (pred == gold) and (pred is not None) and (pred != "")
            correct += int(is_ok)
            results.append({"qid": qid, "true_answer": gold, "generated_answer": pred,
                            "correct": bool(is_ok), "full_response": resp})
        bar.update(len(batch["question"]))
        bar.set_postfix({"acc": f"{correct/len(results)*100:.2f}%"})
    bar.close()

    acc = correct / total if total else 0.0
    seed_m = re.search(r"-s(\d+)-", os.path.basename(run_dir.rstrip("/")))
    metrics = {"arm": arm, "seed": int(seed_m.group(1)) if seed_m else None,
               "accuracy": acc, "correct": correct, "total": total,
               "base_model": base_model, "checkpoint": checkpoint,
               "run_dir": run_dir, "timestamp": datetime.now().isoformat()}
    with open(os.path.join(run_dir, "eval_results.json"), "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)
    print(f"[eval] arm={arm} seed={metrics['seed']} accuracy={acc:.4f} ({correct}/{total}) "
          f"-> {run_dir}/eval_results.json")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--sample", action="store_true", help="Sample instead of greedy (default greedy).")
    p.add_argument("--num_samples", type=int, default=None, help="Limit test set (debug).")
    args = p.parse_args()
    evaluate_run(args.run_dir, temperature=args.temperature, greedy=not args.sample,
                 batch_size=args.batch_size, num_samples=args.num_samples)
