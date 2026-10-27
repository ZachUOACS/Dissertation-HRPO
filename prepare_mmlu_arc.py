"""
prepare_mmlu_arc.py - build the MMLU / ARC-C training subset of the cross-model study, once, on the
login node (CPU, needs the proxy for the Hugging Face download). See mmlu_data.py for the setting.

  1. cais/mmlu auxiliary_train (99,842) + allenai/ai2_arc ARC-Challenge train (1,119), merged in that
     order, as HRPO merged them.
  2. A fixed random order (random.Random(SUBSET_SEED).shuffle of the merged indices), walked until
     SUBSET_SIZE questions are kept. A question is skipped if its full chat prompt (system prompt +
     template, exactly as training builds it) is longer than MAX_PROMPT_TOKENS for ANY backbone in
     model_registry.py, so no training prompt is ever truncated, whichever model trains on it.
  3. Saved with datasets.save_to_disk to mmlu_data.TRAIN_ROOT, plus manifest.json (sources, counts,
     skipped, the merged indices kept, tokenisers, and a SHA-256 of the kept questions).

    python prepare_mmlu_arc.py            # refuses to overwrite an existing subset
"""

import argparse
import hashlib
import json
import os
import random

import mmlu_data as md
from model_registry import MODEL_NAMES, pin_chat_template_date
from utils import SYSTEM_PROMPT


def prompt_lengths(tokenizers, question, choices):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": md.format_prompt(question, choices)}]
    return {name: len(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
            for name, tok in tokenizers.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=md.TRAIN_ROOT)
    p.add_argument("--force", action="store_true", help="overwrite an existing subset")
    a = p.parse_args()
    if os.path.exists(a.out) and not a.force:
        raise SystemExit(f"{a.out} exists; runs may already depend on it. Use --force only before any run.")

    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer

    aux = [md.from_mmlu_aux(r) for r in load_dataset(md.MMLU_AUX_REPO, md.MMLU_AUX_CONFIG)["train"]]
    arc = [md.from_arc(r, source="arc_c_train") for r in load_dataset(md.ARC_REPO, md.ARC_CONFIG)["train"]]
    merged = aux + arc
    print(f"merged: {len(aux)} MMLU auxiliary + {len(arc)} ARC-C train = {len(merged)}")

    # One tokeniser per distinct backbone tokeniser (the Qwen sizes share one).
    tokenizers = {}
    for name in sorted(set(MODEL_NAMES.values())):
        key = "Qwen/Qwen2.5" if name.startswith("Qwen/Qwen2.5-") else name
        if key not in tokenizers:
            tok = AutoTokenizer.from_pretrained(name)
            pin_chat_template_date(tok)            # the prompts training actually builds
            tokenizers[key] = tok
    print("tokenisers:", sorted(tokenizers))

    order = list(range(len(merged)))
    random.Random(md.SUBSET_SEED).shuffle(order)
    kept, skipped, longest = [], 0, 0
    for i in order:
        r = merged[i]
        lengths = prompt_lengths(tokenizers, r["question"], r["choices"])
        if max(lengths.values()) > md.MAX_PROMPT_TOKENS:
            skipped += 1
            continue
        longest = max(longest, max(lengths.values()))
        kept.append(i)
        if len(kept) == md.SUBSET_SIZE:
            break
    if len(kept) < md.SUBSET_SIZE:
        raise SystemExit(f"only {len(kept)} questions fit {md.MAX_PROMPT_TOKENS} tokens")

    rows = [merged[i] for i in kept]
    sources = {}
    for r in rows:
        sources[r["source"]] = sources.get(r["source"], 0) + 1
    digest = hashlib.sha256("\n".join(md.format_prompt(r["question"], r["choices"]) + f"|{r['answer']}"
                                      for r in rows).encode("utf-8")).hexdigest()
    Dataset.from_list(rows).save_to_disk(a.out)
    manifest = {
        "train": {"mmlu_aux": f"{md.MMLU_AUX_REPO} {md.MMLU_AUX_CONFIG}", "arc": f"{md.ARC_REPO} {md.ARC_CONFIG} train"},
        "merged_size": len(merged), "subset_size": len(rows), "seed": md.SUBSET_SEED,
        "max_prompt_tokens": md.MAX_PROMPT_TOKENS, "longest_kept_prompt_tokens": longest,
        "skipped_too_long_before_full": skipped, "sources": sources, "tokenisers": sorted(tokenizers),
        "sha256_questions": digest, "merged_indices_kept": kept,
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"kept {len(rows)} ({sources}); skipped {skipped} prompts over {md.MAX_PROMPT_TOKENS} tokens; "
          f"longest kept {longest}; sha256 {digest[:16]}; saved to {a.out}")


if __name__ == "__main__":
    main()
