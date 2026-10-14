"""
eval_tokens.py - test-time token counts for saved greedy evaluations. CPU only (tokeniser, no model).

eval_results.json stores each answer's text (full_response, cut at the first end-of-sequence
token) but not its length in tokens, and the length-penalty arms are judged on exactly that. This
re-tokenises full_response with the Qwen2.5 tokeniser (shared by 0.5B/1.5B/3B) for every finished
evaluation, arms and baselines alike, so both sides of each comparison are counted the same way,
and writes eval_tokens.json beside each eval_results.json:
    {"source": "eval_results.json", "tokens": [per question], "mean": ..., "mean_correct": ...}
analyse_d2.py reports the per-seed change in mean test tokens when these files exist.

    python eval_tokens.py                    # every experiments/*/checkpoint-*/eval_results.json
    python eval_tokens.py --filter dsmath    # only directories containing the substring
"""

import argparse
import glob
import json
import os

TOKENIZER = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--filter", default="")
    parser.add_argument("--force", action="store_true", help="recount files already counted")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    done = 0
    for path in sorted(glob.glob(os.path.join(args.root, "*", "checkpoint-*", "eval_results.json"))):
        if args.filter not in path:
            continue
        out = os.path.join(os.path.dirname(path), "eval_tokens.json")
        if os.path.exists(out) and not args.force and os.path.getmtime(out) >= os.path.getmtime(path):
            continue
        with open(path) as f:
            res = json.load(f)["results"]
        tokens = [len(tok(r["full_response"], add_special_tokens=False)["input_ids"]) for r in res]
        correct = [t for t, r in zip(tokens, res) if r["correct"]]
        with open(out, "w") as f:
            json.dump({"source": "eval_results.json", "tokenizer": TOKENIZER, "tokens": tokens,
                       "mean": sum(tokens) / len(tokens) if tokens else None,
                       "mean_correct": sum(correct) / len(correct) if correct else None}, f)
        done += 1
        print(f"{out}: mean {sum(tokens) / max(1, len(tokens)):.1f} tokens over {len(tokens)}")
    print(f"counted {done} evaluation(s)")


if __name__ == "__main__":
    main()
