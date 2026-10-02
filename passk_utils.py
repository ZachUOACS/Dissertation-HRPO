"""
passk_utils.py - pure helpers for the sampled (pass@k) evaluation.

Kept free of torch / unsloth imports so the scoring rules can be unit-tested on a
CPU node (tests/test_passk_utils.py) and reused by analyse_d2.py.

Metric definitions (the ones TARPO, LEPO and Soft Tokens report):
  avg@k   mean over the k samples of "this sample is correct", then mean over
          questions. It estimates single-sample accuracy and does not depend on k,
          so avg@8 is comparable with a paper's "Pass@1 averaged over 32 samples".
  pass@k  a question counts if ANY of its k samples is correct. Depends on k:
          never compare pass@8 with a paper's pass@32.
  maj@k   a question counts if the most frequent extracted answer is correct
          (ties broken by first occurrence; samples with no extractable answer
          do not vote).
"""

from collections import Counter


def aggregate_samples(generated_answers, true_answer):
    """Score one question from its k sampled answers.

    generated_answers: list of extracted answers (None when nothing was extractable).
    Returns dict(correct_flags, avg_acc, pass_at_k, maj_answer, maj_at_k)."""
    flags = [a is not None and a == true_answer for a in generated_answers]
    votes = [a for a in generated_answers if a is not None]
    maj_answer = Counter(votes).most_common(1)[0][0] if votes else None
    return {
        "correct_flags": flags,
        "avg_acc": sum(flags) / len(flags) if flags else 0.0,
        "pass_at_k": any(flags),
        "maj_answer": maj_answer,
        "maj_at_k": maj_answer is not None and maj_answer == true_answer,
    }


def generated_length(token_ids, eos_token_id):
    """Number of generated tokens up to and including the first EOS (or all of them
    if no EOS was produced, i.e. the completion hit the length cap)."""
    for i, tok in enumerate(token_ids):
        if tok == eos_token_id:
            return i + 1
    return len(token_ids)


def summarise_questions(per_question):
    """Aggregate per-question records into the headline metrics."""
    n = len(per_question)
    if n == 0:
        return {"avg_at_k": 0.0, "pass_at_k": 0.0, "maj_at_k": 0.0, "total": 0}
    return {
        "avg_at_k": sum(q["avg_acc"] for q in per_question) / n,
        "pass_at_k": sum(1 for q in per_question if q["pass_at_k"]) / n,
        "maj_at_k": sum(1 for q in per_question if q["maj_at_k"]) / n,
        "mean_tokens": sum(q["mean_tokens"] for q in per_question) / n,
        "total": n,
    }


def paired_t(diffs):
    """Paired t statistic over per-question differences. Returns (mean, se, t).
    The per-question avg@k difference is the right unit: pairing removes question
    difficulty. It does NOT remove training-run variance, so a significant t for one
    seed is a statement about two checkpoints, not about two methods."""
    n = len(diffs)
    if n < 2:
        return (diffs[0] if diffs else 0.0), float("nan"), float("nan")
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = (var / n) ** 0.5
    return mean, se, (mean / se if se > 0 else float("nan"))
