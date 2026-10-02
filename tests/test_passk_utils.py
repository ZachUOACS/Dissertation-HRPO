"""Scoring rules of the sampled (pass@k) evaluation. CPU only, no torch.

Run from the repo root:  python -m pytest tests/test_passk_utils.py -q
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from passk_utils import (aggregate_samples, generated_length, paired_t,  # noqa: E402
                         summarise_questions)


def test_avg_pass_and_majority():
    s = aggregate_samples(["18", "18", "20", "18", None, "20", "18", "7"], "18")
    assert s["correct_flags"] == [True, True, False, True, False, False, True, False]
    assert s["avg_acc"] == 4 / 8
    assert s["pass_at_k"] is True
    assert s["maj_answer"] == "18" and s["maj_at_k"] is True


def test_pass_without_majority():
    # One lucky sample: counts for pass@k, not for maj@k.
    s = aggregate_samples(["5", "5", "5", "18"], "18")
    assert s["pass_at_k"] is True and s["maj_at_k"] is False and s["avg_acc"] == 0.25


def test_unextractable_answers_never_count_or_vote():
    s = aggregate_samples([None, None, None], None)
    assert s["avg_acc"] == 0.0 and s["pass_at_k"] is False and s["maj_at_k"] is False
    # None samples do not outvote a real answer.
    s = aggregate_samples([None, None, "18"], "18")
    assert s["maj_answer"] == "18" and s["maj_at_k"] is True and s["avg_acc"] == 1 / 3


def test_majority_tie_is_first_seen():
    assert aggregate_samples(["3", "4", "4", "3"], "3")["maj_answer"] == "3"


def test_generated_length_counts_to_first_eos():
    assert generated_length([5, 6, 7, 99, 99, 99], eos_token_id=99) == 4
    assert generated_length([5, 6, 7], eos_token_id=99) == 3      # hit the cap, no EOS
    assert generated_length([], eos_token_id=99) == 0


def test_summary_averages_over_questions():
    qs = [
        {"avg_acc": 1.0, "pass_at_k": True, "maj_at_k": True, "mean_tokens": 100.0},
        {"avg_acc": 0.25, "pass_at_k": True, "maj_at_k": False, "mean_tokens": 200.0},
        {"avg_acc": 0.0, "pass_at_k": False, "maj_at_k": False, "mean_tokens": 300.0},
    ]
    m = summarise_questions(qs)
    assert m["total"] == 3
    assert math.isclose(m["avg_at_k"], 1.25 / 3)
    assert math.isclose(m["pass_at_k"], 2 / 3)
    assert math.isclose(m["maj_at_k"], 1 / 3)
    assert math.isclose(m["mean_tokens"], 200.0)
    assert summarise_questions([])["total"] == 0


def test_paired_t():
    mean, se, t = paired_t([0.1, 0.1, 0.1, -0.1])
    assert math.isclose(mean, 0.05)
    assert math.isclose(se, 0.05)
    assert math.isclose(t, 1.0)
    mean, se, t = paired_t([0.0, 0.0, 0.0])          # no variance -> undefined t, not a crash
    assert mean == 0.0 and math.isnan(t)
