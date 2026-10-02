"""MATH Stage 0 (tier12) — the warm start's data path.

Three properties decide whether a MATH warm start teaches what RL then rewards:

  1. The target's answer is BOXED. utils.process_math_answer understands only a
     boxed answer or one in LaTeX delimiters; a bare '#### X' falls through to
     "last number in the string". On the real MATH train set the bare form scores
     1.0 under the RL reward on 5,080 of 7,500 problems; the boxed form on all.
  2. Long examples are DROPPED, not truncated — a solution cut before its '####'
     would teach the model never to answer.
  3. A warm start only feeds RL on its own task (load_stage0_init guard).

stage0_warmstart imports unsloth at module level (it patches CUDA kernels), so
its functions are extracted with ast and exec'd, as in test_dr_grpo.py.
"""
import ast
import contextlib
import io
import json
import os
from types import SimpleNamespace

import pytest

import utils
from utils import (ANSWER_START, SYSTEM_PROMPT, extract_boxed_answer,
                   get_reward_func, math_sft_target, process_math_answer)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _score(completion, gold):
    reward = get_reward_func(process_math_answer)
    with contextlib.redirect_stdout(io.StringIO()):     # reward_func prints a sample
        return reward([[{"content": completion}]], [gold])[0]


def _stage0_defs(*names):
    path = os.path.join(ROOT, "stage0_warmstart.py")
    src = open(path).read()
    ns = {"ANSWER_START": ANSWER_START, "SYSTEM_PROMPT": SYSTEM_PROMPT,
          "math_sft_target": math_sft_target}
    found = set()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.get_source_segment(src, node), path, "exec"), ns)
            found.add(node.name)
    assert found == set(names), f"missing: {set(names) - found}"
    return ns


SOLUTIONS = [
    r"Adding gives $2+3=5$, so the answer is $\boxed{5}$.",
    r"Converting, $\theta = \frac{\pi}{2}$, so the angle is $\boxed{\frac{\pi}{2}}$.",
    r"In polar coordinates the point is $\boxed{\left( 3, \frac{\pi}{2} \right)}$.",
    r"Counting forward, the day is $\boxed{\text{Sunday}}$.",
    r"Simplifying the radical gives $\boxed{\sqrt{\frac{1}{2}}}$.",
    r"The sum is $\boxed{-12.5}$.",
]


# --- 1. the target format ------------------------------------------------------

@pytest.mark.parametrize("solution", SOLUTIONS)
def test_boxed_target_earns_full_reward(solution):
    target = math_sft_target(solution)
    assert target.count(ANSWER_START) == 1          # the reward needs exactly one
    assert target.startswith(solution.strip())      # reference reasoning, unaltered
    assert _score(target, extract_boxed_answer(solution)) == 1.0


def test_bare_answer_is_the_trap_this_format_avoids():
    solution = SOLUTIONS[1]
    bare = f"{solution}\n{ANSWER_START} " + r"\frac{\pi}{2}"
    assert process_math_answer(r"\frac{\pi}{2}") == "2"    # read as the last number
    assert _score(bare, extract_boxed_answer(solution)) == 0.0


def test_unanswerable_solutions_are_rejected():
    assert math_sft_target("There is no boxed answer in this solution.") is None
    assert math_sft_target(r"A stray #### marker, then $\boxed{3}$.") is None
    assert math_sft_target(r"An empty $\boxed{}$ answer.") is None


def test_extract_boxed_answer_cannot_detect_a_missing_answer():
    """Why has_boxed_answer exists: the extractor invents an answer from prose."""
    assert extract_boxed_answer("There is no boxed answer here.") == r"\boxed{answer here.}"
    assert not utils.has_boxed_answer("There is no boxed answer here.")
    assert utils.has_boxed_answer(r"so $x^2 + y^2 = \boxed 9$.")     # brace-less form
    assert extract_boxed_answer(r"so $x^2 + y^2 = \boxed 9$.") == r"\boxed{9}"


def test_real_math_train_sample_earns_full_reward():
    try:
        raw = utils.load_math_raw("train")
    except Exception as exc:                                # noqa: BLE001
        pytest.skip(f"MATH train not available offline ({exc!r})")
    step = max(1, len(raw) // 300)
    rows = [raw[i] for i in range(0, len(raw), step)
            if utils.has_boxed_answer(raw[i]["solution"])]
    bad = [r["solution"][-80:] for r in rows
           if _score(math_sft_target(r["solution"]),
                     extract_boxed_answer(r["solution"])) != 1.0]
    assert not bad, f"{len(bad)}/{len(rows)} targets would score 0, e.g. {bad[:2]}"


# --- 2. the encoder --------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    except Exception as exc:                                # noqa: BLE001
        pytest.skip(f"Qwen tokenizer not available offline ({exc!r})")


def _encoder():
    return _stage0_defs("_prompt_ids", "_think_span", "encode_math_example")


def test_think_span_stops_exactly_at_the_answer_marker(tokenizer):
    enc = _encoder()["encode_math_example"]
    problem = "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates."
    out = enc(tokenizer, problem, SOLUTIONS[2], 512, 1024)
    ids, labels, think = out["input_ids"], out["labels"], out["think"]
    n_prompt = labels.index(next(l for l in labels if l != -100))

    assert labels[:n_prompt] == [-100] * n_prompt          # loss on the answer only
    assert labels[n_prompt:] == ids[n_prompt:]
    assert not any(think[:n_prompt])                       # no think span in the prompt
    first_answer = think.index(False, n_prompt)
    assert first_answer > n_prompt                         # the span is not empty
    assert all(think[n_prompt:first_answer])
    assert ANSWER_START not in tokenizer.decode(ids[n_prompt:first_answer])
    assert ANSWER_START in tokenizer.decode(ids[n_prompt:first_answer + 1])


def test_overlong_examples_are_dropped_not_truncated(tokenizer):
    enc = _encoder()["encode_math_example"]
    problem = "What is $2+3$?"
    assert enc(tokenizer, problem, SOLUTIONS[0], 512, 1024) is not None
    assert enc(tokenizer, problem, SOLUTIONS[0], 512, 8) is None     # target too long
    assert enc(tokenizer, problem, SOLUTIONS[0], 8, 1024) is None    # prompt too long
    assert enc(tokenizer, problem, "no boxed answer", 512, 1024) is None


# --- 3. naming and the task guard ------------------------------------------------

def test_math_warm_starts_get_their_own_directory():
    stage0_dir = _stage0_defs("stage0_dir")["stage0_dir"]
    a = dict(model_size="1.5b", blend="on", residual_r_min=0.98, seed=42)
    assert stage0_dir(SimpleNamespace(**a, dataset="gsm8k")) == \
        "./experiments/1.5b-stage0-on-rmin0.98-seed42"             # existing name kept
    assert stage0_dir(SimpleNamespace(**a)) == \
        "./experiments/1.5b-stage0-on-rmin0.98-seed42"             # pre-flag namespace
    assert stage0_dir(SimpleNamespace(**a, dataset="math")) == \
        "./experiments/1.5b-stage0-on-rmin0.98-seed42-dsmath"


def _fake_stage0(tmp_path, summary):
    d = tmp_path / "stage0"
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(b"")
    if summary is not None:
        (d / "stage0_summary.json").write_text(json.dumps(summary))
    return str(d)


@pytest.mark.parametrize("summary, run_dataset", [
    ({"args": {"dataset": "gsm8k"}}, "math"),
    ({"args": {}}, "math"),          # summary written before --dataset existed = GSM8K
    (None, "math"),                  # no summary at all = GSM8K
    ({"args": {"dataset": "math"}}, "gsm8k"),
])
def test_warm_start_from_the_wrong_task_is_refused(tmp_path, summary, run_dataset):
    from variant_common import load_stage0_init
    with pytest.raises(RuntimeError, match="warm start was trained on"):
        load_stage0_init(None, _fake_stage0(tmp_path, summary), dataset=run_dataset)


@pytest.mark.parametrize("summary, run_dataset", [
    ({"args": {"dataset": "math"}}, "math"),
    ({"args": {}}, "gsm8k"),         # every existing GSM8K warm start still loads
])
def test_matching_task_passes_the_guard(tmp_path, summary, run_dataset):
    from variant_common import load_stage0_init
    with pytest.raises(Exception) as err:        # model=None fails AFTER the guard
        load_stage0_init(None, _fake_stage0(tmp_path, summary), dataset=run_dataset)
    assert "warm start was trained on" not in str(err.value)
