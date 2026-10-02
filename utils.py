import re
import string

ANSWER_START = "####"

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and "
    "the assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. The "
    "final answer is provided after the " + ANSWER_START + " tag, i.e., "
    "{reasoning process} " + ANSWER_START + " {answer}."
)


def extract_from_response(text: str) -> str:
    try:
        answer = text.split(ANSWER_START)[-1].strip()
        if answer.endswith("."):
            answer = answer[:-1].strip()
        return answer
    except IndexError:
        return ""


def extract_hash_answer(text: str) -> str | None:
    try:
        return text.split("####")[1].strip()
    except IndexError:
        return None


def extract_boxed_answer(text: str) -> str | None:
    try:  # wrap in boxed for process_math_answer
        return "\\boxed{" + find_box(text).strip() + "}"
    except IndexError:
        return None


# Reward-shaping constants (ablation variants v5/v6). Both default to 0.0
# (disabled) so the stock outcome-only reward below is numerically unchanged.
FORMAT_REWARD = 0.0
REASONING_BONUS = 0.0

# Minimum characters of reasoning text before ANSWER_START for REASONING_BONUS.
_MIN_REASONING_CHARS = 10


def _is_repetition_loop(
    text: str,
    min_len: int = 8,
    max_len: int = 120,
    min_repeats: int = 4,
    tail_chars: int = 600,
) -> bool:
    """Detect degenerate repetition loops near the end of a completion.

    Flags the completion when its final `tail_chars` characters contain the
    same substring (length `min_len`..`max_len`) repeated `min_repeats` or
    more times back-to-back. Character-level so CJK loops (no whitespace)
    are caught too.
    """
    if not text:
        return False
    tail = text[-tail_chars:]
    pattern = r"(.{%d,%d})\1{%d,}" % (min_len, max_len, min_repeats - 1)
    return re.search(pattern, tail, re.DOTALL) is not None


def get_reward_func(
    process_answer_func,
    format_reward: float = None,
    reasoning_bonus: float = None,
    zero_repetition_loops: bool = False,
):
    """Outcome reward with optional shaping (v5) and repetition zeroing (v6).

    Defaults (None/None/False) reproduce the original reward exactly:
    1.0 iff the extracted answer is correct AND ANSWER_START occurs exactly
    once, else 0.0.
    """
    fmt_reward = FORMAT_REWARD if format_reward is None else format_reward
    rsn_bonus = REASONING_BONUS if reasoning_bonus is None else reasoning_bonus

    def reward_func(completions, answer, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]

        ans = [process_answer_func(a) for a in answer]
        extracted = [extract_from_response(r) for r in responses]
        predictions = [process_answer_func(r) for r in extracted]
        accuracy = [True if r == a else False for r, a in zip(predictions, ans)]

        escaped_answer_start = re.escape(ANSWER_START)
        pattern = f"^(?:(?!{escaped_answer_start}).)*{escaped_answer_start}(?:(?!{escaped_answer_start}).)*$"
        matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]

        rewards = []
        for response, a, m in zip(responses, accuracy, matches):
            reward = 1.0 if a and m else 0.0
            if m and fmt_reward:
                reward += fmt_reward
            if m and rsn_bonus:
                reasoning_text = response.split(ANSWER_START)[0].strip()
                if len(reasoning_text) >= _MIN_REASONING_CHARS:
                    reward += rsn_bonus
            if zero_repetition_loops and _is_repetition_loop(response):
                reward = 0.0
            rewards.append(reward)

        print(
            "=" * 50,
            f"\nBatch accuracy: " + "".join("Y" if r > 0 else "N" for r in rewards),
            f"\n1/{len(completions)} responses (answer: {ans[0]}):\n{responses[0]}",
            "\n" + "=" * 50,
        )
        return rewards

    return reward_func


# def exact_match(prediction, golden_answers):
#     if isinstance(golden_answers, str):
#         golden_answers = [golden_answers]
#     normalized_prediction = normalize_answer(prediction)
#     score = 0
#     for golden_answer in golden_answers:
#         golden_answer = normalize_answer(golden_answer)
#         if golden_answer == normalized_prediction:
#             score = 1
#             break
#     return score


def reward_func_rag(completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]

    extracted = [extract_from_response(r) for r in responses]
    predictions = [process_qa_answer(r) for r in extracted]
    # accuracy = [True if r == a else False for r, a in zip(predictions, ans)]
    accuracy = []
    for normalized_pred, answer_list in zip(predictions, answer):
        cur_accuracy = False
        for golden_answer in answer_list:
            normalized_answer = process_qa_answer(golden_answer)
            if normalized_answer == normalized_pred:
                cur_accuracy = True
        accuracy.append(cur_accuracy)

    escaped_answer_start = re.escape(ANSWER_START)
    pattern = f"^(?:(?!{escaped_answer_start}).)*{escaped_answer_start}(?:(?!{escaped_answer_start}).)*$"
    matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]

    rewards = [1.0 if a and m else 0.0 for a, m in zip(accuracy, matches)]

    print(
        "=" * 50,
        f"\nBatch accuracy: " + "".join("Y" if r > 0 else "N" for r in rewards),
        f"\n1/{len(completions)} responses (answer: {answer[0]}):\n{responses[0]}",
        "\n" + "=" * 50,
    )
    return rewards


def delete_extra_zero(n):
    try:
        n=float(n)
    except:
        try:
            n = eval(n)
        except:
            print("Conversion to floating number fails: {}".format(n))
            return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip("0")
        n = int(n.rstrip(".")) if n.endswith(".") else float(n)
        n=str(n)
        return n


def find_box(pred_str: str):
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if (ans[0] == "{"):
        stack = 1
        a = ""
        for c in ans[1:]:
            if (c == "{"):
                stack += 1
                a += c
            elif (c == "}"):
                stack -= 1
                if (stack == 0): break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def find_latex(pred_str: str):
    pattern = re.compile(
        r"""
        (\\\[(?P<display>[\s\S]+?)\\\])                        # \[ ... \]
        |(\\\((?P<inline>[\s\S]+?)\\\))                        # \( ... \)
        |((?P<dollar>\$\$?)(?P<dcontent>[\s\S]+?)(?P=dollar))  # $...$ or $$...$$
        """,
        re.VERBOSE
    )
    matches = list(pattern.finditer(pred_str))
    if not matches: return ""
    return (matches[-1].group("display") or matches[-1].group("inline") 
            or matches[-1].group("dcontent")).strip()


def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    else:
        return string


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr:
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.strip("$")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)
    return string


def process_gsm8k_answer(pred: str) -> str:
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ")
    pred = [delete_extra_zero(s.replace(",", "")) 
            for s in re.findall(r"-?\d+/?\.?\d*", pred)]

    if len(pred) == 0:
        pred = ""
    else:
        pred = pred[-1].rstrip(".").rstrip("/")
    return pred


def process_math_answer(pred: str) -> str:
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ")
    if "boxed" in pred:
        pred = find_box(pred)
    elif find_latex(pred):
        pred = find_latex(pred)
    else:
        preds = re.findall(r"-?\d*\.?\d+", pred)
        if(len(preds) >= 1):
            pred = preds[-1]
        else:
            pred = ""

    pred = _strip_string(pred).rstrip(".").rstrip("/")
    pred = re.sub(r"\\text\{([^}]*)\}", r"\1", pred).lower()
    return pred


def process_mmlu_answer(pred: str) -> str:
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ")
    tmp = re.findall(r"\b(A|B|C|D|E|F|G|H|I|J)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]

    if len(pred) == 0:
        pred = ""
    else:
        pred = pred[-1].rstrip(".").rstrip("/")
    return pred


def process_qa_answer(pred: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(pred))))


def process_gsm8k(batch):
    prompts = [[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q.strip()},
    ] for q in batch["question"]]

    return {
        "prompt": prompts,
        "answer": [extract_hash_answer(a) for a in batch["answer"]]
    }


def process_math(batch):
    prompts = [[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q.strip()},
    ] for q in batch["problem"]]

    return {
        "prompt": prompts,
        "answer": [extract_boxed_answer(a) for a in batch["solution"]]
    }


# --- MATH loading -----------------------------------------------------------
# The original release (Hendrycks et al.) ships as root/<split>/<category>/*.json
# with 'problem' and 'solution' fields, which is what hrpo_math.py and
# eval_math.py expect via their own preprocess_math(). The HuggingFace mirrors
# expose the same two fields, so both paths converge on process_math() above and
# yield an identical {prompt, answer} dataset. MATH-500 is the 500-problem
# evaluation subset used as the reported test set and needs no local copy.
# Imports are local to each function so utils' import surface is unchanged for
# the training scripts that load it on every run.
MATH_LOCAL_ROOTS = ("../MATH", "./MATH", "/data/ztay632/MATH")
MATH_HF_TRAIN = "EleutherAI/hendrycks_math"     # one config per subject
MATH_HF_TEST = "HuggingFaceH4/MATH-500"


def _math_from_local(root, split):
    """root/<split>/<category>/*.json -> Dataset(problem, solution)."""
    import json
    import os
    from datasets import Dataset
    problems, solutions = [], []
    split_dir = os.path.join(root, split)
    for folder in sorted(os.listdir(split_dir)):
        sub = os.path.join(split_dir, folder)
        if not os.path.isdir(sub):
            continue
        for file in sorted(os.listdir(sub)):
            if file.endswith(".json"):
                with open(os.path.join(sub, file)) as f:
                    entry = json.load(f)
                problems.append(entry["problem"])
                solutions.append(entry["solution"])
    if not problems:
        raise FileNotFoundError(f"no MATH .json files under {split_dir}")
    return Dataset.from_dict({"problem": problems, "solution": solutions})


def _math_from_hub(split):
    from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
    if split == "test":
        return load_dataset(MATH_HF_TEST)["test"].select_columns(["problem", "solution"])
    parts = [load_dataset(MATH_HF_TRAIN, c)[split].select_columns(["problem", "solution"])
             for c in get_dataset_config_names(MATH_HF_TRAIN)]
    return concatenate_datasets(parts)


def load_math_raw(split="train", root=None):
    """MATH as Dataset(problem, solution): a local copy if present, else the
    HuggingFace mirror. Raises with both options spelled out rather than
    silently training on the wrong thing."""
    import os
    roots = [root] if root else list(MATH_LOCAL_ROOTS)
    for r in roots:
        if r and os.path.isdir(os.path.join(r, split)):
            print(f"[math] local copy: {os.path.join(r, split)}")
            return _math_from_local(r, split)
    try:
        print(f"[math] no local copy in {roots}; loading from the hub")
        return _math_from_hub(split)
    except Exception as exc:                                   # noqa: BLE001
        raise RuntimeError(
            f"could not load MATH '{split}'. Either point --math_root at a "
            f"directory holding {split}/<category>/*.json, or make "
            f"{MATH_HF_TRAIN} / {MATH_HF_TEST} reachable (compute nodes need "
            f"the squid proxy and HF_HOME exported). Cause: {exc!r}") from exc


def preprocess_math_dataset(split="train", chunk_size=1000, root=None):
    r"""MATH in the {prompt, answer} shape every variant script consumes.

    Nothing is dropped, deliberately. Two MATH train solutions (indices 5341 and
    5343) contain an empty '\boxed{}' — the answer was lost upstream — so their
    gold answer is empty. They were in the training set of every completed MATH
    run (tier10), and removing them would reshuffle the whole data order, so a
    later run would differ from tier10 in more than the one thing it varies.
    Stage 0 does drop them (math_sft_target): a warm start must never teach
    '#### \boxed{}'."""
    return load_math_raw(split, root).map(
        process_math, batched=True, batch_size=chunk_size, load_from_cache_file=False)


def has_boxed_answer(solution: str) -> bool:
    r"""True when a MATH solution carries a real \boxed answer.

    extract_boxed_answer cannot answer this: it splits on the substring "boxed"
    and never returns None, so prose with no \boxed at all comes back as an
    invented answer — "no boxed answer here" yields '\boxed{answer here}'.
    On MATH train, 7,498 of 7,500 pass: the two failures contain an empty
    '\boxed{}' (see preprocess_math_dataset). Two others use the brace-less
    '\boxed 9', which extract_boxed_answer reads correctly and which pass."""
    return ("\\boxed" in solution
            and extract_boxed_answer(solution) not in (None, "\\boxed{}"))


def math_sft_target(solution: str):
    r"""Stage-0 (teacher-forced) target for a MATH solution: the reference
    reasoning, then '#### \boxed{X}'.

    The answer MUST stay boxed. process_math_answer understands only a boxed
    answer or one in LaTeX delimiters; a bare '#### X' falls through to "take the
    last number in the string", so '#### \frac{\pi}{2}' is read as '2'. Across
    the 7,500 MATH training problems the bare form scores 1.0 under the RL reward
    on only 5,080; the boxed form on all 7,500 (tests/test_math_stage0.py).

    Returns None when the solution has no boxed answer, or already contains the
    answer marker (the reward requires exactly one)."""
    if not has_boxed_answer(solution) or ANSWER_START in solution:
        return None
    return f"{solution.strip()}\n{ANSWER_START} {extract_boxed_answer(solution)}"


def load_math500():
    """The 500-problem MATH evaluation subset, always from the hub. Evaluation
    deliberately does NOT use a local MATH test split: that is the full ~5000
    problem set, and mixing the two would change the reported denominator."""
    from datasets import load_dataset
    return load_dataset(MATH_HF_TEST)["test"].select_columns(["problem", "solution"])


def process_mmlu(batch):
    def get_prompt(question, choices):
        prompt = f"Question: {question}\nOptions:\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        return prompt

    prompts = [[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": get_prompt(q, c).strip()},
    ] for q, c in zip(batch["question"], batch["choices"])]

    return {
        "prompt": prompts,
        "answer": [f"{chr(65 + a)}" for a in batch["answer"]]
    }


def process_rag(batch, topk=3):
    def get_prompt(question, contexts):
        prompt = "Context (which may or may not be relevant):\n"
        for context in contexts[:topk]:
            cur_context = context.split("\n")
            cur_context[0] = cur_context[0].strip('"')
            prompt += "::::".join(cur_context) + "\n"
        prompt += f"\nQuestion: {question}"
        return prompt

    prompts = [[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": get_prompt(q, c).strip()},
    ] for q, c in zip(batch["question"], batch["contexts"])]

    return {
        "prompt": prompts,
        "answer": batch["golden_answers"],
    }