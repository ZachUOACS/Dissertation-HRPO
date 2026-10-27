"""
mmlu_data.py - the third task of the cross-model study (CROSS_MODEL_PLAN.md): HRPO's MMLU / ARC-C
setting, built the same way for training (prepare_mmlu_arc.py, variant_common.build_train_dataset)
and evaluation (eval_gsm8k.get_task_spec).

Training data, as in HRPO (Yue et al. 2025, Section 4.2: "For MMLU-ST and ARC-C, we train on the
merged auxiliary MMLU and ARC-C training sets"): cais/mmlu auxiliary_train (99,842 questions, rows
nested under a "train" key) plus allenai/ai2_arc ARC-Challenge train (1,119). HRPO trained on all of
it; one epoch of that is about 12,600 steps here, so this study trains on a fixed random subset the
size of GSM8K's training set (7,473 questions, 934 steps). That departure is reported.

Test data, as in HRPO's eval_mmlust.py / eval_arcc.py: TIGER-Lab/MMLU-STEM test (3,153) and
allenai/ai2_arc ARC-Challenge test (1,172), scored as one set of 4,325 with each answer's source kept,
so the two tests can also be reported separately.

Row format everywhere: {"question", "choices": [str], "answer": int, "source"}. The prompt text is
exactly utils.process_mmlu's ("Question: ...\\nOptions:\\nA. ...\\n", stripped).
"""

ARC_REPO, ARC_CONFIG = "allenai/ai2_arc", "ARC-Challenge"
MMLU_AUX_REPO, MMLU_AUX_CONFIG = "cais/mmlu", "auxiliary_train"
MMLU_STEM_REPO = "TIGER-Lab/MMLU-STEM"
TRAIN_ROOT = "data/mmlu_arc_train_7473"          # written by prepare_mmlu_arc.py
SUBSET_SIZE = 7473                                # = len(openai/gsm8k main train)
SUBSET_SEED = 0
MAX_PROMPT_TOKENS = 1024                          # HRPO's hrpo_mmlu.py --max_prompt_length


def format_prompt(question, choices):
    """utils.process_mmlu's prompt text (tests/test_family.py checks they agree)."""
    prompt = f"Question: {question}\nOptions:\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65 + i)}. {choice}\n"
    return prompt.strip()


def from_mmlu_aux(row):
    """cais/mmlu auxiliary_train nests each example under "train"."""
    r = row["train"] if "train" in row and isinstance(row["train"], dict) else row
    return {"question": r["question"], "choices": list(r["choices"]), "answer": int(r["answer"]),
            "source": "mmlu_aux"}


def from_arc(row, source="arc_c"):
    """ARC stores choices as {"text": [...], "label": [...]} and the key as a label ("A" or "1")."""
    labels, texts = list(row["choices"]["label"]), list(row["choices"]["text"])
    return {"question": row["question"], "choices": texts, "answer": labels.index(row["answerKey"]),
            "source": source}


def from_mmlu_stem(row):
    return {"question": row["question"], "choices": list(row["choices"]), "answer": int(row["answer"]),
            "source": "mmlu_stem"}


def load_test_set():
    """MMLU-STEM test then ARC-C test, in their published order, as one datasets.Dataset with the
    evaluator's columns: prompt_text, answer_letter, source."""
    from datasets import Dataset, load_dataset
    rows = [from_mmlu_stem(r) for r in load_dataset(MMLU_STEM_REPO, "default")["test"]]
    rows += [from_arc(r) for r in load_dataset(ARC_REPO, ARC_CONFIG)["test"]]
    return Dataset.from_dict({
        "prompt_text": [format_prompt(r["question"], r["choices"]) for r in rows],
        "answer_letter": [chr(65 + r["answer"]) for r in rows],
        "source": [r["source"] for r in rows],
    })
