"""
model_registry.py - the one list of backbones the study trains, and the size keys that name them.
Standard library only, so the analysis scripts can import it anywhere.

A size key is the first token of every experiment directory (experiments/<key>-<variant>...), so
it must not contain "-". Qwen keys keep their original spelling ("1.5b"), so no existing directory
name changes. The cross-model study (CROSS_MODEL_PLAN.md) adds other families under their own key.

Gemma is deliberately absent: unsloth runs Gemma 2's generation through its own
Gemma2Model_fast_forward_inference, which does not yet contain HRPO's rollout blend, so a Gemma
HRPO run would train with the latent but roll out without it. Add "gemma2b" only after that port.
"""

import re

MODEL_NAMES = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
    # Cross-model study. unsloth's copies: same weights as meta-llama/*, without the licence gate
    # on the download (the Llama 3.2 licence still applies).
    "llama1b": "unsloth/Llama-3.2-1B-Instruct",
    "llama3b": "unsloth/Llama-3.2-3B-Instruct",
    # Optional third lineage: Hugging Face's SmolLM2, LlamaForCausalLM with tied embeddings (config.json,
    # 23 October), so it runs on the same Llama code path with no port; its own pretraining data and
    # tokeniser (49,152 tokens). Ungated.
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}

# Model family of each key, for tables and plots.
FAMILY = {
    "0.5b": "qwen2.5", "1.5b": "qwen2.5", "3b": "qwen2.5",
    "llama1b": "llama3.2", "llama3b": "llama3.2", "smollm2": "smollm2",
}

# Tasks: the directory suffix experiment_dir appends for each (--dataset) and the size of the test
# set a complete evaluation has. mmlu = HRPO's merged MMLU auxiliary + ARC-Challenge training set
# (a fixed GSM8K-sized subset, mmlu_data.py), scored on MMLU-STEM (3,153) + ARC-C test (1,172).
TASK_SUFFIX = {"gsm8k": "", "math": "-dsmath", "mmlu": "-dsmmlu"}
TASK_TEST_SIZE = {"gsm8k": 1319, "math": 500, "mmlu": 3153 + 1172}


def task_from_name(name):
    """The task a run directory (or its suffix) trained on, from its -ds token."""
    for task, suffix in TASK_SUFFIX.items():
        if suffix and suffix in name:
            return task
    return "gsm8k"


def baseline_tag(task):
    """The cold Dr. GRPO baseline tag of a task, as analyse_d2 keys arms ("drboth", "drboth-dsmath")."""
    return "drboth" + TASK_SUFFIX[task]


# Regex alternation matching exactly one size key; every directory regex in the repo uses this.
SIZE_ALT = "|".join(re.escape(k) for k in sorted(MODEL_NAMES, key=len, reverse=True))

# Llama 3.2's chat template writes the current date into every system prompt
# (date_string = strftime_now("%d %b %Y")), so the same question would get a different prompt on
# each day a run trains or is scored. Pinned to the template's own fallback date, for training and
# evaluation alike. A template without that logic (Qwen2.5) is left exactly as it is.
PINNED_DATE = "26 Jul 2024"


def pin_chat_template_date(tokenizer):
    """Fix date_string in a date-stamping chat template. Returns True if the template changed."""
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or "strftime_now" not in template or "date_string" not in template:
        return False
    if template.startswith("{%- set date_string ="):
        return False                      # already pinned
    tokenizer.chat_template = '{%- set date_string = "' + PINNED_DATE + '" %}\n' + template
    return True
