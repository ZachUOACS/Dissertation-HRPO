"""
variant_common.py
=================
Shared plumbing for the ablation variant training scripts (v1-v6, Task 1 of the
dissertation spec). This module deliberately contains only *variant-neutral*
helpers; everything that differs between variants lives explicitly in the
individual `hrpo_*.py` scripts so each variant remains independently auditable.

No unsloth/trl imports here — the variant scripts must import unsloth first
(it patches transformers/trl at import time), exactly as `hrpo_gsm8k.py` does.
"""

import argparse
import datetime
import glob
import json
import os
import re
import time
import types

import torch
from transformers import TrainerCallback

MODEL_NAMES = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
}

GATE_MODULES = [
    "thinking_residual_gate_r",
    "thinking_residual_gate_i",
    "thinking_residual_Lambda",
]

# Causal chain order (spec Section 6).
VARIANT_CHAIN = [
    "v1_grpo_baseline",
    "v2_fixed_blend",
    "v3_learned_lambda",
    "v4_full_hrpo_gate",
    "v5_hrpo_format_reward",
    "v6_hrpo_full",
]

# Controls: trainable variants that are NOT links in the causal chain. Kept out
# of VARIANT_CHAIN on purpose — analyse_gate's Lambda plots and generate_report's
# primary table/figure iterate VARIANT_CHAIN, so a control listed there is drawn
# as a seventh ablation step. Its gate mode still lives in GATE_MODES, which is
# all evaluation needs.
CONTROL_VARIANTS = [
    "v7_constant_vector",   # tier-8 C1: v3 with the Eq.3 projection -> learned u
]

# Gate configuration each variant was trained with (changes.md mapping #7);
# evaluation and analysis must mirror it.
GATE_MODES = {
    "v1_grpo_baseline": "disabled",
    "v2_fixed_blend": "unit",
    "v3_learned_lambda": "unit",
    "v4_full_hrpo_gate": "standard",
    "v5_hrpo_format_reward": "standard",
    "v6_hrpo_full": "standard",
    # C1 is v3 with the latent SOURCE swapped (projection -> constant vector).
    # The gate configuration is v3's (unit gates), but evaluation must ALSO
    # install the constant-latent patch, or the checkpoint is scored with the
    # Eq.3 projection it was never trained with. "constant" = unit gates + u.
    "v7_constant_vector": "constant",
}


# ---------------------------------------------------------------------------
# Datasets — GSM8K is the study's task; MATH is the second-task extension
# ---------------------------------------------------------------------------
# Everything except the loader and the answer comparator is shared: both tasks
# use SYSTEM_PROMPT, the same "####" ANSWER_START, extract_from_response and
# get_reward_func. A variant script therefore differs across datasets in exactly
# two calls, below.
#
# Lengths are deliberately NOT defaulted per dataset — the submission script
# passes them explicitly so run_config.json records what each run trained with
# (the HRPO paper's MATH row uses 1024-token completions against GSM8K's 512).
DATASETS = ("gsm8k", "math")


def build_train_dataset(dataset: str, chunk_size: int = 500, math_root: str = None):
    """The training split as {prompt, answer} — identical in shape across tasks."""
    import utils
    if dataset == "gsm8k":
        from datasets import load_dataset
        return load_dataset("openai/gsm8k", "main")["train"].map(
            utils.process_gsm8k, batched=True, batch_size=chunk_size,
            load_from_cache_file=False)
    if dataset == "math":
        return utils.preprocess_math_dataset("train", chunk_size=chunk_size,
                                             root=math_root)
    raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")


def reward_answer_fn(dataset: str):
    """The answer comparator get_reward_func scores with: GSM8K compares the
    string after '####', MATH compares the contents of a boxed expression."""
    import utils
    return {"gsm8k": utils.process_gsm8k_answer,
            "math": utils.process_math_answer}[dataset]


# ---------------------------------------------------------------------------
# Argument parsing — Section 5 fixed hyperparameters as defaults
# ---------------------------------------------------------------------------

def build_arg_parser(variant_id: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"HRPO ablation variant {variant_id}")
    parser.add_argument("--model_size", type=str, default="1.5b",
                        choices=sorted(MODEL_NAMES.keys()))

    # Task. GSM8K is the study's dataset and stays the default so every existing
    # directory name, arm key and result is unchanged; "math" is the second-task
    # extension and lands in its own -dsmath directories.
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=DATASETS)
    parser.add_argument("--math_root", type=str, default=None,
                        help="directory holding MATH as <split>/<category>/*.json; "
                             "omit to use a known local path or the HF mirror")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_targets", type=str, default="full",
                        choices=sorted(LORA_TARGETS.keys()),
                        help="full = attention+MLP (main ablation); "
                             "attention = q/k/v/o only, matching HRPO paper Table 4")

    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.005)
    # C1 control. "projection" = the published Eq.3 softmax-over-embeddings
    # projection (default: every existing variant). "constant" = replace that
    # output with a single learned vector u, isolating "does the latent need to
    # be input-dependent at all, or would any on-manifold vector do?".
    parser.add_argument("--latent_source", type=str, default="projection",
                        choices=["projection", "constant"])
    # How the C1 constant vector u is initialised. "mean" is the brief's
    # specification (centroid of the embedding matrix). It is on-manifold in
    # DIRECTION but not necessarily in SCALE: for a peaked probs the real Eq.3
    # output is close to a single token embedding, whereas the centroid's norm is
    # much smaller because the rows largely cancel. "mean_scaled" keeps the
    # centroid direction but rescales to the mean row norm; "random_row" takes an
    # actual embedding row, which is on-manifold in both senses. The registration
    # reports all three norms so the choice can be made on evidence.
    parser.add_argument("--constant_init", type=str, default="mean",
                        choices=["mean", "mean_scaled", "random_row"])
    parser.add_argument("--residual_r_min", type=float, default=0.98)
    parser.add_argument("--residual_r_max", type=float, default=0.999)
    parser.add_argument("--lr_residual_gate", type=float, default=1e-4)
    parser.add_argument("--lr_residual_Lambda", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--optimizer", type=str, default="paged_adamw_8bit")
    parser.add_argument("--max_grad_norm", type=float, default=0.1)

    # HRPO paper Table 4/5 (Yue et al. 2025), GSM8k row:
    #   group size 4, grad-accum 4, TOTAL train batch 32 -> per_device 8
    #   (total = per_device x grad_accum), prompt/completion 512 / 512.
    # The dissertation spec's Section 5 listed per_device 4 and 1024/1024,
    # which cost 4x per completion (each optimizer step makes grad_accum
    # generate() calls regardless of batch size, so 1024-token decodes double
    # the step time while a batch of 4 halves the completions produced) and
    # mismatched eval_gsm8k.py's 512-token decode. Paper values used.
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=512)
    # Rollout-sampling repetition penalty. The published HRPO method has none, so
    # 1.0 is the only value used for any reported result. The default was 1.2
    # until the D2 programme; it is now 1.0 so a forgotten flag cannot silently
    # train an off-paper run. experiment_dir still suffixes "-rp1" for 1.0 (it
    # only omits the suffix at 1.2), which is exactly how every settled rp=1.0
    # directory is already named.
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    # Dr. GRPO de-bias (control C2; Liu et al. 2025). "len" divides the policy
    # loss by the constant max_completion_length instead of each response's own
    # length; "std" drops the per-group std from the advantage; "both" is Dr.
    # GRPO. "off" is the original GRPO objective — every run before C2 used it,
    # and with "off" the loss/advantage code is bit-identical to before (pinned
    # by tests/test_dr_grpo.py). Applied to the trainer by apply_dr_grpo().
    parser.add_argument("--dr_grpo", type=str, default="off",
                        choices=["off", "len", "std", "both"])

    # D2: parameter-space exploration noise (QeRL-style adaptive quantization
    # noise; see noise_ext.py). Gaussian noise on the frozen RMSNorm weights, one
    # draw per micro-batch, alive for the rollout AND the loss forward, removed
    # before checkpointing/eval. sigma decays exponentially from _start to _end
    # over --noise_stages equal stages (QeRL: 5e-2 -> 5e-4, 10 stages).
    # _start = 0 (default) is OFF and leaves the trainer untouched. _end defaults
    # to _start (constant noise). A noisy run is a different arm, so
    # experiment_dir files it under a "new-" prefix that the legacy analysis
    # (EXPERIMENT_NAME_RE) cannot see, rather than pooling it with its control.
    parser.add_argument("--noise_sigma_start", type=float, default=0.0)
    parser.add_argument("--noise_sigma_end", type=float, default=None)
    parser.add_argument("--noise_stages", type=int, default=10)
    parser.add_argument("--noise_mode", type=str, default="add",
                        choices=["add", "mult"])

    # Decoupled latent temperature. HRPO's Eq.3 mixture is formed from the sampling
    # distribution, so the sampling temperature (0.5) also sharpens the latent. With
    # this set, the mixture is formed from the same top-k scores re-tempered to this
    # value, while tokens are still sampled at --temperature (see
    # transformers/generation/utils.py::hrpo_latent_probs). Rollout only: the loss
    # replays the stored blended inputs, and evaluation must form the mixture the
    # same way (eval_d2 reads it from train_summary.json). None (default) is the
    # original behaviour. A run with it set is a new arm ("new-...-lt{value}").
    parser.add_argument("--latent_temperature", type=float, default=None)

    # D3: uncertainty-gated blend. Replaces HRPO's Eq.4 blend (whose Lambda and gates were shown
    # to be inert) with a convex blend whose weight depends on how far the latent departs from
    # the embedding of the token it accompanies:
    #   d_t = 1 - cos(h_t, e_t),  beta_t = blend_max * clamp((d_t - d0) / (d1 - d0), 0, 1),
    #   input_t = (1 - beta_t) * e_t + beta_t * h_t.
    # Where the model was confident, h_t ~ e_t, d_t ~ 0 and the input is the plain token; where it
    # was uncertain the latent is blended in strongly. d_t needs only (e_t, h_t), which both the
    # rollout and the loss forward already have, so the patch covers both call sites with no new
    # buffer. d0/d1 come from the calibration in probe_latent_temp.py --calibrate. "off" (default)
    # leaves the variant's own blend untouched. A run with it on is a new arm ("new-...-dg{max}").
    # C1: stochastic latent with its own gradient (after LEPO, Zhou et al. 2026). In sampled
    # rollouts the token is drawn by the Gumbel-max trick (same distribution as before) and the
    # latent is the Gumbel-softmax of the SAME noise at this temperature, so the latent differs
    # between the completions of a group. In the loss, positions whose latent was used take the
    # soft label sum_k z_k log pi(k) as the policy-gradient target (LEPO Eq. 9) instead of the
    # one-hot token; elsewhere the target is the token, exactly as before. Greedy evaluation
    # keeps the deterministic Eq.3 mixture. None (default) is off. LEPO used 0.3 (3B) and 0.5
    # (7B) and reports gains for 0.3 to 0.6. A run with it set is a new arm ("new-...-sl{T}").
    parser.add_argument("--stochastic_latent", type=float, default=None)
    # Length penalty (Arora & Zanette 2025, arXiv 2502.04463; length_penalty.py): a correct
    # response earns 1 - alpha * sigmoid((len - mean) / std), mean and std over the correct
    # responses of its own group; incorrect responses earn 0 as before. alpha in (0, 1); the paper
    # used 0.05, 0.1, 0.2 and 0.4 with an RLOO estimator, which Dr. GRPO's advantage matches. None
    # (default) is off. A run with it set is a new arm ("new-...-lp{alpha}").
    parser.add_argument("--length_penalty_alpha", type=float, default=None)
    parser.add_argument("--blend_gate", type=str, default="off", choices=["off", "divergence"])
    parser.add_argument("--blend_max", type=float, default=0.5)
    parser.add_argument("--blend_d0", type=float, default=None)
    parser.add_argument("--blend_d1", type=float, default=None)

    # Stage-0 warm start: directory produced by stage0_warmstart.py whose LoRA
    # adapter + Lambda are loaded before RL begins. Recorded in the config
    # fingerprint so a run started from a warm start can never silently resume as
    # a cold one (or from a different warm start).
    parser.add_argument("--init_from", type=str, default=None,
                        help="experiments/<size>-stage0-... directory to "
                             "initialise the adapter and Lambda from")

    parser.add_argument("--init_only", action="store_true",
                        help="load --init_from, report Lambda, and exit without "
                             "training (Stage-0 brief verification check 3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="wandb")
    return parser


# ---------------------------------------------------------------------------
# Experiment directory handling
# ---------------------------------------------------------------------------

# LoRA target sets. "full" is what the main ablation used; "attention" matches
# the HRPO paper's Table 4 ("query, key, value, dense"), used by the faithful
# reproduction run.
LORA_TARGETS = {
    "full": ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"],
    "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def experiment_dir(model_size: str, variant_id: str, args=None) -> str:
    """experiments/{size}-{variant}, plus a suffix for any follow-up run that
    departs from the main ablation's configuration.

    Without the suffix a seed-43 replicate would resolve to the seed-42
    directory, hit exit_if_complete() and silently do nothing.
    """
    name = f"{model_size}-{variant_id}"
    if args is not None:
        if getattr(args, "seed", 42) != 42:
            name += f"-seed{args.seed}"
        if getattr(args, "repetition_penalty", 1.2) != 1.2:
            name += f"-rp{args.repetition_penalty:g}"
        if getattr(args, "lora_targets", "full") != "full":
            name += f"-lora{args.lora_targets}"
        # A Stage-0 warm start is a DIFFERENT arm from its cold counterpart.
        # Without this suffix a warm-started v3 resolves to
        # experiments/1.5b-v3_learned_lambda-rp1 — the settled A1 result — and
        # would either exit_if_complete (masking the run) or, worse, train into
        # a directory holding a dissertation number. Appended last so no
        # existing directory name changes.
        init_from = getattr(args, "init_from", None)
        if init_from:
            m = re.search(r"stage0-([a-z0-9]+)", os.path.basename(
                os.path.normpath(init_from)))
            name += f"-warm{m.group(1) if m else 'x'}"
        # Control C2 changes the objective itself — a different arm. Appended
        # last, and only when on, so no existing directory name changes.
        dr_grpo = getattr(args, "dr_grpo", "off")
        if dr_grpo != "off":
            name += f"-dr{dr_grpo}"
        # Second task. Appended last and only when != gsm8k, so every existing
        # GSM8K directory name is untouched. A MATH run must never resolve to a
        # GSM8K directory: it would skip against a completed GSM8K checkpoint,
        # or train into one holding a reported number.
        dataset = getattr(args, "dataset", "gsm8k")
        if dataset != "gsm8k":
            name += f"-ds{dataset}"

        # D2 programme axes: rollout temperature, learning rate, exploration noise.
        # Every pre-existing arm used temperature 0.5, lr 5e-6 and no noise, so the
        # tokens are only appended when an axis moves. A run on any moved axis is
        # additionally prefixed "new-": EXPERIMENT_NAME_RE is anchored on the model
        # size, so the legacy pipeline (eval_all / analyse_replicates /
        # generate_report) never sees these directories. That is deliberate. Their
        # parse_suffix would ignore the unknown tokens and pool a noisy run with
        # its own Dr. GRPO control as extra "seeds" of one arm. eval_d2.py and
        # analyse_d2.py own these directories.
        new_tokens = []
        temperature = getattr(args, "temperature", 0.5)
        if temperature != 0.5:
            new_tokens.append(f"-t{temperature:g}")
        lr = getattr(args, "lr", 5e-6)
        if lr != 5e-6:
            new_tokens.append(f"-lr{lr:g}")
        sigma_start = getattr(args, "noise_sigma_start", 0.0)
        if sigma_start and sigma_start > 0:
            sigma_end = getattr(args, "noise_sigma_end", None)
            if sigma_end is None:
                sigma_end = sigma_start
            tag = f"-noise{sigma_start:g}to{sigma_end:g}"
            if getattr(args, "noise_mode", "add") != "add":
                tag += args.noise_mode
            new_tokens.append(tag)
        latent_temperature = getattr(args, "latent_temperature", None)
        if latent_temperature is not None:
            new_tokens.append(f"-lt{latent_temperature:g}")
        stochastic_latent = getattr(args, "stochastic_latent", None)
        if stochastic_latent is not None:
            new_tokens.append(f"-sl{stochastic_latent:g}")
        if getattr(args, "blend_gate", "off") != "off":
            new_tokens.append(f"-dg{getattr(args, 'blend_max', 0.5):g}"
                              f"d{args.blend_d0:g}to{args.blend_d1:g}")
        length_alpha = getattr(args, "length_penalty_alpha", None)
        if length_alpha is not None:
            new_tokens.append(f"-lp{length_alpha:g}")
        if new_tokens:
            name = "new-" + name + "".join(new_tokens)
    return f"./experiments/{name}"


# experiments/{size}-{base_variant}[-suffix], where suffix marks a follow-up run
# (-seed43, -rp1, -loraattention, -warmon, -drboth, -dsmath, or a combination).
EXPERIMENT_NAME_RE = re.compile(
    r"^(?P<size>0\.5b|1\.5b|3b)-(?P<variant>v\d_[a-z_]+?)(?P<suffix>-(?:seed|rp|lora|warm|dr|ds).*)?$"
)


def parse_experiment_name(name: str):
    """('1.5b-v3_learned_lambda-seed43') -> ('1.5b', 'v3_learned_lambda',
    '-seed43'), or None if the directory is not an ablation experiment.

    Analysis code must key gate mode / chain position off the BASE variant
    while still reporting the suffix, otherwise follow-up runs are either
    ignored entirely or mis-evaluated with the wrong gate configuration.
    """
    m = EXPERIMENT_NAME_RE.match(name)
    if not m:
        return None
    return m.group("size"), m.group("variant"), (m.group("suffix") or "")


def experiment_is_complete(exp_dir: str) -> bool:
    """A run is complete iff some checkpoint's trainer_state.json reports
    global_step == max_steps. (Stricter than the exists+non-empty guard in
    hrpo_gsm8k.py, which would treat a crashed run as complete.)"""
    for state_path in glob.glob(os.path.join(exp_dir, "checkpoint-*", "trainer_state.json")):
        try:
            with open(state_path) as f:
                state = json.load(f)
            if state.get("max_steps") and state.get("global_step") == state["max_steps"]:
                return True
        except (json.JSONDecodeError, OSError):
            continue
    return False


def exit_if_complete(exp_dir: str):
    if experiment_is_complete(exp_dir):
        print(f"Experiment {exp_dir} already has a completed checkpoint. Exiting...")
        raise SystemExit(0)


def latest_checkpoint(exp_dir: str):
    """Highest-step checkpoint in exp_dir, or None. Used to resume a run that
    was cut short (e.g. killed at the SLURM wall-time limit) instead of
    restarting it from scratch."""
    candidates = []
    for path in glob.glob(os.path.join(exp_dir, "checkpoint-*")):
        m = re.match(r"checkpoint-(\d+)$", os.path.basename(path))
        if m and os.path.exists(os.path.join(path, "trainer_state.json")):
            candidates.append((int(m.group(1)), path))
    return max(candidates)[1] if candidates else None


# Hyperparameters that must not change across a resume: a checkpoint trained
# with different values cannot be continued without silently mixing configs.
_CONFIG_FINGERPRINT = [
    "model_size", "lora_rank", "lr", "beta", "seed",
    "residual_r_min", "residual_r_max", "group_size", "temperature",
    "gradient_accumulation_steps", "per_device_train_batch_size",
    "max_prompt_length", "max_completion_length", "repetition_penalty",
    "lora_targets", "latent_source", "constant_init", "init_from",
    "dr_grpo", "dataset",
    "noise_sigma_start", "noise_sigma_end", "noise_stages", "noise_mode",
    "latent_temperature", "stochastic_latent",
    "blend_gate", "blend_max", "blend_d0", "blend_d1",
    "length_penalty_alpha",
]
# Value to assume when a run_config.json predates a fingerprint key. Without it
# every checkpoint written before C2 would refuse to resume ("dr_grpo:
# checkpoint=None now='off'") although it trained with exactly that setting.
# The noise keys follow the same rule: every earlier run trained without noise.
_FINGERPRINT_DEFAULTS = {"dr_grpo": "off", "dataset": "gsm8k",
                         "noise_sigma_start": 0.0, "noise_sigma_end": None,
                         "noise_stages": 10, "noise_mode": "add",
                         "latent_temperature": None, "stochastic_latent": None,
                         "blend_gate": "off", "blend_max": 0.5, "blend_d0": None, "blend_d1": None,
                         "length_penalty_alpha": None}

_DR_GRPO_MODES = ("off", "len", "std", "both")


def apply_dr_grpo(trainer, mode: str):
    """Switch on the Dr. GRPO de-bias (control C2; Liu et al. 2025) for a trainer.

    The switches are trainer attributes, read with getattr(..., default) by the
    vendored loss (unsloth/models/rl_replacements.py) and advantage code
    (trl/trainer/grpo_trainer.py). "off" sets NOTHING, so the trainer runs the
    exact pre-C2 code path. They ride on the trainer rather than GRPOConfig
    because Unsloth regenerates its config class from source.

    Both switches also shrink the update at an unchanged learning rate, not just
    de-bias it: dividing by max_completion_length (512) instead of the ~250-token
    mean length roughly halves the loss scale, and dropping /std (group std is at
    most ~0.58 for binary rewards in a group of 4) shrinks every advantage. That is
    inherent to Dr. GRPO as published and must be stated next to any C2 result."""
    if mode not in _DR_GRPO_MODES:
        raise ValueError(f"dr_grpo must be one of {_DR_GRPO_MODES}, got {mode!r}")
    if mode == "off":
        return
    if not hasattr(trainer, "num_generations"):
        raise RuntimeError(
            "apply_dr_grpo needs a GRPOTrainer (no num_generations on this "
            "trainer) — the de-bias would silently not apply")
    if mode in ("len", "both"):
        trainer.dr_grpo_len_constant = float(trainer.args.max_completion_length)
    if mode in ("std", "both"):
        trainer.dr_grpo_scale_rewards = False
    print(f"[dr_grpo] mode={mode}: loss normaliser="
          f"{getattr(trainer, 'dr_grpo_len_constant', 'per-response length')}, "
          f"advantage /std={getattr(trainer, 'dr_grpo_scale_rewards', True)}")


def apply_exploration_noise(trainer, args):
    """D2: switch on QeRL-style parameter noise (noise_ext.py) for a trainer.

    OFF (noise_sigma_start == 0, the default) touches nothing, so every
    pre-existing arm runs the exact code path it always did. Imported lazily so
    a run without noise never needs the module."""
    sigma_start = getattr(args, "noise_sigma_start", 0.0) or 0.0
    if sigma_start <= 0:
        return
    if not hasattr(trainer, "training_step"):
        raise RuntimeError("apply_exploration_noise needs a Trainer with training_step")
    from noise_ext import patch_trainer_noise
    sigma_end = getattr(args, "noise_sigma_end", None)
    if sigma_end is None:
        sigma_end = sigma_start
    patch_trainer_noise(
        trainer, sigma_start=sigma_start, sigma_end=sigma_end,
        stages=getattr(args, "noise_stages", 10),
        mode=getattr(args, "noise_mode", "add"), seed=getattr(args, "seed", 42))
    print(f"[noise] RMSNorm weight noise on: sigma {sigma_start:g} -> {sigma_end:g} "
          f"over {getattr(args, 'noise_stages', 10)} stages, mode="
          f"{getattr(args, 'noise_mode', 'add')}")


def set_latent_temperature(model, latent_temperature):
    """Set the Eq.3 mixture temperature on the module whose _sample loop builds the
    mixture: the causal-LM object that carries `answer_start` (set by every variant
    script and by the evaluation loader before any PEFT wrapping). Returns the number
    of modules set.

    None sets nothing, so the rollout keeps mixing over the sampling distribution.
    A latent temperature on a model whose answer_start is "" (v1, or latent-off
    evaluation) would be accepted and do nothing, so that combination is refused."""
    if latent_temperature is None:
        return 0
    if latent_temperature <= 0:
        raise ValueError(f"latent_temperature must be > 0, got {latent_temperature}")
    targets = [m for m in model.modules() if "answer_start" in vars(m)]
    if not targets:
        raise RuntimeError("set_latent_temperature: no module carries answer_start; the "
                           "latent temperature would not reach the rollout")
    if all(m.answer_start == "" for m in targets):
        raise RuntimeError("set_latent_temperature: answer_start is empty (v1 or latent-off), "
                           "so there is no latent mixture for the temperature to act on")
    for m in targets:
        m.latent_temperature = float(latent_temperature)
    return len(targets)


def divergence_gate_weights(embeds, residual, blend_max, d0, d1):
    """D3 blend weight per position: beta = blend_max * clamp((d - d0) / (d1 - d0), 0, 1) with
    d = 1 - cos(latent, token embedding), computed in fp32. No parameter receives gradient through
    it (the embedding table is frozen and the latent is a stored constant in the loss forward).
    Returns beta with a trailing singleton dimension, broadcastable over the hidden size."""
    cos = torch.nn.functional.cosine_similarity(embeds.float(), residual.float(), dim=-1)
    d = 1.0 - cos
    beta = blend_max * torch.clamp((d - d0) / (d1 - d0), 0.0, 1.0)
    return beta.unsqueeze(-1)


def _divergence_gate_thinking_residual(self, embeds, residual, eps=1e-8):
    """D3 blend, replacing Eq.4 at BOTH call sites (rollout unsloth/models/llama.py:944 and the
    loss forward llama.py:666): input = (1 - beta) * e + beta * h. Returns (input, a_t) with
    a_t = 1 - beta, the token's weight, so the existing embeds_ratio logging records the blend
    (positions with beta = 0 report 1, exactly like non-latent positions)."""
    beta = divergence_gate_weights(embeds, residual, self.blend_max, self.blend_d0, self.blend_d1)
    beta = beta.to(embeds.dtype)
    a_t = (1 - beta).expand_as(embeds)
    return (1 - beta) * embeds + beta * residual.to(embeds.dtype), a_t


def set_blend_gate(model, blend_gate="off", blend_max=0.5, blend_d0=None, blend_d1=None):
    """Install the D3 blend on the module that owns thinking_residual. "off" does nothing.
    Must run AFTER any gate-mode patch (the v3 unit-gate patch also overrides
    thinking_residual) and at evaluation as well as training. Returns True when installed."""
    if blend_gate in (None, "off"):
        return False
    if blend_gate != "divergence":
        raise ValueError(f"unknown blend_gate {blend_gate!r}")
    if blend_d0 is None or blend_d1 is None or not (0 <= blend_d0 < blend_d1):
        raise ValueError(f"blend_gate needs 0 <= blend_d0 < blend_d1, got {blend_d0}, {blend_d1}")
    if not (0 < blend_max <= 1):
        raise ValueError(f"blend_max must be in (0, 1], got {blend_max}")
    holders = [m for m in model.modules() if "answer_start" in vars(m)]
    if holders and all(m.answer_start == "" for m in holders):
        raise RuntimeError("set_blend_gate: answer_start is empty (v1 or latent-off), so no "
                           "latent is blended and the gate would do nothing")
    inner = get_inner_model(model)
    inner.blend_max, inner.blend_d0, inner.blend_d1 = float(blend_max), float(blend_d0), float(blend_d1)
    inner.thinking_residual = types.MethodType(_divergence_gate_thinking_residual, inner)
    return True


def apply_blend_gate(trainer, args):
    """D3 for a trainer. Off (default) touches nothing. Also flags the trainer so the loss logs
    the share of latent positions that were blended (unsloth/models/rl_replacements.py)."""
    if getattr(args, "blend_gate", "off") == "off":
        return
    set_blend_gate(trainer.model, args.blend_gate, args.blend_max, args.blend_d0, args.blend_d1)
    trainer.blend_gate_on = True
    print(f"[blend_gate] D3 divergence-gated blend: beta = {args.blend_max:g} * clamp((d - "
          f"{args.blend_d0:g}) / ({args.blend_d1:g} - {args.blend_d0:g}), 0, 1), "
          f"d = 1 - cos(latent, token); replaces the Eq.4 blend at rollout and in the loss")


def set_stochastic_latent(model, gumbel_temperature):
    """Set the Gumbel temperature of the C1 stochastic latent on the module that runs _sample
    (the one carrying answer_start). None sets nothing. Refused where there is no latent (v1,
    latent-off). Returns the number of modules set."""
    if gumbel_temperature is None:
        return 0
    if gumbel_temperature <= 0:
        raise ValueError(f"stochastic_latent must be > 0, got {gumbel_temperature}")
    targets = [m for m in model.modules() if "answer_start" in vars(m)]
    if not targets:
        raise RuntimeError("set_stochastic_latent: no module carries answer_start; the setting "
                           "would not reach the rollout")
    if all(m.answer_start == "" for m in targets):
        raise RuntimeError("set_stochastic_latent: answer_start is empty (v1 or latent-off), so "
                           "there is no latent to make stochastic")
    for m in targets:
        m.stochastic_latent = float(gumbel_temperature)
    return len(targets)


def apply_stochastic_latent(trainer, args):
    """C1 for a trainer. Off (None) touches nothing. Flags the trainer so GRPOTrainer collects
    the rollout's soft labels and the loss logs how soft they were."""
    tau = getattr(args, "stochastic_latent", None)
    if tau is None:
        return
    if getattr(args, "latent_temperature", None) is not None:
        raise SystemExit("--stochastic_latent and --latent_temperature both form the latent; "
                         "use one of them")
    n = set_stochastic_latent(trainer.model, tau)
    trainer.stochastic_latent_on = True
    print(f"[stochastic_latent] Gumbel latent at temperature {tau:g} with the soft-label "
          f"policy-gradient term at latent positions; set on {n} module(s)")


def apply_length_penalty(trainer, args):
    """Length-penalised reward (length_penalty.py) for a trainer. Off (None) touches nothing.
    GRPOTrainer._prepare_inputs reads trainer.length_penalty_alpha; the penalty itself refuses
    any reward that is not 0/1, so it can never silently reshape v5/v6's shaped rewards."""
    alpha = getattr(args, "length_penalty_alpha", None)
    if alpha is None:
        return
    if not 0.0 < alpha < 1.0:
        raise SystemExit(f"--length_penalty_alpha must be in (0, 1), got {alpha}")
    if not hasattr(trainer, "num_generations"):
        raise RuntimeError("apply_length_penalty needs a GRPOTrainer (no num_generations)")
    trainer.length_penalty_alpha = float(alpha)
    print(f"[length_penalty] correct responses earn 1 - {alpha:g} * sigmoid(normalised length within "
          f"the group's correct responses); advantage /std="
          f"{getattr(trainer, 'dr_grpo_scale_rewards', True)}")


def apply_latent_temperature(trainer, args):
    """Decoupled latent temperature for a trainer's rollouts. Off (None) touches nothing."""
    latent_temperature = getattr(args, "latent_temperature", None)
    if latent_temperature is None:
        return
    n = set_latent_temperature(trainer.model, latent_temperature)
    print(f"[latent_temperature] Eq.3 mixture at temperature {latent_temperature:g} "
          f"(sampling stays at {getattr(args, 'temperature', 0.5):g}); set on {n} module(s)")


def training_incomplete(exp_dir: str):
    """True only when we can POSITIVELY show training stopped early.

    eval_all evaluates whichever checkpoint is newest, regardless of whether
    training finished — so a cancelled run enters the results looking like a
    finished arm. The MATH seed-43 runs (stopped at 500/937) scored HIGHER than
    every completed run, because they were snapshotted before the collapse.
    Absent or unreadable trainer_state means "unknown", treated as complete so
    a settled arm is never silently dropped."""
    best = None
    for d in glob.glob(os.path.join(exp_dir, "checkpoint-*")):
        m = re.match(r"checkpoint-(\d+)$", os.path.basename(d))
        f = os.path.join(d, "trainer_state.json")
        if m and os.path.exists(f) and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), f)
    if not best:
        return False
    try:
        with open(best[1]) as fh:
            st = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return False
    mx, step = st.get("max_steps"), st.get("global_step")
    return bool(mx and step and step < mx)


def train_with_resume(trainer, exp_dir: str, args):
    """trainer.train(), resuming from the newest checkpoint if one exists.
    Checkpoints are written every save_steps=250, so a job killed at the SLURM
    wall-time limit loses at most 250 steps when resubmitted.

    Refuses to resume a checkpoint whose run_config.json disagrees with the
    current hyperparameters — e.g. a partial run from before the switch to the
    HRPO paper's per_device 8 / 512-token config. Resuming across a config
    change would mix hyperparameters mid-run and invalidate the variant."""
    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, "run_config.json")
    current = {k: getattr(args, k) for k in _CONFIG_FINGERPRINT if hasattr(args, k)}

    resume = latest_checkpoint(exp_dir)
    if resume:
        stored = None
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    stored = json.load(f)
            except (json.JSONDecodeError, OSError):
                stored = None
        diffs = []
        if stored is None:
            diffs = ["run_config.json missing (checkpoint predates this guard)"]
        else:
            for key, value in current.items():
                stored_value = stored.get(key, _FINGERPRINT_DEFAULTS.get(key))
                if stored_value != value:
                    diffs.append(f"{key}: checkpoint={stored_value!r} now={value!r}")
        if diffs:
            raise SystemExit(
                f"\nREFUSING TO RESUME {resume}\n"
                f"Its hyperparameters differ from the current run:\n  "
                + "\n  ".join(diffs)
                + f"\n\nResuming would mix configs mid-run. Delete the stale "
                  f"experiment directory and start fresh:\n"
                  f"  rm -rf {exp_dir}\n")
        print(f"Resuming training from {resume} (config matches)")
    else:
        print(f"No checkpoint in {exp_dir}; starting from scratch")

    with open(config_path, "w") as f:
        json.dump(current, f, indent=2)

    # Control C2. Applied here — the one call every variant script shares — so
    # the flag cannot be accepted, suffix the directory, and then silently not
    # reach the trainer. "off" sets nothing (the original objective).
    apply_dr_grpo(trainer, getattr(args, "dr_grpo", "off"))
    # D2 exploration noise, same reason: the one call every variant shares.
    apply_exploration_noise(trainer, args)
    # Decoupled latent temperature, same reason.
    apply_latent_temperature(trainer, args)
    # D3 uncertainty-gated blend, same reason (after the variant script's own gate patch).
    apply_blend_gate(trainer, args)
    # C1 stochastic latent with its own gradient, same reason.
    apply_stochastic_latent(trainer, args)
    # Length penalty, same reason.
    apply_length_penalty(trainer, args)

    trainer.train(resume_from_checkpoint=resume)


# ---------------------------------------------------------------------------
# Gate configuration helpers
# ---------------------------------------------------------------------------

def get_inner_model(model):
    """Resolve the innermost Qwen2Model that owns the thinking_residual_*
    modules, through however many wrappers (PeftModelForCausalLM ->
    Qwen2ForCausalLM -> Qwen2Model) are present."""
    obj = model
    for _ in range(6):
        if hasattr(obj, "thinking_residual_Lambda"):
            # Guard against PeftModel.__getattr__ forwarding: only accept the
            # module that has it in _modules (i.e. actually owns it).
            if "thinking_residual_Lambda" in getattr(obj, "_modules", {}):
                return obj
        if hasattr(obj, "model") and obj.model is not obj:
            obj = obj.model
        else:
            break
    raise AttributeError("Could not locate the module owning thinking_residual_Lambda")


def _unit_gate_thinking_residual(self, embeds, residual, eps=1e-8):
    """Equation 4 with r_t = 1 and i_t = 1 (v2/v3): the input-dependent gates
    are bypassed; only Lambda controls the blend.
    a_t = exp(-c * softplus(-Lambda) * 1);  out = a_t*e + sqrt(1-a_t^2+eps)*h.
    Identical maths to Qwen2Model.thinking_residual with unit gates."""
    ones = torch.ones_like(embeds)
    a_t = self.thinking_residual_Lambda(ones)
    return a_t * embeds + torch.sqrt(1 - a_t.pow(2) + eps) * residual, a_t


def apply_unit_gate_patch(model):
    """Instance-level override so gate_r/gate_i never execute (v2/v3). Must be
    applied at BOTH training and evaluation time (the random gate weights are
    not saved in v3 adapters). Returns the patched inner model."""
    inner = get_inner_model(model)
    inner.thinking_residual = types.MethodType(_unit_gate_thinking_residual, inner)
    return inner


def _constant_latent_thinking_residual(self, embeds, residual, eps=1e-8):
    """C1: Equation 4 with unit gates (the v3 configuration) AND the Equation 3
    projection output replaced by a single learned constant vector u.

    The incoming `residual` (the projection output) is deliberately IGNORED.
    Patching at this level covers BOTH call sites in one place — the generation
    loop (unsloth/models/llama.py:944) and the log-prob forward (llama.py:666) —
    and, critically, places u in the LIVE graph of the loss forward so it
    actually receives gradient. Substituting inside the generation loop instead
    would not: generate() runs under no_grad and the recorded thinking_embeds are
    detached constants by the time the loss is computed, so u would train on a
    zero gradient forever."""
    ones = torch.ones_like(embeds)
    a_t = self.thinking_residual_Lambda(ones)
    u = self.thinking_residual_Lambda.u.to(embeds.dtype).expand_as(embeds)
    return a_t * embeds + torch.sqrt(1 - a_t.pow(2) + eps) * u, a_t


def constant_init_stats(embed_weight):
    """Norms that decide whether a given u init is scale-matched to what it
    replaces. The Eq.3 output is W_e^T p / ||p||; for a peaked p that is close to
    a single embedding row, so `mean_row_norm` is the scale u should sit near.
    `mean_norm` is the centroid's norm — typically far smaller, because the rows
    cancel. Returned (not just printed) so tests and logs can assert on it."""
    e = embed_weight.detach().float()
    mean_norm = float(e.mean(dim=0).norm())
    mean_row_norm = float(e.norm(dim=1).mean())
    return {
        "mean_norm": mean_norm,
        "mean_row_norm": mean_row_norm,
        "ratio_mean_to_row": (mean_norm / mean_row_norm) if mean_row_norm else float("nan"),
    }


def register_constant_latent_vector(model, init="mean", seed=0):
    """Create u on the BASE model, BEFORE get_peft_model. Returns (u, stats).

    Ordering matters: modules_to_save=["thinking_residual_Lambda"] makes PEFT
    deep-copy that module at get_peft_model time. Registering u afterwards would
    attach it to the ModulesToSaveWrapper rather than the saved copy, and u would
    never reach the adapter. Registering it here means the copy already contains
    it, so u is written into the adapter alongside Lambda.

    u lives on thinking_residual_Lambda for two properties that then come free:
    its parameter name contains "thinking_residual_Lambda", so patch.py puts it
    in Lambda's optimizer group (lr 1e-3); and that module is already in
    GATE_MODULES. It is registered here rather than in
    ThinkingResidualLambda.__init__ so v1-v6 keep byte-identical state dicts and
    every existing checkpoint keeps loading unchanged.

    init: "mean" (the brief's spec — embedding centroid), "mean_scaled" (centroid
    direction rescaled to the mean row norm) or "random_row" (an actual embedding
    row: on-manifold in direction AND scale). The norm diagnostics are printed so
    a scale mismatch is visible before any GPU time is spent."""
    lam = get_inner_model(model).thinking_residual_Lambda
    embed = model.get_input_embeddings().weight
    stats = constant_init_stats(embed)

    if not hasattr(lam, "u"):
        e = embed.detach()
        if init == "mean":
            vec = e.mean(dim=0)
        elif init == "mean_scaled":
            m = e.mean(dim=0)
            vec = m / m.norm().clamp_min(1e-12) * stats["mean_row_norm"]
        elif init == "random_row":
            g = torch.Generator(device="cpu").manual_seed(seed)
            vec = e[int(torch.randint(e.shape[0], (1,), generator=g))]
        else:
            raise ValueError(f"unknown constant_init: {init}")
        lam.register_parameter("u", torch.nn.Parameter(
            vec.clone().to(dtype=lam.Lambda.dtype, device=lam.Lambda.device)))

    stats["init"] = init
    stats["u_norm"] = float(lam.u.detach().float().norm())
    stats["u_to_row_ratio"] = (stats["u_norm"] / stats["mean_row_norm"]
                               if stats["mean_row_norm"] else float("nan"))
    print(f"[C1] constant latent u init={init}: ||u||={stats['u_norm']:.4f}, "
          f"mean||e_i||={stats['mean_row_norm']:.4f} "
          f"(ratio {stats['u_to_row_ratio']:.4f}); embedding centroid norm "
          f"={stats['mean_norm']:.4f}")
    if stats["u_to_row_ratio"] < 0.1:
        print("[C1] WARNING: u starts >10x smaller than a typical embedding row — "
              "the latent contributes almost nothing at init and must climb at "
              "lr_residual_Lambda. Consider --constant_init mean_scaled/random_row.")
    return lam.u, stats


def apply_constant_latent_patch(model):
    """Install the C1 thinking_residual override (AFTER get_peft_model, mirroring
    apply_unit_gate_patch). Must be applied at BOTH training and evaluation time.
    register_constant_latent_vector() must have been called first."""
    inner = get_inner_model(model)
    if not hasattr(inner.thinking_residual_Lambda, "u"):
        raise RuntimeError(
            "constant latent vector u is not registered — call "
            "register_constant_latent_vector(model) BEFORE get_peft_model()")
    inner.thinking_residual = types.MethodType(
        _constant_latent_thinking_residual, inner)
    return inner


def disable_gate(model):
    """v1 (pure GRPO): with answer_start == "" every decoded string 'contains'
    it, so is_thinking is always False -> the Eq.4 blend is skipped for every
    token in both rollouts (generation/utils.py:3368) and the loss forward
    (unsloth/models/llama.py:663-669). embeds_ratio stays 1 everywhere."""
    model.answer_start = ""


def load_stage0_init(model, path, dataset="gsm8k"):
    """Initialise a PEFT model's LoRA adapter + Lambda from a Stage-0 warm start
    (stage0_warmstart.py output). Call AFTER get_peft_model and AFTER
    reset_lambda_parameters — this overwrites Lambda with the warm-started value,
    which is the entire point.

    Raises on unexpected keys rather than warning: a silent key mismatch would
    leave the run cold-started while every log claimed it was warm-started, and
    the resulting arm would be quietly incomparable. Missing keys are reported
    but tolerated (a Stage-0 run with --blend off legitimately has no gate
    entries beyond Lambda)."""
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict

    src = os.path.join(path, "adapter_model.safetensors")
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"--init_from {path}: no adapter_model.safetensors "
            f"(is it a finished stage0_warmstart.py output?)")

    # The -warm{on,off} directory suffix records THAT a warm start was used, not
    # which task it was trained on — so a GSM8K warm start fed to a MATH run
    # would land in the same directory as a genuine MATH warm start and be
    # reported as one. Refuse the mismatch. Summaries written before --dataset
    # existed carry no key; those warm starts are all GSM8K.
    s0_dataset = "gsm8k"
    summary = os.path.join(path, "stage0_summary.json")
    if os.path.exists(summary):
        with open(summary) as f:
            s0_dataset = (json.load(f).get("args") or {}).get("dataset", "gsm8k")
    if s0_dataset != dataset:
        raise RuntimeError(
            f"--init_from {path}: this warm start was trained on {s0_dataset}, but "
            f"the run is --dataset {dataset}. Use a Stage-0 run of the same task "
            f"(DATASET={dataset} sbatch cluster/run_stage0.slurm).")

    before = get_inner_model(model).thinking_residual_Lambda.Lambda.detach().clone()
    sd = load_file(src)
    # Whether this warm start carries Lambda at all decides how the checks below
    # are read: a --blend off (plain LoRA SFT) run legitimately has none.
    has_lambda = any("thinking_residual_Lambda" in k for k in sd)
    result = set_peft_model_state_dict(model, sd)

    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    # "missing" is the frozen BASE model, which an adapter never contains — it is
    # the same count whether or not the file carries Lambda. Reported, not acted
    # on. Unexpected keys are the real failure and raise.
    missing = list(getattr(result, "missing_keys", []) or [])
    if unexpected:
        raise RuntimeError(
            f"--init_from {path}: {len(unexpected)} unexpected key(s), e.g. "
            f"{unexpected[:3]} — the warm start does not match this model/config")

    after = get_inner_model(model).thinking_residual_Lambda.Lambda.detach()
    moved = float((after - before).abs().max())
    lam_mean = float(after.detach().float().mean())
    if has_lambda:
        lam_desc = f"max |ΔLambda| = {moved:.6g}; resulting lambda_mean = {lam_mean!r}"
    else:
        # v1 never calls reset_lambda_parameters, so its Lambda holds whatever
        # the fast init left there — often NaN. Inert, because disable_gate makes
        # is_thinking always False and the blend never runs. Say so rather than
        # printing a bare nan that reads like a failure.
        lam_desc = ("no Lambda in this warm start (expected for --blend off; "
                    "the receiving variant's Lambda is unused)")
    print(f"[init_from] loaded {len(sd)} tensors from {path}; "
          f"{len(missing)} missing base-model key(s); {lam_desc}")

    # Only meaningful when the file actually carries Lambda. NaN must be caught
    # explicitly: `nan == 0.0` is False, so a Lambda that never arrived would
    # otherwise slip through this guard silently.
    if has_lambda and not (moved > 0.0):
        raise RuntimeError(
            f"--init_from {path}: the warm start carries Lambda but the model's "
            f"Lambda did not change (max |delta| = {moved!r}) — it was not "
            f"applied, so this run would train from a COLD Lambda while every "
            f"log claimed it was warm-started. Check modules_to_save in both runs.")
    return {"n_tensors": len(sd), "missing": len(missing), "lambda_delta": moved,
            "has_lambda": has_lambda}


def freeze_gate_modules(model, module_names=GATE_MODULES):
    """v2: freeze the gate modules (adapter copies included) after
    get_peft_model/reset_lambda_parameters have initialised them.

    NOTE: peft's ModulesToSaveWrapper.enable_adapters — triggered on every
    exit of the `disable_adapter()` context the GRPO trainer uses for the
    reference log-probs — re-sets requires_grad=True on the adapter copies
    during training. The freeze is nevertheless effective because the
    optimizer param groups are built (in patch.py / create_optimizer) while
    requires_grad is False, so the gate parameters are never registered with
    the optimizer and can never be updated. assert_optimizer_gate_params()
    verifies this, and the smoke test additionally checks the parameter
    values are bit-identical after training."""
    inner = get_inner_model(model)
    frozen = []
    for name in module_names:
        module = getattr(inner, name)
        module.requires_grad_(False)
        frozen.append(name)
    return frozen


def assert_optimizer_gate_params(trainer, expect_lambda: bool, expect_gates: bool):
    """Post-construction sanity check on optimizer param groups."""
    trainer.create_optimizer()
    named = {id(p): n for n, p in trainer.model.named_parameters()}
    lambda_params, gate_params = [], []
    for group in trainer.optimizer.param_groups:
        for p in group["params"]:
            n = named.get(id(p), "?")
            if "thinking_residual_Lambda" in n:
                lambda_params.append(n)
            elif "thinking_residual_gate" in n:
                gate_params.append(n)
    assert bool(lambda_params) == expect_lambda, \
        f"Lambda in optimizer: {lambda_params} (expected {expect_lambda})"
    assert bool(gate_params) == expect_gates, \
        f"gates in optimizer: {gate_params} (expected {expect_gates})"
    print(f"Optimizer check OK — Lambda params: {len(lambda_params)}, "
          f"gate params: {len(gate_params)}")


def set_rollout_repetition_penalty(trainer, repetition_penalty: float):
    """GRPOConfig has no repetition_penalty field (see changes.md mapping #8);
    set it on the trainer's GenerationConfig used for rollouts."""
    if getattr(trainer, "generation_config", None) is not None:
        trainer.generation_config.repetition_penalty = repetition_penalty
        print(f"Rollout repetition_penalty set to {repetition_penalty}")
    elif repetition_penalty != 1.0:
        raise RuntimeError("Trainer has no generation_config (vLLM path?); "
                           "cannot apply repetition_penalty")


# ---------------------------------------------------------------------------
# Gradient-flow logging (Task 4: |grad Lambda| vs |grad gate_r/gate_i|)
# ---------------------------------------------------------------------------

class GateGradientLogger(TrainerCallback):
    """Appends per-optimizer-step gradient L2 norms of the gate parameters to
    <exp_dir>/gradient_log.jsonl (attached for v3-v6; v1/v2 have no trainable
    gate parameters, so there is nothing to log).

    Fires on_pre_optimizer_step, which HF Trainer calls AFTER gradient
    accumulation and AFTER clip_grad_norm_ (transformers/trainer.py:2589 vs
    :2605) — so the recorded norms are POST-clipping. Clipping rescales every
    gradient by one common factor, so Lambda-vs-gate ratios (the quantity the
    Task 4 gradient-flow analysis asks about) are exact; absolute magnitudes
    are scaled down whenever clipping is active (visible as the LoRA norm
    pinning at max_grad_norm). analyse_gate.gradient_flow() reconstructs the
    pre-clip magnitudes using the pre-clip total norm that the Trainer records
    as `grad_norm` in trainer_state.json."""

    def __init__(self, model, exp_dir: str, every: int = 1):
        self.exp_dir = exp_dir
        self.every = every
        self.path = os.path.join(exp_dir, "gradient_log.jsonl")
        self._params = {"Lambda": [], "u": [], "gate_r": [], "gate_i": [], "lora": []}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # The C1 constant vector lives INSIDE thinking_residual_Lambda, so it
            # matches the Lambda substring below. Bucket it separately first, or
            # v7's gradient_log.jsonl would report Lambda+u pooled and the
            # gradient-flow diagnostic would be contaminated. Empty (logged null)
            # for v1-v6, which have no u — exactly as gate_r/gate_i are for v3.
            if "thinking_residual_Lambda" in name and name.endswith(".u"):
                self._params["u"].append(p)
            elif "thinking_residual_Lambda" in name:
                self._params["Lambda"].append(p)
            elif "thinking_residual_gate_r" in name:
                self._params["gate_r"].append(p)
            elif "thinking_residual_gate_i" in name:
                self._params["gate_i"].append(p)
            elif "lora_" in name:
                self._params["lora"].append(p)

    @staticmethod
    def _grad_norm(params):
        sq = 0.0
        seen = False
        for p in params:
            if p.grad is not None:
                sq += float(p.grad.detach().float().pow(2).sum().item())
                seen = True
        return (sq ** 0.5) if seen else None

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if state.global_step % self.every != 0:
            return
        record = {"step": state.global_step}
        for key, params in self._params.items():
            record[key] = self._grad_norm(params)
        os.makedirs(self.exp_dir, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


def merge_training_log(exp_dir: str, model_size: str, variant_id: str,
                       start_time: str, end_time: str, status: str,
                       log_path: str = "training_log.json"):
    """Merge one run's outcome (plus its train_summary.json, if present) into
    the shared training_log.json used by run_ablations.sh. Guarded by an
    exclusive flock so concurrent SLURM jobs finishing together cannot
    clobber each other's entries."""
    import fcntl
    lock_path = log_path + ".lock"
    lock_file = open(lock_path, "w")
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    try:
        _merge_training_log_locked(exp_dir, model_size, variant_id,
                                   start_time, end_time, status, log_path)
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def _merge_training_log_locked(exp_dir, model_size, variant_id,
                               start_time, end_time, status, log_path):
    entry = {
        "model_size": model_size,
        "variant_id": variant_id,
        "start_time": start_time,
        "end_time": end_time,
        "status": status,
    }
    summary_path = os.path.join(exp_dir, "train_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            for key in ("wall_clock_seconds", "peak_gpu_memory_allocated_gb",
                        "peak_gpu_memory_reserved_gb", "global_step"):
                if key in summary:
                    entry[key] = summary[key]
        except (json.JSONDecodeError, OSError):
            pass
    data = {}
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    data[f"{model_size}-{variant_id}"] = entry
    with open(log_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"training_log.json updated: {model_size}-{variant_id} -> {status}")


# ---------------------------------------------------------------------------
# Run bookkeeping (start/end time, peak GPU memory) for run_ablations.sh
# ---------------------------------------------------------------------------

class RunLogger:
    def __init__(self, exp_dir: str, variant_id: str, model_size: str, args):
        self.exp_dir = exp_dir
        self.summary = {
            "variant_id": variant_id,
            "model_size": model_size,
            "args": {k: v for k, v in vars(args).items()},
            "start_time": datetime.datetime.now().isoformat(),
        }
        self._t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def finish(self, trainer=None):
        self.summary["end_time"] = datetime.datetime.now().isoformat()
        self.summary["wall_clock_seconds"] = round(time.time() - self._t0, 1)
        if torch.cuda.is_available():
            self.summary["peak_gpu_memory_allocated_gb"] = round(
                torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
            self.summary["peak_gpu_memory_reserved_gb"] = round(
                torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
        if trainer is not None:
            self.summary["global_step"] = int(trainer.state.global_step)
        os.makedirs(self.exp_dir, exist_ok=True)
        path = os.path.join(self.exp_dir, "train_summary.json")
        with open(path, "w") as f:
            json.dump(self.summary, f, indent=2)
        print(f"Run summary written to {path}")
        return self.summary
