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

# Gate configuration each variant was trained with (changes.md mapping #7);
# evaluation and analysis must mirror it.
GATE_MODES = {
    "v1_grpo_baseline": "disabled",
    "v2_fixed_blend": "unit",
    "v3_learned_lambda": "unit",
    "v4_full_hrpo_gate": "standard",
    "v5_hrpo_format_reward": "standard",
    "v6_hrpo_full": "standard",
}


# ---------------------------------------------------------------------------
# Argument parsing — Section 5 fixed hyperparameters as defaults
# ---------------------------------------------------------------------------

def build_arg_parser(variant_id: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"HRPO ablation variant {variant_id}")
    parser.add_argument("--model_size", type=str, default="1.5b",
                        choices=sorted(MODEL_NAMES.keys()))
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_targets", type=str, default="full",
                        choices=sorted(LORA_TARGETS.keys()),
                        help="full = attention+MLP (main ablation); "
                             "attention = q/k/v/o only, matching HRPO paper Table 4")

    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.005)
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
    # Rollout-sampling repetition penalty (Section 5). Set 1.0 to disable
    # (that reproduces the pre-existing 1.5b-v1_grpo_baseline checkpoint).
    parser.add_argument("--repetition_penalty", type=float, default=1.2)

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
    return f"./experiments/{name}"


# experiments/{size}-{base_variant}[-suffix], where suffix marks a follow-up
# run (-seed43, -rp1, -loraattention, or a combination).
EXPERIMENT_NAME_RE = re.compile(
    r"^(?P<size>0\.5b|1\.5b|3b)-(?P<variant>v\d_[a-z_]+?)(?P<suffix>-(?:seed|rp|lora).*)?$"
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
    "lora_targets",
]


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
                if stored.get(key) != value:
                    diffs.append(f"{key}: checkpoint={stored.get(key)!r} now={value!r}")
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


def disable_gate(model):
    """v1 (pure GRPO): with answer_start == "" every decoded string 'contains'
    it, so is_thinking is always False -> the Eq.4 blend is skipped for every
    token in both rollouts (generation/utils.py:3368) and the loss forward
    (unsloth/models/llama.py:663-669). embeds_ratio stays 1 everywhere."""
    model.answer_start = ""


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
        self._params = {"Lambda": [], "gate_r": [], "gate_i": [], "lora": []}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "thinking_residual_Lambda" in name:
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
