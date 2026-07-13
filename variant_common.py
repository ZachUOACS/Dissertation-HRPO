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

    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    # Rollout-sampling repetition penalty (Section 5). Set 1.0 to disable
    # (that reproduces the pre-existing 1.5b-v1_grpo_baseline checkpoint).
    parser.add_argument("--repetition_penalty", type=float, default=1.2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="wandb")
    return parser


# ---------------------------------------------------------------------------
# Experiment directory handling
# ---------------------------------------------------------------------------

def experiment_dir(model_size: str, variant_id: str) -> str:
    return f"./experiments/{model_size}-{variant_id}"


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
    if os.path.exists(exp_dir) and len(os.listdir(exp_dir)) > 0:
        print(f"WARNING: {exp_dir} exists but is incomplete; the Trainer will "
              f"write fresh checkpoints into it.")


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
    <exp_dir>/gradient_log.jsonl. Fires on_pre_optimizer_step, i.e. after
    gradient accumulation, before clipping and the optimizer step."""

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
