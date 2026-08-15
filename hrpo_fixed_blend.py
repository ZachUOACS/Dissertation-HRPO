"""
v2_fixed_blend — fixed blend ratio, no learnable gates (ablation Section 6).

The Eq.3 projection runs every step; Lambda is initialised from
[r_min, r_max] exactly as in HRPO and then frozen, and the input-dependent
gates are fixed at r_t = i_t = 1 via the unit-gate patch (the gate linear
layers never execute — see changes.md mapping #7 for why omitting them from
modules_to_save is not sufficient). The blend ratio therefore never moves
from its initialised value: the closest equivalent to a fixed-residual
approach (Wang et al. 2026, AAAI).

Per the spec's implementation note, all three gate modules are still placed
in modules_to_save so they are initialised/saved exactly like v4, then frozen.
"""

import unsloth
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import os
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, Dataset
from patch import patch_trainer_optimizer
from utils import *
from variant_common import (
    train_with_resume, LORA_TARGETS,
    MODEL_NAMES, GATE_MODULES, build_arg_parser, experiment_dir,
    exit_if_complete, apply_unit_gate_patch, freeze_gate_modules,
    assert_optimizer_gate_params, set_rollout_repetition_penalty, RunLogger,
)

VARIANT_ID = "v2_fixed_blend"

os.environ["WANDB_PROJECT"] = "latent-reasoning"


def preprocess_gsm8k(split="train", chunk_size=1000) -> Dataset:
    dataset = load_dataset('openai/gsm8k', 'main')[split]
    return dataset.map(process_gsm8k, batched=True,
                       batch_size=chunk_size, load_from_cache_file=False)


def main(args):
    exp_name = experiment_dir(args.model_size, VARIANT_ID, args)
    exit_if_complete(exp_name)
    model_name = MODEL_NAMES[args.model_size]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = args.max_prompt_length + args.max_completion_length,
        load_in_4bit = False,
        load_in_8bit = False,
        fast_inference = False,
    )
    model.answer_start = ANSWER_START

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_rank,
        target_modules = LORA_TARGETS[args.lora_targets],
        # v2: all three gate modules so they initialise/save like v4 ...
        modules_to_save = list(GATE_MODULES),
        lora_alpha = args.lora_rank * 2,
        use_gradient_checkpointing = "unsloth",
        random_state = args.seed,
    )
    model.model.model.thinking_residual_Lambda.reset_lambda_parameters(
        r_min = args.residual_r_min, r_max = args.residual_r_max,
    )
    # ... then r_t = i_t = 1 (gates bypassed) and everything frozen.
    apply_unit_gate_patch(model)
    freeze_gate_modules(model, GATE_MODULES)

    training_args = GRPOConfig(
        use_vllm = False,
        learning_rate = args.lr,
        beta = args.beta,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = args.weight_decay,
        warmup_ratio = args.warmup_ratio,
        lr_scheduler_type = args.lr_scheduler_type,
        optim = args.optimizer,
        max_grad_norm = args.max_grad_norm,
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        temperature = args.temperature,
        num_generations = args.group_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        per_device_train_batch_size = args.per_device_train_batch_size,
        max_prompt_length = args.max_prompt_length,
        max_completion_length = args.max_completion_length,
        num_train_epochs = 1,
        save_steps = 250,
        save_total_limit = 3,
        report_to = args.report_to,
        output_dir = exp_name,
    )

    dataset = preprocess_gsm8k('train', chunk_size=500)
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            get_reward_func(process_gsm8k_answer),  # outcome-only reward
        ],
        args = training_args,
        train_dataset = dataset,
    )
    patch_trainer_optimizer(
        trainer,
        args.lr_residual_gate,
        args.lr_residual_Lambda,
    )
    set_rollout_repetition_penalty(trainer, args.repetition_penalty)
    # v2: nothing gate-related may be trainable.
    assert_optimizer_gate_params(trainer, expect_lambda=False, expect_gates=False)

    run_logger = RunLogger(exp_name, VARIANT_ID, args.model_size, args)
    train_with_resume(trainer, exp_name, args)
    run_logger.finish(trainer)


if __name__ == "__main__":
    args = build_arg_parser(VARIANT_ID).parse_args()
    main(args)
