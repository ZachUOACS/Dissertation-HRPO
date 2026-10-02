"""
v1_grpo_baseline — pure GRPO, no gate (ablation Section 6).

Standalone copy of hrpo_gsm8k.py with the gate entirely removed:
  * no gate modules in modules_to_save (LoRA only),
  * no reset_lambda_parameters call,
  * no gate parameter groups (patch_trainer_optimizer is not called),
  * the Eq.4 blend is disabled for every token via disable_gate()
    (answer_start == "" -> is_thinking always False, embeds_ratio == 1).

Reproducing the pre-existing experiments/1.5b-v1_grpo_baseline checkpoint
requires --repetition_penalty 1.0 (that run predates the Section 5 default).
"""

import unsloth
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import os
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, Dataset
from utils import *
from variant_common import (
    build_train_dataset, reward_answer_fn,
    load_stage0_init,
    train_with_resume, LORA_TARGETS,
    MODEL_NAMES, build_arg_parser, experiment_dir, exit_if_complete,
    disable_gate, assert_optimizer_gate_params, set_rollout_repetition_penalty,
    RunLogger,
)

VARIANT_ID = "v1_grpo_baseline"

os.environ["WANDB_PROJECT"] = "latent-reasoning"




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
    # v1: no hybrid reasoning — answer_start = "" makes is_thinking always
    # False, so the Eq.4 blend never fires in rollouts or the loss forward.
    disable_gate(model)

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_rank,
        target_modules = LORA_TARGETS[args.lora_targets],
        # v1: no gate modules saved or trained.
        modules_to_save = None,
        lora_alpha = args.lora_rank * 2,
        use_gradient_checkpointing = "unsloth",
        random_state = args.seed,
    )

    # Stage-0 warm start (tier8 arm C3). Must follow get_peft_model: the loader
    # writes into the PEFT adapter. v1 has modules_to_save=None, so the warm start
    # it is given must be a --blend off (plain LoRA SFT) run, which carries no
    # Lambda; a --blend on warm start would raise on unexpected Lambda keys.
    if args.init_from:
        load_stage0_init(model, args.init_from, dataset=args.dataset)
        if args.init_only:
            print("[init_only] warm start loaded successfully; exiting before training")
            raise SystemExit(0)

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

    dataset = build_train_dataset(args.dataset, chunk_size=500,
                                  math_root=args.math_root)
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            get_reward_func(reward_answer_fn(args.dataset)),  # outcome-only reward
        ],
        args = training_args,
        train_dataset = dataset,
    )
    # v1: patch_trainer_optimizer intentionally NOT called — no gate params.
    set_rollout_repetition_penalty(trainer, args.repetition_penalty)
    assert_optimizer_gate_params(trainer, expect_lambda=False, expect_gates=False)

    run_logger = RunLogger(exp_name, VARIANT_ID, args.model_size, args)
    train_with_resume(trainer, exp_name, args)
    run_logger.finish(trainer)


if __name__ == "__main__":
    args = build_arg_parser(VARIANT_ID).parse_args()
    main(args)
