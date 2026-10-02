"""
v7_constant_vector (control C1) — v3 with the latent SOURCE swapped.

Identical to v3_learned_lambda (learned Lambda, unit input gates) in every
respect except one: with --latent_source constant, the Equation 3 projection
output is replaced by a single learned vector u. This isolates the question
"does the latent channel need to be input-dependent at all, or would any
on-manifold vector do?" from the question v3 answers.

--latent_source projection (the default) reproduces v3 EXACTLY — same code path,
same patch, same modules_to_save — which is what verification check 1 relies on.

u is registered on thinking_residual_Lambda BEFORE get_peft_model so that the
modules_to_save deep-copy contains it (otherwise it never reaches the adapter),
and its name puts it in Lambda's 1e-3 optimizer group. The substitution is made
in thinking_residual itself rather than in the generation loop: generate() runs
under no_grad and its recorded latents are detached by the time the loss is
computed, so a generation-only substitution would leave u with zero gradient.
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
    build_train_dataset, reward_answer_fn,
    train_with_resume, LORA_TARGETS,
    MODEL_NAMES, build_arg_parser, experiment_dir, exit_if_complete,
    apply_unit_gate_patch, assert_optimizer_gate_params,
    register_constant_latent_vector, apply_constant_latent_patch,
    set_rollout_repetition_penalty, GateGradientLogger, RunLogger,
)

VARIANT_ID = "v7_constant_vector"

os.environ["WANDB_PROJECT"] = "latent-reasoning"




def main(args):
    # experiment_dir has no --latent_source axis, so projection mode resolves to
    # the SAME directory as the constant arm (e.g. 1.5b-v7_constant_vector-rp1).
    # Training in projection mode would therefore either skip against C1's
    # completed run or train into it. The v3-equivalence claim that mode exists
    # to support is verified in tests/test_constant_vector.py, in-process and
    # without an experiment directory, so refuse the training run outright.
    if args.latent_source != "constant":
        raise SystemExit(
            f"{VARIANT_ID} is arm C1 and must be run with --latent_source "
            f"constant (got '{args.latent_source}'). Projection mode would "
            f"resolve to the same experiment directory as the constant arm; "
            f"its equivalence to v3 is covered by tests/test_constant_vector.py.")

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

    # Must precede get_peft_model: modules_to_save deep-copies the Lambda module
    # here, and u has to be inside that copy to reach the adapter.
    if args.latent_source == "constant":
        register_constant_latent_vector(
            model, init=args.constant_init, seed=args.seed)

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_rank,
        target_modules = LORA_TARGETS[args.lora_targets],
        # As v3: only Lambda (and, when enabled, the u registered inside it).
        modules_to_save = ["thinking_residual_Lambda"],
        lora_alpha = args.lora_rank * 2,
        use_gradient_checkpointing = "unsloth",
        random_state = args.seed,
    )
    model.model.model.thinking_residual_Lambda.reset_lambda_parameters(
        r_min = args.residual_r_min, r_max = args.residual_r_max,
    )
    # r_t = i_t = 1 in both cases; the constant patch additionally replaces the
    # Eq.3 projection output with u.
    if args.latent_source == "constant":
        apply_constant_latent_patch(model)
    else:
        apply_unit_gate_patch(model)

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
            get_reward_func(reward_answer_fn(args.dataset)),  # outcome-only reward, as v3
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
    # As v3: Lambda trainable, gate linears absent from the optimizer.
    assert_optimizer_gate_params(trainer, expect_lambda=True, expect_gates=False)
    trainer.add_callback(GateGradientLogger(trainer.model, exp_name))

    run_logger = RunLogger(exp_name, VARIANT_ID, args.model_size, args)
    train_with_resume(trainer, exp_name, args)
    run_logger.finish(trainer)


if __name__ == "__main__":
    args = build_arg_parser(VARIANT_ID).parse_args()
    main(args)
