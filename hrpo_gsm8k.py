import unsloth
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import os
import argparse
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, Dataset
from patch import patch_trainer_optimizer
from utils import *

os.environ["WANDB_PROJECT"] = "latent-reasoning"


def preprocess_gsm8k(split="train", chunk_size=1000) -> Dataset:
    dataset = load_dataset('openai/gsm8k', 'main')[split]
    return dataset.map(process_gsm8k, batched=True, 
                       batch_size=chunk_size, load_from_cache_file=False)


def main(args):
    exp_name = args.exp_name or (
        f"./experiments/{args.model_name.split('/')[-1]}-gsm8k-group{args.group_size}"
        f"-lora{args.lora_rank}-rmin{args.residual_r_min}-temp{args.temperature}"
    )
    # Resumability: if checkpoints already exist, resume (when --resume) instead of clobbering.
    resume_ckpt = False
    if os.path.exists(exp_name) and len(os.listdir(exp_name)) > 0:
        has_ckpt = any(d.startswith("checkpoint-") for d in os.listdir(exp_name))
        if args.resume and has_ckpt:
            resume_ckpt = True
            print(f"Experiment {exp_name} exists with checkpoints -> resuming.")
        elif not args.resume:
            print(f"Experiment {exp_name} already exists. Exiting...")
            exit()

    model_path, use_exact = resolve_model_path(args.model_name)  # offline-safe local snapshot
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = args.max_prompt_length + args.max_completion_length,
        load_in_4bit = False,
        load_in_8bit = False,
        fast_inference = False,
        use_exact_model_name = use_exact,
    )
    model.answer_start = ANSWER_START

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # ==== Change A: Context-Prediction Fusion setup (no-op unless --fusion_enabled) ====
    # Fusion assumes tied input/output embeddings (Qwen2.5). If untied, fall back to the
    # LT-Tuning Eq. 20 adapter path (built BEFORE get_peft_model so it can be saved/trained).
    if args.fusion_enabled:
        tied = bool(getattr(model.config, "tie_word_embeddings", False))
        print(f"[fusion] tie_word_embeddings = {tied} (expected True for Qwen2.5)")
        if not tied and not args.fusion_use_adapter:
            print("[fusion] WARNING: untied embeddings -> enabling the Eq. 20 adapter path.")
            args.fusion_use_adapter = True
        if args.fusion_use_adapter:
            model.model.init_fusion_adapter(bottleneck_dim = args.fusion_adapter_dim)

    modules_to_save = [
        "thinking_residual_gate_r",
        "thinking_residual_gate_i",
        "thinking_residual_Lambda",
    ]
    if args.fusion_enabled:
        modules_to_save.append("thinking_residual_gate_beta")
        if args.fusion_use_adapter:
            modules_to_save.append("thinking_residual_gate_beta_adapter")

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_rank,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save = modules_to_save,
        lora_alpha = args.lora_rank * 2,
        use_gradient_checkpointing = "unsloth",
        random_state = args.seed,
    )
    model.model.model.thinking_residual_Lambda.reset_lambda_parameters(
        r_min = args.residual_r_min, r_max = args.residual_r_max,
    )

    # Configure the fusion gate on the inner Qwen2Model (attrs persist through PEFT wrapping).
    inner = model.model.model
    inner.fusion_enabled     = args.fusion_enabled
    inner.fusion_layer_I     = args.fusion_layer_I
    inner.fusion_gate_input  = args.fusion_gate_input
    inner.fusion_normalize_h = args.fusion_normalize_h
    inner.fusion_use_adapter = args.fusion_use_adapter
    if args.fusion_enabled:
        # Init beta ~= sigmoid(fusion_beta_init) ~= 0 (baseline-like start). Walk params so BOTH the
        # PEFT original_module and modules_to_save.default copies of the beta gate are initialized;
        # the adapter (if any) is excluded.
        import torch as _torch
        with _torch.no_grad():
            for _n, _p in model.named_parameters():
                if "thinking_residual_gate_beta" in _n and "adapter" not in _n:
                    if _n.endswith("bias"):     _p.fill_(float(args.fusion_beta_init))
                    elif _n.endswith("weight"): _p.zero_()

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
        max_steps = args.max_steps,
        save_steps = args.save_steps,
        save_total_limit = 2,
        report_to = "wandb",
        output_dir = exp_name,
        # Seed the FULL trainer pipeline (set_seed, data sampler, generation sampling), not just the
        # LoRA init (random_state above). Default is 42, so seed-42 runs are unchanged; seeds 43/44 now
        # vary data order + rollout sampling too, making the multi-seed comparison a real robustness check.
        seed = args.seed,
        # Change B: Dr. GRPO debias flags (defaults reproduce vanilla HRPO).
        length_norm = args.length_norm,
        advantage_std_norm = args.advantage_std_norm,
        length_norm_constant = args.length_norm_constant,
    )

    dataset = preprocess_gsm8k('train', chunk_size=500)
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            get_reward_func(process_gsm8k_answer),
        ],
        args = training_args,
        train_dataset = dataset,
    )
    patch_trainer_optimizer(
        trainer,
        args.lr_residual_gate,
        args.lr_residual_Lambda,
        lr_thinking_residual_gate_beta = args.lr_residual_gate_beta,
    )

    # Fusion sanity guard: build the optimizer now and verify the beta gate is actually trainable
    # AND landed in an optimizer param group. (A prior run silently trained a FROZEN gate because
    # gate_beta was excluded from the optimizer -> fusion_beta stuck at sigmoid(b_beta_init).)
    if args.fusion_enabled:
        beta_named = [(n, p) for n, p in model.named_parameters()
                      if "thinking_residual_gate_beta" in n and "adapter" not in n]
        n_trainable = sum(1 for _, p in beta_named if p.requires_grad)
        print(f"[fusion] gate_beta params: {len(beta_named)}, trainable: {n_trainable}")
        for n, p in beta_named:
            print(f"[fusion]   {n}  requires_grad={p.requires_grad}  shape={tuple(p.shape)}")
        trainer.create_optimizer()  # patched create_optimizer; reused by train()
        trainable_ids = {id(p) for _, p in beta_named if p.requires_grad}
        in_opt = sum(1 for g in trainer.optimizer.param_groups for p in g["params"]
                     if id(p) in trainable_ids)
        print(f"[fusion] gate_beta params in optimizer: {in_opt}/{len(trainable_ids)}")
        assert n_trainable > 0 and in_opt == len(trainable_ids) and in_opt > 0, (
            "FUSION GATE NOT OPTIMIZED: thinking_residual_gate_beta is not trainable / not in the "
            "optimizer -> beta would stay frozen at init. Fix modules_to_save / requires_grad before "
            "training a fusion arm.")
        print("[fusion] gate is trainable and optimized. OK.")

    trainer.train(resume_from_checkpoint=resume_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_rank", type=int, default=32)

    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--residual_r_min", type=float, default=0.99)
    parser.add_argument("--residual_r_max", type=float, default=0.999)
    parser.add_argument("--lr_residual_gate", type=float, default=1e-4)
    parser.add_argument("--lr_residual_Lambda", type=float, default=1e-3)
    parser.add_argument("--lr_residual_gate_beta", type=float, default=1e-3,
                        help="LR for the fusion beta gate (higher than the base gate so beta can "
                             "actually open from a near-closed init; fusion arms only).")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--optimizer", type=str, default="paged_adamw_8bit")
    parser.add_argument("--max_grad_norm", type=float, default=0.1)

    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=1024)

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seed", type=int, default=42)

    # Run management (used by run_ablation.sh to route arms + do smoke runs / resume).
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Output dir override; default = auto-named under ./experiments/.")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Cap training steps (-1 = full epoch). run_ablation --smoke sets this low.")
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False,
                        help="Resume from the latest checkpoint in exp_name instead of exiting.")

    # ==== Change A: Context-Prediction Fusion (all default to baseline HRPO) ====
    parser.add_argument("--fusion_enabled", action=argparse.BooleanOptionalAction, default=False,
                        help="Add the contextual-hidden-state channel to HRPO's gate (Change A).")
    parser.add_argument("--fusion_layer_I", type=int, default=-2,
                        help="Which hidden layer supplies the context h_I (negative = from the end).")
    parser.add_argument("--fusion_gate_input", type=str, default="hidden", choices=["hidden", "token"],
                        help="Gate beta on h_I ('hidden') or on the sampled-token embedding ('token').")
    parser.add_argument("--fusion_beta_init", type=float, default=-4.0,
                        help="b_beta init; beta ~= sigmoid(b_beta) ~= 0 so training starts ~= baseline.")
    parser.add_argument("--fusion_use_adapter", action=argparse.BooleanOptionalAction, default=False,
                        help="LT-Tuning Eq. 20 adapter on h_I (untied-embedding backbones only).")
    parser.add_argument("--fusion_normalize_h", action=argparse.BooleanOptionalAction, default=True,
                        help="Normalize h_I to h_I/||h_I|| before fusion.")
    parser.add_argument("--fusion_adapter_dim", type=int, default=1024,
                        help="Bottleneck dim for the fusion adapter (untied only).")

    # ==== Change B: Dr. GRPO debias (all default to baseline HRPO) ====
    parser.add_argument("--length_norm", type=str, default="response", choices=["response", "constant"],
                        help="'response' = per-|y_i| (baseline); 'constant' = divide loss by C (Dr. GRPO).")
    parser.add_argument("--advantage_std_norm", action=argparse.BooleanOptionalAction, default=True,
                        help="True = /std (baseline); False = mean-center only (Dr. GRPO).")
    parser.add_argument("--length_norm_constant", type=int, default=None,
                        help="C for length_norm='constant'; default = max_completion_length.")

    args = parser.parse_args()

    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen2.5-3B-Instruct"
    # "meta-llama/Llama-3.2-1B-Instruct"
    # "meta-llama/Llama-3.2-3B-Instruct"

    main(args)