"""
stage0_warmstart.py — one teacher-forced epoch over GSM8K-train (or MATH-train
with --dataset math), producing a checkpoint that any hrpo_*.py script can
resume from via --init_from. A warm start only feeds RL on its own task:
load_stage0_init refuses a GSM8K warm start for a MATH run and vice versa.

Why this exists: the RL variants start from a cold LoRA, so Lambda spends much of
the run climbing before the latent channel carries anything. Stage 0 gives the
blend a supervised warm start first.

--blend on   the Eq.3 blend runs in the think span, exactly as in the RL log-prob
             forward: the SAME projection helper the rollout uses
             (transformers.generation.utils.hrpo_project_latent) and the SAME
             blend path (unsloth/models/llama.py:663-669, reached by passing
             input_ids + inputs_embeds + thinking_mask together).
--blend off  plain LoRA SFT, no gate code path at all — control C3's warm start.

Configuration is v3's: unit input gates (r_t = i_t = 1) with Lambda trainable,
gate linears frozen. Lambda init uses the same --residual_r_min mechanism as RL
(default 0.98, matching the existing -rp1 arms).

TEACHER-FORCED BLEND, in two passes (there is no generation loop to produce the
latents):
  1. plain forward -> per-position logits -> softmax -> Eq.3 projection -> latents
     This pass runs under no_grad and the latents are DETACHED. That is not a
     shortcut: in RL the latents come from generate(), which is itself no_grad, so
     they are already detached constants by the time the loss forward runs.
     Mirroring that keeps Stage 0 faithful to the RL path (and avoids building a
     graph through a vocab-sized einsum).
  2. forward again with inputs_embeds=<shifted latents> + thinking_mask, so the
     unsloth blend fires at think-span positions; cross-entropy on that output.

The projection is chunked over positions: probs is (batch, seq, vocab) and vocab
is ~152k, so materialising it whole would cost gigabytes.

Output: experiments/{size}-stage0-{on|off}-rmin{r}-seed{N}/ containing the LoRA
adapter + Lambda in the layout the RL scripts save, plus stage0_log.jsonl. Note
the directory name deliberately does NOT match EXPERIMENT_NAME_RE, so eval_all.py
and analyse_replicates.py will not mistake a warm start for an ablation result.
"""

import unsloth
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import argparse
import json
import os
import re

import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.generation.utils import hrpo_project_latent

from patch import patch_trainer_optimizer
from utils import ANSWER_START, SYSTEM_PROMPT, load_math_raw, math_sft_target
from variant_common import (
    LORA_TARGETS, MODEL_NAMES, build_arg_parser,
    apply_unit_gate_patch, freeze_gate_modules, get_inner_model,
)

os.environ["WANDB_PROJECT"] = "latent-reasoning"

# Execution counter for the Eq.4 blend (see main()).
BLEND_CALLS = {"n": 0, "masked_rows": 0}
# Counts gradient arrivals at the trainable Lambda. A backward hook is used
# rather than torch.autograd.grad(): the latter raises under reentrant
# gradient checkpointing ("incompatible with .grad()"), which is precisely
# what happens once Lambda IS connected through the checkpointed layers.
LAMBDA_GRAD = {"hits": 0, "max_norm": 0.0}

# GSM8K reasoning carries calculator annotations like <<48/2=24>>. The RL model
# is never rewarded for emitting those, so teaching them here would pull the
# warm start away from the behaviour RL then optimises.
_CALC = re.compile(r"<<[^>]*>>")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _prompt_ids(tokenizer, question):
    """Byte-identical to the RL prompt: system=SYSTEM_PROMPT, user=question
    (utils.process_gsm8k / utils.process_math)."""
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": question.strip()}],
        tokenize=True, add_generation_prompt=True,
    )


def _think_span(tokenizer, ids, n_prompt):
    """Think span: assistant positions BEFORE the '####' marker. This mirrors
    the RL rule (generation/utils.py:3368 — is_thinking is True while the
    decoded string does not yet contain ANSWER_START), evaluated here on the
    teacher-forced target instead of a sampled string."""
    hash_at = len(ids)
    for i in range(n_prompt, len(ids)):
        if tokenizer.decode(ids[n_prompt:i + 1]).find(ANSWER_START) != -1:
            hash_at = i
            break
    think = [False] * len(ids)
    for i in range(n_prompt, min(hash_at, len(ids))):
        think[i] = True
    return think


def encode_math_example(tokenizer, problem, solution, max_prompt, max_completion):
    """One MATH example as Stage-0 features, or None to drop it.

    Dropped rather than truncated. GSM8K solutions never approach the length
    limit, so the GSM8K path can truncate safely; a MATH solution cut off before
    its '####' would teach the model never to answer. A prompt longer than
    max_prompt is dropped too, since GRPO truncates prompts and the warm start
    would otherwise train on a prompt RL never shows. The target keeps the
    answer boxed so the RL reward scores it (utils.math_sft_target)."""
    target = math_sft_target(solution)
    if target is None:
        return None
    prompt_ids = _prompt_ids(tokenizer, problem)
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    target_ids = target_ids + [tokenizer.eos_token_id]
    if len(prompt_ids) > max_prompt or len(target_ids) > max_completion:
        return None
    ids = prompt_ids + target_ids
    return {"input_ids": ids,
            "labels": [-100] * len(prompt_ids) + target_ids,
            "think": _think_span(tokenizer, ids, len(prompt_ids))}


def build_dataset(tokenizer, max_len, split="train", dataset="gsm8k",
                  max_prompt=None, max_completion=None, math_root=None):
    """GSM8K: target is the reference reasoning plus '#### <answer>'. MATH:
    encode_math_example. Either way the prompt matches RL byte-for-byte and the
    loss is masked to the assistant span only."""
    if dataset == "math":
        raw = load_math_raw(split, math_root)
        placeholder = {"input_ids": [tokenizer.eos_token_id], "labels": [-100],
                       "think": [False], "keep": False}

        def _encode_math(ex):
            out = encode_math_example(tokenizer, ex["problem"], ex["solution"],
                                      max_prompt, max_completion)
            # Dropped rows carry a typed placeholder rather than empty lists, so
            # every column keeps one Arrow type across batches until the filter.
            return placeholder if out is None else {**out, "keep": True}

        encoded = raw.map(_encode_math, remove_columns=raw.column_names,
                          load_from_cache_file=False)
        kept = encoded.filter(lambda e: e["keep"]).remove_columns("keep")
        print(f"[stage0] MATH {split}: kept {len(kept)}/{len(encoded)} "
              f"(dropped {len(encoded) - len(kept)}: no boxed answer, prompt > "
              f"{max_prompt} tokens, or target > {max_completion} tokens)")
        return kept

    raw = load_dataset("openai/gsm8k", "main")[split]

    def _encode(ex):
        prompt_ids = _prompt_ids(tokenizer, ex["question"])
        body, _, ans = ex["answer"].partition(ANSWER_START)
        target = f"{_CALC.sub('', body).strip()}\n{ANSWER_START} {ans.strip()}"
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        target_ids = target_ids + [tokenizer.eos_token_id]

        ids = (prompt_ids + target_ids)[:max_len]
        labels = ([-100] * len(prompt_ids) + target_ids)[:max_len]
        return {"input_ids": ids, "labels": labels,
                "think": _think_span(tokenizer, ids, len(prompt_ids))}

    cols = raw.column_names
    return raw.map(_encode, remove_columns=cols, load_from_cache_file=False)


def collate(features, pad_id):
    n = max(len(f["input_ids"]) for f in features)
    out = {"input_ids": [], "labels": [], "attention_mask": [], "thinking_mask": []}
    for f in features:
        k = n - len(f["input_ids"])
        out["input_ids"].append(f["input_ids"] + [pad_id] * k)
        out["labels"].append(f["labels"] + [-100] * k)
        out["attention_mask"].append([1] * len(f["input_ids"]) + [0] * k)
        # Position 0 can never be a think position: its latent would come from
        # position -1, which does not exist.
        think = list(f["think"]) + [False] * k
        think[0] = False
        out["thinking_mask"].append(think)
    return {
        "input_ids": torch.tensor(out["input_ids"], dtype=torch.long),
        "labels": torch.tensor(out["labels"], dtype=torch.long),
        "attention_mask": torch.tensor(out["attention_mask"], dtype=torch.long),
        "thinking_mask": torch.tensor(out["thinking_mask"], dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def lambda_diagnostics(model):
    """Lambda mean, per-dimension CV of the hidden ratio, and the mean hidden
    ratio. Hidden ratio at r_t = 1 is sqrt(1 - a_t^2) with
    a_t = exp(-c * softplus(-Lambda)) — the same convention analyse_gate uses."""
    lam_mod = get_inner_model(model).thinking_residual_Lambda
    lam = lam_mod.Lambda.detach().float()
    c = float(getattr(lam_mod, "c", 8.0))
    a = torch.exp(-c * torch.nn.functional.softplus(-lam))
    ratio = torch.sqrt((1 - a.pow(2)).clamp_min(0))
    return {
        "lambda_mean": float(lam.mean()),
        "hidden_ratio_mean": float(ratio.mean()),
        "hidden_ratio_cv": float(ratio.std() / ratio.mean()) if float(ratio.mean()) else None,
    }


class Stage0Logger(TrainerCallback):
    def __init__(self, model, out_dir, every=25):
        self.model, self.out_dir, self.every = model, out_dir, every
        self.path = os.path.join(out_dir, "stage0_log.jsonl")
        self._lam = [p for n, p in model.named_parameters()
                     if "thinking_residual_Lambda" in n and not n.endswith(".u")]

    def on_pre_optimizer_step(self, args, state, control, **kw):
        # Fire on the first few steps too: a 10-step smoke run would
        # otherwise only ever log step 0 and tell us nothing.
        if state.global_step > 3 and state.global_step % self.every:
            return
        gn = None
        sq, seen = 0.0, False
        for p in self._lam:
            if p.grad is not None:
                sq += float(p.grad.detach().float().pow(2).sum()); seen = True
        if seen:
            gn = sq ** 0.5
        rec = {"step": state.global_step, "lambda_grad_norm": gn}
        rec.update(lambda_diagnostics(self.model))
        if state.log_history:
            rec["loss"] = state.log_history[-1].get("loss")
        os.makedirs(self.out_dir, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Stage0Trainer(Trainer):
    def __init__(self, *a, blend=True, latent_temperature=0.5,
                 proj_chunk=64, **kw):
        super().__init__(*a, **kw)
        self.blend = blend
        self.latent_temperature = latent_temperature
        self.proj_chunk = proj_chunk

    @staticmethod
    def _ce(logits, labels):
        return torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

    def _project(self, logits, embed_weight):
        """Eq.3 per position, chunked: probs is (B, S, vocab) and vocab ~152k, so
        the whole thing at once costs gigabytes. Uses the SAME helper as the
        rollout, so the two paths cannot drift."""
        outs = []
        for i in range(0, logits.size(1), self.proj_chunk):
            sl = logits[:, i:i + self.proj_chunk].float()
            if self.latent_temperature and self.latent_temperature != 1.0:
                sl = sl / self.latent_temperature
            # Softmax in fp32 for stability, then cast BACK to the embedding
            # dtype before the einsum. Two reasons: einsum requires matching
            # dtypes (fp32 probs against a bf16 W_e raises "expected scalar type
            # BFloat16 but found Float"), and generation itself does the
            # projection in the model dtype — so casting here keeps the einsum
            # identical to the rollout path rather than diverging from it.
            probs = torch.softmax(sl, dim=-1).to(embed_weight.dtype)
            outs.append(hrpo_project_latent(probs, embed_weight))
        return torch.cat(outs, dim=1)

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        ids = inputs["input_ids"]
        am = inputs["attention_mask"]
        labels = inputs["labels"]

        # NOTE: attention_mask is deliberately NOT passed to the model.
        # unsloth/models/llama.py:707 detaches inputs_embeds when
        #   _has_no_labels (we compute CE ourselves, so labels is None)
        #   AND attention_mask is not None
        #   AND past_key_values is None
        #   AND embed_tokens is frozen
        # all of which hold here. That detach severs the graph immediately after
        # the Eq.4 blend, orphaning a_t and Lambda while a fresh
        # requires_grad_(True) leaf lets LoRA keep training — which is exactly
        # the "loss falls but Lambda never moves" failure we chased. The RL
        # log-prob call (rl_replacements.py:409) omits attention_mask for the
        # same reason, and unsloth nulls it during training anyway
        # (llama.py:726-728), so dropping it changes no attention behaviour.
        if not self.blend:
            out = model(input_ids=ids)
            loss = self._ce(out.logits, labels)
            return (loss, out) if return_outputs else loss

        embed_weight = model.get_input_embeddings().weight
        # Pass 1 — latents. Detached, mirroring RL where they come from no_grad
        # generation and are already constants in the loss forward.
        with torch.no_grad():
            logits1 = model(input_ids=ids).logits
            latents = self._project(logits1, embed_weight)
            # latent for input position t is the projection at t-1
            shifted = torch.zeros_like(latents)
            shifted[:, 1:] = latents[:, :-1]

        # Pass 2 — blended forward. Passing input_ids AND inputs_embeds together
        # is what routes through unsloth's blend (llama.py:606-607, 663-669),
        # exactly as rl_replacements.py:410 does for the RL log-prob forward.
        out = model(input_ids=ids, inputs_embeds=shifted,
                    thinking_mask=inputs["thinking_mask"])
        loss = self._ce(out.logits, labels)

        return (loss, out) if return_outputs else loss


# ---------------------------------------------------------------------------

def stage0_dir(args):
    # GSM8K keeps its original name so the existing warm starts still resolve;
    # any other task gets -ds<task>, which cluster/run_stage0.slurm mirrors.
    ds = getattr(args, "dataset", "gsm8k")
    return (f"./experiments/{args.model_size}-stage0-{args.blend}"
            f"-rmin{args.residual_r_min:g}-seed{args.seed}"
            f"{'' if ds == 'gsm8k' else '-ds' + ds}")


def main(args):
    out_dir = stage0_dir(args)
    model_name = MODEL_NAMES[args.model_size]
    max_len = args.max_prompt_length + args.max_completion_length

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=max_len,
        load_in_4bit=False, load_in_8bit=False, fast_inference=False,
    )
    model.answer_start = ANSWER_START
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    blend_on = (args.blend == "on")

    # --blend off must be a PLAIN LoRA SFT: no Lambda saved, no gate code path.
    # That is what makes it a valid control C3 warm start — v1 has
    # modules_to_save=None, so an adapter carrying Lambda could not be loaded
    # into it — and what makes "reproduces a plain PEFT SFT step bit-for-bit"
    # true rather than merely approximately true.
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=LORA_TARGETS[args.lora_targets],
        modules_to_save=["thinking_residual_Lambda"] if blend_on else None,
        lora_alpha=args.lora_rank * 2,
        use_gradient_checkpointing=(False if args.grad_checkpoint == "off"
                                    else args.grad_checkpoint),
        random_state=args.seed,
    )
    if blend_on:
        model.model.model.thinking_residual_Lambda.reset_lambda_parameters(
            r_min=args.residual_r_min, r_max=args.residual_r_max,
        )
        apply_unit_gate_patch(model)      # r_t = i_t = 1 (v3 configuration)
        # Count real executions of the blend. If this stays 0 the forward never
        # reaches thinking_residual at all, which is the only remaining
        # explanation for Lambda getting no gradient once the think mask and the
        # optimizer have both been ruled out.
        _inner = get_inner_model(model)
        _orig_tr = _inner.thinking_residual
        def _counting_thinking_residual(embeds, residual, eps=1e-8):
            BLEND_CALLS["n"] += 1
            BLEND_CALLS["masked_rows"] += int(embeds.shape[0])
            out, a_t = _orig_tr(embeds, residual, eps)
            BLEND_CALLS["last_out"] = out
            if BLEND_CALLS["n"] == 1:
                # Localise the break. If a_t does not require grad, the Lambda
                # used inside the blend is NOT the trainable one — PEFT's
                # ModulesToSaveWrapper falls back to original_module (frozen)
                # whenever adapters are disabled, which would explain a live
                # blend whose gradient never reaches the optimised Lambda.
                lam_mod = _inner.thinking_residual_Lambda
                dispatched = type(getattr(lam_mod, "original_module", lam_mod)).__name__
                trainable = [n for n, q in lam_mod.named_parameters() if q.requires_grad]
                print(f"[stage0][blend-probe] embeds.rg={embeds.requires_grad} "
                      f"residual.rg={residual.requires_grad} "
                      f"a_t.rg={a_t.requires_grad} out.rg={out.requires_grad}")
                print(f"[stage0][blend-probe] Lambda module={type(lam_mod).__name__} "
                      f"(orig={dispatched}) trainable_inside={trainable}")
                print(f"[stage0][blend-probe] disable_adapters="
                      f"{getattr(lam_mod, 'disable_adapters', 'n/a')} "
                      f"active={getattr(lam_mod, 'active_adapter', 'n/a')} "
                      f"keys={list(getattr(lam_mod, 'modules_to_save', {}) or {})}")
            return out, a_t
        _inner.thinking_residual = _counting_thinking_residual
        freeze_gate_modules(model, ["thinking_residual_gate_r",
                                    "thinking_residual_gate_i"])
    else:
        # Belt and braces: with no thinking_mask ever passed the blend cannot
        # fire anyway, but this makes is_thinking False by construction too.
        model.answer_start = ""

    dataset = build_dataset(tokenizer, max_len, dataset=args.dataset,
                            max_prompt=args.max_prompt_length,
                            max_completion=args.max_completion_length,
                            math_root=args.math_root)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    # Think-span diagnostics. If this mask is empty the Eq.4 blend never fires,
    # Lambda receives no gradient and Stage 0 silently degenerates to plain SFT.
    _n = min(200, len(dataset))
    _cnt = [sum(dataset[i]['think']) for i in range(_n)]
    _len = [len(dataset[i]['input_ids']) for i in range(_n)]
    _zero = sum(1 for c in _cnt if c == 0)
    print(f"[stage0] think-span over {_n} examples: mean {sum(_cnt)/_n:.1f} tokens, "
          f"min {min(_cnt)}, max {max(_cnt)}; mean seq len {sum(_len)/_n:.1f}; "
          f"examples with EMPTY think span: {_zero}/{_n}")
    if blend_on and _zero == _n:
        raise RuntimeError('every think span is empty — the blend can never fire')
    if blend_on and args.freeze_lambda:
        # check 2: with Lambda frozen it must be bit-identical after N steps.
        get_inner_model(model).thinking_residual_Lambda.requires_grad_(False)

    targs = TrainingArguments(
        output_dir=out_dir,
        learning_rate=args.lr,                      # LoRA lr (Lambda gets its own group)
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optimizer,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        report_to=args.report_to,
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Stage0Trainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=lambda f: collate(f, tokenizer.pad_token_id),
        blend=(args.blend == "on"),
        latent_temperature=args.temperature,
        proj_chunk=args.proj_chunk,
    )
    # Same param-group split as RL: LoRA at args.lr, Lambda at 1e-3.
    patch_trainer_optimizer(trainer, args.lr_residual_gate, args.lr_residual_Lambda)
    if blend_on:
        # Lambda must actually be trainable AND in the optimizer, or Stage 0
        # silently degenerates to plain SFT with a decorative blend. The first
        # smoke run showed Lambda bit-identical before/after with a null grad
        # norm, which is exactly what that failure looks like.
        # Skipped under --freeze_lambda, where requires_grad=False is the point.
        _named = [n for n, p in trainer.model.named_parameters()
                  if "thinking_residual_Lambda" in n and p.requires_grad]
        print(f"[stage0] trainable Lambda params: {_named}")
        _lam_ids = {id(p) for n, p in trainer.model.named_parameters()
                    if "thinking_residual_Lambda" in n}
        trainer.create_optimizer()
        _in_opt = sum(1 for g in trainer.optimizer.param_groups
                      for p in g["params"] if id(p) in _lam_ids)
        print(f"[stage0] Lambda tensors in optimizer: {_in_opt}")
        if not args.freeze_lambda:
            if not _named:
                raise RuntimeError("Lambda is not trainable — it could never move")
            if _in_opt == 0:
                raise RuntimeError("Lambda is not in any optimizer param group")
        _lam_p = next(p for n, p in trainer.model.named_parameters()
                      if "thinking_residual_Lambda" in n
                      and not n.endswith(".u") and p.requires_grad)
        def _lam_hook(g):
            LAMBDA_GRAD["hits"] += 1
            LAMBDA_GRAD["max_norm"] = max(
                LAMBDA_GRAD["max_norm"], float(g.detach().float().norm()))
            return g
        _lam_p.register_hook(_lam_hook)
        trainer.add_callback(Stage0Logger(trainer.model, out_dir))

    print(f"[stage0] blend={args.blend} r_min={args.residual_r_min} "
          f"seed={args.seed} -> {out_dir}")
    if blend_on:
        print(f"[stage0] initial {lambda_diagnostics(trainer.model)}")
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    if blend_on:
        print(f"[stage0] Lambda grad arrivals: {LAMBDA_GRAD['hits']}, "
              f"max |grad| = {LAMBDA_GRAD['max_norm']:.3e}")
        if LAMBDA_GRAD["hits"] == 0:
            raise RuntimeError(
                "no gradient EVER reached Lambda — the blend is decorative and "
                "this warm start would be worthless (check nothing re-introduced "
                "attention_mask into the model call; see llama.py:707)")
        print(f"[stage0] blend executions: {BLEND_CALLS['n']} calls, "
              f"{BLEND_CALLS['masked_rows']} masked rows blended")
        if BLEND_CALLS["n"] == 0:
            print("[stage0] ERROR: thinking_residual was NEVER called — the "
                  "forward is not reaching the blend, so Lambda cannot learn.")
    final = lambda_diagnostics(trainer.model) if blend_on else None
    print(f"[stage0] final {final}")
    with open(os.path.join(out_dir, "stage0_summary.json"), "w") as f:
        json.dump({"args": vars(args), "final_lambda": final}, f, indent=2)


if __name__ == "__main__":
    parser = build_arg_parser("stage0")
    parser.add_argument("--blend", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--proj_chunk", type=int, default=64,
                        help="positions per Eq.3 projection chunk (memory knob)")
    # Smoke-test knobs: verify the machinery on a handful of steps before
    # committing a full epoch of GPU time.
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N training examples")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="stop after N optimizer steps (-1 = full epoch)")
    parser.add_argument("--freeze_lambda", action="store_true",
                        help="verification check 2: Lambda must not move")
    # The blend happens at the EMBEDDING level, upstream of the checkpointed
    # decoder layers. If the checkpoint boundary does not propagate gradient to
    # its input, Lambda gets nothing while LoRA (inside the segments) trains
    # normally — exactly the observed signature. Default OFF for Stage 0:
    # memory is not the constraint here (80GB at batch 4) and correctness is.
    parser.add_argument("--grad_checkpoint", type=str, default="off",
                        choices=["off", "unsloth", "True"])
    # Stage 0 is supervised, not RL: the brief specifies a LoRA lr of 1e-5.
    parser.set_defaults(lr=1e-5)
    args = parser.parse_args()
    # build_arg_parser hands Stage 0 the RL flags too. --dr_grpo changes the GRPO
    # objective, which a supervised warm start does not have: refuse rather than
    # accept it and silently do nothing.
    if args.dr_grpo != "off":
        raise SystemExit("--dr_grpo applies to GRPO training only; Stage 0 is a "
                         "supervised warm start with no GRPO objective")
    main(args)
