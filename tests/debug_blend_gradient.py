"""Standalone gradient-path diagnostic for the Stage-0 blend.

Runs in seconds on one GPU and tests EVERY remaining hypothesis at once, instead
of one per cluster round-trip:

  * gradient checkpointing OFF vs "unsloth"
  * my Stage-0 style call (input_ids + inputs_embeds + thinking_mask +
    attention_mask) vs the RL log-prob style call (no attention_mask, with
    logits_to_keep) — the RL one demonstrably DOES train Lambda, so if it works
    here and mine does not, the difference is in the call, not the blend.

For each combination it reports where the autograd chain breaks:
  grad(loss, Lambda)        None => Lambda not connected to the loss
  grad(loss, blend_output)  None => break is DOWNSTREAM of the blend
  grad(blend_output, Lambda) None => break is INSIDE the blend

Run it as a batch job (needs HF_HOME + proxy):
    sbatch cluster/run_blend_debug.slurm
"""
import torch

import unsloth
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from variant_common import (
    LORA_TARGETS, MODEL_NAMES, apply_unit_gate_patch, get_inner_model,
)
from utils import ANSWER_START

SIZE, B, S = "1.5b", 2, 24
CAPTURE = {}


def build(grad_ckpt):
    model, tok = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAMES[SIZE], max_seq_length=128,
        load_in_4bit=False, load_in_8bit=False, fast_inference=False,
    )
    model.answer_start = ANSWER_START
    tok.pad_token = tok.pad_token or tok.eos_token
    model = FastLanguageModel.get_peft_model(
        model, r=8, target_modules=LORA_TARGETS["full"],
        modules_to_save=["thinking_residual_Lambda"],
        lora_alpha=16, use_gradient_checkpointing=grad_ckpt, random_state=0,
    )
    model.model.model.thinking_residual_Lambda.reset_lambda_parameters(
        r_min=0.98, r_max=0.999)
    apply_unit_gate_patch(model)

    # capture the blend output so we can bisect blend->loss vs blend->Lambda
    inner = get_inner_model(model)
    orig = inner.thinking_residual

    def capturing(embeds, residual, eps=1e-8):
        out, a_t = orig(embeds, residual, eps)
        CAPTURE["out"] = out
        CAPTURE["a_t"] = a_t
        return out, a_t
    inner.thinking_residual = capturing
    model.train()
    return model, tok


def lam_param(model):
    return next(p for n, p in model.named_parameters()
                if "thinking_residual_Lambda" in n and p.requires_grad)


def probe(model, style):
    CAPTURE.clear()
    dev = next(model.parameters()).device
    ids = torch.randint(10, 1000, (B, S), device=dev)
    mask = torch.zeros(B, S, dtype=torch.bool, device=dev)
    mask[:, 5:15] = True                      # a dense think span
    latents = torch.randn(B, S, model.config.hidden_size, device=dev,
                          dtype=next(model.parameters()).dtype)

    kw = dict(input_ids=ids, inputs_embeds=latents, thinking_mask=mask)
    if style == "stage0":
        kw["attention_mask"] = torch.ones(B, S, dtype=torch.long, device=dev)
    else:                                     # RL log-prob style
        kw["logits_to_keep"] = S

    out = model(**kw)
    logits = out.logits.float()
    loss = logits.log_softmax(-1).mean()

    lam = lam_param(model)
    blend = CAPTURE.get("out")
    res = {"blend_called": blend is not None,
           "loss_rg": bool(loss.requires_grad)}
    res["g_loss_lambda"] = torch.autograd.grad(
        loss, lam, retain_graph=True, allow_unused=True)[0] is not None
    if blend is not None:
        res["g_loss_blend"] = torch.autograd.grad(
            loss, blend, retain_graph=True, allow_unused=True)[0] is not None
        res["g_blend_lambda"] = torch.autograd.grad(
            blend.sum(), lam, retain_graph=True, allow_unused=True)[0] is not None
    return res


def main():
    print(f"{'grad_ckpt':<10} {'call':<8} {'blend?':<7} {'loss.rg':<8} "
          f"{'g(loss,Λ)':<10} {'g(loss,blend)':<14} {'g(blend,Λ)':<11}")
    for gc in (False, "unsloth"):
        model, _ = build(gc)
        for style in ("stage0", "rl"):
            try:
                r = probe(model, style)
                print(f"{str(gc):<10} {style:<8} {str(r['blend_called']):<7} "
                      f"{str(r['loss_rg']):<8} {str(r['g_loss_lambda']):<10} "
                      f"{str(r.get('g_loss_blend')):<14} "
                      f"{str(r.get('g_blend_lambda')):<11}")
            except Exception as exc:                      # noqa: BLE001
                print(f"{str(gc):<10} {style:<8} FAILED: {type(exc).__name__}: {exc}")
        del model
        torch.cuda.empty_cache()
    print("\nReading: g(loss,Λ)=True is the goal. If g(loss,blend)=False the break "
          "is downstream of the blend; if g(blend,Λ)=False it is inside it.")


if __name__ == "__main__":
    main()
