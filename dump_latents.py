"""Dump per-think-position latent representations for PCA-based feature-collapse analysis
(LT-Tuning Figure 5 style).

For each generated sample we collect, at every latent (<thinking>) position, three vectors:
  - e_pred : the predictive vocabulary interpolation (HRPO Eq. 3)          -> Soft-Thinking style
  - h_I    : the raw contextual hidden state at layer I (Change A, fusion)  -> Coconut style
  - latent : the actual fused latent INPUT embedding fed to the next step   -> LT-Tuning style
             i.e. thinking_residual(e_hat, fuse(e_pred, h_I)).
Reconstructed with the model's OWN fuse + gate so it matches training exactly.

Saves an .npz (and optional 3D PCA coords) so feature collapse across reasoning steps can be
visualized offline (whether latents from different samples converge at later step indices).

Run via sbatch (needs the model on GPU). Fusion flags must match the trained run.
"""
import unsloth
from unsloth import FastLanguageModel

import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import GenerationConfig

from utils import ANSWER_START, SYSTEM_PROMPT, resolve_model_path


def dump_latents(model_path, adapter_path, out_path, num_samples=20, temperature=0.5,
                 max_new_tokens=256, fusion_enabled=False, fusion_layer_I=-2,
                 fusion_gate_input="hidden", fusion_normalize_h=True, run_pca=True):
    resolved, use_exact = resolve_model_path(model_path)  # offline-safe local snapshot
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=resolved, max_seq_length=1024, load_in_4bit=False, fast_inference=False,
        use_exact_model_name=use_exact,
    )
    model.answer_start = ANSWER_START
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    if adapter_path:
        model.load_adapter(adapter_path)

    # Configure fusion on the inner model (must match the trained arm).
    inner = model.model.model
    inner.fusion_enabled = fusion_enabled
    inner.fusion_layer_I = fusion_layer_I
    inner.fusion_gate_input = fusion_gate_input
    inner.fusion_normalize_h = fusion_normalize_h
    inner.fusion_use_adapter = getattr(inner, "fusion_use_adapter", False)

    dataset = load_dataset('openai/gsm8k', 'main')['test'].shuffle(seed=42).select(range(num_samples))

    all_latent, all_epred, all_hI, all_sample, all_step = [], [], [], [], []

    for s in range(num_samples):
        q = dataset[s]['question'].strip()
        prompt = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': q}],
            tokenize=False, add_generation_prompt=True,
        )
        enc = tokenizer(prompt, return_tensors="pt", padding=True, padding_side="left",
                        add_special_tokens=False)
        prompt_ids = enc["input_ids"].to(model.device)
        prompt_mask = enc["attention_mask"].to(model.device)

        seqs, thinking_embeds, thinking_mask, embeds_ratio, thinking_hidden = model.generate(
            prompt_ids, attention_mask=prompt_mask,
            generation_config=GenerationConfig(do_sample=True, temperature=temperature,
                                               max_new_tokens=max_new_tokens),
            processing_class=tokenizer, return_thinking_embeds=True,
        )

        # Reconstruct the latent input embeddings at think positions using the model's own methods.
        with torch.inference_mode():
            e_hat = model.get_input_embeddings()(seqs).to(thinking_embeds.dtype)
            e_pred = thinking_embeds
            if fusion_enabled and thinking_hidden is not None:
                residual, _ = inner.fuse_context_prediction(e_pred, thinking_hidden, e_hat=e_hat)
                h_I = thinking_hidden
            else:
                residual = e_pred
                h_I = torch.zeros_like(e_pred)
            latent, _ = inner.thinking_residual(e_hat, residual)

        mask = thinking_mask[0].bool().cpu().numpy()           # (T,)
        idxs = np.nonzero(mask)[0]
        for step, p in enumerate(idxs):
            all_latent.append(latent[0, p].float().cpu().numpy())
            all_epred.append(e_pred[0, p].float().cpu().numpy())
            all_hI.append(h_I[0, p].float().cpu().numpy())
            all_sample.append(s)
            all_step.append(step)

    data = dict(
        latent=np.asarray(all_latent), e_pred=np.asarray(all_epred), h_I=np.asarray(all_hI),
        sample_id=np.asarray(all_sample), step_idx=np.asarray(all_step),
    )

    if run_pca and len(all_latent) > 3:
        # Simple 3-component PCA on each representation for a quick collapse check.
        for key in ("latent", "e_pred", "h_I"):
            X = data[key].astype(np.float64)
            X = X - X.mean(0, keepdims=True)
            # SVD-based PCA (top 3 comps).
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            data[f"{key}_pca3"] = (U[:, :3] * S[:3])

    np.savez(out_path, **data)
    print(f"Saved {len(all_latent)} latent vectors from {num_samples} samples to {out_path}")
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter_path", type=str, default=None)
    p.add_argument("--out_path", type=str, default="latents.npz")
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--fusion_enabled", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--fusion_layer_I", type=int, default=-2)
    p.add_argument("--fusion_gate_input", type=str, default="hidden", choices=["hidden", "token"])
    p.add_argument("--fusion_normalize_h", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    dump_latents(
        args.model_path, args.adapter_path, args.out_path, args.num_samples, args.temperature,
        args.max_new_tokens, args.fusion_enabled, args.fusion_layer_I, args.fusion_gate_input,
        args.fusion_normalize_h,
    )
