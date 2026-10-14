"""
weight_quant.py - weight-only post-training quantisation of a trained checkpoint, simulated.

Question it serves: does a hybrid latent model (HRPO, v3) lose more or less than a token-only
GRPO model (v1) when its weights are quantised for deployment, in accuracy and in reasoning
length? Lotfi et al. (arXiv 2606.00206) find that post-training quantisation lowers accuracy and
lengthens chains of thought in token reasoning models, with the quantisation error concentrated at
high-entropy positions; those are the positions where HRPO's Eq.3 latent differs from the token.

What is done, in order, after the adapter is loaded and before FastLanguageModel.for_inference:
  1. every LoRA adapter on a transformer-block projection (q, k, v, o, gate, up, down) is merged
     into its base weight, as it would be for deployment;
  2. each of those merged weights is replaced by quantise -> dequantise under the chosen scheme.
Embeddings (tied with lm_head in Qwen2.5-1.5B/3B), norms, biases and the HRPO gate parameters stay
in bf16, as in GPTQ/AWQ-style weight-only quantisation, so the latent h = W_e^T p / ||p|| is built
from unquantised embeddings and quantisation reaches it only through p.

Simulated ("fake") quantisation reproduces the numerical effect of the quantised weights exactly
(up to the bf16 rounding of the dequantised values) but not the memory or speed of a quantised
kernel. Weight memory is therefore reported from the storage format (effective_bits), not measured.

Schemes:
  bf16  merge only: the reference every quantised scheme is compared with (merging alone changes
        bf16 rounding, and greedy decoding is sensitive to that, so the unmerged eval_results.json
        is not a fair reference);
  int8 / int4 / int3
        asymmetric round-to-nearest (RTN) per group of 128 input weights, min-max range, stored
        as an fp16 scale and b-bit zero point per group: the RTN baseline of the GPTQ and AWQ
        papers (rtn_quantise follows GPTQ's reference quantiser);
  nf4   bitsandbytes NF4, blocksize 64, double-quantised statistics: what QLoRA and
        FastLanguageModel.from_pretrained(load_in_4bit=True) use (unsloth/models/llama.py).
"""

import re

import torch

SCHEMES = ("bf16", "int8", "int4", "int3", "nf4")
RTN_BITS = {"int8": 8, "int4": 4, "int3": 3}
GROUP_SIZE = 128
NF4_BLOCKSIZE = 64
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
TARGET_RE = re.compile(r"(?:^|\.)layers\.\d+\.(?:self_attn|mlp)\.(" + "|".join(PROJECTIONS) + r")$")


def effective_bits(scheme):
    """Stored bits per quantised weight, including the per-group statistics."""
    if scheme == "bf16":
        return 16.0
    if scheme in RTN_BITS:
        b = RTN_BITS[scheme]
        return b + (16 + b) / GROUP_SIZE              # fp16 scale + b-bit zero point per group
    if scheme == "nf4":
        # QLoRA (Dettmers et al., 2023), double quantisation: 8-bit absmax per block of 64 plus
        # one fp32 constant per 256 blocks.
        return 4 + 8 / NF4_BLOCKSIZE + 32 / (NF4_BLOCKSIZE * 256)
    raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")


def rtn_quantise(w, bits, group_size=GROUP_SIZE):
    """Asymmetric min-max round-to-nearest per group of `group_size` consecutive input weights.
    Returns the dequantised weight in w's dtype and shape.

    This is GPTQ's Quantizer (github.com/IST-DASLab/gptq, quant.py: find_params with sym=False,
    then quantize) applied to each group: the range is widened to include 0, an all-zero group
    gets the range [-1, 1], and the arithmetic is fp32. On a group whose weights straddle 0, as
    every group of a trained projection does, it gives the same grid as AWQ's
    pseudo_quantize_tensor (llm-awq, awq/quantize/quantizer.py). The scale is not rounded to fp16
    here; the cast of the result to the model's bf16 rounds more coarsely than that would."""
    out_f, in_f = w.shape
    if in_f % group_size:
        raise ValueError(f"in_features {in_f} is not a multiple of the group size {group_size}")
    g = w.float().reshape(out_f, in_f // group_size, group_size)
    xmin = g.amin(-1, keepdim=True).clamp(max=0)
    xmax = g.amax(-1, keepdim=True).clamp(min=0)
    empty = (xmin == 0) & (xmax == 0)
    xmin = torch.where(empty, torch.full_like(xmin, -1.0), xmin)
    xmax = torch.where(empty, torch.full_like(xmax, 1.0), xmax)
    qmax = 2 ** bits - 1
    scale = (xmax - xmin) / qmax
    zero = torch.round(-xmin / scale)
    q = torch.clamp(torch.round(g / scale) + zero, 0, qmax)
    return (scale * (q - zero)).reshape(out_f, in_f).to(w.dtype)


def nf4_quantise(w):
    import bitsandbytes.functional as bnbf
    q, state = bnbf.quantize_4bit(w.contiguous(), blocksize=NF4_BLOCKSIZE,
                                  compress_statistics=True, quant_type="nf4")
    return bnbf.dequantize_4bit(q, state).to(w.dtype).reshape(w.shape)


def quantise_weight(w, scheme):
    if scheme == "bf16":
        return w
    if scheme in RTN_BITS:
        return rtn_quantise(w, RTN_BITS[scheme])
    if scheme == "nf4":
        return nf4_quantise(w)
    raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")


def apply_weight_quant(model, scheme, num_layers=None):
    """Merge the LoRA adapters of every block projection and quantise the merged weights in
    place. Returns a stats dict for the results file. Raises if the number of projections found
    is not 7 per layer, so a pattern that matched nothing can never pass off a bf16 evaluation as
    a quantised one."""
    effective_bits(scheme)                                        # validates the name early
    if num_layers is None:
        num_layers = model.config.num_hidden_layers
    targets = [(n, m) for n, m in model.named_modules() if TARGET_RE.search(n)]
    if len(targets) != len(PROJECTIONS) * num_layers:
        raise RuntimeError(f"weight_quant: found {len(targets)} block projections, expected "
                           f"{len(PROJECTIONS) * num_layers} ({num_layers} layers x 7)")
    n_quant, n_merged, rel_err = 0, 0, []
    with torch.no_grad():
        for name, module in targets:
            if hasattr(module, "merge") and getattr(module, "lora_A", None):
                if not module.merged:
                    module.merge()
                if not module.merged:
                    raise RuntimeError(f"weight_quant: LoRA on {name} did not merge")
                n_merged += 1
            weight = getattr(module, "base_layer", module).weight
            new = quantise_weight(weight.data, scheme)
            if scheme != "bf16":
                rel_err.append(((new.float() - weight.data.float()).norm()
                                / weight.data.float().norm().clamp(min=1e-12)).item())
                weight.data.copy_(new)
            n_quant += weight.numel()
    # Deployed weight memory: LoRA factors are gone after merging, PEFT's untrained copies of the
    # modules_to_save (the HRPO gate) are not deployed, and everything else stays bf16.
    n_other = sum(p.numel() for n, p in model.named_parameters()
                  if "lora_" not in n and ".original_module." not in n) - n_quant
    bits = effective_bits(scheme)
    return {
        "scheme": scheme,
        "projections": len(targets),
        "lora_merged": n_merged,
        "quantised_params": n_quant,
        "other_params_bf16": n_other,
        "effective_bits": round(bits, 4),
        "weight_gb": round((n_quant * bits + n_other * 16) / 8 / 1e9, 4),
        "weight_gb_bf16": round((n_quant + n_other) * 2 / 1e9, 4),
        "mean_relative_error": round(sum(rel_err) / len(rel_err), 6) if rel_err else 0.0,
    }
