"""
hrpo_overhead.py
================
Measures the computational overhead introduced by the HRPO latent reasoning
gate relative to a frozen baseline (standard GRPO / plain Qwen inference).

Three measurements:
  1. Parameter overhead  — extra trainable params from gate vs LoRA alone
  2. FLOPs overhead      — extra multiply-adds per forward pass from the gate
  3. Wall-clock overhead — actual time difference per token, batch, and epoch

All results are saved to overhead_report.json and printed as a table.

Usage — single model
---------------------
python hrpo_overhead.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --checkpoint_path experiments/Qwen2.5-1.5B-Instruct-gsm8k-group4-lora32-rmin0.99-temp0.5 \
    --seq_len 512 \
    --batch_size 8 \
    --n_timing_runs 50

Usage — all three Qwen model sizes
-------------------------------------
python hrpo_overhead.py --all_scales --seq_len 512 --batch_size 4 --n_timing_runs 30

The script freezes the base model and LoRA weights so only the gate is active,
letting you isolate its contribution precisely.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import math
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_parameters(model, requires_grad_only: bool = False) -> dict:
    """
    Count parameters broken down by component.
    Returns a dict of {component_name: param_count}.
    """
    counts = {}
    for name, param in model.named_parameters():
        if requires_grad_only and not param.requires_grad:
            continue
        # Assign to component bucket
        if "thinking_residual_Lambda" in name:
            key = "gate_Lambda"
        elif "thinking_residual_gate_r" in name:
            key = "gate_r"
        elif "thinking_residual_gate_i" in name:
            key = "gate_i"
        elif "lora_A" in name or "lora_B" in name:
            key = "lora"
        elif "modules_to_save" in name:
            key = "modules_to_save_other"
        else:
            key = "base_model"
        counts[key] = counts.get(key, 0) + param.numel()
    return counts


def freeze_all_except_gate(model) -> list:
    """
    Freeze everything except thinking_residual_* parameters.
    Returns list of (name, original_requires_grad) for restoration.
    """
    states = []
    for name, param in model.named_parameters():
        states.append((name, param.requires_grad))
        if "thinking_residual" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    return states


def restore_grad_states(model, states: list):
    """Restore requires_grad states saved by freeze_all_except_gate."""
    state_dict = dict(states)
    for name, param in model.named_parameters():
        if name in state_dict:
            param.requires_grad_(state_dict[name])


@contextmanager
def timer(label: str, results: dict):
    """Context manager that records elapsed time in seconds."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    yield
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start
    results[label] = elapsed


# ---------------------------------------------------------------------------
# FLOPs estimation
# ---------------------------------------------------------------------------

def estimate_gate_flops(model, seq_len: int, batch_size: int) -> dict:
    """
    Estimate FLOPs added by the HRPO gate per forward pass.

    SOURCE FOR FLOP COUNTING CONVENTION:
    We use multiply-accumulate (MAC) counting where one MAC = 2 FLOPs
    (one multiply + one add).  This is the convention used in:
      - Kaplan et al. (2020) "Scaling Laws for Neural Language Models"
        https://arxiv.org/abs/2001.08361  (Section 2.1, Appendix D)
      - Hoffmann et al. (2022) "Training Compute-Optimal Large Language Models"
        https://arxiv.org/abs/2203.15556
    Both use 6*N*D as the FLOPs estimate for a transformer forward+backward
    pass (factor of 6 = 2 for MACs * 3 for forward+backward).  We use 6*N*D
    for the baseline to match this convention.

    GATE FLOP DERIVATION (from HRPO paper Equations 3 and 4,
    Yue et al. 2025, arXiv:2505.18454):

    Equation 4 gate operations per think-span token:

    1. gate_r linear (W_a @ e_{t+1} + b_a):
       A matrix-vector multiply of shape (hidden_dim, hidden_dim).
       FLOPs = 2 * hidden_dim * hidden_dim  [standard linear layer formula]

    2. gate_i linear (W_x @ e_{t+1} + b_x):
       Same shape as gate_r.
       FLOPs = 2 * hidden_dim * hidden_dim

    3. Two sigmoid activations (r_t and i_t):
       sigmoid(x) = 1 / (1 + exp(-x)) requires: negation, exp, add, divide
       = 4 FLOPs per element, times 2 gates, times hidden_dim elements.
       FLOPs = 2 * hidden_dim * 4
       Note: this is an approximation — hardware may fuse these ops.

    4. softplus(Lambda) + exp for a_t:
       softplus(x) = log(1 + exp(x)) requires: exp, add, log = ~3 FLOPs
       Then multiply by -c (1 FLOP), then outer exp (1 FLOP) = ~5 total.
       We use 8 as a conservative upper bound including the element-wise
       multiply with r_t (one more FLOP per element).
       FLOPs = hidden_dim * 8

    5. Blend operation (a_t ⊙ e_{t+1} + sqrt(1-a_t^2) ⊙ (i_t ⊙ h_{t+1})):
       sqrt: 1, square: 1, subtract: 1, three element-wise multiplies: 3,
       one add: 1 = 7 FLOPs per element. We use 6 as a lower bound
       (sqrt may be fused with the subtract on modern hardware).
       FLOPs = hidden_dim * 6

    Equation 3 projection operation per think-span token:

    6. Projection h_{t+1} = W_e^T * (p_{t+1} / ||p_{t+1}||):
       First compute p_{t+1} = softmax(Head(h_t) / tau):
         - Head(h_t): hidden_dim -> vocab_size linear = 2*vocab_size*hidden_dim
           (but this is already done in the normal forward pass, not extra)
         - divide by tau: vocab_size FLOPs
         - softmax exp + normalise: ~3 * vocab_size FLOPs
       Then weighted sum over embedding matrix W_e (vocab_size, hidden_dim):
         - matrix-vector: 2 * vocab_size * hidden_dim FLOPs
       Then normalise: hidden_dim FLOPs (negligible)
       Total new FLOPs = vocab_size * 3 + 2 * vocab_size * hidden_dim
       The 2*vocab_size*hidden_dim term dominates since hidden_dim >> 3.

    THINK SPAN FRACTION:
    We assume 80% of tokens are in the think span (before ####).
    This is conservative — actual fraction depends on response length.
    From the HRPO paper Figure 4, completion lengths are ~150-250 tokens
    with most being reasoning. 80% is used as a documented assumption.
    """
    # ---- Read model dimensions from the embedding layer --------------------
    # Priority 1: read directly from embedding weight shape (most reliable)
    hidden_dim = None
    vocab_size = None

    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight.dim() == 2:
            if "embed_tokens" in name or "embedding" in name.lower():
                vocab_size = module.weight.shape[0]
                hidden_dim = module.weight.shape[1]
                print(f"  Dimensions from embedding layer '{name}': "
                      f"hidden_dim={hidden_dim}, vocab_size={vocab_size}")
                break

    # Priority 2: read from model.config (HuggingFace standard attribute)
    if hidden_dim is None:
        try:
            # Access config through various common paths
            cfg = getattr(model, "config", None) or \
                  getattr(getattr(model, "base_causallm", None), "config", None) or \
                  getattr(getattr(model, "model", None), "config", None)
            if cfg is not None:
                hidden_dim = cfg.hidden_size
                vocab_size = cfg.vocab_size
                print(f"  Dimensions from model.config: "
                      f"hidden_dim={hidden_dim}, vocab_size={vocab_size}")
        except AttributeError:
            pass

    # Priority 3: fail loudly rather than silently use wrong values
    if hidden_dim is None:
        raise RuntimeError(
            "Could not determine hidden_dim and vocab_size from the model. "
            "Neither the embedding layer nor model.config were accessible. "
            "Please check that the model loaded correctly."
        )

    # ---- FLOPs per think-span token ----------------------------------------
    think_frac = 0.8   # documented assumption: 80% of tokens in think span
    think_tokens = int(seq_len * think_frac)

    # Eq 4: gate linear layers
    gate_r_flops  = 2 * hidden_dim * hidden_dim   # W_a @ e_{t+1}
    gate_i_flops  = 2 * hidden_dim * hidden_dim   # W_x @ e_{t+1}
    # Eq 4: activation functions (approximations, see docstring)
    sigmoid_flops = 2 * hidden_dim * 4            # two sigmoids
    lambda_flops  = hidden_dim * 8                # softplus + exp + multiply
    blend_flops   = hidden_dim * 6                # sqrt, multiply, add
    # Eq 3: projection back onto embedding manifold
    proj_softmax  = vocab_size * 3                # divide by tau, exp, normalise
    proj_matmul   = 2 * vocab_size * hidden_dim  # weighted sum over W_e

    per_token_flops = (
        gate_r_flops + gate_i_flops + sigmoid_flops +
        lambda_flops + blend_flops + proj_softmax + proj_matmul
    )

    total_gate_flops = per_token_flops * think_tokens * batch_size

    # ---- Baseline transformer FLOPs ----------------------------------------
    # 6 * N * D convention from Kaplan et al. (2020), where N = parameter
    # count and D = sequence length.  This estimates the total forward pass
    # FLOPs and is used as the denominator for gate_overhead_percent.
    n_params = sum(p.numel() for p in model.parameters())
    baseline_flops = 6 * n_params * seq_len * batch_size

    return {
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "think_span_fraction_assumed": think_frac,
        "think_tokens_per_seq": think_tokens,
        "gate_r_flops_per_token": gate_r_flops,
        "gate_i_flops_per_token": gate_i_flops,
        "projection_flops_per_token": proj_softmax + proj_matmul,
        "blend_flops_per_token": sigmoid_flops + lambda_flops + blend_flops,
        "total_gate_flops_per_token": per_token_flops,
        "total_gate_flops_per_batch": total_gate_flops,
        "baseline_transformer_flops_per_batch": baseline_flops,
        "gate_overhead_percent": 100 * total_gate_flops / max(baseline_flops, 1),
        "flop_counting_convention": "MAC (1 multiply-add = 2 FLOPs), Kaplan et al. 2020",
        "baseline_formula": "6 * n_params * seq_len * batch_size",
    }


# ---------------------------------------------------------------------------
# Wall-clock timing
# ---------------------------------------------------------------------------

def _time_model(model, tokenizer, seq_len, batch_size, n_runs, device, label=""):
    """
    Time a model's forward pass with OOM-safe batch size reduction.
    Device is inferred from the model's first parameter so this works
    correctly with unsloth models that manage their own device placement.
    Returns dict with mean_ms, std_ms, batch_size.
    """
    model.eval()

    # Infer device from model — safer than trusting the passed device arg
    # for unsloth models which may have layers on different devices
    try:
        inferred_device = next(model.parameters()).device
    except StopIteration:
        inferred_device = torch.device(device)

    actual_batch_size = batch_size
    dummy_ids, dummy_mask = None, None

    while actual_batch_size >= 1:
        try:
            torch.cuda.empty_cache() if inferred_device.type != "cpu" else None
            dummy_ids = torch.randint(
                0, tokenizer.vocab_size,
                (actual_batch_size, seq_len), device=inferred_device
            )
            dummy_mask = torch.ones(
                actual_batch_size, seq_len,
                device=inferred_device, dtype=torch.long
            )
            with torch.no_grad():
                _ = model(input_ids=dummy_ids, attention_mask=dummy_mask)
            torch.cuda.synchronize() if inferred_device.type != "cpu" else None
            if actual_batch_size != batch_size:
                print(f"  [{label}] OOM at batch={batch_size}, "
                      f"using batch={actual_batch_size}")
            else:
                print(f"  [{label}] batch_size={actual_batch_size} fits OK")
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            actual_batch_size //= 2
            dummy_ids, dummy_mask = None, None

    if dummy_ids is None:
        raise RuntimeError(
            f"Could not fit even batch_size=1 with seq_len={seq_len}."
        )

    times = []
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids=dummy_ids, attention_mask=dummy_mask)
        torch.cuda.synchronize() if inferred_device.type != "cpu" else None

        for _ in range(n_runs):
            torch.cuda.synchronize() if inferred_device.type != "cpu" else None
            t0 = time.perf_counter()
            _ = model(input_ids=dummy_ids, attention_mask=dummy_mask)
            torch.cuda.synchronize() if inferred_device.type != "cpu" else None
            times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "batch_size": actual_batch_size,
    }


def measure_wall_clock(
    model,
    tokenizer,
    seq_len: int,
    batch_size: int,
    n_runs: int,
    device: str,
) -> dict:
    """
    Measure wall-clock time for:
      - baseline: model with gate frozen to a_t = 1 (pure tokens, no latent)
      - hrpo:     model with gate active

    Uses random token IDs — GPU matrix multiplication time is determined by
    tensor shape, not content, so this is equivalent to real data for
    overhead measurement purposes.

    If the requested batch_size causes OOM, automatically halves it and
    retries until it fits or batch_size reaches 1.

    Returns mean and std of per-run times in milliseconds.
    """
    model.eval()

    # --- Find largest batch size that fits in GPU memory -------------------
    actual_batch_size = batch_size
    dummy_ids, dummy_mask = None, None

    while actual_batch_size >= 1:
        try:
            torch.cuda.empty_cache() if device != "cpu" else None
            dummy_ids, dummy_mask = _make_dummy_batch(
                tokenizer, actual_batch_size, seq_len, device
            )
            # Test forward pass before committing to full timing run
            with torch.no_grad():
                _ = model(input_ids=dummy_ids, attention_mask=dummy_mask)
            torch.cuda.synchronize() if device != "cpu" else None
            if actual_batch_size != batch_size:
                print(f"  OOM at batch_size={batch_size*2}, "
                      f"reduced to batch_size={actual_batch_size}")
            else:
                print(f"  batch_size={actual_batch_size} fits OK")
            break
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at batch_size={actual_batch_size}, halving...")
            torch.cuda.empty_cache()
            actual_batch_size //= 2
            dummy_ids, dummy_mask = None, None

    if dummy_ids is None:
        raise RuntimeError(
            f"Could not fit even batch_size=1 with seq_len={seq_len} "
            f"on this GPU. Try reducing --seq_len."
        )

    # --- HRPO timing (gate active) -----------------------------------------
    hrpo_times = []
    with torch.no_grad():
        for _ in range(3):   # warmup
            _ = model(input_ids=dummy_ids, attention_mask=dummy_mask)
        torch.cuda.synchronize() if device != "cpu" else None

        for _ in range(n_runs):
            torch.cuda.synchronize() if device != "cpu" else None
            t0 = time.perf_counter()
            _ = model(input_ids=dummy_ids, attention_mask=dummy_mask)
            torch.cuda.synchronize() if device != "cpu" else None
            hrpo_times.append((time.perf_counter() - t0) * 1000)

    # --- Baseline timing (gate mathematically disabled) --------------------
    baseline_times = []
    lambda_backup = {}

    try:
        for name, module in model.named_modules():
            if "thinking_residual_Lambda" in name and hasattr(module, "Lambda"):
                lambda_backup[name] = module.Lambda.data.clone()
                module.Lambda.data.fill_(-100.0)

        with torch.no_grad():
            for _ in range(3):   # warmup
                _ = model(input_ids=dummy_ids, attention_mask=dummy_mask)
            torch.cuda.synchronize() if device != "cpu" else None

            for _ in range(n_runs):
                torch.cuda.synchronize() if device != "cpu" else None
                t0 = time.perf_counter()
                _ = model(input_ids=dummy_ids, attention_mask=dummy_mask)
                torch.cuda.synchronize() if device != "cpu" else None
                baseline_times.append((time.perf_counter() - t0) * 1000)

    finally:
        for name, module in model.named_modules():
            if name in lambda_backup and hasattr(module, "Lambda"):
                module.Lambda.data.copy_(lambda_backup[name])

    hrpo_arr = np.array(hrpo_times)
    base_arr = np.array(baseline_times)

    return {
        "hrpo_mean_ms": float(hrpo_arr.mean()),
        "hrpo_std_ms": float(hrpo_arr.std()),
        "baseline_mean_ms": float(base_arr.mean()),
        "baseline_std_ms": float(base_arr.std()),
        "overhead_ms": float(hrpo_arr.mean() - base_arr.mean()),
        "overhead_percent": float(
            100 * (hrpo_arr.mean() - base_arr.mean()) / max(base_arr.mean(), 1e-6)
        ),
        "overhead_per_token_us": float(
            1000 * (hrpo_arr.mean() - base_arr.mean())
            / (seq_len * actual_batch_size)
        ),
        "n_runs": n_runs,
        "batch_size": actual_batch_size,
        "batch_size_requested": batch_size,
        "seq_len": seq_len,
        "data_source": "random",
    }


# ---------------------------------------------------------------------------
# Memory overhead
# ---------------------------------------------------------------------------

def measure_memory_overhead(model, tokenizer, seq_len: int, batch_size: int,
                             device: str, input_ids: torch.Tensor = None,
                             attention_mask: torch.Tensor = None) -> dict:
    """
    Measure peak GPU memory with gate active vs frozen.
    Only meaningful on CUDA.

    Key fix: torch.cuda.empty_cache() is called before each measurement to
    flush PyTorch's allocator cache.  Without this, the second measurement
    (baseline) finds tensors already allocated from the first (HRPO) run,
    making it appear to use LESS memory — producing a spurious negative
    overhead.  empty_cache() forces each run to allocate from scratch.

    We average over N_MEMORY_RUNS passes for stability.
    """
    if device == "cpu":
        return {"note": "Memory measurement requires CUDA"}

    if input_ids is None:
        input_ids = torch.randint(
            0, tokenizer.vocab_size, (batch_size, seq_len), device=device
        )
        attention_mask = torch.ones(
            batch_size, seq_len, device=device, dtype=torch.long
        )

    N_MEMORY_RUNS = 5
    results = {}

    # ---- HRPO active -------------------------------------------------------
    hrpo_peaks = []
    for _ in range(N_MEMORY_RUNS):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize()
        hrpo_peaks.append(torch.cuda.max_memory_allocated() / 1e6)

    results["hrpo_peak_mb"] = float(np.mean(hrpo_peaks))
    results["hrpo_peak_mb_std"] = float(np.std(hrpo_peaks))

    # ---- Baseline (gate mathematically disabled) ---------------------------
    lambda_backup = {}
    try:
        for name, module in model.named_modules():
            if "thinking_residual_Lambda" in name and hasattr(module, "Lambda"):
                lambda_backup[name] = module.Lambda.data.clone()
                module.Lambda.data.fill_(-100.0)

        baseline_peaks = []
        for _ in range(N_MEMORY_RUNS):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()
            baseline_peaks.append(torch.cuda.max_memory_allocated() / 1e6)

        results["baseline_peak_mb"] = float(np.mean(baseline_peaks))
        results["baseline_peak_mb_std"] = float(np.std(baseline_peaks))

    finally:
        for name, module in model.named_modules():
            if name in lambda_backup and hasattr(module, "Lambda"):
                module.Lambda.data.copy_(lambda_backup[name])

    results["memory_overhead_mb"] = (
        results["hrpo_peak_mb"] - results["baseline_peak_mb"]
    )
    results["memory_overhead_percent"] = (
        100 * results["memory_overhead_mb"]
        / max(results["baseline_peak_mb"], 1)
    )
    results["n_runs"] = N_MEMORY_RUNS
    results["note"] = (
        "Each run preceded by torch.cuda.empty_cache() to prevent "
        "allocator cache from inflating baseline measurements."
    )
    return results


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_report(param_counts: dict, flops: dict, timing: dict, memory: dict):
    sep = "=" * 60

    print(f"\n{sep}")
    print("  HRPO COMPUTATIONAL OVERHEAD REPORT")
    print(sep)

    print("\n── Parameter Counts ──────────────────────────────────────")
    total = sum(param_counts.values())
    gate_total = sum(v for k, v in param_counts.items() if k.startswith("gate"))
    lora_total = param_counts.get("lora", 0)
    base_total = param_counts.get("base_model", 0)

    for k, v in sorted(param_counts.items(), key=lambda x: -x[1]):
        pct = 100 * v / max(total, 1)
        print(f"  {k:<30} {v:>12,}  ({pct:5.2f}%)")
    print(f"  {'TOTAL':<30} {total:>12,}")
    print(f"\n  Gate params (r + i + Lambda):  {gate_total:>10,}")
    print(f"  LoRA params:                   {lora_total:>10,}")
    print(f"  Gate / LoRA ratio:             {gate_total/max(lora_total,1):>10.4f}x")
    print(f"  Gate / Total ratio:            {gate_total/max(total,1)*100:>9.4f}%")

    print("\n── FLOPs Overhead ────────────────────────────────────────")
    print(f"  Convention:                    {flops['flop_counting_convention']}")
    print(f"  Baseline formula:              {flops['baseline_formula']}")
    print(f"  Hidden dim:                    {flops['hidden_dim']:>10,}")
    print(f"  Vocab size:                    {flops['vocab_size']:>10,}")
    print(f"  Think span fraction assumed:   {flops['think_span_fraction_assumed']:>10.0%}")
    print(f"  Think tokens per seq:          {flops['think_tokens_per_seq']:>10,}")
    print(f"  Gate FLOPs per token:          {flops['total_gate_flops_per_token']:>10,}")
    print(f"    of which — projection:       {flops['projection_flops_per_token']:>10,}")
    print(f"    of which — gate r/i/blend:   {flops['gate_r_flops_per_token']+flops['gate_i_flops_per_token']+flops['blend_flops_per_token']:>10,}")
    print(f"  Total gate FLOPs per batch:    {flops['total_gate_flops_per_batch']:>10,.0f}")
    print(f"  Baseline transformer FLOPs:    {flops['baseline_transformer_flops_per_batch']:>10,.0f}")
    print(f"  Gate overhead:                 {flops['gate_overhead_percent']:>9.3f}%")

    print("\n── Wall-Clock Timing ─────────────────────────────────────")
    print(f"  Input data:                    random (dummy tokens)")
    actual_bs = timing['batch_size']
    requested_bs = timing.get('batch_size_requested', actual_bs)
    if actual_bs != requested_bs:
        print(f"  Batch size (requested):        {requested_bs}  →  reduced to {actual_bs} (OOM)")
    else:
        print(f"  Batch size / seq len:          {actual_bs} / {timing['seq_len']}")
    print(f"  Timing runs:                   {timing['n_runs']}")
    print(f"  HRPO forward:                  {timing['hrpo_mean_ms']:>8.2f} ± {timing['hrpo_std_ms']:.2f} ms")
    print(f"  Baseline forward:              {timing['baseline_mean_ms']:>8.2f} ± {timing['baseline_std_ms']:.2f} ms")
    print(f"  Overhead (absolute):           {timing['overhead_ms']:>8.2f} ms")
    print(f"  Overhead (relative):           {timing['overhead_percent']:>8.2f}%")
    print(f"  Overhead per token:            {timing['overhead_per_token_us']:>8.3f} µs")

    print("\n── Memory Overhead ───────────────────────────────────────")
    if "note" in memory and "requires CUDA" in memory["note"]:
        print(f"  {memory['note']}")
    else:
        n = memory.get("n_runs", 1)
        print(f"  Measurement runs:              {n} (avg of {n} passes)")
        print(f"  HRPO peak memory:     {memory['hrpo_peak_mb']:>8.1f} ± {memory.get('hrpo_peak_mb_std', 0):.1f} MB")
        print(f"  Baseline peak memory: {memory['baseline_peak_mb']:>8.1f} ± {memory.get('baseline_peak_mb_std', 0):.1f} MB")
        print(f"  Memory overhead:               {memory['memory_overhead_mb']:>8.1f} MB")
        print(f"  Memory overhead (relative):    {memory['memory_overhead_percent']:>8.2f}%")
        if "note" in memory:
            print(f"  Note: {memory['note']}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Cross-scale comparison
# ---------------------------------------------------------------------------

_QWEN_SCALE_MODELS = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b":   "Qwen/Qwen2.5-3B-Instruct",
}


def _discover_best_checkpoint(model_size: str) -> Optional[str]:
    """Find the highest-step checkpoint for a model size, preferring v5_hrpo_full."""
    import glob
    preferred_order = ["v5_hrpo_full", "v2_hrpo_gate_only"]
    exp_dirs = sorted(glob.glob(os.path.join("experiments", f"{model_size}-v*")))

    preferred = [d for d in exp_dirs if any(p in d for p in preferred_order)]
    search_order = preferred + [d for d in exp_dirs if d not in preferred]

    for exp_dir in search_order:
        if not os.path.isdir(exp_dir):
            continue
        candidates = []
        for d in os.listdir(exp_dir):
            full = os.path.join(exp_dir, d)
            if (os.path.isdir(full) and
                    os.path.isfile(os.path.join(full, "adapter_config.json"))):
                digits = "".join(c for c in d if c.isdigit())
                step = int(digits) if digits else 0
                candidates.append((step, full))
        if candidates:
            return max(candidates)[1]
    return None


def _run_isolated_timing(model_name, checkpoint_path,
                         seq_len, batch_size, n_runs, label):
    """
    Run a forward-pass timing loop in a fresh Python subprocess so each model
    gets a clean GPU context.  Returns dict with mean_ms, std_ms, batch_size.
    """
    script = f"""
import torch, numpy as np, json, time

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name={repr(model_name)},
        max_seq_length={seq_len},
        load_in_4bit=False,
        fast_inference=False,
    )
    if {repr(checkpoint_path)} is not None:
        model.load_adapter({repr(checkpoint_path)})
    model = FastLanguageModel.for_inference(model)
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
    raise

model.eval()

try:
    inferred_device = next(model.parameters()).device
except StopIteration:
    inferred_device = torch.device(device)

actual_batch = {batch_size}
dummy_ids = dummy_mask = None

while actual_batch >= 1:
    try:
        torch.cuda.empty_cache()
        dummy_ids = torch.randint(0, tokenizer.vocab_size,
                                  (actual_batch, {seq_len}),
                                  device=inferred_device)
        dummy_mask = torch.ones(actual_batch, {seq_len},
                                device=inferred_device, dtype=torch.long)
        with torch.no_grad():
            _ = model(input_ids=dummy_ids, attention_mask=dummy_mask)
        torch.cuda.synchronize()
        break
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        actual_batch //= 2
        dummy_ids = dummy_mask = None

if dummy_ids is None:
    print(json.dumps({{"error": "OOM at batch_size=1"}}))
    exit(1)

times = []
with torch.no_grad():
    for _ in range(3):
        _ = model(input_ids=dummy_ids, attention_mask=dummy_mask)
    torch.cuda.synchronize()
    for _ in range({n_runs}):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(input_ids=dummy_ids, attention_mask=dummy_mask)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

arr = np.array(times)
print(json.dumps({{
    "mean_ms": float(arr.mean()),
    "std_ms":  float(arr.std()),
    "batch_size": actual_batch,
    "label": {repr(label)},
}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True
    )
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    for line in reversed(lines):
        try:
            data = json.loads(line)
            if "error" in data:
                raise RuntimeError(
                    f"Subprocess error for {label}: {data['error']}\n"
                    f"stderr: {result.stderr[-500:]}"
                )
            return data
        except json.JSONDecodeError:
            continue
    raise RuntimeError(
        f"No JSON result from {label} subprocess.\n"
        f"stdout: {result.stdout[-500:]}\n"
        f"stderr: {result.stderr[-500:]}"
    )


def run_all_scales(args):
    """
    Measure gate overhead for all three Qwen model sizes sequentially.
    Each model is loaded, measured for params/FLOPs/memory, then freed before
    subprocess timing so no two models compete for GPU memory at once.

    Saves results/overhead_comparison.json and results/plots/overhead_comparison.png.
    """
    os.makedirs("results", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    for size, base_model in _QWEN_SCALE_MODELS.items():
        print(f"\n{'='*60}")
        print(f"  {size}  ({base_model})")
        print(f"{'='*60}")

        checkpoint_path = _discover_best_checkpoint(size)
        if checkpoint_path:
            print(f"  Checkpoint: {checkpoint_path}")
        else:
            print(f"  No checkpoint found — using base model")

        # ── Load model for param count, FLOPs, memory ─────────────────────
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model,
                max_seq_length=args.seq_len,
                load_in_4bit=False,
                fast_inference=False,
            )
            if checkpoint_path:
                model.load_adapter(checkpoint_path)
            model = FastLanguageModel.for_inference(model)
        except ImportError:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=torch.float16
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            if checkpoint_path:
                from peft import PeftModel
                model = PeftModel.from_pretrained(
                    model, checkpoint_path, local_files_only=True
                )
        model.eval()

        # Param counts
        param_counts = count_parameters(model, requires_grad_only=False)
        total_params = sum(param_counts.values())
        gate_params  = sum(v for k, v in param_counts.items() if k.startswith("gate"))
        lora_params  = param_counts.get("lora", 0)
        gate_pct     = 100.0 * gate_params / max(total_params, 1)

        # FLOPs
        print("  Estimating FLOPs...")
        flops = estimate_gate_flops(model, args.seq_len, args.batch_size)

        # Memory
        print("  Measuring memory overhead...")
        memory = measure_memory_overhead(
            model, tokenizer, args.seq_len, args.batch_size, device
        )

        # Free GPU memory before subprocess timing
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Subprocess timing ─────────────────────────────────────────────
        print("  Timing HRPO (subprocess)...")
        t_hrpo = _run_isolated_timing(
            base_model, checkpoint_path,
            args.seq_len, args.batch_size, args.n_timing_runs, "HRPO"
        )
        print(f"    HRPO: {t_hrpo['mean_ms']:.1f} ± {t_hrpo['std_ms']:.1f} ms")

        print("  Timing baseline (subprocess)...")
        t_base = _run_isolated_timing(
            base_model, None,
            args.seq_len, args.batch_size, args.n_timing_runs, "Base"
        )
        print(f"    Base: {t_base['mean_ms']:.1f} ± {t_base['std_ms']:.1f} ms")

        overhead_ms  = t_hrpo["mean_ms"] - t_base["mean_ms"]
        overhead_pct = 100.0 * overhead_ms / max(t_base["mean_ms"], 1e-6)

        row = {
            "model_size": size,
            "base_model": base_model,
            "checkpoint": checkpoint_path or "none",
            "total_params": total_params,
            "gate_params": gate_params,
            "lora_params": lora_params,
            "gate_param_pct": round(gate_pct, 4),
            "flops_overhead_pct": round(flops["gate_overhead_percent"], 4),
            "wall_clock_overhead_pct": round(overhead_pct, 2),
            "wall_clock_overhead_ms": round(overhead_ms, 3),
            "hrpo_mean_ms": round(t_hrpo["mean_ms"], 2),
            "baseline_mean_ms": round(t_base["mean_ms"], 2),
            "hidden_dim": flops["hidden_dim"],
            "vocab_size": flops["vocab_size"],
        }
        if isinstance(memory, dict) and "memory_overhead_percent" in memory:
            row["memory_overhead_pct"] = round(memory["memory_overhead_percent"], 2)
            row["memory_overhead_mb"]  = round(memory["memory_overhead_mb"], 1)

        rows.append(row)

        print(f"\n  {size} summary:")
        print(f"    Gate params:         {gate_params:>12,}  ({gate_pct:.4f}%)")
        print(f"    FLOPs overhead:      {flops['gate_overhead_percent']:>11.4f}%")
        print(f"    Wall-clock overhead: {overhead_pct:>11.2f}%")
        if "memory_overhead_pct" in row:
            print(f"    Memory overhead:     {row['memory_overhead_pct']:>11.2f}%")

    # ── Print table ───────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  Cross-Scale Overhead Comparison")
    print(f"{'='*72}")
    print(f"  {'Size':<8} {'Gate params%':>14} {'FLOPs%':>10} {'WallClock%':>12} {'Mem%':>8}")
    print(f"  {'-'*8} {'-'*14} {'-'*10} {'-'*12} {'-'*8}")
    for row in rows:
        mem = f"{row.get('memory_overhead_pct', float('nan')):>7.2f}%"
        print(f"  {row['model_size']:<8} {row['gate_param_pct']:>13.4f}% "
              f"{row['flops_overhead_pct']:>9.4f}% "
              f"{row['wall_clock_overhead_pct']:>11.2f}% {mem}")

    # ── Save JSON ─────────────────────────────────────────────────────────
    out_json = args.scales_output
    os.makedirs(os.path.dirname(out_json) if os.path.dirname(out_json) else ".", exist_ok=True)
    payload = {
        "config": {
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "n_timing_runs": args.n_timing_runs,
        },
        "rows": rows,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_json}")

    # ── Bar-chart plot ────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sizes = [r["model_size"] for r in rows]
        metrics = [
            ("Gate params (%)",          [r["gate_param_pct"] for r in rows]),
            ("FLOPs overhead (%)",        [r["flops_overhead_pct"] for r in rows]),
            ("Wall-clock overhead (%)",   [r["wall_clock_overhead_pct"] for r in rows]),
        ]
        if all("memory_overhead_pct" in r for r in rows):
            metrics.append(
                ("Memory overhead (%)", [r["memory_overhead_pct"] for r in rows])
            )

        n_m = len(metrics)
        fig, axes = plt.subplots(1, n_m, figsize=(4 * n_m, 5))
        if n_m == 1:
            axes = [axes]
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

        for ax, (label_, vals), color in zip(axes, metrics, colors):
            bars = ax.bar(sizes, vals, color=color, alpha=0.85, edgecolor="white")
            ax.set_title(label_, fontsize=10)
            ax.set_xlabel("Model size")
            ax.set_ylabel("%")
            mx = max(vals) if vals else 1
            ax.set_ylim(0, mx * 1.3)
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + mx * 0.03,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=9,
                )

        fig.suptitle("HRPO Gate Overhead by Model Scale", fontsize=13, fontweight="bold")
        plt.tight_layout()
        os.makedirs("results/plots", exist_ok=True)
        plot_path = "results/plots/overhead_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved: {plot_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")


# ---------------------------------------------------------------------------
# Main (single model)
# ---------------------------------------------------------------------------

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")
    print(f"Loading base model: {args.base_model}")

    # ---- Resolve checkpoint path ------------------------------------------
    checkpoint_path = None
    if args.checkpoint_path:
        checkpoint_path = os.path.abspath(args.checkpoint_path)
        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint directory not found: {checkpoint_path}"
            )

        # If adapter_config.json is directly here, use it as-is
        if os.path.isfile(os.path.join(checkpoint_path, "adapter_config.json")):
            print(f"Checkpoint resolved to: {checkpoint_path}")

        else:
            # Search all subdirectories for adapter_config.json and pick
            # the one with the highest step number.
            candidates = []
            for d in os.listdir(checkpoint_path):
                full = os.path.join(checkpoint_path, d)
                if (os.path.isdir(full) and
                        os.path.isfile(os.path.join(full, "adapter_config.json"))):
                    # Extract trailing integer for sorting — works for
                    # checkpoint-250, checkpoint-1868, step_500, etc.
                    digits = "".join(c for c in d if c.isdigit())
                    step = int(digits) if digits else 0
                    candidates.append((step, full))

            if not candidates:
                contents = os.listdir(checkpoint_path)
                raise FileNotFoundError(
                    f"No adapter_config.json found in {checkpoint_path} "
                    f"or any subdirectory.\n"
                    f"Directory contents: {contents}\n"
                    f"Pass the full path to the specific checkpoint folder "
                    f"(e.g. .../checkpoint-1868)."
                )

            candidates.sort(key=lambda x: x[0])
            checkpoint_path = candidates[-1][1]
            print(f"adapter_config.json not at top level.")
            print(f"Using latest checkpoint (step {candidates[-1][0]}): {checkpoint_path}")

    # ---- Load model --------------------------------------------------------
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=args.seq_len,
            load_in_4bit=False,
            fast_inference=False,
        )
        if checkpoint_path:
            print(f"Loading adapter from: {checkpoint_path}")
            model.load_adapter(checkpoint_path)
        model = FastLanguageModel.for_inference(model)
        # Do NOT call .to(device) — unsloth handles device placement
        print("Loaded via unsloth.")
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.float16
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if checkpoint_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                model, checkpoint_path, local_files_only=True,
            )
        print("Loaded via HuggingFace (no unsloth).")

    model.eval()

    # ---- 1. Parameter counts -----------------------------------------------
    print("\n[1/4] Counting parameters...")
    param_counts = count_parameters(model, requires_grad_only=False)

    # Also count trainable-only
    trainable_counts = count_parameters(model, requires_grad_only=True)
    print(f"  Total params:     {sum(param_counts.values()):,}")
    print(f"  Trainable params: {sum(trainable_counts.values()):,}")

    # ---- 2. Freeze everything except gate ----------------------------------
    print("\n[2/4] Freezing base model + LoRA, isolating gate...")
    grad_states = freeze_all_except_gate(model)
    gate_only_counts = count_parameters(model, requires_grad_only=True)
    gate_total = sum(gate_only_counts.values())
    print(f"  Gate-only trainable params: {gate_total:,}")
    # Restore for timing (we want the full model for timing)
    restore_grad_states(model, grad_states)

    # ---- 3. FLOPs estimate -------------------------------------------------
    print("\n[3/4] Estimating FLOPs...")
    flops = estimate_gate_flops(model, args.seq_len, args.batch_size)

    # ---- 4. Wall-clock timing ----------------------------------------------
    # Each model is timed in a SEPARATE SUBPROCESS so they never share GPU
    # context, memory allocator state, or cache warmth.  Running both models
    # in the same process causes whichever runs second to get a warmer GPU
    # state, making results order-dependent and unreliable (overhead flips
    # sign between runs).
    #
    # Each subprocess loads only its own model, times it, prints a JSON
    # result line, and exits — fully releasing GPU memory before the next
    # subprocess starts.
    print(f"\n[4/5] Timing models in isolated subprocesses "
          f"({args.n_timing_runs} runs each)...")

    print("  Timing HRPO model (subprocess 1/2)...")
    timing_hrpo = _run_isolated_timing(
        model_name=args.base_model,
        checkpoint_path=checkpoint_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        n_runs=args.n_timing_runs,
        label="HRPO",
    )
    print(f"  HRPO:     {timing_hrpo['mean_ms']:.1f} ± {timing_hrpo['std_ms']:.1f} ms"
          f"  (batch={timing_hrpo['batch_size']})")

    print("  Timing base model (subprocess 2/2)...")
    timing_base = _run_isolated_timing(
        model_name=args.base_model,
        checkpoint_path=None,          # no adapter = raw base model
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        n_runs=args.n_timing_runs,
        label="Base",
    )
    print(f"  Baseline: {timing_base['mean_ms']:.1f} ± {timing_base['std_ms']:.1f} ms"
          f"  (batch={timing_base['batch_size']})")

    actual_bs = timing_hrpo["batch_size"]
    overhead_ms = timing_hrpo["mean_ms"] - timing_base["mean_ms"]
    timing = {
        "hrpo_mean_ms": timing_hrpo["mean_ms"],
        "hrpo_std_ms": timing_hrpo["std_ms"],
        "baseline_mean_ms": timing_base["mean_ms"],
        "baseline_std_ms": timing_base["std_ms"],
        "overhead_ms": overhead_ms,
        "overhead_percent": 100 * overhead_ms / max(timing_base["mean_ms"], 1e-6),
        "overhead_per_token_us": (
            1000 * overhead_ms / (args.seq_len * actual_bs)
        ),
        "n_runs": args.n_timing_runs,
        "batch_size": actual_bs,
        "batch_size_requested": args.batch_size,
        "seq_len": args.seq_len,
        "baseline": "raw base model (no gate, no LoRA), isolated subprocess",
        "note": (
            "Each model timed in a separate subprocess to prevent shared "
            "GPU context/cache from biasing results. HRPO includes LoRA + gate; "
            "baseline is raw pretrained model only."
        ),
    }

    # ---- 5. Memory ---------------------------------------------------------
    memory = measure_memory_overhead(
        model, tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
    )

    # ---- Report ------------------------------------------------------------
    print_report(param_counts, flops, timing, memory)

    # Save JSON
    report = {
        "config": vars(args),
        "device": device,
        "param_counts": param_counts,
        "trainable_counts": trainable_counts,
        "gate_only_counts": gate_only_counts,
        "flops": flops,
        "timing": timing,
        "memory": memory,
    }
    out_path = args.output or "overhead_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full report saved to: {out_path}")


# ---------------------------------------------------------------------------
# Task 5 extension: model-size x variant overhead matrix (spec Section 11)
# ---------------------------------------------------------------------------
#
# The pre-existing wall-clock methodology times bare `model(input_ids=...)`
# forward passes. That call path never passes `is_thinking` /
# `last_thinking_states`, so the Eq.4 gate blend NEVER executes in it and the
# Eq.3 projection (which lives in the generation loop, transformers/
# generation/utils.py:3369-3372) is not exercised either — it measured LoRA
# adapter overhead only. The variant matrix therefore times real `generate()`
# calls at a fixed decode length, hybrid path ON vs OFF, inside one isolated
# subprocess per (size, variant), and additionally microbenchmarks the Eq.3
# projection einsum (which in this fork runs unconditionally in _sample, even
# for v1).

VARIANT_MATRIX_ALL = [
    "v1_grpo_baseline", "v2_fixed_blend", "v3_learned_lambda",
    "v4_full_hrpo_gate", "v5_hrpo_format_reward", "v6_hrpo_full",
]


def _gate_param_counts(hidden_dim: int) -> dict:
    """Analytic per-module gate parameter counts."""
    linear = hidden_dim * hidden_dim + hidden_dim  # weight + bias
    return {"gate_r": linear, "gate_i": linear, "Lambda": hidden_dim,
            "total": 2 * linear + hidden_dim}


def variant_param_breakdown(hidden_dim: int, adapter_path: str, variant: str) -> dict:
    """Total vs trainable gate parameters per variant (spec Section 11):
    v2 has the same total gate params as v4 but zero trainable after freezing;
    v3 trains only Lambda; v1 has no gate at all."""
    counts = _gate_param_counts(hidden_dim)
    trainable = {
        "v1_grpo_baseline": 0,
        "v2_fixed_blend": 0,
        "v3_learned_lambda": counts["Lambda"],
        "v4_full_hrpo_gate": counts["total"],
        "v5_hrpo_format_reward": counts["total"],
        "v6_hrpo_full": counts["total"],
    }.get(variant, counts["total"])

    saved = 0
    lora = 0
    try:
        from safetensors.torch import load_file
        sd = load_file(os.path.join(adapter_path, "adapter_model.safetensors"))
        for k, v in sd.items():
            if "thinking_residual" in k:
                saved += v.numel()
            elif "lora_" in k:
                lora += v.numel()
    except (FileNotFoundError, OSError):
        saved = None
        lora = None
    return {
        "gate_params_total": counts["total"],
        "gate_params_trainable": trainable,
        "gate_params_saved_in_adapter": saved,
        "lora_params_in_adapter": lora,
        "per_module": {k: v for k, v in counts.items() if k != "total"},
    }


def estimate_gate_flops_variant(hidden_dim: int, vocab_size: int, n_params: int,
                                seq_len: int, batch_size: int, variant: str) -> dict:
    """Variant-aware version of estimate_gate_flops (same MAC convention and
    component derivations — see estimate_gate_flops docstring):
      v1     — no projection, no blend, no gate linears (all zero);
      v2/v3  — projection + Lambda/blend, NO gate_r/gate_i linears (unit gates
               are bypassed entirely by the instance patch);
      v4-v6  — everything.
    """
    think_frac = 0.8
    think_tokens = int(seq_len * think_frac)

    gate_linears = 2 * (2 * hidden_dim * hidden_dim) + 2 * hidden_dim * 4
    lambda_blend = hidden_dim * 8 + hidden_dim * 6
    projection = vocab_size * 3 + 2 * vocab_size * hidden_dim

    if variant == "v1_grpo_baseline":
        per_token = 0
        parts = {"gate_linears": 0, "lambda_blend": 0, "projection": 0}
    elif variant in ("v2_fixed_blend", "v3_learned_lambda"):
        per_token = lambda_blend + projection
        parts = {"gate_linears": 0, "lambda_blend": lambda_blend,
                 "projection": projection}
    else:
        per_token = gate_linears + lambda_blend + projection
        parts = {"gate_linears": gate_linears, "lambda_blend": lambda_blend,
                 "projection": projection}

    total_gate = per_token * think_tokens * batch_size
    baseline = 6 * n_params * seq_len * batch_size
    return {
        "per_token_flops": per_token,
        "components_per_token": parts,
        "total_gate_flops_per_batch": total_gate,
        "baseline_transformer_flops_per_batch": baseline,
        "gate_overhead_percent": 100 * total_gate / max(baseline, 1),
        "think_span_fraction_assumed": think_frac,
        "flop_counting_convention": "MAC (1 multiply-add = 2 FLOPs), Kaplan et al. 2020",
        "baseline_formula": "6 * n_params * seq_len * batch_size",
    }


def _run_isolated_generation_timing(model_name, checkpoint_path, gate_mode,
                                    prompt_len, gen_tokens, batch_size,
                                    n_runs, temperature=0.5):
    """Time generate() with the hybrid path ON (answer_start never matched ->
    is_thinking True for every decoded token) vs OFF (answer_start "" ->
    is_thinking always False) in one fresh subprocess (fresh CUDA context per
    (size, variant), same-model configs share the context deliberately so the
    ON/OFF difference is not confounded by allocator state). Decode length is
    pinned with min_new_tokens == max_new_tokens so both configs execute an
    identical number of steps. Also microbenchmarks the Eq.3 projection
    einsum, and records peak memory per config (empty_cache + reset before
    each run, averaged — the pre-existing memory methodology)."""
    script = f"""
import json, time
import numpy as np
import torch

try:
    from unsloth import FastLanguageModel
    from transformers import GenerationConfig

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name={repr(model_name)},
        max_seq_length={prompt_len + gen_tokens},
        load_in_4bit=False,
        fast_inference=False,
    )
    model.answer_start = "####"
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    if {repr(checkpoint_path)} is not None:
        model.load_adapter({repr(checkpoint_path)})
    if {repr(gate_mode)} == "unit":
        from variant_common import apply_unit_gate_patch
        apply_unit_gate_patch(model)
    model = FastLanguageModel.for_inference(model)
except Exception as e:
    print(json.dumps({{"error": repr(e)}}))
    raise

device = next(model.parameters()).device
torch.manual_seed(42)
prompt_ids = torch.randint(0, tokenizer.vocab_size, ({batch_size}, {prompt_len}), device=device)
prompt_mask = torch.ones_like(prompt_ids)
gen_cfg = GenerationConfig(
    do_sample=True, temperature={temperature},
    min_new_tokens={gen_tokens}, max_new_tokens={gen_tokens},
    pad_token_id=tokenizer.pad_token_id,
)

def run_config(answer_start, n_runs):
    model.answer_start = answer_start
    times, peaks = [], []
    for _ in range(2):  # warmup
        _ = model.generate(prompt_ids, attention_mask=prompt_mask,
                           generation_config=gen_cfg,
                           processing_class=tokenizer, is_inference=True)
    torch.cuda.synchronize()
    for _ in range(n_runs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.generate(prompt_ids, attention_mask=prompt_mask,
                           generation_config=gen_cfg,
                           processing_class=tokenizer, is_inference=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        peaks.append(torch.cuda.max_memory_allocated() / 1e6)
    return times, peaks

# Hybrid path ON: answer_start that can never appear in decoded text.
on_times, on_peaks = run_config(chr(0xFFF0), {n_runs})
# Hybrid path OFF: empty answer_start is contained in every string.
off_times, off_peaks = run_config("", {n_runs})

# Eq.3 projection einsum microbenchmark (runs unconditionally in _sample;
# in production it executes under unsloth's bf16 autocast, so cast to match).
W = model.get_input_embeddings().weight
probs = torch.softmax(torch.randn({batch_size}, W.shape[0], device=device,
                                  dtype=torch.float32), -1).to(W.dtype)
for _ in range(3):
    _ = torch.einsum('bv,vd->bd', probs, W)
torch.cuda.synchronize()
t0 = time.perf_counter()
REPS = 50
for _ in range(REPS):
    _ = torch.einsum('bv,vd->bd', probs, W)
torch.cuda.synchronize()
proj_ms = (time.perf_counter() - t0) * 1000 / REPS

print(json.dumps({{
    "gate_on_ms_mean": float(np.mean(on_times)),
    "gate_on_ms_std": float(np.std(on_times)),
    "gate_off_ms_mean": float(np.mean(off_times)),
    "gate_off_ms_std": float(np.std(off_times)),
    "gate_on_peak_mb": float(np.mean(on_peaks)),
    "gate_off_peak_mb": float(np.mean(off_peaks)),
    "projection_einsum_ms_per_step": proj_ms,
    "gen_tokens": {gen_tokens},
    "batch_size": {batch_size},
}}))
"""
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True)
    for line in reversed([l.strip() for l in result.stdout.splitlines() if l.strip()]):
        try:
            data = json.loads(line)
            if "error" in data:
                raise RuntimeError(f"Subprocess error: {data['error']}\n"
                                   f"stderr: {result.stderr[-500:]}")
            return data
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"No JSON from generation-timing subprocess.\n"
                       f"stdout: {result.stdout[-800:]}\nstderr: {result.stderr[-800:]}")


def _matrix_checkpoint(model_size: str, variant: str):
    import glob as _glob
    exp_dir = os.path.join("experiments", f"{model_size}-{variant}")
    candidates = []
    for d in _glob.glob(os.path.join(exp_dir, "checkpoint-*")):
        digits = "".join(c for c in os.path.basename(d) if c.isdigit())
        if digits and os.path.isfile(os.path.join(d, "adapter_config.json")):
            candidates.append((int(digits), d))
    return max(candidates)[1] if candidates else None


def run_variant_matrix(args):
    """Spec Section 11: loop over model sizes and variants; report gate params
    (total AND trainable), variant-aware FLOPs estimate, generation wall-clock
    overhead, and memory overhead. Saves results/overhead_matrix.json and
    prints the size x variant wall-clock table plus the parameter breakdown."""
    from variant_common import GATE_MODES

    sizes = args.matrix_sizes
    variants = args.matrix_variants
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: variant matrix timing requires CUDA; "
              "wall-clock/memory will be skipped.")

    os.makedirs("results", exist_ok=True)
    cells = {}
    size_meta = {}

    for size in sizes:
        base_model = _QWEN_SCALE_MODELS[size]

        # Model dims + total params, loaded once per size then freed.
        print(f"\n=== {size} ({base_model}) — reading dimensions ===")
        from unsloth import FastLanguageModel
        model, _tok = FastLanguageModel.from_pretrained(
            model_name=base_model, max_seq_length=args.seq_len,
            load_in_4bit=False, fast_inference=False)
        cfg = model.config
        n_params = sum(p.numel() for p in model.parameters())
        size_meta[size] = {"hidden_dim": cfg.hidden_size,
                           "vocab_size": cfg.vocab_size,
                           "n_params": n_params}
        del model, _tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for variant in variants:
            ckpt = _matrix_checkpoint(size, variant)
            if ckpt is None:
                print(f"[matrix] {size}-{variant}: no checkpoint — skipped")
                continue
            print(f"\n=== {size}-{variant} ({ckpt}) ===")
            meta = size_meta[size]

            params = variant_param_breakdown(meta["hidden_dim"], ckpt, variant)
            params["gate_param_pct_of_total"] = round(
                100 * params["gate_params_total"] / meta["n_params"], 4)
            params["gate_trainable_pct_of_total"] = round(
                100 * params["gate_params_trainable"] / meta["n_params"], 4)

            flops = estimate_gate_flops_variant(
                meta["hidden_dim"], meta["vocab_size"], meta["n_params"],
                args.seq_len, args.gen_batch, variant)

            cell = {"checkpoint": ckpt, "params": params, "flops": flops}

            if variant == "v1_grpo_baseline":
                # v1 never runs the blend: overhead is zero by construction.
                cell["timing"] = {"wall_clock_overhead_pct": 0.0,
                                  "note": "gate disabled in v1; overhead 0 by construction"}
            elif device == "cuda":
                gate_mode = GATE_MODES.get(variant, "standard")
                print(f"  timing generation (gate ON vs OFF, "
                      f"{args.n_gen_runs} runs x {args.gen_tokens} tokens)...")
                t = _run_isolated_generation_timing(
                    base_model, ckpt, gate_mode,
                    args.gen_prompt_len, args.gen_tokens, args.gen_batch,
                    args.n_gen_runs)
                overhead_ms = t["gate_on_ms_mean"] - t["gate_off_ms_mean"]
                t["wall_clock_overhead_pct"] = round(
                    100 * overhead_ms / max(t["gate_off_ms_mean"], 1e-6), 3)
                t["wall_clock_overhead_ms"] = round(overhead_ms, 3)
                t["memory_overhead_mb"] = round(
                    t["gate_on_peak_mb"] - t["gate_off_peak_mb"], 2)
                t["memory_overhead_pct"] = round(
                    100 * t["memory_overhead_mb"] / max(t["gate_off_peak_mb"], 1), 3)
                cell["timing"] = t
                print(f"    ON {t['gate_on_ms_mean']:.1f}±{t['gate_on_ms_std']:.1f} ms, "
                      f"OFF {t['gate_off_ms_mean']:.1f}±{t['gate_off_ms_std']:.1f} ms "
                      f"-> {t['wall_clock_overhead_pct']}%  "
                      f"(projection einsum {t['projection_einsum_ms_per_step']:.3f} ms/step)")

            cells[f"{size}-{variant}"] = cell

    # ---- tables --------------------------------------------------------------
    def cell_of(size, variant):
        return cells.get(f"{size}-{variant}")

    print(f"\n{'='*100}")
    print("  Wall-clock generation overhead % (gate ON vs OFF), rows = sizes, cols = variants")
    print(f"{'='*100}")
    header = "  {:<6}".format("size") + "".join(f"{v.split('_')[0]:>10}" for v in variants)
    print(header)
    for size in sizes:
        row = f"  {size:<6}"
        for v in variants:
            c = cell_of(size, v)
            row += (f"{c['timing']['wall_clock_overhead_pct']:>9.2f}%"
                    if c and "timing" in c and
                    "wall_clock_overhead_pct" in c["timing"] else f"{'-':>10}")
        print(row)

    print(f"\n  FLOPs overhead % (analytical, MAC convention)")
    print(header)
    for size in sizes:
        row = f"  {size:<6}"
        for v in variants:
            c = cell_of(size, v)
            row += (f"{c['flops']['gate_overhead_percent']:>9.4f}%" if c else f"{'-':>10}")
        print(row)

    print(f"\n  Gate parameters per variant (total / trainable)")
    for size in sizes:
        for v in variants:
            c = cell_of(size, v)
            if not c:
                continue
            p = c["params"]
            print(f"  {size}-{v:<24} total {p['gate_params_total']:>12,} "
                  f"({p['gate_param_pct_of_total']:.4f}%)   trainable "
                  f"{p['gate_params_trainable']:>12,} "
                  f"({p['gate_trainable_pct_of_total']:.4f}%)")

    payload = {
        "config": {"seq_len": args.seq_len, "gen_prompt_len": args.gen_prompt_len,
                   "gen_tokens": args.gen_tokens, "gen_batch": args.gen_batch,
                   "n_gen_runs": args.n_gen_runs},
        "size_meta": size_meta,
        "cells": cells,
    }
    with open(args.matrix_output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {args.matrix_output}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure computational overhead of the HRPO latent reasoning gate"
    )
    parser.add_argument(
        "--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name or local path of the base model"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="Path to the HRPO adapter checkpoint (optional — measures gate "
             "overhead even without a trained adapter)"
    )
    parser.add_argument(
        "--seq_len", type=int, default=512,
        help="Sequence length to use for timing and FLOPs estimation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for timing measurements"
    )
    parser.add_argument(
        "--n_timing_runs", type=int, default=50,
        help="Number of forward passes to time (higher = more stable estimate)"
    )
    parser.add_argument(
        "--output", type=str, default="overhead_report.json",
        help="Where to save the single-model JSON report"
    )
    parser.add_argument(
        "--all_scales", action="store_true",
        help="Measure overhead for all 3 Qwen sizes (0.5b / 1.5b / 3b) and "
             "produce a cross-scale comparison table + plot."
    )
    parser.add_argument(
        "--scales_output", type=str, default="results/overhead_comparison.json",
        help="Output path for the --all_scales comparison JSON."
    )
    parser.add_argument(
        "--variant_matrix", action="store_true",
        help="Task 5: model-size x variant overhead matrix over the ablation "
             "checkpoints under experiments/{size}-{variant}/ (generation "
             "wall-clock gate ON vs OFF, variant-aware FLOPs, params "
             "total/trainable, memory)."
    )
    parser.add_argument("--matrix_sizes", nargs="+",
                        default=["0.5b", "1.5b", "3b"],
                        choices=["0.5b", "1.5b", "3b"])
    parser.add_argument("--matrix_variants", nargs="+",
                        default=VARIANT_MATRIX_ALL)
    parser.add_argument("--gen_prompt_len", type=int, default=128,
                        help="Prompt length for generation timing")
    parser.add_argument("--gen_tokens", type=int, default=128,
                        help="Fixed decode length (min == max new tokens)")
    parser.add_argument("--gen_batch", type=int, default=4,
                        help="Batch size for generation timing")
    parser.add_argument("--n_gen_runs", type=int, default=5,
                        help="Timed generation runs per config")
    parser.add_argument("--matrix_output", type=str,
                        default="results/overhead_matrix.json")
    args = parser.parse_args()

    if args.variant_matrix:
        run_variant_matrix(args)
    elif args.all_scales:
        run_all_scales(args)
    else:
        main(args)
