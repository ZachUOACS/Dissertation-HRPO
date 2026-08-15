"""Numerical arm-1 equivalence harness (Phase 3 test #1).

Runs ONE deterministic policy forward + backward on a FIXED batch and dumps a small summary
(loss, forward sums, shared-param gradient norm). Used to prove:

  (a) with all default flags, the NEW code == the PRE-CHANGE code  (--arm baseline), and
  (b) with fusion on but b_beta = -inf, the NEW code == baseline    (--arm beta_neg_inf),

by dumping from a clean pre-change git worktree AND from the current tree, then `--compare`.
See slurm/equivalence.sbatch for the one-command driver.

Determinism: every shared trainable param is overwritten from a PER-NAME seed, so ref and new
get byte-identical shared weights regardless of construction order or whether the fusion gate
exists. The forward has no RNG. Runs in the model's native bf16 (as training does); a real
regression shifts the summary by >> the bf16 cross-process kernel jitter (guarded by --rtol).

This exercises the model-forward surface (where Change A lives). Change B's advantage + loss
denominator are pure functions covered exactly by tests/test_fusion_debias.py.

Cluster only (needs a GPU). Not run locally.
"""
import argparse
import hashlib
import os

import torch


def _resolve_model_path(model_name):
    """Self-contained (this file is copied into the pre-change worktree, whose utils.py is old):
    resolve an HF repo id to a local cache snapshot so unsloth loads offline (no HfFileSystem glob).
    Returns (name_or_path, use_exact_model_name)."""
    if os.path.isdir(model_name):
        return model_name, True
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return model_name, False
    candidates = []
    if "/" in model_name and not model_name.startswith("unsloth/"):
        candidates.append("unsloth/" + model_name.split("/", 1)[1])
    candidates.append(model_name)
    for repo in candidates:
        try:
            return snapshot_download(repo, local_files_only=True), True
        except Exception:
            continue
    return model_name, False


def _name_seed(name):
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little") % (2 ** 31)


def _fill_shared_params(model, std=0.02):
    """Overwrite every shared trainable param from a per-name seed (order-independent, so the
    presence of the extra fusion gate in the new tree cannot shift the reference weights)."""
    n = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "thinking_residual_gate_beta" in name or "adapter" in name:
            continue  # fusion-only params: excluded so the shared draw sequence matches the ref
        g = torch.Generator(device="cpu").manual_seed(_name_seed(name))
        p.data.copy_((torch.randn(p.shape, generator=g) * std).to(p.dtype).to(p.device))
        n += 1
    return n


def _fixed_tensor(name, shape, device, dtype, std=1.0):
    g = torch.Generator(device="cpu").manual_seed(_name_seed(name))
    return (torch.randn(*shape, generator=g) * std).to(dtype).to(device)


def build_and_run(arm, out_path):
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    torch.manual_seed(0)
    fusion = (arm == "beta_neg_inf")

    model_path, use_exact = _resolve_model_path("Qwen/Qwen2.5-1.5B-Instruct")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=64, load_in_4bit=False, load_in_8bit=False, fast_inference=False,
        use_exact_model_name=use_exact,
    )

    modules_to_save = ["thinking_residual_gate_r", "thinking_residual_gate_i", "thinking_residual_Lambda"]
    if fusion:
        modules_to_save.append("thinking_residual_gate_beta")

    model = FastLanguageModel.get_peft_model(
        model, r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=modules_to_save, lora_alpha=16,
        use_gradient_checkpointing="unsloth", random_state=0,
    )
    model.model.model.thinking_residual_Lambda.reset_lambda_parameters(r_min=0.99, r_max=0.999)

    if fusion:
        inner = model.model.model
        inner.fusion_enabled = True
        inner.fusion_layer_I = -2
        inner.fusion_gate_input = "hidden"
        inner.fusion_normalize_h = True
        inner.fusion_use_adapter = False
        # b_beta = -inf, W_beta = 0  ->  beta = 0  ->  h_tilde == e_pred exactly (baseline).
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "thinking_residual_gate_beta" in n and "adapter" not in n:
                    if n.endswith("bias"):     p.fill_(float("-inf"))
                    elif n.endswith("weight"): p.zero_()

    n_shared = _fill_shared_params(model)
    model.train()

    dev = model.device
    H = model.config.hidden_size
    V = model.config.vocab_size
    B, P, C = 2, 6, 8
    seq = P + C
    dtype = next(model.parameters()).dtype

    # Fixed batch (per-name-seeded so ref and new are identical).
    input_ids = (_fixed_tensor("ids", (B, seq), dev, torch.float32, std=1.0).abs() * (V - 10)).long() % (V - 10) + 5
    thinking_embeds = _fixed_tensor("epred", (B, seq, H), dev, dtype, std=0.05)
    thinking_hidden = _fixed_tensor("hI", (B, seq, H), dev, dtype, std=50.0)  # hidden-state scale
    thinking_mask = torch.zeros(B, seq, dtype=torch.bool, device=dev)
    thinking_mask[:, P + 1] = True
    thinking_mask[:, P + 3] = True
    labels = _fixed_tensor("labels", (B, C), dev, torch.float32).abs().long() % (V - 10) + 5

    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    mixed = torch.float16 if os.environ.get("ACCELERATE_MIXED_PRECISION", "bf16") == "fp16" else torch.bfloat16

    kw = dict(input_ids=input_ids, inputs_embeds=thinking_embeds, thinking_mask=thinking_mask,
              logits_to_keep=C + 1)
    if fusion:
        kw["thinking_hidden"] = thinking_hidden

    with torch.amp.autocast(device_type="cuda", dtype=mixed):
        new_hidden = model(**kw).logits           # (B, C+1, H) hidden states
    lm_head = model.get_output_embeddings().weight
    new_logits = torch.matmul(new_hidden.float(), lm_head.float().t())[:, :-1, :]  # (B, C, V)
    loss = torch.nn.functional.cross_entropy(new_logits.reshape(-1, V), labels.reshape(-1))
    loss.backward()

    shared_grad_sq = 0.0
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        if "thinking_residual_gate_beta" in name or "adapter" in name:
            continue
        shared_grad_sq += p.grad.detach().float().pow(2).sum().item()

    summary = {
        "arm": arm,
        "loss": float(loss.detach().float().item()),
        "hidden_sum": float(new_hidden.detach().float().sum().item()),
        "logits_sum": float(new_logits.detach().float().sum().item()),
        "shared_grad_norm": float(shared_grad_sq ** 0.5),
        "n_shared_params": int(n_shared),
    }
    torch.save(summary, out_path)
    print(f"[{arm}] {summary}  ->  {out_path}")
    return summary


def compare(ref_path, new_path, rtol, atol):
    ref = torch.load(ref_path, weights_only=False)
    new = torch.load(new_path, weights_only=False)
    keys = ["loss", "hidden_sum", "logits_sum", "shared_grad_norm"]
    print(f"comparing {ref_path}  vs  {new_path}   (rtol={rtol}, atol={atol})")
    ok = True
    for k in keys:
        a, b = ref[k], new[k]
        close = abs(a - b) <= atol + rtol * abs(a)
        flag = "OK " if close else "FAIL"
        print(f"  {flag}  {k:18s} ref={a:+.6e}  new={b:+.6e}  |d|={abs(a-b):.3e}")
        ok = ok and close
    if ref["n_shared_params"] != new["n_shared_params"]:
        print(f"  FAIL  n_shared_params ref={ref['n_shared_params']} new={new['n_shared_params']}")
        ok = False
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "beta_neg_inf"], default="baseline")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--compare", nargs=2, metavar=("REF", "NEW"), default=None)
    ap.add_argument("--rtol", type=float, default=1e-2)   # bf16 cross-process jitter << a real bug
    ap.add_argument("--atol", type=float, default=1e-3)
    args = ap.parse_args()

    if args.compare is not None:
        ok = compare(args.compare[0], args.compare[1], args.rtol, args.atol)
        raise SystemExit(0 if ok else 1)
    assert args.out is not None, "--out required when not comparing"
    build_and_run(args.arm, args.out)
