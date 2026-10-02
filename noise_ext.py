"""Parameter-space exploration noise for RL rollouts (QeRL-style "adaptive quantization noise").

QeRL (Huang et al., 2025, arXiv:2510.11696, Sec. 3.3) perturbs the RMSNorm weight vectors that sit in front
of the attention and MLP projections with Gaussian noise whose scale decays exponentially over training:

    sigma(k) = sigma_start * (sigma_end / sigma_start) ** ((k - 1) / (K - 1)),   k = 1..K stages

with K = 10 stages evenly spaced over training and sigma going from 5e-2 to 5e-4 in their GSM8K runs.

Why this module is small
------------------------
* RMSNorm weights are frozen under LoRA, so perturbing and restoring them never touches the optimiser.
* HF `Trainer.training_step` runs `_prepare_inputs` (which, for GRPOTrainer, does the rollout and the
  reference log-probs), then `compute_loss`, then `backward`. Wrapping `training_step` therefore keeps
  ONE noise draw alive for the rollout, the loss forward and the (checkpointed) backward recompute of a
  micro-batch. The policy that generated the tokens is the policy the loss differentiates, so no
  importance ratio or noise replay is needed (HRPO has no ratio).
* The noise is removed in a `finally` block, so checkpoints and evaluation always see clean weights.

Caveats (read before trusting results)
--------------------------------------
* bf16 has about 3 significant decimal digits. An additive sigma of 5e-4 on a weight near 1.0 is below
  bf16 resolution (about 4e-3 at 1.0) and is largely rounded away. Noise is computed in fp32 and cast back,
  but the effective noise at the low end of QeRL's schedule is smaller than nominal. Prefer sigma_end >= 4e-3
  unless the norm weights are kept in fp32.
* This is a re-implementation of QeRL's noise injection only. It does NOT quantise the model.
* Every noise-based gain must be compared against a higher-temperature control: noise on the norms tends to
  flatten the output distribution, which a temperature increase does too.
"""

import types

import torch

DEFAULT_TARGETS = ("input_layernorm", "post_attention_layernorm")


def sigma_at(step, total_steps, sigma_start, sigma_end, stages):
    """Noise scale at optimiser step `step` (0-based) under QeRL's staged exponential decay."""
    if sigma_start <= 0:
        return 0.0
    if stages <= 1 or sigma_end == sigma_start:
        return float(sigma_start)
    stage = min(int(step * stages / max(total_steps, 1)), stages - 1)  # 0 .. K-1
    return float(sigma_start * (sigma_end / sigma_start) ** (stage / (stages - 1)))


def find_norm_weights(model, targets=DEFAULT_TARGETS):
    """(name, weight tensor) for every module whose last name component is in `targets`."""
    found = []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        weight = getattr(module, "weight", None)
        if leaf in targets and isinstance(weight, torch.Tensor):
            found.append((name, weight))
    return found


class NoiseHandle:
    """Holds the clean weights so the perturbation can be undone exactly."""

    def __init__(self, saved):
        self.saved = saved

    @torch.no_grad()
    def restore(self):
        for weight, original in self.saved:
            weight.data.copy_(original)
        self.saved = []


@torch.no_grad()
def apply_norm_noise(model, sigma, seed, mode="add", targets=DEFAULT_TARGETS):
    """Perturb the target RMSNorm weights in place and return a handle that restores them.

    mode="add":  w <- w + sigma * z          (QeRL: additive noise on the norm weight)
    mode="mult": w <- w * (1 + sigma * z)    (relative noise, comparable across layers)

    The Gaussian draw uses a CPU generator seeded with `seed`, so the same (seed, sigma) gives the same
    perturbation on any device.
    """
    if sigma <= 0:
        return NoiseHandle([])
    if mode not in ("add", "mult"):
        raise ValueError(f"mode must be 'add' or 'mult', got {mode!r}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2 ** 62))
    saved = []
    for _, weight in find_norm_weights(model, targets):
        original = weight.data.clone()
        z = torch.randn(weight.shape, generator=generator, dtype=torch.float32).to(weight.device)
        base = original.float()
        perturbed = base + sigma * z if mode == "add" else base * (1.0 + sigma * z)
        weight.data.copy_(perturbed.to(weight.dtype))
        saved.append((weight, original))
    return NoiseHandle(saved)


def patch_trainer_noise(trainer, sigma_start, sigma_end, stages=10, mode="add", seed=0,
                        targets=DEFAULT_TARGETS, total_steps=None, verbose=True):
    """Wrap `trainer.training_step` so every micro-batch rolls out and trains under one noise draw.

    Call after the trainer is built (next to `patch_trainer_optimizer`). A no-op if sigma_start <= 0.
    """
    if sigma_start <= 0:
        return trainer
    original_step = trainer.training_step
    state = {"calls": 0, "stage_sigma": None}

    def training_step(self, model, inputs, *args, **kwargs):
        steps = total_steps or (self.state.max_steps if self.state.max_steps and self.state.max_steps > 0 else 1)
        sigma = sigma_at(self.state.global_step, steps, sigma_start, sigma_end, stages)
        draw_seed = seed * 1_000_003 + state["calls"]
        state["calls"] += 1
        if verbose and sigma != state["stage_sigma"]:
            print(f"[noise_ext] step {self.state.global_step}: sigma = {sigma:.6g} ({mode})", flush=True)
            state["stage_sigma"] = sigma
        self.last_noise_sigma = sigma
        handle = apply_norm_noise(model, sigma, draw_seed, mode=mode, targets=targets)
        try:
            return original_step(model, inputs, *args, **kwargs)
        finally:
            handle.restore()

    trainer.training_step = types.MethodType(training_step, trainer)
    return trainer
