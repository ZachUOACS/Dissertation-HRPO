"""Pure, dependency-light (torch-only) helpers for the HRPO fusion + debias extension.

Kept in one small module so the exact math the production code runs is unit-testable on CPU
without importing the full transformers/unsloth/model stack. The production sites import these
via CALL-TIME imports (robust to PatchFastRL's trainer rebuild; repo root is on sys.path when
launching `python hrpo_*.py`). See CHANGES.md / docs/fusion_debias_design.md for provenance.

  - context_prediction_fusion : Change A  (LT-Tuning-style fusion, learned per-dim gate)
  - compute_group_advantages  : Change B  (Dr. GRPO bias 2 - the +/- std toggle)
  - masked_length_normalized_loss / select_length_denominator : Change B (Dr. GRPO bias 1)

The two changes never share a code path here: fusion touches only the input-side latent,
the debias helpers touch only the advantage and the loss denominator.
"""
import torch


# ============================== Change A: fusion ==============================
def context_prediction_fusion(e_pred, h_I, gate_beta, e_hat=None,
                              gate_input="hidden", normalize_h=True, adapter=None, eps=1e-8):
    """LT-Tuning Context-Prediction Fusion generalized to a learned per-dimension gate.

    h_tilde = (1 - beta) * e_pred + beta * h_bar_I,  beta = sigmoid(gate_beta(gate_source)).

    e_pred      : predictive vocabulary interpolation (HRPO Eq. 3), already ||p||-normalized.
    h_I         : raw contextual hidden state at layer I (DETACHED by the caller: Option A).
    gate_beta   : nn.Linear (W_beta, b_beta). beta plays LT-Tuning's alpha (weight ON the hidden
                  state), NOT (1 - alpha).
    e_hat       : sampled-token embedding, used only when gate_input == "token".
    adapter     : optional LT-Tuning Eq. 20 adapter applied to h_I (untied-embedding models).

    Returns (h_tilde, beta). At b_beta -> -inf, beta -> 0 and h_tilde == e_pred exactly
    (baseline HRPO); this is the strict-generalization guarantee.
    """
    h_ctx = h_I
    if adapter is not None:
        h_ctx = adapter(h_ctx)
    if normalize_h:
        h_ctx = h_ctx / (h_ctx.norm(dim=-1, keepdim=True) + eps)
    gate_source = h_I if gate_input == "hidden" else e_hat
    beta = torch.sigmoid(gate_beta(gate_source))
    h_tilde = (1.0 - beta) * e_pred + beta * h_ctx
    return h_tilde, beta


# ============================== Change B: debias ==============================
def compute_group_advantages(rewards, num_generations, advantage_std_norm, eps=1e-4):
    """Group-relative advantages. advantage_std_norm=True is vanilla HRPO/GRPO (Dr. GRPO's
    question-level difficulty bias); False keeps mean-centering only (Dr. GRPO fix, = RLOO up
    to a constant scale). A fully tied group yields exactly 0 under both settings (no NaN)."""
    mean_g = rewards.view(-1, num_generations).mean(dim=1)
    std_g = rewards.view(-1, num_generations).std(dim=1)
    mean_g = mean_g.repeat_interleave(num_generations, dim=0)
    std_g = std_g.repeat_interleave(num_generations, dim=0)
    if advantage_std_norm:
        return (rewards - mean_g) / (std_g + eps)
    return rewards - mean_g


def select_length_denominator(n_mask_per_reward, length_norm_constant):
    """Per-response length |y_i| (vanilla, the length bias) vs a fixed constant C (Dr. GRPO fix).
    None or <=0 -> per-response. Mirrors the inline choice in grpo_compute_loss."""
    if length_norm_constant is None or length_norm_constant <= 0:
        return n_mask_per_reward
    return length_norm_constant


def masked_length_normalized_loss(loss_i, mask, length_norm_constant):
    """Per-response length-normalized mean of a per-token loss, matching grpo_compute_loss's
    reduction. Under 'constant' every token contributes equally regardless of |y_i|."""
    mask = mask.to(loss_i.dtype)
    n_mask_per_reward = mask.sum(1)
    denom = select_length_denominator(n_mask_per_reward, length_norm_constant)
    return ((loss_i * mask).sum(1) / denom).mean()
