"""
length_penalty.py - length-penalised reward of Arora & Zanette (2025), "Training Language Models to
Reason Efficiently" (arXiv 2502.04463), Eq. (lenpen) with its sigmoid-normalised f:

    r_i = 1{y_i correct} * (1 - alpha * sigmoid((len(y_i) - MEAN(x)) / STD(x)))

MEAN(x) and STD(x) are the per-prompt mean and standard deviation of the length of the CORRECT
responses, "directly estimated from the rollouts during online training", i.e. from the correct
completions in the prompt's own group. Only correct responses are penalised, and for alpha in
[0, 1) a correct response always earns more than an incorrect one. Their estimator is RLOO
(mean-centred advantages, no std division), which is what Dr. GRPO's advantage reduces to, so
their alpha values (0.05, 0.1, 0.2, 0.4) carry over.

Choices the paper leaves open, fixed here:
  * length = generated tokens up to and including the first end-of-sequence token
    (completion_mask.sum(1) in GRPOTrainer), the quantity logged as completion_length; latent
    positions still emit a token, so they count;
  * STD is the population standard deviation of the group's correct lengths; when a group has a
    single correct response, or all its correct responses have the same length, the normalised
    length is 0 (f = 1/2), so there is no length signal within that group;
  * "correct" is reward 1: the default reward is 1.0 for a correct, well-formed answer and 0.0
    otherwise (utils.get_reward_func); any other reward value raises, so a shaped reward (v5, v6)
    can never be silently rescaled.
"""

import torch


def length_penalised_rewards(rewards, lengths, group_size, alpha, eps=1e-6):
    """rewards, lengths: (N,) with N = n_prompts * group_size, groups contiguous as in
    GRPOTrainer. Returns (shaped rewards, f), f the sigmoid factor (0 where incorrect)."""
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    r = rewards.view(-1, group_size)
    if not ((r == 0) | (r == 1)).all():
        raise ValueError("length penalty expects 0/1 outcome rewards; a shaped reward (v5/v6) "
                         "would be rescaled, not length-penalised")
    n = lengths.float().view(-1, group_size)
    correct = (r > 0.5).float()
    n_correct = correct.sum(1, keepdim=True)
    mean = (n * correct).sum(1, keepdim=True) / n_correct.clamp(min=1)
    var = (((n - mean) ** 2) * correct).sum(1, keepdim=True) / n_correct.clamp(min=1)
    std = var.sqrt()
    z = torch.where(std > eps, (n - mean) / std.clamp(min=eps), torch.zeros_like(n))
    f = torch.sigmoid(z) * correct
    shaped = r * (1.0 - alpha * f)
    return shaped.view(-1), f.view(-1)
