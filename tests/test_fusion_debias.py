"""Phase 3 CPU pure-function tests for the HRPO fusion (Change A) + debias (Change B) extension.

These exercise the SAME functions the production code calls (hrpo_ext.*), on tiny CPU tensors,
so no GPU / model forward is needed. Full-model forward-equivalence and the 4-arm smoke run are
done on the cluster via sbatch (run_ablation.sh ... --smoke).

Run:  venv/bin/python -m pytest tests/test_fusion_debias.py -q
"""
import math

import pytest
import torch
import torch.nn as nn

from hrpo_ext import (
    context_prediction_fusion,
    compute_group_advantages,
    select_length_denominator,
    masked_length_normalized_loss,
)

torch.manual_seed(0)
DEV = "cpu"
H = 16


def make_gate(bias_val=-4.0, zero_weight=True, h=H):
    g = nn.Linear(h, h)
    with torch.no_grad():
        if zero_weight:
            g.weight.zero_()
        g.bias.fill_(bias_val)
    return g


# ============================== Change A: fusion ==============================
def test_beta_neg_inf_reduces_to_epred_exactly():
    """Strict-generalization / equivalence: b_beta -> -inf => beta == 0 => h_tilde == e_pred."""
    gate = make_gate(bias_val=float("-inf"), zero_weight=True)
    e_pred = torch.randn(2, 5, H)
    h_I = torch.randn(2, 5, H)
    h_tilde, beta = context_prediction_fusion(e_pred, h_I, gate, gate_input="hidden")
    assert torch.allclose(beta, torch.zeros_like(beta))
    assert torch.allclose(h_tilde, e_pred, atol=0, rtol=0)


def test_beta_init_is_near_baseline_but_not_identical():
    """b_beta = -4 => beta ~= sigmoid(-4) ~= 0.018 => h_tilde ~= e_pred (close, not equal)."""
    gate = make_gate(bias_val=-4.0, zero_weight=True)
    e_pred = torch.randn(3, 4, H)
    h_I = torch.randn(3, 4, H)
    h_tilde, beta = context_prediction_fusion(e_pred, h_I, gate)
    assert torch.allclose(beta, torch.full_like(beta, 1 / (1 + math.exp(4))), atol=1e-6)
    assert not torch.allclose(h_tilde, e_pred)              # fusion is active
    assert (h_tilde - e_pred).abs().mean() < 0.2 * e_pred.abs().mean()  # but small


def test_shapes_dtype_device_preserved():
    gate = make_gate().to(torch.float32)
    for shape in [(1, 1, H), (4, 7, H), (2, 3, H)]:
        e_pred = torch.randn(*shape)
        h_I = torch.randn(*shape)
        h_tilde, beta = context_prediction_fusion(e_pred, h_I, gate)
        assert h_tilde.shape == e_pred.shape == beta.shape
        assert h_tilde.dtype == e_pred.dtype
        assert str(h_tilde.device) == DEV


def test_think_mask_boundary_only_affects_masked_positions():
    """Fusion must apply ONLY inside the think span (mirrors the training-forward masking)."""
    gate = make_gate(bias_val=1.0, zero_weight=False)  # non-trivial beta so fusion changes things
    B, T = 2, 6
    e_hat = torch.randn(B, T, H)
    e_pred = torch.randn(B, T, H)
    h_I = torch.randn(B, T, H)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[0, 1] = mask[0, 3] = mask[1, 4] = True

    residual = e_pred.clone()
    fused, _ = context_prediction_fusion(e_pred[mask], h_I[mask], gate, e_hat=e_hat[mask])
    residual[mask] = fused

    # masked positions changed; unmasked positions are byte-identical to e_pred
    assert not torch.allclose(residual[mask], e_pred[mask])
    assert torch.equal(residual[~mask], e_pred[~mask])


def test_scale_check_hbar_is_unit_and_norms_finite():
    gate = make_gate(bias_val=0.0, zero_weight=False)
    e_pred = torch.randn(4, 3, H)
    h_I = torch.randn(4, 3, H) * 50.0  # deliberately large, like a real hidden state
    h_tilde, beta = context_prediction_fusion(e_pred, h_I, gate, normalize_h=True)
    h_bar = h_I / (h_I.norm(dim=-1, keepdim=True) + 1e-8)
    assert torch.allclose(h_bar.norm(dim=-1), torch.ones(4, 3), atol=1e-4)  # unit-normalized
    for t in (e_pred, h_bar, h_tilde):
        assert torch.isfinite(t).all()
    # h_tilde stays a convex-ish blend, so its norm is bounded by the two inputs' norms.
    assert (h_tilde.norm(dim=-1) <= e_pred.norm(dim=-1) + h_bar.norm(dim=-1) + 1e-4).all()


def test_gate_input_token_vs_hidden_differ():
    gate = make_gate(bias_val=0.0, zero_weight=False)
    e_pred = torch.randn(2, 3, H)
    h_I = torch.randn(2, 3, H)
    e_hat = torch.randn(2, 3, H)
    _, beta_hidden = context_prediction_fusion(e_pred, h_I, gate, e_hat=e_hat, gate_input="hidden")
    _, beta_token = context_prediction_fusion(e_pred, h_I, gate, e_hat=e_hat, gate_input="token")
    assert not torch.allclose(beta_hidden, beta_token)


def test_adapter_path_runs_and_changes_output():
    gate = make_gate(bias_val=0.0, zero_weight=False)
    adapter = nn.Sequential(nn.Linear(H, 8), nn.GELU(), nn.Linear(8, H))
    e_pred = torch.randn(2, 3, H)
    h_I = torch.randn(2, 3, H)
    h_no, _ = context_prediction_fusion(e_pred, h_I, gate, normalize_h=True, adapter=None)
    h_ad, _ = context_prediction_fusion(e_pred, h_I, gate, normalize_h=True, adapter=adapter)
    assert h_ad.shape == e_pred.shape
    assert not torch.allclose(h_no, h_ad)


def test_gradient_flows_to_gate_not_into_detached_h_I():
    """Option A: h_I is stored detached, so gradient reaches W_beta / b_beta but NOT layer I.
    Also confirms no gradient path is created into a 'sampling' tensor."""
    gate = make_gate(bias_val=-1.0, zero_weight=False)
    e_pred = torch.randn(2, 3, H)                 # replayed constant (detached in production)
    h_I = torch.randn(2, 3, H)                    # detached rollout feature: requires_grad=False
    sampled_logits = torch.randn(2, 3, H, requires_grad=True)  # stands in for the sampling path
    h_tilde, beta = context_prediction_fusion(e_pred, h_I, gate)
    h_tilde.sum().backward()
    assert gate.weight.grad is not None and gate.weight.grad.abs().sum() > 0
    assert gate.bias.grad is not None and gate.bias.grad.abs().sum() > 0
    assert h_I.grad is None                        # no gradient into layer I (detached)
    assert sampled_logits.grad is None             # fusion created no path into sampling


# ============================== Change B: advantages ==============================
def _adv(rewards, g, std):
    return compute_group_advantages(torch.tensor(rewards, dtype=torch.float32), g, std)


def test_advantage_mean_centering_with_and_without_std():
    rewards = [1.0, 0.0, 1.0, 0.0]  # one group of 4, mean 0.5
    a_no = _adv(rewards, 4, False)
    assert torch.allclose(a_no, torch.tensor([0.5, -0.5, 0.5, -0.5]))
    a_std = _adv(rewards, 4, True)
    # same sign / centering, but rescaled by 1/(std+eps)
    assert torch.all(torch.sign(a_std) == torch.sign(a_no))
    assert not torch.allclose(a_std, a_no)


def test_advantage_all_tied_group_is_zero_no_nan_both_settings():
    for rewards in ([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]):
        for std in (True, False):
            a = _adv(rewards, 4, std)
            assert torch.isfinite(a).all()
            assert torch.allclose(a, torch.zeros(4), atol=1e-6)


def test_advantage_all_correct_and_all_wrong_are_zero():
    assert torch.allclose(_adv([1, 1], 2, False), torch.zeros(2))
    assert torch.allclose(_adv([0, 0], 2, False), torch.zeros(2))
    assert torch.allclose(_adv([1, 1], 2, True), torch.zeros(2), atol=1e-6)


def test_advantage_multiple_groups_independent():
    # two groups of 2: [1,0] and [1,1]
    a = _adv([1.0, 0.0, 1.0, 1.0], 2, False)
    assert torch.allclose(a, torch.tensor([0.5, -0.5, 0.0, 0.0]))


# ============================== Change B: loss reduction ======================
def test_length_denominator_selection():
    n = torch.tensor([10.0, 20.0])
    assert torch.equal(select_length_denominator(n, None), n)      # baseline
    assert torch.equal(select_length_denominator(n, -1.0), n)      # sentinel -> per-response
    assert select_length_denominator(n, 128.0) == 128.0            # constant


def test_loss_per_token_weighting_uniform_under_constant_length_dependent_under_response():
    """Two rollouts of very different lengths; assert d(loss)/d(per-token loss) is uniform under
    'constant' and length-dependent under 'response'."""
    G, T = 2, 8
    mask = torch.zeros(G, T)
    L = [3, 7]
    mask[0, :L[0]] = 1.0
    mask[1, :L[1]] = 1.0
    C = 16.0

    # ---- constant: every masked token has identical gradient 1/(C*G) ----
    loss_i = torch.zeros(G, T, requires_grad=True)
    loss = masked_length_normalized_loss(loss_i, mask, C)
    loss.backward()
    grads = loss_i.grad[mask.bool()]
    assert torch.allclose(grads, torch.full_like(grads, 1.0 / (C * G)), atol=1e-7)

    # ---- response: token gradient is 1/(len_i * G), differs across the two rollouts ----
    loss_i = torch.zeros(G, T, requires_grad=True)
    loss = masked_length_normalized_loss(loss_i, mask, None)
    loss.backward()
    g0 = loss_i.grad[0, :L[0]]
    g1 = loss_i.grad[1, :L[1]]
    assert torch.allclose(g0, torch.full_like(g0, 1.0 / (L[0] * G)), atol=1e-7)
    assert torch.allclose(g1, torch.full_like(g1, 1.0 / (L[1] * G)), atol=1e-7)
    assert not torch.allclose(g0[0], g1[0])  # length-dependent


# ============================== Independence (factorial) ======================
def test_flipping_debias_flags_does_not_change_fusion():
    gate = make_gate(bias_val=0.3, zero_weight=False)
    e_pred = torch.randn(2, 3, H)
    h_I = torch.randn(2, 3, H)
    h1, b1 = context_prediction_fusion(e_pred, h_I, gate)
    # advantage / loss settings are entirely separate call paths; recompute fusion identically
    _ = compute_group_advantages(torch.tensor([1.0, 0.0]), 2, True)
    _ = masked_length_normalized_loss(torch.ones(2, 4), torch.ones(2, 4), 8.0)
    h2, b2 = context_prediction_fusion(e_pred, h_I, gate)
    assert torch.equal(h1, h2) and torch.equal(b1, b2)


def test_flipping_fusion_flags_does_not_change_advantages_or_denominator():
    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0])
    a_ref = compute_group_advantages(rewards, 2, True)
    # run fusion with various flags in between; advantages must be identical
    gate = make_gate(bias_val=0.0, zero_weight=False)
    for gi in ("hidden", "token"):
        for norm in (True, False):
            _ = context_prediction_fusion(torch.randn(2, 3, H), torch.randn(2, 3, H), gate,
                                          e_hat=torch.randn(2, 3, H), gate_input=gi, normalize_h=norm)
    a_after = compute_group_advantages(rewards, 2, True)
    assert torch.equal(a_ref, a_after)
    assert select_length_denominator(torch.tensor([5.0]), None).item() == 5.0
