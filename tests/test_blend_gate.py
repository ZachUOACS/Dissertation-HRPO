"""D3 uncertainty-gated blend (--blend_gate divergence).

The gate replaces thinking_residual at BOTH call sites, so its maths must be exact and its
default ("off") must touch nothing. The weight depends only on (token embedding, latent), so the
rollout and the loss forward compute the same blend from the same two tensors.

CPU only. Run from the repo root:  python -m pytest tests/test_blend_gate.py -q
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import variant_common as vc  # noqa: E402

DIM = 16


def test_weight_is_zero_when_the_latent_is_the_token_and_full_when_orthogonal():
    e = torch.randn(5, DIM)
    beta_same = vc.divergence_gate_weights(e, e.clone(), 0.5, 0.02, 0.1)
    assert torch.allclose(beta_same, torch.zeros(5, 1), atol=1e-6)
    h = torch.zeros(5, DIM)
    h[:, 0] = 1.0
    e2 = torch.zeros(5, DIM)
    e2[:, 1] = 1.0                                         # cos 0 -> d 1 -> above d1
    assert torch.allclose(vc.divergence_gate_weights(e2, h, 0.5, 0.02, 0.1),
                          torch.full((5, 1), 0.5))


def test_weight_ramps_linearly_between_d0_and_d1():
    # build pairs with known cosine c: h = c e + sqrt(1-c^2) u, u orthogonal to e
    e = torch.zeros(3, DIM); e[:, 0] = 1.0
    u = torch.zeros(3, DIM); u[:, 1] = 1.0
    ds = torch.tensor([0.02, 0.06, 0.10])                  # d0, midpoint, d1
    c = (1 - ds).unsqueeze(1)
    h = c * e + torch.sqrt(1 - c ** 2) * u
    beta = vc.divergence_gate_weights(e, h, 0.5, 0.02, 0.10).squeeze(1)
    assert torch.allclose(beta, torch.tensor([0.0, 0.25, 0.5]), atol=1e-5)


class _Lambda(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Lambda = torch.nn.Parameter(torch.zeros(DIM))


class _Inner(torch.nn.Module):            # stands in for Qwen2Model
    def __init__(self):
        super().__init__()
        self.thinking_residual_Lambda = _Lambda()

    def thinking_residual(self, embeds, residual, eps=1e-8):    # stock blend, never used by D3
        return embeds, torch.ones_like(embeds)


class _CausalLM(torch.nn.Module):
    def __init__(self, answer_start="####"):
        super().__init__()
        self.answer_start = answer_start
        self.model = _Inner()


@pytest.mark.parametrize("shape", [(7, DIM), (4, 1, DIM)])  # loss forward (N, d) and rollout (B, 1, d)
def test_installed_blend_is_convex_and_reports_the_token_weight(shape):
    lm = _CausalLM()
    assert vc.set_blend_gate(lm, "divergence", 0.5, 0.02, 0.1)
    inner = lm.model
    e = torch.randn(*shape)
    h = torch.randn(*shape)
    out, a_t = inner.thinking_residual(e, h)
    beta = vc.divergence_gate_weights(e, h, 0.5, 0.02, 0.1)
    assert torch.allclose(out, (1 - beta) * e + beta * h, atol=1e-6)
    assert a_t.shape == e.shape
    assert torch.allclose(a_t, (1 - beta).expand_as(e))
    # the Eq.4 parameters play no part: no gradient can reach Lambda through the D3 blend
    assert not out.requires_grad


def test_same_inputs_give_the_same_blend_at_both_call_sites():
    """The rollout passes (B, 1, d); the loss forward passes the same rows flattened to (N, d)."""
    lm = _CausalLM()
    vc.set_blend_gate(lm, "divergence", 0.5, 0.02, 0.1)
    e = torch.randn(6, 1, DIM)
    h = e + 0.3 * torch.randn(6, 1, DIM)
    roll, _ = lm.model.thinking_residual(e, h)
    loss, _ = lm.model.thinking_residual(e.squeeze(1), h.squeeze(1))
    assert torch.allclose(roll.squeeze(1), loss, atol=1e-6)


def test_bf16_inputs_stay_bf16():
    lm = _CausalLM()
    vc.set_blend_gate(lm, "divergence", 0.5, 0.02, 0.1)
    e = torch.randn(3, DIM).to(torch.bfloat16)
    h = torch.randn(3, DIM)                                 # latent may arrive in fp32
    out, a_t = lm.model.thinking_residual(e, h)
    assert out.dtype == torch.bfloat16 and a_t.dtype == torch.bfloat16


def test_off_and_invalid_settings():
    lm = _CausalLM()
    original = lm.model.thinking_residual
    assert vc.set_blend_gate(lm, "off") is False
    assert lm.model.thinking_residual == original
    with pytest.raises(ValueError):
        vc.set_blend_gate(lm, "divergence", 0.5, None, 0.1)
    with pytest.raises(ValueError):
        vc.set_blend_gate(lm, "divergence", 0.5, 0.1, 0.05)     # d0 >= d1
    with pytest.raises(ValueError):
        vc.set_blend_gate(lm, "divergence", 1.5, 0.02, 0.1)     # blend_max > 1
    with pytest.raises(RuntimeError):
        vc.set_blend_gate(_CausalLM(""), "divergence", 0.5, 0.02, 0.1)   # v1 / latent-off


def test_default_args_leave_the_trainer_untouched():
    lm = _CausalLM()
    original = lm.model.thinking_residual

    class T:
        model = lm
    t = T()
    vc.apply_blend_gate(t, vc.build_arg_parser("v3_learned_lambda").parse_args([]))
    assert lm.model.thinking_residual == original
    assert not getattr(t, "blend_gate_on", False)


def make(*extra):
    args = vc.build_arg_parser("v3_learned_lambda").parse_args(["--model_size", "1.5b", *extra])
    return os.path.basename(vc.experiment_dir("1.5b", "v3_learned_lambda", args))


def test_gated_arm_is_a_new_arm_and_fingerprinted():
    name = make("--dataset", "math", "--dr_grpo", "both", "--seed", "43",
                "--latent_temperature", "1.0", "--blend_gate", "divergence",
                "--blend_d0", "0.02", "--blend_d1", "0.1")
    assert name == "new-1.5b-v3_learned_lambda-seed43-rp1-drboth-dsmath-lt1-dg0.5d0.02to0.1"
    assert vc.parse_experiment_name(name) is None
    for key in ("blend_gate", "blend_max", "blend_d0", "blend_d1"):
        assert key in vc._CONFIG_FINGERPRINT and key in vc._FINGERPRINT_DEFAULTS
    defaults = vc.build_arg_parser("v3_learned_lambda").parse_args([])
    assert defaults.blend_gate == vc._FINGERPRINT_DEFAULTS["blend_gate"] == "off"
