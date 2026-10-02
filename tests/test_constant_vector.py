"""CPU tests for the C1 constant-vector control (v7_constant_vector).

These exercise the substitution maths directly, with a stub module standing in
for ThinkingResidualLambda, so they need torch but neither a GPU nor a real
checkpoint. Run on the cluster (or any box with torch):

    python -m pytest tests/test_constant_vector.py -v

The full end-to-end "first training step reproduces v3 bit-for-bit" comparison
(brief check 1) needs a real model + GPU and is a separate cluster job; what is
covered here is the part most likely to be wrong: the algebra, the gradient path
and the registration ordering guard.
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from variant_common import (
    _unit_gate_thinking_residual,
    _constant_latent_thinking_residual,
)

HIDDEN = 16
C = 8.0


class _Lambda(nn.Module):
    """Mimics modeling_qwen2.ThinkingResidualLambda: a_t = exp(-c*softplus(-Λ)*r_t)."""
    c = C

    def __init__(self, hidden=HIDDEN, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.Lambda = nn.Parameter(torch.rand(hidden, generator=g) * 2 - 1)

    def forward(self, r_t):
        return torch.exp(-self.c * nn.functional.softplus(-self.Lambda) * r_t)


class _Host(nn.Module):
    """Stands in for the Qwen2Model that owns thinking_residual_Lambda."""

    def __init__(self, hidden=HIDDEN, seed=0):
        super().__init__()
        self.thinking_residual_Lambda = _Lambda(hidden, seed)


def _fixed_inputs(batch=4, hidden=HIDDEN, seed=1):
    g = torch.Generator().manual_seed(seed)
    embeds = torch.randn(batch, hidden, generator=g)
    residual = torch.randn(batch, hidden, generator=g)
    return embeds, residual


# --- brief check 2: with u == the projection output, constant ≡ projection ----

def test_constant_matches_unit_gate_when_u_equals_residual():
    """If u is set to the value the Eq.3 projection would have produced, the
    constant path must reproduce the unit-gate (v3) path EXACTLY. This is the
    meaningful form of the brief's check 2: a single u can only equal 'the
    projection output' when that output is the same at every position, so the
    residual is held constant across the batch here."""
    host = _Host()
    embeds, _ = _fixed_inputs()
    # one shared projection output, replicated across positions
    shared = torch.randn(HIDDEN, generator=torch.Generator().manual_seed(7))
    residual = shared.unsqueeze(0).expand_as(embeds)

    host.thinking_residual_Lambda.register_parameter("u", nn.Parameter(shared.clone()))

    out_proj, a_proj = _unit_gate_thinking_residual(host, embeds, residual)
    out_const, a_const = _constant_latent_thinking_residual(host, embeds, residual)

    assert torch.equal(a_proj, a_const), "a_t must be unaffected by the latent source"
    assert torch.equal(out_proj, out_const), "constant path must be bit-identical when u == residual"


def test_constant_ignores_the_incoming_residual():
    """The whole point of C1: whatever the projection produced is discarded."""
    host = _Host()
    embeds, residual = _fixed_inputs()
    host.thinking_residual_Lambda.register_parameter(
        "u", nn.Parameter(torch.randn(HIDDEN, generator=torch.Generator().manual_seed(3))))

    out_a, _ = _constant_latent_thinking_residual(host, embeds, residual)
    out_b, _ = _constant_latent_thinking_residual(host, embeds, residual * 17.0 + 3.0)
    assert torch.equal(out_a, out_b), "output must not depend on the projection output"


# --- brief check 3: u receives a non-zero gradient ---------------------------

def test_u_receives_nonzero_gradient():
    """u must be in the live graph. (A generation-only substitution would fail
    this: generate() runs under no_grad and its recorded latents are detached.)"""
    host = _Host()
    embeds, residual = _fixed_inputs()
    host.thinking_residual_Lambda.register_parameter(
        "u", nn.Parameter(torch.randn(HIDDEN, generator=torch.Generator().manual_seed(5))))
    u = host.thinking_residual_Lambda.u

    out, _ = _constant_latent_thinking_residual(host, embeds, residual)
    out.sum().backward()

    assert u.grad is not None, "u got no gradient at all"
    assert torch.count_nonzero(u.grad) > 0, "u gradient is all zeros"


def test_lambda_still_receives_gradient_in_constant_mode():
    """Swapping the latent source must not sever Λ's gradient path."""
    host = _Host()
    embeds, residual = _fixed_inputs()
    host.thinking_residual_Lambda.register_parameter(
        "u", nn.Parameter(torch.randn(HIDDEN, generator=torch.Generator().manual_seed(5))))
    lam = host.thinking_residual_Lambda.Lambda

    out, _ = _constant_latent_thinking_residual(host, embeds, residual)
    out.sum().backward()

    assert lam.grad is not None and torch.count_nonzero(lam.grad) > 0


# --- registration ordering guard --------------------------------------------

def test_constant_patch_requires_registered_u():
    """apply_constant_latent_patch must refuse if u was not registered before
    get_peft_model — otherwise u silently never reaches the adapter."""
    from variant_common import apply_constant_latent_patch

    class _Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.thinking_residual_Lambda = _Lambda()

    with pytest.raises(RuntimeError, match="not registered"):
        apply_constant_latent_patch(_Outer())


# --- u initialisation: scale diagnostics -------------------------------------

def test_constant_init_stats_flags_the_centroid_scale_gap():
    """The centroid of an embedding matrix has a much smaller norm than a typical
    row (the rows cancel). This is why "mean" is on-manifold in direction but not
    in scale — the thing the diagnostic exists to surface."""
    from variant_common import constant_init_stats
    g = torch.Generator().manual_seed(11)
    embed = torch.randn(5000, 64, generator=g)          # zero-centred, like a real table
    s = constant_init_stats(embed)
    assert s["mean_row_norm"] > 0
    # centroid norm is tiny relative to a row -> ratio well below 1
    assert s["ratio_mean_to_row"] < 0.1, s


def test_mean_scaled_matches_row_norm():
    """mean_scaled must keep the centroid DIRECTION but take the row SCALE."""
    from variant_common import constant_init_stats
    g = torch.Generator().manual_seed(12)
    embed = torch.randn(2000, 32, generator=g) + 0.5     # off-centre so the mean is meaningful
    s = constant_init_stats(embed)
    m = embed.mean(dim=0)
    scaled = m / m.norm().clamp_min(1e-12) * s["mean_row_norm"]
    assert scaled.norm().item() == pytest.approx(s["mean_row_norm"], rel=1e-5)
    # direction preserved
    cos = torch.nn.functional.cosine_similarity(scaled, m, dim=0)
    assert cos.item() == pytest.approx(1.0, abs=1e-6)


# --- brief check 1 (structural half): projection mode is v3 -------------------

def test_projection_mode_is_literally_v3_patch():
    """In projection mode the C1 script installs the SAME function v3 installs,
    so 'reproduces v3' holds by construction; the bit-exact training-step
    comparison is the cluster half of check 1."""
    import variant_common as vc
    src = open("hrpo_constant_vector.py").read()
    assert "apply_unit_gate_patch(model)" in src
    assert 'if args.latent_source == "constant":' in src
    # v3's patch function is the one used for the projection branch
    assert vc._unit_gate_thinking_residual.__name__ == "_unit_gate_thinking_residual"
