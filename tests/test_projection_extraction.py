"""The Eq.3 projection was extracted from transformers/generation/utils.py's
_sample loop into hrpo_project_latent() so the teacher-forced Stage-0 path can
reuse it instead of reimplementing it.

That refactor touches the code path EVERY existing arm uses for its rollouts, so
it has to be provably identical, not just plausibly so. These tests pin it
against a verbatim copy of the original inline expression.

    python -m pytest tests/test_projection_extraction.py -v
"""
import pytest

torch = pytest.importorskip("torch")

from transformers.generation.utils import hrpo_project_latent

VOCAB, DIM = 512, 64


def _original_inline(probs, embed_weight):
    """Verbatim copy of the pre-extraction code (utils.py, before the refactor):

        last_thinking_states = torch.einsum('bv,vd->bd', probs, W)
        last_thinking_states /= torch.sqrt((probs ** 2).sum(-1, keepdim=True))
                                     .to(last_thinking_states.dtype)
    """
    out = torch.einsum('bv,vd->bd', probs, embed_weight)
    out /= torch.sqrt((probs ** 2).sum(-1, keepdim=True)).to(out.dtype)
    return out


def _fixtures(batch=6, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, VOCAB, generator=g, dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1).to(dtype)
    embed = torch.randn(VOCAB, DIM, generator=g, dtype=torch.float32).to(dtype)
    return probs, embed


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matches_original_bitwise(dtype):
    """The single-decode-step case generation actually uses."""
    probs, embed = _fixtures(dtype=dtype)
    assert torch.equal(hrpo_project_latent(probs, embed),
                       _original_inline(probs, embed)), \
        "extraction changed generation behaviour — every existing arm would shift"


def test_matches_original_for_peaked_and_uniform_distributions():
    """Edge cases: a near-one-hot p (what greedy decoding produces) and a uniform
    p (maximum entropy). Both must match the original exactly."""
    embed = torch.randn(VOCAB, DIM, generator=torch.Generator().manual_seed(2))

    peaked = torch.full((3, VOCAB), 1e-9)
    peaked[0, 5] = 1.0; peaked[1, 100] = 1.0; peaked[2, 511] = 1.0
    peaked = peaked / peaked.sum(-1, keepdim=True)

    uniform = torch.full((3, VOCAB), 1.0 / VOCAB)

    for p in (peaked, uniform):
        assert torch.equal(hrpo_project_latent(p, embed), _original_inline(p, embed))


def test_sequence_shape_equals_per_position_loop():
    """Stage 0 needs (batch, seq, vocab) -> (batch, seq, dim), and it must agree
    with projecting each sequence separately through the 2-D path generation uses.

    Agreement here is NUMERICAL, not bitwise, and deliberately so: the batched
    form issues one (B*S, V) @ (V, D) GEMM while the loop issues B separate
    (S, V) @ (V, D) GEMMs, and BLAS does not promise the same accumulation order
    across different shapes. Observed deviation is ~1e-6 in fp32 — floating-point
    noise over a 512-term reduction.

    The tolerance is still tight enough to catch a real defect: a wrong reshape,
    a transposed einsum or a mis-broadcast norm moves values by O(0.1-1), which is
    5+ orders of magnitude above this bound. The bitwise guarantee that matters
    for not perturbing existing arms — the 2-D generation shape — is asserted in
    test_matches_original_bitwise above."""
    g = torch.Generator().manual_seed(3)
    B, S = 4, 7
    probs = torch.softmax(torch.randn(B, S, VOCAB, generator=g), dim=-1)
    embed = torch.randn(VOCAB, DIM, generator=g)

    batched = hrpo_project_latent(probs, embed)
    assert batched.shape == (B, S, DIM)

    worst = 0.0
    for b in range(B):
        per_pos = _original_inline(probs[b], embed)      # (S, DIM)
        worst = max(worst, (batched[b] - per_pos).abs().max().item())
        assert torch.allclose(batched[b], per_pos, rtol=1e-5, atol=1e-5)
    assert worst < 1e-4, f"deviation {worst:.2e} is too large to be FP noise"


def test_peaked_projection_is_close_to_the_embedding_row():
    """Sanity on the maths itself: for a one-hot p, W_e^T p / ||p|| is exactly
    that embedding row. This is why a typical projection output sits at ROW
    scale, which is what the C1 u-init scale discussion was about."""
    embed = torch.randn(VOCAB, DIM, generator=torch.Generator().manual_seed(4))
    p = torch.zeros(1, VOCAB); p[0, 42] = 1.0
    out = hrpo_project_latent(p, embed)
    assert torch.allclose(out[0], embed[42], atol=1e-6)


def test_mixed_dtype_probs_and_embedding_must_be_cast_by_the_caller():
    """Regression: stage0's _project upcast logits to fp32 for a stable softmax
    and then fed fp32 probs into the einsum against a bf16 embedding table,
    killing both --blend on smoke runs with

        RuntimeError: expected scalar type BFloat16 but found Float

    The helper deliberately does NOT paper over this (a silent internal cast
    would both hide caller bugs and change the dtype behaviour of the rollout
    path). The contract is: the CALLER casts probs to the embedding dtype."""
    embed_bf16 = torch.randn(VOCAB, DIM).to(torch.bfloat16)
    probs_fp32 = torch.softmax(torch.randn(3, VOCAB), dim=-1)     # fp32

    with pytest.raises(RuntimeError):
        hrpo_project_latent(probs_fp32, embed_bf16)

    # what stage0 now does: stable fp32 softmax, cast back before the einsum
    out = hrpo_project_latent(probs_fp32.to(embed_bf16.dtype), embed_bf16)
    assert out.dtype == embed_bf16.dtype
    assert out.shape == (3, DIM)
    assert torch.isfinite(out.float()).all()
