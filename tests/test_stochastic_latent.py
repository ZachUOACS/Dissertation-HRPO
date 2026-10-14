"""C1: stochastic latent with its own gradient (--stochastic_latent, after LEPO).

Three properties carry the arm:
  * the Gumbel-max draw samples the token from p exactly as torch.multinomial does, so the
    discrete rollout distribution is unchanged and only the latent becomes stochastic;
  * a one-hot soft label gives exactly the ordinary token loss (value and gradient), so positions
    without a latent, and every other arm, train as before;
  * a soft label gives the LEPO Eq. 9 gradient, A * sum_k w_k grad log pi(k).

CPU only; the loss functions are extracted with ast as in tests/test_dr_grpo.py, so unsloth is
never imported.   python -m pytest tests/test_stochastic_latent.py -q
"""
import ast
import os
import sys

import pytest

torch = pytest.importorskip("torch")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformers.generation.utils import hrpo_gumbel_latent_step, hrpo_soft_labels  # noqa: E402

import variant_common as vc  # noqa: E402

RL_REPL = "unsloth/models/rl_replacements.py"


def _loss_defs():
    with open(os.path.join(ROOT, RL_REPL)) as f:
        source = f.read()
    ns = {"torch": torch, "torch_compile_options": {}}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and \
                node.name in ("grpo_compute_loss", "UnslothEfficientGRPO"):
            exec(compile(ast.get_source_segment(source, node), RL_REPL, "exec"), ns)
    return ns


@pytest.fixture(autouse=True)
def _eager(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn=None, **kw: fn)
    monkeypatch.setattr(torch._dynamo, "mark_dynamic", lambda *a, **k: None)


# --- rollout helpers ------------------------------------------------------------------------------

def test_gumbel_max_samples_tokens_from_p():
    torch.manual_seed(0)
    p = torch.tensor([0.5, 0.3, 0.15, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    probs = p.expand(60000, -1).clone()
    tokens, z = hrpo_gumbel_latent_step(probs, 0.5)
    freq = torch.bincount(tokens, minlength=10).float() / tokens.numel()
    assert torch.allclose(freq, p, atol=0.01)
    assert freq[4:].sum() == 0                                  # outside the support: never drawn
    assert torch.allclose(z.sum(-1), torch.ones(60000), atol=1e-5)
    assert torch.equal(z.argmax(-1), tokens)                     # the latent agrees with the token
    assert (z[:, 4:] == 0).all()


def test_gumbel_temperature_controls_softness():
    torch.manual_seed(1)
    probs = torch.tensor([[0.6, 0.3, 0.1] + [0.0] * 7]).expand(20000, -1).clone()
    _, z_soft = hrpo_gumbel_latent_step(probs, 1.0)
    torch.manual_seed(1)
    _, z_hard = hrpo_gumbel_latent_step(probs, 0.1)
    assert z_hard.max(-1).values.mean() > z_soft.max(-1).values.mean()


def test_soft_labels_are_one_hot_where_no_latent_is_used():
    torch.manual_seed(2)
    probs = torch.softmax(torch.randn(6, 20), -1)
    tokens, z = hrpo_gumbel_latent_step(probs, 0.5)
    use = torch.tensor([True, False, True, False, True, True])
    idx, w = hrpo_soft_labels(z, tokens, use, k=8)
    assert idx.shape == w.shape == (6, 8)
    for r in range(6):
        assert idx[r, 0] == tokens[r]
        assert torch.isclose(w[r].sum(), torch.tensor(1.0))
        if not use[r]:
            assert (idx[r] == tokens[r]).all() and w[r, 0] == 1 and (w[r, 1:] == 0).all()
        else:
            assert (w[r, :-1] >= w[r, 1:]).all()                 # top-k, largest first


# --- loss -------------------------------------------------------------------------------------------

def _logits(seed=3, B=3, T=5, V=12):
    g = torch.Generator().manual_seed(seed)
    old = torch.randn(B, T, V, generator=g)
    new = torch.randn(B, T, V, generator=g)
    ids = torch.randint(0, V, (B, T), generator=g)
    mask = (torch.arange(T)[None] < torch.tensor([[5], [3], [4]])).long()
    adv = torch.randn(B, generator=g)
    return old, new, ids, mask, adv


def _one_hot_labels(ids, k=8):
    idx = ids.unsqueeze(-1).expand(*ids.shape, k).clone()
    w = torch.zeros(*ids.shape, k)
    w[..., 0] = 1.0
    return idx, w


@pytest.mark.parametrize("constant", [None, 512.0])
def test_one_hot_soft_labels_reproduce_the_token_loss_exactly(constant):
    f = _loss_defs()["grpo_compute_loss"]
    old, new, ids, mask, adv = _logits()
    a = new.clone().requires_grad_(True)
    b = new.clone().requires_grad_(True)
    ref = f(old, a, ids, mask, 0.04, adv, constant)
    got = f(old, b, ids, mask, 0.04, adv, constant, *_one_hot_labels(ids))
    for r, x in zip(ref, got):
        assert torch.equal(r, x)
    ref[0].backward()
    got[0].backward()
    assert torch.allclose(a.grad, b.grad, atol=1e-7)


def test_soft_labels_give_the_lepo_gradient():
    f = _loss_defs()["grpo_compute_loss"]
    old, new, ids, mask, adv = _logits(seed=4)
    B, T, V = new.shape
    g = torch.Generator().manual_seed(5)
    idx = torch.randint(0, V, (B, T, 4), generator=g)
    idx[..., 0] = ids
    w = torch.softmax(torch.randn(B, T, 4, generator=g), -1)
    x = new.clone().requires_grad_(True)
    loss, _, _ = f(old, x, ids, mask, 0.0, adv, None, idx, w)     # beta 0: policy term only
    loss.backward()
    y = new.clone().requires_grad_(True)
    soft_logp = (torch.gather(y.log_softmax(-1), -1, idx) * w).sum(-1)
    m = mask.float()
    expected = (-(soft_logp * adv[:, None] * m).sum(1) / m.sum(1)).mean()
    expected.backward()
    assert torch.allclose(x.grad, y.grad, atol=1e-6)
    # the loss VALUE does not depend on the labels (exp(x - x.detach()) == 1)
    base, _, _ = f(old, new, ids, mask, 0.0, adv, None)
    assert torch.allclose(loss.detach(), base, atol=1e-7)


@pytest.mark.parametrize("n_chunks", [1, 2, 4])   # factors of the batch, as grpo_accumulated_loss picks
def test_efficient_grpo_with_soft_labels_is_chunk_invariant_and_one_hot_is_neutral(n_chunks):
    cls = _loss_defs()["UnslothEfficientGRPO"]
    g = torch.Generator().manual_seed(6)
    B, T, H, V = 4, 5, 8, 11
    new_h = torch.randn(B, T + 1, H, generator=g)
    old_h = new_h + 0.05 * torch.randn(B, T + 1, H, generator=g)
    lm_head = torch.randn(V, H, generator=g)
    ids = torch.randint(0, V, (B, T), generator=g)
    mask = (torch.arange(T)[None] < torch.tensor([[5], [2], [4], [3]])).long()
    adv = torch.randn(B, generator=g)
    idx = torch.randint(0, V, (B, T, 4), generator=g)
    idx[..., 0] = ids
    w = torch.softmax(torch.randn(B, T, 4, generator=g), -1)

    def run(n, *extra):
        x = new_h.clone().requires_grad_(True)
        out = cls.apply(x, old_h, lm_head, ids, mask, adv, 0.04, None, n, None, *extra)
        out[0].sum().backward()
        return out[0].detach(), x.grad

    base_loss, base_grad = run(1)
    oh_loss, oh_grad = run(n_chunks, *_one_hot_labels(ids, 4))
    assert torch.allclose(oh_loss, base_loss, atol=1e-6)
    assert torch.allclose(oh_grad, base_grad, atol=1e-6)
    s1_loss, s1_grad = run(1, idx, w)
    sn_loss, sn_grad = run(n_chunks, idx, w)
    assert torch.allclose(s1_grad, sn_grad, atol=1e-6)
    assert not torch.allclose(s1_grad, base_grad, atol=1e-6)    # soft labels do change the update


# --- wiring -----------------------------------------------------------------------------------------

class _Inner(torch.nn.Module):
    pass


class _CausalLM(torch.nn.Module):
    def __init__(self, answer_start="####"):
        super().__init__()
        self.answer_start = answer_start
        self.model = _Inner()


def test_setter_and_refusals():
    lm = _CausalLM()
    assert vc.set_stochastic_latent(lm, None) == 0 and not hasattr(lm, "stochastic_latent")
    assert vc.set_stochastic_latent(lm, 0.5) == 1 and lm.stochastic_latent == 0.5
    with pytest.raises(RuntimeError):
        vc.set_stochastic_latent(_CausalLM(""), 0.5)
    with pytest.raises(ValueError):
        vc.set_stochastic_latent(_CausalLM(), 0.0)

    class T:
        model = _CausalLM()
    args = vc.build_arg_parser("v3_learned_lambda").parse_args(
        ["--stochastic_latent", "0.5", "--latent_temperature", "1.0"])
    with pytest.raises(SystemExit):
        vc.apply_stochastic_latent(T(), args)
    t = T()
    vc.apply_stochastic_latent(t, vc.build_arg_parser("v3_learned_lambda").parse_args([]))
    assert not getattr(t, "stochastic_latent_on", False)


def test_naming_and_fingerprint():
    args = vc.build_arg_parser("v3_learned_lambda").parse_args(
        ["--model_size", "1.5b", "--dataset", "math", "--dr_grpo", "both", "--seed", "43",
         "--stochastic_latent", "0.5"])
    name = os.path.basename(vc.experiment_dir("1.5b", "v3_learned_lambda", args))
    assert name == "new-1.5b-v3_learned_lambda-seed43-rp1-drboth-dsmath-sl0.5"
    assert vc.parse_experiment_name(name) is None
    assert "stochastic_latent" in vc._CONFIG_FINGERPRINT
    assert vc._FINGERPRINT_DEFAULTS["stochastic_latent"] is None
