"""Control C2 — Dr. GRPO de-bias (Liu et al. 2025, "Understanding R1-Zero-Like
Training: A Critical Perspective").

Two switches, both OFF by default, living in code that EVERY arm trains through:
  * len : policy loss divided by the constant max_completion_length instead of
          each response's own token count  (unsloth/models/rl_replacements.py)
  * std : advantage = reward - group mean, without the / group std
          (trl/trainer/grpo_trainer.py)

The load-bearing property is that OFF is bit-identical to the pre-C2 code: every
settled result was trained through these functions. Those tests execute the
pre-C2 source straight out of git (commit PRE_C2) side by side with the current
source, so they compare against the real previous code, not a hand-copied one.

CPU-only: the functions are extracted with ast and exec'd, so unsloth (which
patches CUDA kernels at import) is never imported.
"""
import ast
import inspect
import os
import subprocess
from types import SimpleNamespace

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_REPL = "unsloth/models/rl_replacements.py"
TRL_GRPO = "trl/trainer/grpo_trainer.py"
PRE_C2 = "8fae960"   # last commit before the Dr. GRPO switches were added
NAMES = ["grpo_compute_loss", "UnslothEfficientGRPO"]


def _exec_defs(source, names):
    ns = {"torch": torch, "torch_compile_options": {}}
    found = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            exec(compile(ast.get_source_segment(source, node), RL_REPL, "exec"), ns)
            found.add(node.name)
    assert found == set(names), f"missing definitions: {set(names) - found}"
    return ns


def _git_show(path):
    try:
        return subprocess.run(["git", "show", f"{PRE_C2}:{path}"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("pre-C2 source unavailable (not a git checkout)")


def _current():
    with open(os.path.join(ROOT, RL_REPL)) as f:
        return _exec_defs(f.read(), NAMES)


def _pre_c2():
    return _exec_defs(_git_show(RL_REPL), NAMES)


@pytest.fixture(autouse=True)
def _eager(monkeypatch):
    # UnslothEfficientGRPO wraps its chunk step in torch.compile; the cluster
    # trains with TORCHDYNAMO_DISABLE=1, so eager is the configuration that runs.
    monkeypatch.setattr(torch, "compile", lambda fn=None, **kw: fn)
    monkeypatch.setattr(torch._dynamo, "mark_dynamic", lambda *a, **k: None)


def _batch(seed=0, B=4, T=6, H=8, V=11):
    g = torch.Generator().manual_seed(seed)
    new_h = torch.randn(B, T + 1, H, generator=g)
    old_h = new_h + 0.05 * torch.randn(B, T + 1, H, generator=g)
    lm_head = torch.randn(V, H, generator=g)
    ids = torch.randint(0, V, (B, T), generator=g)
    lengths = torch.tensor([2, 6, 3, 5])[:B]
    mask = (torch.arange(T)[None, :] < lengths[:, None]).long()
    adv = torch.randn(B, generator=g)
    return new_h, old_h, lm_head, ids, mask, adv


def _run(cls, batch, n_chunks, *extra, beta=0.04):
    new_h, old_h, lm_head, ids, mask, adv = batch
    x = new_h.clone().requires_grad_(True)
    loss, clen, kl = cls.apply(x, old_h, lm_head, ids, mask, adv, beta, None, n_chunks, *extra)
    loss.sum().backward()
    return loss.detach(), clen, kl, x.grad


# --- OFF must be bit-identical to the code every settled run trained with ------

def test_loss_function_off_is_bit_identical_to_pre_c2():
    old, new = _pre_c2()["grpo_compute_loss"], _current()["grpo_compute_loss"]
    g = torch.Generator().manual_seed(1)
    B, T, V = 3, 7, 13
    ol, nl = torch.randn(B, T, V, generator=g), torch.randn(B, T, V, generator=g)
    ids = torch.randint(0, V, (B, T), generator=g)
    mask = (torch.arange(T)[None] < torch.tensor([[7], [2], [4]])).long()
    adv = torch.randn(B, generator=g)
    ref = old(ol, nl, ids, mask, 0.04, adv)
    for got in (new(ol, nl, ids, mask, 0.04, adv), new(ol, nl, ids, mask, 0.04, adv, None)):
        for r, x in zip(ref, got):
            assert torch.equal(r, x)


@pytest.mark.parametrize("n_chunks", [1, 2, 4])
def test_efficient_grpo_off_is_bit_identical_to_pre_c2(n_chunks):
    old, new = _pre_c2()["UnslothEfficientGRPO"], _current()["UnslothEfficientGRPO"]
    b = _batch()
    ref = _run(old, b, n_chunks)
    # both the 9-argument call (constant omitted) and the explicit None
    for got in (_run(new, b, n_chunks), _run(new, b, n_chunks, None)):
        for r, x in zip(ref, got):   # loss, completion_length, mean_kl, gradient
            assert torch.equal(r, x)


def test_advantage_off_branch_is_the_pre_c2_expression():
    original = "advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)"
    assert _git_show(TRL_GRPO).count(original) == 1
    with open(os.path.join(ROOT, TRL_GRPO)) as f:
        src = f.read()
    guard = 'if getattr(self, "dr_grpo_scale_rewards", True):'
    debiased = "advantages = rewards - mean_grouped_rewards"
    assert src.count(guard) == 1 and src.count(original) == 1 and src.count(debiased) == 1
    assert src.index(guard) < src.index(original) < src.index(debiased)


# --- ON must implement Dr. GRPO -------------------------------------------------

def test_len_mode_matches_dr_grpo_formula_and_is_chunk_invariant():
    cls = _current()["UnslothEfficientGRPO"]
    b = new_h, old_h, lm_head, ids, mask, adv = _batch(seed=2)
    L, beta = 512.0, 0.04
    new = torch.gather((new_h @ lm_head.t())[:, :-1].log_softmax(-1), -1, ids[..., None])[..., 0]
    old = torch.gather((old_h @ lm_head.t())[:, :-1].log_softmax(-1), -1, ids[..., None])[..., 0]
    kl = torch.exp(old - new) - (old - new) - 1.0
    expected = ((-(adv[:, None] - beta * kl) * mask).sum(1) / L).mean()
    for n in (1, 2, 4):
        assert torch.allclose(_run(cls, b, n, L)[0].squeeze(), expected, rtol=1e-5, atol=1e-7)
    # the logged completion_length and KL still use true per-response lengths
    base, dr = _run(cls, b, 2), _run(cls, b, 2, L)
    assert torch.equal(base[1], dr[1]) and torch.equal(base[2], dr[2])


@pytest.mark.parametrize("constant, expected_ratio", [(None, 3.0), (9.0, 1.0)])
def test_token_weight_depends_on_response_length_only_without_constant(constant, expected_ratio):
    """The bias itself: under original GRPO a token in a 3-token response gets 3x
    the gradient of a token in a 9-token response with the same advantage. Under
    Dr. GRPO every token weighs the same."""
    f = _current()["grpo_compute_loss"]
    B, T, V = 2, 9, 5
    new = torch.zeros(B, T, V, requires_grad=True)
    mask = torch.zeros(B, T)
    mask[0, :3] = 1
    mask[1, :] = 1
    loss, _, _ = f(torch.zeros(B, T, V), new, torch.zeros(B, T, dtype=torch.long),
                   mask, 0.0, torch.ones(B), constant)
    loss.backward()
    ratio = new.grad[0, 0, 0].abs() / new.grad[1, 0, 0].abs()
    assert torch.isclose(ratio, torch.tensor(expected_ratio))


# --- wiring ---------------------------------------------------------------------

def _trainer():
    return SimpleNamespace(args=SimpleNamespace(max_completion_length=512), num_generations=4)


def test_apply_dr_grpo_off_leaves_trainer_untouched():
    from variant_common import apply_dr_grpo
    t = _trainer()
    apply_dr_grpo(t, "off")
    assert not hasattr(t, "dr_grpo_len_constant")
    assert not hasattr(t, "dr_grpo_scale_rewards")


@pytest.mark.parametrize("mode, const, scale",
                         [("len", 512.0, True), ("std", None, False), ("both", 512.0, False)])
def test_apply_dr_grpo_modes(mode, const, scale):
    from variant_common import apply_dr_grpo
    t = _trainer()
    apply_dr_grpo(t, mode)
    assert getattr(t, "dr_grpo_len_constant", None) == const
    assert getattr(t, "dr_grpo_scale_rewards", True) is scale


def test_apply_dr_grpo_rejects_bad_mode_and_non_grpo_trainer():
    from variant_common import apply_dr_grpo
    with pytest.raises(ValueError):
        apply_dr_grpo(_trainer(), "yes")
    with pytest.raises(RuntimeError):
        apply_dr_grpo(SimpleNamespace(args=SimpleNamespace()), "both")


class _RecordingTrainer:
    def __init__(self):
        self.args = SimpleNamespace(max_completion_length=512)
        self.num_generations = 4
        self.seen = None

    def train(self, resume_from_checkpoint=None):
        # the trainer's state at the moment training actually starts
        self.seen = (resume_from_checkpoint,
                     getattr(self, "dr_grpo_len_constant", None),
                     getattr(self, "dr_grpo_scale_rewards", True))


@pytest.mark.parametrize("mode, const, scale", [("off", None, True), ("both", 512.0, False)])
def test_train_with_resume_applies_dr_grpo_before_training(tmp_path, mode, const, scale):
    """The switch must reach the trainer BEFORE train(). A -drboth directory whose
    trainer ran the original objective would be a mislabeled arm — which is
    exactly what the first version of this wiring produced."""
    from variant_common import train_with_resume
    t = _RecordingTrainer()
    train_with_resume(t, str(tmp_path / "exp"), SimpleNamespace(seed=42, dr_grpo=mode))
    assert t.seen == (None, const, scale)


def _pre_c2_checkpoint(exp):
    ck = exp / "checkpoint-10"
    ck.mkdir(parents=True)
    (ck / "trainer_state.json").write_text("{}")
    (exp / "run_config.json").write_text('{"seed": 42}')   # written before C2: no dr_grpo key
    return ck


def test_pre_c2_checkpoint_still_resumes_as_off(tmp_path):
    from variant_common import train_with_resume
    exp = tmp_path / "exp"
    ck = _pre_c2_checkpoint(exp)
    t = _RecordingTrainer()
    train_with_resume(t, str(exp), SimpleNamespace(seed=42, dr_grpo="off"))
    assert t.seen[0] == str(ck)


def test_pre_c2_checkpoint_refuses_to_resume_as_dr_grpo(tmp_path):
    from variant_common import train_with_resume
    exp = tmp_path / "exp"
    _pre_c2_checkpoint(exp)
    with pytest.raises(SystemExit):
        train_with_resume(_RecordingTrainer(), str(exp), SimpleNamespace(seed=42, dr_grpo="both"))


def test_dr_grpo_runs_get_their_own_directory_and_arm():
    from variant_common import build_arg_parser, experiment_dir, parse_experiment_name
    from analyse_replicates import parse_suffix
    p = build_arg_parser("v1_grpo_baseline")
    base = ["--model_size", "1.5b", "--repetition_penalty", "1.0"]
    cold = experiment_dir("1.5b", "v1_grpo_baseline", p.parse_args(base))
    dr = experiment_dir("1.5b", "v1_grpo_baseline", p.parse_args(base + ["--dr_grpo", "both"]))
    assert cold == "./experiments/1.5b-v1_grpo_baseline-rp1"       # existing dirs unchanged
    assert dr == "./experiments/1.5b-v1_grpo_baseline-rp1-drboth"
    size, variant, suffix = parse_experiment_name(os.path.basename(dr))
    assert (size, variant) == ("1.5b", "v1_grpo_baseline")
    assert parse_suffix(suffix) == (42, 1.0, "full", None, "both", None)
    assert parse_suffix("-rp1") == (42, 1.0, "full", None, None, None)
    assert parse_suffix("-seed43-rp1-warmoff-drboth") == (43, 1.0, "full", "off", "both", None)
    assert parse_experiment_name("1.5b-v3_learned_lambda-drlen")[2] == "-drlen"
    # second task: -dsmath must parse, and must not be read as -dr
    math_dir = experiment_dir("1.5b", "v3_learned_lambda",
                              p.parse_args(base + ["--dataset", "math"]))
    assert math_dir == "./experiments/1.5b-v3_learned_lambda-rp1-dsmath"
    assert parse_suffix("-rp1-dsmath") == (42, 1.0, "full", None, None, "math")
    assert parse_experiment_name("1.5b-v1_grpo_baseline-dsmath")[2] == "-dsmath"
