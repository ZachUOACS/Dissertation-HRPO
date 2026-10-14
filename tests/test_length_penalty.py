"""Length-penalised reward (length_penalty.py; Arora & Zanette 2025, arXiv 2502.04463).

What must hold:
  * it is the paper's formula: correct responses earn 1 - alpha * sigmoid((len - mean) / std), mean
    and (population) std over the correct responses of the same group; incorrect earn 0;
  * a correct response always earns more than an incorrect one (alpha < 1), and a shorter correct
    response more than a longer one in the same group;
  * no within-group length signal from a single correct response or equal lengths (f = 1/2);
  * groups are independent; shaped (non 0/1) rewards and alpha outside [0, 1) raise;
  * the trainer applies it only when the attribute is set and keeps logging the correctness reward;
  * the arm gets its own "new-...-lp{alpha}" directory, fingerprint key, and eval_d2 parses it.

    python -m pytest tests/test_length_penalty.py -q
"""
import ast
import math
import os
import re
import statistics
import sys
import types

import pytest

torch = pytest.importorskip("torch")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from length_penalty import length_penalised_rewards  # noqa: E402


def reference(rewards, lengths, alpha):
    """The paper's formula written out per element, for one group."""
    corr = [n for r, n in zip(rewards, lengths) if r == 1]
    mean = statistics.fmean(corr) if corr else 0.0
    std = statistics.pstdev(corr) if len(corr) > 1 else 0.0
    out = []
    for r, n in zip(rewards, lengths):
        if r != 1:
            out.append(0.0)
            continue
        z = (n - mean) / std if std > 1e-6 else 0.0
        out.append(1 - alpha * (1 / (1 + math.exp(-z))))
    return out


def test_matches_formula():
    rewards = torch.tensor([1., 1., 0., 1., 0., 1., 1., 0.])
    lengths = torch.tensor([100, 200, 50, 300, 400, 120, 80, 10])
    shaped, f = length_penalised_rewards(rewards, lengths, 4, 0.2)
    want = reference([1, 1, 0, 1], [100, 200, 50, 300], 0.2) + reference([0, 1, 1, 0], [400, 120, 80, 10], 0.2)
    assert torch.allclose(shaped, torch.tensor(want), atol=1e-6)
    assert torch.equal(f == 0, rewards == 0)


def test_ordering_properties():
    torch.manual_seed(0)
    for _ in range(200):
        rewards = (torch.rand(16) < 0.6).float()
        lengths = torch.randint(20, 1024, (16,))
        shaped, _ = length_penalised_rewards(rewards, lengths, 4, 0.9)
        for g in range(4):
            r, n, s = rewards[4 * g:4 * g + 4], lengths[4 * g:4 * g + 4], shaped[4 * g:4 * g + 4]
            if (r == 1).any() and (r == 0).any():
                assert s[r == 1].min() > s[r == 0].max()          # correct always beats incorrect
            idx = [i for i in range(4) if r[i] == 1]
            for i in idx:
                for j in idx:
                    if n[i] < n[j]:
                        assert s[i] > s[j]                        # shorter correct earns more


def test_no_signal_cases():
    shaped, f = length_penalised_rewards(torch.tensor([1., 0., 0., 0.]), torch.tensor([300, 10, 20, 30]), 4, 0.2)
    assert shaped[0].item() == pytest.approx(0.9) and f[0].item() == pytest.approx(0.5)
    shaped, _ = length_penalised_rewards(torch.tensor([1., 1., 0., 0.]), torch.tensor([300, 300, 1, 2]), 4, 0.2)
    assert torch.allclose(shaped[:2], torch.tensor([0.9, 0.9]))
    shaped, _ = length_penalised_rewards(torch.zeros(4), torch.tensor([1, 2, 3, 4]), 4, 0.2)
    assert torch.equal(shaped, torch.zeros(4))


def test_groups_are_independent():
    rewards = torch.tensor([1., 1., 1., 0., 1., 1., 0., 1.])
    a, _ = length_penalised_rewards(rewards, torch.tensor([10, 20, 30, 40, 50, 60, 70, 80]), 4, 0.3)
    b, _ = length_penalised_rewards(rewards, torch.tensor([10, 20, 30, 40, 900, 5, 70, 300]), 4, 0.3)
    assert torch.equal(a[:4], b[:4])


def test_refusals():
    with pytest.raises(ValueError):
        length_penalised_rewards(torch.tensor([1., 0.5, 0., 0.]), torch.tensor([1, 2, 3, 4]), 4, 0.2)
    with pytest.raises(ValueError):
        length_penalised_rewards(torch.ones(4), torch.tensor([1, 2, 3, 4]), 4, 1.0)


def test_trainer_hook_is_guarded_and_logs_correctness_reward():
    src = open(os.path.join(ROOT, "trl/trainer/grpo_trainer.py"), encoding="utf-8").read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_prepare_inputs")
    body = ast.get_source_segment(src, fn)
    assert 'getattr(self, "length_penalty_alpha", None)' in body
    assert 'self._metrics["reward"].append(correctness_rewards.mean().item())' in body
    # the penalty is applied before the group mean that the advantages subtract
    assert body.index("length_penalised_rewards(") < body.index("mean_grouped_rewards = rewards.view")


def test_naming_fingerprint_and_setup():
    import variant_common as vc
    args = vc.build_arg_parser("v3_learned_lambda").parse_args(
        ["--seed", "43", "--repetition_penalty", "1.0", "--dr_grpo", "both", "--dataset", "math",
         "--length_penalty_alpha", "0.2"])
    name = os.path.basename(vc.experiment_dir("1.5b", "v3_learned_lambda", args))
    assert name == "new-1.5b-v3_learned_lambda-seed43-rp1-drboth-dsmath-lp0.2"
    assert "length_penalty_alpha" in vc._CONFIG_FINGERPRINT
    assert vc._FINGERPRINT_DEFAULTS["length_penalty_alpha"] is None
    off = vc.build_arg_parser("v3_learned_lambda").parse_args(["--seed", "43"])
    assert "-lp" not in vc.experiment_dir("1.5b", "v3_learned_lambda", off)

    trainer = types.SimpleNamespace(num_generations=4)
    vc.apply_length_penalty(trainer, off)
    assert not hasattr(trainer, "length_penalty_alpha")
    vc.apply_length_penalty(trainer, args)
    assert trainer.length_penalty_alpha == 0.2
    with pytest.raises(SystemExit):
        vc.apply_length_penalty(trainer, types.SimpleNamespace(length_penalty_alpha=1.0))

    src = open(os.path.join(ROOT, "eval_d2.py"), encoding="utf-8").read()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "NEW_RE")
    new_re = eval(compile(ast.Expression(node.value), "eval_d2.py", "eval"), {"re": re})
    m = new_re.match(name)
    assert m and m.group("variant") == "v3_learned_lambda" and m.group("suffix").endswith("-lp0.2")
