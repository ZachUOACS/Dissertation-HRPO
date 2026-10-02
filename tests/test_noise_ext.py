"""CPU tests for noise_ext. Run from the repo root:  python -m pytest tests/test_noise_ext.py -q

NOT YET RUN: written without a local torch install. Run it on the cluster (login node is fine, CPU only)
before using noise_ext in training.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noise_ext import apply_norm_noise, find_norm_weights, patch_trainer_noise, sigma_at  # noqa: E402


class TinyNormBlock(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.input_layernorm = torch.nn.LayerNorm(d)
        self.post_attention_layernorm = torch.nn.LayerNorm(d)
        self.other = torch.nn.Linear(d, d)


class TinyModel(torch.nn.Module):
    def __init__(self, d=16, layers=3):
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyNormBlock(d) for _ in range(layers)])
        self.norm = torch.nn.LayerNorm(d)


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_schedule_matches_qerl_formula():
    start, end, K, T = 5e-2, 5e-4, 10, 1000
    assert sigma_at(0, T, start, end, K) == pytest.approx(start)
    assert sigma_at(T - 1, T, start, end, K) == pytest.approx(end)
    # stage k (0-based) = start * (end/start) ** (k / (K - 1))
    assert sigma_at(450, T, start, end, K) == pytest.approx(start * (end / start) ** (4 / 9))
    values = [sigma_at(s, T, start, end, K) for s in range(0, T, 10)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_zero_sigma_is_a_no_op():
    assert sigma_at(10, 100, 0.0, 0.0, 10) == 0.0
    model = TinyModel()
    before = snapshot(model)
    apply_norm_noise(model, 0.0, seed=1).restore()
    after = snapshot(model)
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_only_target_norms_are_perturbed_and_restore_is_exact():
    model = TinyModel()
    before = snapshot(model)
    handle = apply_norm_noise(model, 0.05, seed=3)
    during = snapshot(model)
    changed = {k for k in before if not torch.equal(before[k], during[k])}
    assert changed and all(k.endswith("input_layernorm.weight") or k.endswith("post_attention_layernorm.weight")
                           for k in changed)
    assert "norm.weight" not in changed and "layers.0.other.weight" not in changed
    handle.restore()
    after = snapshot(model)
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert len(find_norm_weights(model)) == 6


def test_same_seed_same_noise_different_seed_different_noise():
    a, b, c = TinyModel(), TinyModel(), TinyModel()
    b.load_state_dict(a.state_dict())
    c.load_state_dict(a.state_dict())
    apply_norm_noise(a, 0.05, seed=7)
    apply_norm_noise(b, 0.05, seed=7)
    apply_norm_noise(c, 0.05, seed=8)
    assert torch.equal(a.layers[0].input_layernorm.weight, b.layers[0].input_layernorm.weight)
    assert not torch.equal(a.layers[0].input_layernorm.weight, c.layers[0].input_layernorm.weight)


def test_mult_mode_scales_with_weight():
    model = TinyModel()
    with torch.no_grad():
        model.layers[0].input_layernorm.weight.zero_()
    apply_norm_noise(model, 0.5, seed=1, mode="mult")
    assert torch.count_nonzero(model.layers[0].input_layernorm.weight) == 0  # multiplicative noise keeps zeros


def test_bf16_weights_round_trip():
    model = TinyModel().to(torch.bfloat16)
    before = snapshot(model)
    handle = apply_norm_noise(model, 0.05, seed=2)
    handle.restore()
    after = snapshot(model)
    assert all(torch.equal(before[k], after[k]) for k in before)


class FakeState:
    global_step = 0
    max_steps = 100


class FakeTrainer:
    """Mimics what HF Trainer.training_step does: the whole step runs while noise must be live."""

    def __init__(self, model):
        self.model = model
        self.state = FakeState()
        self.seen = []

    def training_step(self, model, inputs, *args, **kwargs):
        self.seen.append(model.layers[0].input_layernorm.weight.detach().clone())
        return torch.tensor(0.0)


def test_wrapper_applies_noise_during_step_and_restores_after():
    model = TinyModel()
    clean = model.layers[0].input_layernorm.weight.detach().clone()
    trainer = FakeTrainer(model)
    patch_trainer_noise(trainer, sigma_start=0.05, sigma_end=0.005, stages=4, seed=5, verbose=False)
    trainer.training_step(model, {})
    assert not torch.equal(trainer.seen[0], clean)
    assert torch.equal(model.layers[0].input_layernorm.weight, clean)
    trainer.training_step(model, {})  # second micro-batch gets a different draw
    assert not torch.equal(trainer.seen[0], trainer.seen[1])


def test_wrapper_restores_on_exception():
    model = TinyModel()
    clean = model.layers[0].input_layernorm.weight.detach().clone()

    class Boom(FakeTrainer):
        def training_step(self, model, inputs, *args, **kwargs):
            raise RuntimeError("OOM")

    trainer = Boom(model)
    patch_trainer_noise(trainer, sigma_start=0.05, sigma_end=0.005, seed=1, verbose=False)
    with pytest.raises(RuntimeError):
        trainer.training_step(model, {})
    assert torch.equal(model.layers[0].input_layernorm.weight, clean)


def test_sigma_start_zero_leaves_trainer_untouched():
    model = TinyModel()
    trainer = FakeTrainer(model)
    original = trainer.training_step
    patch_trainer_noise(trainer, sigma_start=0.0, sigma_end=0.0)
    assert trainer.training_step == original
