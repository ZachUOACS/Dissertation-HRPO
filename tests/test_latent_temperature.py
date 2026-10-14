"""Decoupled latent temperature (--latent_temperature).

hrpo_latent_probs sits on the rollout path of EVERY arm, so the default (None) must hand back the
sampling distribution itself, and the re-tempered distribution must equal what the logits
processors would give at the latent temperature on the same support. Naming and fingerprinting
must keep the new arm apart from the settled ones, and the setter must refuse a model with no
latent mixture (v1, latent-off) rather than silently doing nothing.

CPU only. Run from the repo root:  python -m pytest tests/test_latent_temperature.py -q
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GenerationConfig  # noqa: E402
from transformers.generation.logits_process import (  # noqa: E402
    LogitsProcessorList, RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper,
    TopKLogitsWarper)
from transformers.generation.utils import hrpo_latent_probs  # noqa: E402

import variant_common as vc  # noqa: E402

VOCAB = 1000


def processed(logits, input_ids, temperature, top_k=50, rp=1.0):
    """The scores _sample sees, built with the same processors HF adds for do_sample."""
    procs = LogitsProcessorList()
    if rp != 1.0:
        procs.append(RepetitionPenaltyLogitsProcessor(penalty=rp))
    if temperature != 1.0:
        procs.append(TemperatureLogitsWarper(temperature))
    procs.append(TopKLogitsWarper(top_k=top_k))
    return procs(input_ids, logits.clone())


def setup(seed=0, batch=4):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, VOCAB, generator=g) * 3
    input_ids = torch.randint(0, VOCAB, (batch, 7), generator=g)
    return logits, input_ids


def test_default_returns_the_sampling_distribution_itself():
    logits, ids = setup()
    scores = processed(logits, ids, 0.5)
    probs = torch.softmax(scores, dim=-1)
    cfg = GenerationConfig(do_sample=True, temperature=0.5, top_k=50)
    assert hrpo_latent_probs(probs, scores, cfg, None) is probs


@pytest.mark.parametrize("rp", [1.0, 1.2])
@pytest.mark.parametrize("latent_t", [0.5, 1.0, 2.0])
def test_retempered_equals_processors_at_the_latent_temperature(rp, latent_t):
    logits, ids = setup(seed=1)
    scores = processed(logits, ids, 0.5, rp=rp)
    probs = torch.softmax(scores, dim=-1)
    cfg = GenerationConfig(do_sample=True, temperature=0.5, top_k=50, repetition_penalty=rp)
    got = hrpo_latent_probs(probs, scores, cfg, latent_t)
    want = torch.softmax(processed(logits, ids, latent_t, rp=rp), dim=-1)
    assert torch.allclose(got, want, atol=1e-6)


def test_same_support_and_same_argmax_flatter_distribution():
    logits, ids = setup(seed=2)
    scores = processed(logits, ids, 0.5)
    probs = torch.softmax(scores, dim=-1)
    cfg = GenerationConfig(do_sample=True, temperature=0.5, top_k=50)
    lat = hrpo_latent_probs(probs, scores, cfg, 1.0)
    assert torch.equal(lat > 0, probs > 0)                 # top-k support unchanged
    assert torch.equal(lat.argmax(-1), probs.argmax(-1))   # sampled/greedy token unaffected
    ent = lambda p: torch.special.entr(p).sum(-1)
    assert (ent(lat) > ent(probs)).all()                   # 1.0 is flatter than 0.5


def test_sampling_at_temperature_one_is_handled():
    """No TemperatureLogitsWarper is added at 1.0, so tau_s must be taken as 1."""
    logits, ids = setup(seed=3)
    scores = processed(logits, ids, 1.0)
    probs = torch.softmax(scores, dim=-1)
    cfg = GenerationConfig(do_sample=True, temperature=1.0, top_k=50)
    got = hrpo_latent_probs(probs, scores, cfg, 2.0)
    want = torch.softmax(processed(logits, ids, 2.0), dim=-1)
    assert torch.allclose(got, want, atol=1e-6)


def make(variant, *extra):
    args = vc.build_arg_parser(variant).parse_args(["--model_size", "1.5b", *extra])
    return os.path.basename(vc.experiment_dir("1.5b", variant, args))


def test_latent_arm_is_a_new_arm():
    name = make("v3_learned_lambda", "--dataset", "math", "--dr_grpo", "both", "--seed", "43",
                "--latent_temperature", "1.0")
    assert name == "new-1.5b-v3_learned_lambda-seed43-rp1-drboth-dsmath-lt1"
    assert vc.parse_experiment_name(name) is None
    # unset: the settled MATH name is untouched
    assert make("v3_learned_lambda", "--dataset", "math", "--dr_grpo", "both", "--seed", "43") == \
        "1.5b-v3_learned_lambda-seed43-rp1-drboth-dsmath"


def test_latent_arm_pairs_with_the_math_baseline_in_analyse_d2():
    pytest.importorskip("scipy")
    import analyse_d2
    size, variant, seed, tag, is_new = analyse_d2.split_name(
        "new-1.5b-v3_learned_lambda-seed43-rp1-drboth-dsmath-lt1")
    assert (size, variant, seed, is_new) == ("1.5b", "v3_learned_lambda", 43, True)
    assert tag == "drboth-dsmath-lt1" and "-dsmath" in tag   # -> compared with drboth-dsmath


def test_latent_temperature_is_fingerprinted_with_a_resume_safe_default():
    assert "latent_temperature" in vc._CONFIG_FINGERPRINT
    assert vc._FINGERPRINT_DEFAULTS["latent_temperature"] is None
    assert vc.build_arg_parser("v3_learned_lambda").parse_args([]).latent_temperature is None


class _Inner(torch.nn.Module):
    pass


class _CausalLM(torch.nn.Module):
    def __init__(self, answer_start):
        super().__init__()
        self.answer_start = answer_start
        self.model = _Inner()


class _Wrapper(torch.nn.Module):          # stands in for the PEFT wrapper
    def __init__(self, inner):
        super().__init__()
        self.base_model = inner


def test_setter_reaches_the_module_that_runs_sample():
    lm = _CausalLM("####")
    assert vc.set_latent_temperature(_Wrapper(lm), 1.0) == 1
    assert lm.latent_temperature == 1.0
    assert not hasattr(lm.model, "latent_temperature")


def test_setter_default_touches_nothing():
    lm = _CausalLM("####")
    assert vc.set_latent_temperature(lm, None) == 0
    assert not hasattr(lm, "latent_temperature")

    class T:
        model = lm
    vc.apply_latent_temperature(T(), vc.build_arg_parser("v3_learned_lambda").parse_args([]))
    assert not hasattr(lm, "latent_temperature")


def test_setter_refuses_a_model_without_a_latent_mixture():
    with pytest.raises(RuntimeError):
        vc.set_latent_temperature(_CausalLM(""), 1.0)       # v1 / latent-off
    with pytest.raises(RuntimeError):
        vc.set_latent_temperature(_Inner(), 1.0)            # nothing carries answer_start
    with pytest.raises(ValueError):
        vc.set_latent_temperature(_CausalLM("####"), 0.0)
