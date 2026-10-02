"""D2 programme: directory naming must keep new arms apart from the settled ones.

The hazard (changes.md, the warm-start fix): an unrecognised suffix token is ignored by
analyse_replicates.parse_suffix, so a treatment would be pooled with its own control as extra
"seeds" of one arm, or a run would resolve to a finished directory and silently skip.

CPU only. Run from the repo root:  python -m pytest tests/test_new_arm_naming.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import variant_common as vc  # noqa: E402


def make(variant, *extra):
    args = vc.build_arg_parser(variant).parse_args(["--model_size", "1.5b", *extra])
    return vc.experiment_dir("1.5b", variant, args)


def test_settled_arm_names_are_unchanged():
    assert make("v3_learned_lambda", "--dr_grpo", "both", "--seed", "43") == \
        "./experiments/1.5b-v3_learned_lambda-seed43-rp1-drboth"
    assert make("v1_grpo_baseline", "--repetition_penalty", "1.0") == \
        "./experiments/1.5b-v1_grpo_baseline-rp1"


def test_default_repetition_penalty_is_one():
    assert vc.build_arg_parser("v1_grpo_baseline").parse_args([]).repetition_penalty == 1.0


@pytest.mark.parametrize("extra", [
    ["--noise_sigma_start", "0.05", "--noise_sigma_end", "0.005"],
    ["--noise_sigma_start", "0.01", "--noise_mode", "mult"],
    ["--temperature", "0.8"],
    ["--lr", "1e-5"],
])
def test_new_arms_are_invisible_to_the_legacy_parser(extra):
    name = os.path.basename(make("v1_grpo_baseline", "--dr_grpo", "both", *extra))
    assert name.startswith("new-")
    assert vc.parse_experiment_name(name) is None


def test_distinct_settings_get_distinct_directories():
    settings = [[], ["--temperature", "0.8"], ["--temperature", "1.0"], ["--lr", "1e-5"],
                ["--noise_sigma_start", "0.05", "--noise_sigma_end", "0.005"],
                ["--noise_sigma_start", "0.05", "--noise_sigma_end", "0.005", "--seed", "43"],
                ["--noise_sigma_start", "0.02", "--noise_sigma_end", "0.002"]]
    names = {make("v1_grpo_baseline", "--dr_grpo", "both", *s) for s in settings}
    assert len(names) == len(settings)


def test_noise_keys_are_fingerprinted_with_resume_safe_defaults():
    for key in ("noise_sigma_start", "noise_sigma_end", "noise_stages", "noise_mode"):
        assert key in vc._CONFIG_FINGERPRINT
        assert key in vc._FINGERPRINT_DEFAULTS
    # A checkpoint written before D2 has none of these keys and must still resume.
    old = {k: vc._FINGERPRINT_DEFAULTS[k] for k in
           ("noise_sigma_start", "noise_sigma_end", "noise_stages", "noise_mode")}
    args = vc.build_arg_parser("v1_grpo_baseline").parse_args([])
    for key, default in old.items():
        assert getattr(args, key) == default


def test_noise_off_leaves_the_trainer_untouched():
    class T:
        def training_step(self):
            return 1

    t = T()
    original = t.training_step
    args = vc.build_arg_parser("v1_grpo_baseline").parse_args([])
    vc.apply_exploration_noise(t, args)
    assert t.training_step == original
