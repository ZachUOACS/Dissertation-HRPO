"""Regression tests for the arm key (size, variant, rp, lora, warm, dr).

The tier-8 analysis crashed on the cluster the first time warm-started arms
existed: sorted() on the raw keys compared warm='on' with warm=None
(TypeError), and plot_arms still unpacked keys as 4 values. Both only fire
once an arm and its counterpart differing in an optional field are present
together, which no earlier run had. C2 added a sixth field (dr) — the same
failure is waiting at every site that unpacks or sorts keys.
"""
import pytest

import analyse_replicates as ar

COLD_V3 = ("1.5b", "v3_learned_lambda", 1.0, "full", None, None, None)
WARM_V3 = ("1.5b", "v3_learned_lambda", 1.0, "full", "on", None, None)
COLD_V1 = ("1.5b", "v1_grpo_baseline", 1.0, "full", None, None, None)
DR_V1 = ("1.5b", "v1_grpo_baseline", 1.0, "full", None, "both", None)
C3 = ("1.5b", "v1_grpo_baseline", 1.0, "full", "off", None, None)
C1 = ("1.5b", "v7_constant_vector", 1.0, "full", None, None, None)
OLD_V3 = ("1.5b", "v3_learned_lambda", 1.2, "full", None, None, None)
MATH_V3 = ("1.5b", "v3_learned_lambda", 1.0, "full", None, None, "math")
MATH_V1 = ("1.5b", "v1_grpo_baseline", 1.0, "full", None, None, "math")


def _arms():
    return {
        WARM_V3: {42: (0.6884, None), 43: (0.6846, None), 44: (0.6801, None)},
        COLD_V3: {42: (0.7165, None), 43: (0.6869, None), 44: (0.7111, None)},
        DR_V1: {42: (0.7000, None)},
        COLD_V1: {42: (0.6846, None)},
        C3: {42: (0.6952, None)},
        C1: {42: (0.7028, None)},
        OLD_V3: {42: (0.6702, None)},
        MATH_V3: {42: (0.3100, None), 43: (0.3220, None)},
        MATH_V1: {42: (0.3160, None)},
    }


def test_raw_tuple_sort_is_the_bug():
    """Documents why arm_sort_key exists: the plain sort really does raise."""
    with pytest.raises(TypeError):
        sorted(_arms())


def test_arm_sort_key_orders_optional_fields_without_error():
    order = sorted(_arms(), key=ar.arm_sort_key)
    assert len(order) == len(_arms())
    # each cold arm sorts immediately before its warm / Dr. GRPO counterpart
    assert order.index(COLD_V3) + 1 == order.index(WARM_V3)
    assert order.index(COLD_V1) < order.index(DR_V1)
    # MATH arms group together rather than interleaving with GSM8K ones
    math_positions = [order.index(MATH_V1), order.index(MATH_V3)]
    assert max(math_positions) - min(math_positions) == 1


def test_plot_arms_accepts_full_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "PLOTS_DIR", str(tmp_path))
    ar.plot_arms(_arms())
    assert (tmp_path / "replicate_summary.png").exists()


def test_labels_distinguish_every_arm():
    labels = [ar.arm_label(*k) for k in _arms()]
    assert len(set(labels)) == len(labels)
    assert "Dr. GRPO (both)" in ar.arm_label(*DR_V1)
    assert "warm-start on" in ar.arm_label(*WARM_V3)
    # a MATH number must never be readable as a GSM8K one
    assert "MATH" in ar.arm_label(*MATH_V3)
    assert "GSM8K" in ar.arm_label(*COLD_V3)
