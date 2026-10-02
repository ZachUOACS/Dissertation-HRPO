"""
plot_style.py — shared matplotlib styling for all dissertation analysis plots.

Categorical colors come from a CVD-validated palette (worst adjacent
colorblind delta-E 24.2 on a white surface). Variant colors are assigned in a
FIXED order so the same variant wears the same hue in every figure, regardless
of which variants a given plot happens to include.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical palette (light surface).
PALETTE = {
    "blue":    "#2a78d6",
    "aqua":    "#1baf7a",
    "yellow":  "#eda100",
    "green":   "#008300",
    "violet":  "#4a3aa7",
    "red":     "#e34948",
    "magenta": "#e87ba4",
    "orange":  "#eb6834",
}

# Fixed entity -> hue assignment (never re-ordered per plot).
VARIANT_COLORS = {
    "base_zero_shot":        "#8a8a85",  # neutral gray reference
    "v1_grpo_baseline":      PALETTE["blue"],
    "v2_fixed_blend":        PALETTE["aqua"],
    "v3_learned_lambda":     PALETTE["yellow"],
    "v4_full_hrpo_gate":     PALETTE["green"],
    "v5_hrpo_format_reward": PALETTE["violet"],
    "v6_hrpo_full":          PALETTE["red"],
    "v7_constant_vector":    PALETTE["orange"],  # C1 control
}

VARIANT_SHORT = {
    "base_zero_shot":        "base",
    "v1_grpo_baseline":      "v1 GRPO",
    "v2_fixed_blend":        "v2 fixed blend",
    "v3_learned_lambda":     "v3 learned Λ",
    "v4_full_hrpo_gate":     "v4 full gate",
    "v5_hrpo_format_reward": "v5 +format reward",
    "v6_hrpo_full":          "v6 +rep. handling",
    "v7_constant_vector":    "v7 constant vector",
}

CORRECT_COLOR = PALETTE["green"]
INCORRECT_COLOR = PALETTE["red"]


def variant_color(variant_id: str) -> str:
    return VARIANT_COLORS.get(variant_id, PALETTE["orange"])


def variant_label(variant_id: str) -> str:
    return VARIANT_SHORT.get(variant_id, variant_id)


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e3",       # recessive grid
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 2.0,        # 2px lines
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    return plt
