"""
stats_utils.py — Wilson CIs and McNemar's test shared by eval_all.py,
analyse_failures.py and generate_report.py (kept free of heavy imports so
analysis scripts don't drag in unsloth/torch).
"""

import math

from scipy import stats


def wilson_ci(correct: int, total: int, z: float = 1.96):
    """Wilson score 95% confidence interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    denom = 1 + z ** 2 / total
    centre = p + z ** 2 / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2))
    return (centre - margin) / denom, (centre + margin) / denom


def mcnemar_test(y_a, y_b):
    """McNemar's paired test with continuity correction on aligned binary
    outcome vectors. Returns (chi2, p). b = a-only correct, c = b-only."""
    b = sum(1 for a, bb in zip(y_a, y_b) if a == 1 and bb == 0)
    c = sum(1 for a, bb in zip(y_a, y_b) if a == 0 and bb == 1)
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(stats.chi2.sf(chi2, df=1))
    return chi2, p
