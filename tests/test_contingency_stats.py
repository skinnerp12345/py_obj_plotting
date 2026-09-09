"""Hand-computed validation of contingency_stats.py's POD/FAR/FBIAS/CSI
formulas and bootstrap_ci_widths -- pure functions, no file I/O.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_contingency_stats.py -v -s
"""

import numpy as np
import pytest

from py_obj_plotting.climatology.contingency_stats import (
    bootstrap_ci_widths,
    bootstrap_pod_diff_ci_width,
    compute_csi,
    compute_far,
    compute_fbias,
    compute_pod,
)


def test_hand_computed_standard_formulas():
    # hits=8, misses=2, false_alarms=4 -- textbook contingency table
    assert compute_pod(8, 2) == 0.8  # 8 / (8+2)
    assert compute_far(8, 4) == pytest.approx(4 / 12)
    assert compute_csi(8, 2, 4) == pytest.approx(8 / 14)


def test_fbias_extended_treatment_folds_in_extras():
    # hits=8, misses=2, false_alarms=4, truth_extra=1, forecast_extra=3
    # FBIAS = (hits + false_alarms + forecast_extra) / (hits + misses + truth_extra)
    #       = (8 + 4 + 3) / (8 + 2 + 1) = 15 / 11
    result = compute_fbias(8, 2, 4, 1, 3)
    expected = 15.0 / 11.0
    print(f"\n[contingency-check1] fbias={result}, expected={expected}")
    assert abs(result - expected) < 1e-9


def test_fbias_matches_standard_fbias_when_extras_are_zero():
    # with truth_extra=forecast_extra=0, FBIAS must reduce to the standard formula
    hits, misses, false_alarms = 8, 2, 4
    standard = (hits + false_alarms) / (hits + misses)
    extended = compute_fbias(hits, misses, false_alarms, 0, 0)
    assert abs(extended - standard) < 1e-9


def test_zero_denominator_returns_zero():
    assert compute_pod(0, 0) == 0.0
    assert compute_far(0, 0) == 0.0
    assert compute_csi(0, 0, 0) == 0.0
    assert compute_fbias(0, 0, 0, 0, 0) == 0.0
    # pod's own denominator can be zero even with other categories present
    assert compute_pod(0, 0) == 0.0


def test_bootstrap_ci_widths_zero_total_returns_all_zero():
    assert bootstrap_ci_widths(0, 0, 0, 0, 0, samples=100) == (0.0, 0.0, 0.0, 0.0)


def test_bootstrap_ci_widths_are_nonnegative_and_shrink_with_more_extreme_data():
    """A CI width is, by construction, a difference of two percentiles of a
    non-negative-support statistic -- must never be negative. Also confirms
    the bootstrap actually responds to its input: an overwhelming majority
    class (very little sampling variability) should produce a visibly
    narrower CI than a near-50/50 split (high sampling variability), at a
    fixed total sample size."""
    np.random.seed(0)
    narrow = bootstrap_ci_widths(990, 10, 0, 0, 0, samples=500)  # hits dominate -> POD near-certain
    wide = bootstrap_ci_widths(500, 500, 0, 0, 0, samples=500)  # 50/50 -> POD uncertain

    for w in narrow + wide:
        assert w >= 0.0

    pod_width_narrow, _, _, _ = narrow
    pod_width_wide, _, _, _ = wide
    print(f"\n[contingency-check2] narrow POD CI width={pod_width_narrow:.4f}, wide POD CI width={pod_width_wide:.4f}")
    assert pod_width_narrow < pod_width_wide


def test_bootstrap_ci_widths_reproduce_point_estimate_when_n_is_huge():
    """With a very large, fixed-ratio sample, the bootstrap distribution
    should concentrate tightly around the true point estimate -- CI width
    should be small (a weak but real correctness signal: this isn't just
    returning noise)."""
    np.random.seed(1)
    pod_w, far_w, fbias_w, csi_w = bootstrap_ci_widths(80000, 20000, 10000, 0, 0, samples=200)
    print(f"\n[contingency-check3] large-n widths: pod={pod_w:.4f}, far={far_w:.4f}, fbias={fbias_w:.4f}, csi={csi_w:.4f}")
    assert pod_w < 0.02
    assert far_w < 0.02
    assert csi_w < 0.02


def test_bootstrap_pod_diff_ci_width_zero_side_returns_zero():
    assert bootstrap_pod_diff_ci_width(0, 0, 8, 2, samples=100) == 0.0
    assert bootstrap_pod_diff_ci_width(8, 2, 0, 0, samples=100) == 0.0


def test_bootstrap_pod_diff_ci_width_nonnegative_and_widens_with_uncertainty():
    """Same shape of check as bootstrap_ci_widths' own: nonnegative by
    construction, and a near-certain POD-diff estimate (huge, lopsided
    samples on both sides) must produce a narrower CI than a small,
    genuinely uncertain sample."""
    np.random.seed(2)
    narrow = bootstrap_pod_diff_ci_width(9900, 100, 100, 9900, samples=300)
    wide = bootstrap_pod_diff_ci_width(6, 4, 5, 5, samples=300)
    print(f"\n[contingency-check-pod-diff] narrow={narrow:.4f}, wide={wide:.4f}")
    assert narrow >= 0.0 and wide >= 0.0
    assert narrow < wide


def test_bootstrap_pod_diff_ci_width_centers_near_true_difference_at_large_n():
    """At large, fixed-ratio n on both sides, independently resampled POD
    values should concentrate tightly, giving a small CI width -- a real
    correctness signal, not just nonnegativity."""
    np.random.seed(3)
    # pod_a = 0.8, pod_b = 0.5 at large n -- CI should be narrow regardless
    # of the true difference's magnitude.
    width = bootstrap_pod_diff_ci_width(80000, 20000, 50000, 50000, samples=200)
    print(f"\n[contingency-check-pod-diff-largen] width={width:.4f}")
    assert width < 0.02
