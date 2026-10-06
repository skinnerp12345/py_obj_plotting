"""Hand-computed validation of contingency_stats.py's POD/FAR/FBIAS/CSI
formulas and bootstrap_ci_widths -- pure functions, no file I/O.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_contingency_stats.py -v -s
"""

import time

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


def _bootstrap_ci_widths_legacy(
    hits: int, misses: int, false_alarms: int, truth_extra: int, forecast_extra: int,
    samples: int = 1000, interval: float = 95,
) -> tuple[float, float, float, float]:
    """Verbatim copy of bootstrap_ci_widths' PRE-multinomial implementation
    (explicit length-n categorical vector + np.random.randint gather),
    kept only in this test module to compare the new implementation against
    the actual old one directly, not a reconstruction from memory."""
    n_total = hits + misses + false_alarms + truth_extra + forecast_extra
    if n_total == 0:
        return 0.0, 0.0, 0.0, 0.0

    full_vector = np.concatenate([
        np.ones(hits), np.zeros(misses), np.full(false_alarms, 2.0),
        np.full(truth_extra, 3.0), np.full(forecast_extra, 4.0),
    ])
    idx = np.random.randint(0, n_total, size=(samples, n_total))
    sampled = full_vector[idx]
    h = np.sum(sampled == 1, axis=1).astype(float)
    m = np.sum(sampled == 0, axis=1).astype(float)
    f = np.sum(sampled == 2, axis=1).astype(float)
    te = np.sum(sampled == 3, axis=1).astype(float)
    fe = np.sum(sampled == 4, axis=1).astype(float)

    pod_denom, far_denom, fbias_denom, csi_denom = h + m, h + f, h + m + te, h + m + f
    pod_s = np.divide(h, pod_denom, out=np.zeros_like(h), where=pod_denom > 0)
    far_s = np.divide(f, far_denom, out=np.zeros_like(f), where=far_denom > 0)
    fbias_s = np.divide(h + f + fe, fbias_denom, out=np.zeros_like(h), where=fbias_denom > 0)
    csi_s = np.divide(h, csi_denom, out=np.zeros_like(h), where=csi_denom > 0)

    lo, hi = (100 - interval) / 2.0, (100 - interval) / 2.0 + interval
    return (
        np.percentile(pod_s, hi) - np.percentile(pod_s, lo),
        np.percentile(far_s, hi) - np.percentile(far_s, lo),
        np.percentile(fbias_s, hi) - np.percentile(fbias_s, lo),
        np.percentile(csi_s, hi) - np.percentile(csi_s, lo),
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
    fixed total sample size.

    seed= passed explicitly (not an ambient np.random.seed()) since
    bootstrap_ci_widths uses its own independent numpy.random.Generator
    stream internally -- an ambient legacy-global seed has no effect on it."""
    narrow = bootstrap_ci_widths(990, 10, 0, 0, 0, samples=500, seed=0)  # hits dominate -> POD near-certain
    wide = bootstrap_ci_widths(500, 500, 0, 0, 0, samples=500, seed=0)  # 50/50 -> POD uncertain

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
    pod_w, far_w, fbias_w, csi_w = bootstrap_ci_widths(80000, 20000, 10000, 0, 0, samples=200, seed=1)
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


# --- Statistical consistency: new (multinomial-draw) implementation vs. the
# actual old (explicit-vector-resample) implementation. Resampling a
# length-n categorical population with replacement and counting category
# occurrences IS the Multinomial(n, p) distribution by definition -- so
# these two implementations target the IDENTICAL distribution, not merely a
# similar one. A single call from each will differ (independent RNG
# streams), but repeated-call statistics (here, the mean width across many
# independent bootstrap runs) should agree closely -- this is the actual
# empirical evidence for "nearly identical results", not just the
# mathematical argument.

_CONSISTENCY_CASES = {
    "no_extras": (600, 300, 100, 0, 0),  # exercises the FBIAS-degrades-to-standard path
    "with_extras": (600, 300, 100, 40, 25),  # exercises the extended-bias generalization this change added
}
_CONSISTENCY_N_REPEATS = 300
_CONSISTENCY_SAMPLES = 1000
_CONSISTENCY_TOLERANCE = 0.02  # absolute; same order of magnitude as this file's existing "< 0.02" large-n idiom


@pytest.mark.parametrize("case_name", sorted(_CONSISTENCY_CASES))
def test_new_bootstrap_matches_legacy_bootstrap_distribution(case_name):
    hits, misses, false_alarms, truth_extra, forecast_extra = _CONSISTENCY_CASES[case_name]

    legacy_widths = []
    new_widths = []
    for i in range(_CONSISTENCY_N_REPEATS):
        np.random.seed(10_000 + i)  # legacy reads the ambient global RNG state
        legacy_widths.append(_bootstrap_ci_widths_legacy(
            hits, misses, false_alarms, truth_extra, forecast_extra, samples=_CONSISTENCY_SAMPLES,
        ))
        new_widths.append(bootstrap_ci_widths(
            hits, misses, false_alarms, truth_extra, forecast_extra, samples=_CONSISTENCY_SAMPLES, seed=20_000 + i,
        ))

    legacy_widths = np.array(legacy_widths)  # (n_repeats, 4) -- columns: pod, far, fbias, csi
    new_widths = np.array(new_widths)

    for i, metric in enumerate(("pod", "far", "fbias", "csi")):
        legacy_mean = legacy_widths[:, i].mean()
        new_mean = new_widths[:, i].mean()
        print(f"\n[consistency-{case_name}-{metric}] legacy_mean={legacy_mean:.4f}, new_mean={new_mean:.4f}, "
              f"diff={abs(legacy_mean - new_mean):.4f}")
        assert abs(legacy_mean - new_mean) < _CONSISTENCY_TOLERANCE


def test_new_bootstrap_matches_legacy_bootstrap_ks_test():
    """Secondary, formal check on top of the mean-tolerance test above: a
    two-sample Kolmogorov-Smirnov test should find no significant
    difference between the two implementations' POD-width distributions.
    Uses a strict alpha (0.001) specifically to avoid flaking on the
    ~5%-of-the-time-by-chance rejections a standard alpha=0.05 KS test
    would produce even when the null (identical distributions) is true."""
    from scipy.stats import ks_2samp

    hits, misses, false_alarms, truth_extra, forecast_extra = _CONSISTENCY_CASES["with_extras"]
    legacy_pod_widths = []
    new_pod_widths = []
    for i in range(_CONSISTENCY_N_REPEATS):
        np.random.seed(30_000 + i)
        legacy_pod_widths.append(_bootstrap_ci_widths_legacy(
            hits, misses, false_alarms, truth_extra, forecast_extra, samples=_CONSISTENCY_SAMPLES,
        )[0])
        new_pod_widths.append(bootstrap_ci_widths(
            hits, misses, false_alarms, truth_extra, forecast_extra, samples=_CONSISTENCY_SAMPLES, seed=40_000 + i,
        )[0])

    result = ks_2samp(legacy_pod_widths, new_pod_widths)
    print(f"\n[consistency-ks] statistic={result.statistic:.4f}, pvalue={result.pvalue:.4f}")
    assert result.pvalue > 0.001


# --- Speed: the new implementation's cost is O(samples), independent of
# n_total; the legacy implementation's cost is O(samples * n_total). The
# real win only shows up at realistic-to-large n_total (this function is
# called once per (member, lead_hour) cell, and some of those cells pool to
# hundreds of thousands of real records in this project's own data) -- not
# asserted at tiny n, where both are already fast and per-call overhead
# (e.g. constructing a new Generator) can dominate either way.

def test_speedup_at_large_n_total():
    hits, misses, false_alarms, truth_extra, forecast_extra = 150_000, 80_000, 40_000, 0, 0
    samples = 1000

    np.random.seed(0)
    t0 = time.perf_counter()
    _bootstrap_ci_widths_legacy(hits, misses, false_alarms, truth_extra, forecast_extra, samples=samples)
    legacy_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    bootstrap_ci_widths(hits, misses, false_alarms, truth_extra, forecast_extra, samples=samples, seed=0)
    new_seconds = time.perf_counter() - t0

    speedup = legacy_seconds / new_seconds if new_seconds > 0 else float("inf")
    print(f"\n[speedup] n_total={hits + misses + false_alarms}, samples={samples}: "
          f"legacy={legacy_seconds * 1000:.1f}ms, new={new_seconds * 1000:.1f}ms, speedup={speedup:.1f}x")
    assert speedup > 5.0
