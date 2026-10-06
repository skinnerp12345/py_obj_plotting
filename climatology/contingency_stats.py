"""Pure contingency-table verification statistics (POD, FAR, FBIAS, CSI) and
their bootstrap confidence-interval widths -- no I/O, no dependency on
MatchLeadHourClimatology or any file format, so these are independently
testable against hand-computed values.

Formulas and bootstrap method port python_base/wofs_obj_verif_suite_v2.py's
compute_cont_table_metrics()/_bootstrap_ci_fast() (confirmed against
wofs_verif_cbook.py's original bootstrap_ci(), which _bootstrap_ci_fast is a
numerically-validated vectorized reimplementation of), with two confirmed,
deliberate deviations from that codebase:

- POD/FAR/CSI use the standard 3-category contingency table (hit/miss/
  false_alarm only). FBIAS uses the "extended bias" treatment (folds
  truth_extra into the miss-like denominator term, forecast_extra into the
  false-alarm-like numerator term) -- the one metric where the match
  schema's 5th/4th categories are counted.
- A zero-denominator bin returns 0.0 (matching wofs_verif_cbook.py's
  bootstrap_ci exactly), not NaN.
"""

import numpy as np

DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_CI_INTERVAL = 95


def compute_pod(hits: int, misses: int) -> float:
    denom = hits + misses
    return hits / denom if denom > 0 else 0.0


def compute_far(hits: int, false_alarms: int) -> float:
    denom = hits + false_alarms
    return false_alarms / denom if denom > 0 else 0.0


def compute_csi(hits: int, misses: int, false_alarms: int) -> float:
    denom = hits + misses + false_alarms
    return hits / denom if denom > 0 else 0.0


def compute_fbias(hits: int, misses: int, false_alarms: int, truth_extra: int, forecast_extra: int) -> float:
    """Extended bias: truth_extra folds into the denominator (it's an
    unmatched truth object, the same role as a miss); forecast_extra folds
    into the numerator (it's an unmatched forecast object, the same role as
    a false alarm)."""
    denom = hits + misses + truth_extra
    return (hits + false_alarms + forecast_extra) / denom if denom > 0 else 0.0


def bootstrap_ci_widths(
    hits: int, misses: int, false_alarms: int, truth_extra: int, forecast_extra: int,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES, interval: float = DEFAULT_CI_INTERVAL,
    seed=None,
) -> tuple[float, float, float, float]:
    """Bootstrap CI width (upper percentile - lower percentile of the
    per-replicate statistic distribution, NOT a percentile band around the
    point estimate) for (pod, far, fbias, csi). Returns (0., 0., 0., 0.) if
    there are no records at all.

    Resampling a length-n categorical population with replacement and
    counting each category's occurrences is EXACTLY the Multinomial(n, p)
    distribution by definition of resampling-with-replacement -- not an
    approximation of the older element-by-element approach (concatenate a
    length-n vector, then np.random.randint gather `samples` resamples of
    it), just a direct draw from the same target distribution via numpy's
    own multinomial sampler. This makes bootstrapping O(samples) (five
    category-probability draws per replicate), independent of n, instead of
    O(samples * n) for both compute and the intermediate index array's
    memory -- a real difference at the scale this is called at (once per
    (member, lead_hour) cell in verification_stats.compute_verification_stats(),
    where n can range from tens to hundreds of thousands of pooled records).

    seed: optional, forwarded to numpy.random.default_rng() for
    reproducible output -- None (default) matches the prior implementation's
    non-reproducible-by-default behavior.
    """
    n_total = hits + misses + false_alarms + truth_extra + forecast_extra
    if n_total == 0:
        return 0.0, 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    probs = [hits / n_total, misses / n_total, false_alarms / n_total,
             truth_extra / n_total, forecast_extra / n_total]
    counts = rng.multinomial(n_total, probs, size=samples)
    h, m, f, te, fe = (counts[:, i].astype(float) for i in range(5))

    # np.divide(..., where=...) leaves the output at its `out=` initial value
    # (0.0) wherever the mask is False, without evaluating that division at
    # all -- unlike np.where(cond, a/b, 0.0), which computes a/b eagerly for
    # every element (including 0/0 ones) and only discards the result after,
    # triggering a spurious "invalid value encountered in divide" warning.
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


def bootstrap_pod_diff_ci_width(
    hits_a: int, misses_a: int, hits_b: int, misses_b: int,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES, interval: float = DEFAULT_CI_INTERVAL,
) -> float:
    """Bootstrap CI width of the POD DIFFERENCE (pod_a - pod_b) between two
    INDEPENDENT hit/miss populations -- each side resampled independently
    (not a paired resample), ported from python_base/wofs_verif_cbook.py's
    heat_map_diff_ci() / wofs_obj_verif_suite_v2.py's _heat_map_diff_ci_fast().
    Used to test whether a POD difference between two systems at one
    (property-bin, lead-hour) heatmap cell is statistically significant.
    Returns 0.0 if either side has zero total records (no meaningful POD to
    compare)."""
    n_a, n_b = hits_a + misses_a, hits_b + misses_b
    if n_a == 0 or n_b == 0:
        return 0.0

    vec_a = np.concatenate([np.ones(hits_a), np.zeros(misses_a)])
    vec_b = np.concatenate([np.ones(hits_b), np.zeros(misses_b)])
    idx_a = np.random.randint(0, n_a, size=(samples, n_a))
    idx_b = np.random.randint(0, n_b, size=(samples, n_b))
    diff_samples = np.mean(vec_a[idx_a], axis=1) - np.mean(vec_b[idx_b], axis=1)

    lo, hi = (100 - interval) / 2.0, (100 - interval) / 2.0 + interval
    return np.percentile(diff_samples, hi) - np.percentile(diff_samples, lo)
