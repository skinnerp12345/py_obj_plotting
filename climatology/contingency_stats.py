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
) -> tuple[float, float, float, float]:
    """Bootstrap CI width (upper percentile - lower percentile of the
    per-replicate statistic distribution, NOT a percentile band around the
    point estimate) for (pod, far, fbias, csi), resampling the flattened set
    of individual match records with replacement -- exactly
    wofs_obj_verif_suite_v2.py's _bootstrap_ci_fast method, extended with
    truth_extra/forecast_extra labels since FBIAS (unlike POD/FAR/CSI) uses
    them. Returns (0., 0., 0., 0.) if there are no records at all.
    """
    n_total = hits + misses + false_alarms + truth_extra + forecast_extra
    if n_total == 0:
        return 0.0, 0.0, 0.0, 0.0

    # 1=hit, 0=miss, 2=false_alarm, 3=truth_extra, 4=forecast_extra
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
