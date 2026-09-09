"""Hand-computed validation of distribution_stats.percentile_band_and_median.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_distribution_stats.py -v -s
"""

import math

from py_obj_plotting.climatology.distribution_stats import percentile_band_and_median


def test_hand_computed_percentiles():
    """Default bounds are the interquartile range (25th/75th percentile),
    per the plotting style: shade the IQR, not a wider percentile band."""
    values = list(range(1, 101))  # 1..100
    p25, median, p75 = percentile_band_and_median(values)
    print(f"\n[dist-stats-check1] p25={p25}, median={median}, p75={p75}")
    assert abs(p25 - 25.75) < 1e-9
    assert abs(median - 50.5) < 1e-9
    assert abs(p75 - 75.25) < 1e-9


def test_empty_list_returns_nan():
    p10, median, p90 = percentile_band_and_median([])
    assert math.isnan(p10) and math.isnan(median) and math.isnan(p90)


def test_single_value():
    p10, median, p90 = percentile_band_and_median([5.0])
    assert p10 == median == p90 == 5.0


def test_custom_percentile_bounds():
    """Non-default bounds (10th/90th) still work via explicit lower/upper --
    confirms the function isn't hardcoded to the IQR default."""
    values = list(range(1, 101))
    p10, median, p90 = percentile_band_and_median(values, lower=10.0, upper=90.0)
    print(f"\n[dist-stats-check2] p10={p10}, median={median}, p90={p90}")
    assert abs(p10 - 10.9) < 1e-9
    assert abs(p90 - 90.1) < 1e-9
