"""Pure percentile/median statistics for a pooled list of raw values -- no
I/O, no dependency on DiagnosticDistributionClimatology, independently
testable.
"""

import numpy as np

DEFAULT_LOWER_PERCENTILE = 25.0
DEFAULT_UPPER_PERCENTILE = 75.0


def percentile_band_and_median(
    values: list, lower: float = DEFAULT_LOWER_PERCENTILE, upper: float = DEFAULT_UPPER_PERCENTILE,
) -> tuple[float, float, float]:
    """(p_lower, median, p_upper) of `values` -- (nan, nan, nan) if empty."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, lower)), float(np.median(arr)), float(np.percentile(arr, upper))
