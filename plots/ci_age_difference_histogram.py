"""Convection-initiation (CI) age-difference histogram: for one system's
matched CI pairs (see climatology.aggregate_ci_skill.
build_ci_age_difference_distribution), the distribution of
truth_age_seconds - forecast_age_seconds (observed - forecast). Ages count up
from 0 at an object's first detection, so positive (forecast object younger)
means the model initiated the storm LATER than observed (late/delayed bias),
negative means EARLIER (premature bias). (Before 2026-10-05 this was
forecast - truth while still labeled this way, i.e. the early/late arrows were
reversed.)

One histogram per system -- this is a single-system timing-bias diagnostic,
not a two-system comparison (see plots.two_system_diff_histograms for that
shape, which compares metric VALUES between two systems, a different
question from one system's own early/late tendency)."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .. import cb_colors

_BIN_WIDTH_MINUTES = 2.0
_MIN_HALF_RANGE_MINUTES = 30.0


def _symmetric_bin_edges_minutes(
    diffs_minutes: np.ndarray, floor: float = _MIN_HALF_RANGE_MINUTES, bin_width: float = _BIN_WIDTH_MINUTES,
) -> np.ndarray:
    """Same construction as two_system_diff_histograms._symmetric_bin_edges
    (symmetric around 0, sized to the largest-magnitude real value, floored,
    rounded up to a whole multiple of bin_width) -- kept local rather than
    imported since that helper is private to its own module and this is a
    different unit (minutes, not a bounded [0,1] metric)."""
    max_abs = float(np.max(np.abs(diffs_minutes))) if len(diffs_minutes) else 0.0
    half_range = max(np.ceil(max(max_abs, floor) / bin_width) * bin_width, bin_width)
    n_bins = int(round(2 * half_range / bin_width))
    return np.linspace(-half_range, half_range, n_bins + 1)


def plot_ci_age_difference_histogram(
    distribution,
    out_path: str,
    age_diff_threshold_minutes: float | None = None,
    title: str | None = None,
    color: tuple = cb_colors.blue5,
) -> int:
    """distribution: a CIAgeDifferenceDistribution (or any object exposing
    the same age_diff_seconds attribute, in seconds). age_diff_threshold_minutes,
    if given, draws dashed reference lines at +/- that value (pass the same
    number used for build_ci_skill_climatology) so the fraction of matched
    CI pairs that would count as a CI hit under that threshold is visible
    directly on the plot, without recomputing it here.

    Returns the number of pairs plotted (0 is a valid, real result -- e.g.
    no truth-and-forecast-both-CI-candidate pairs were found -- the caller
    decides whether to skip writing the file in that case, matching this
    project's established "report, don't silently skip" convention)."""
    diffs_minutes = np.asarray(distribution.age_diff_seconds, dtype=float) / 60.0
    n = len(diffs_minutes)

    edges = _symmetric_bin_edges_minutes(diffs_minutes)
    fig, ax = plt.subplots(figsize=(8, 6))
    if n > 0:
        ax.hist(diffs_minutes, bins=edges, edgecolor="k", linewidth=0.5, color=color)
    ax.set_xlim(edges[0], edges[-1])
    ax.axvline(0.0, color="black", linestyle="-", linewidth=1.0, alpha=0.6)
    if age_diff_threshold_minutes is not None:
        ax.axvline(age_diff_threshold_minutes, color=cb_colors.red5, linestyle="--", linewidth=1.5)
        ax.axvline(-age_diff_threshold_minutes, color=cb_colors.red5, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Observed $-$ forecast object age (minutes)\n"
                  r"$\leftarrow$ model too early            model too late $\rightarrow$")
    ax.set_ylabel("Number of matched CI pairs")
    ax.set_title(title if title is not None else f"CI timing-error distribution ({n} matched pair(s))", fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return n
