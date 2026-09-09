"""Two-system metric-difference histograms: for exactly two systems being
compared, one histogram per contingency metric (POD, FAR, FBIAS, CSI)
showing the DISTRIBUTION of (system_a's ensemble-mean metric) - (system_b's
ensemble-mean metric) across every matched case (the intersection of both
systems' own case_label sets, see plots.performance_diagram.matched_cases_report).

Reuses aggregate_matches_per_case.case_metrics() unchanged for the
per-case, per-lead-range ensemble-mean metric values -- no new stats math,
only new data-shaping + histogram rendering. Shares
plots.performance_diagram.matched_cases_report() with the two-system
CSI-difference performance diagram (plot_two_system_case_diff_performance_diagram),
since both features restrict to the identical matched-case population.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .. import cb_colors
from ..climatology.aggregate_matches_per_case import case_metrics
from .performance_diagram import TwoSystemMatchedCasesReport, matched_cases_report

_METRICS = ("pod", "far", "fbias", "csi")
_METRIC_LABELS = {"pod": "POD", "far": "FAR", "fbias": "Frequency Bias", "csi": "CSI"}

_BIN_WIDTH = 0.025

# Minimum half-width the x-axis (and bin range) must span, even if the
# real data's own difference range is smaller -- keeps small-sample
# histograms from looking artificially "zoomed in."
_MIN_HALF_RANGE = {"pod": 0.4, "far": 0.4, "csi": 0.2, "fbias": 0.4}


def _symmetric_bin_edges(diffs: list, floor: float | None, bin_width: float = _BIN_WIDTH) -> np.ndarray:
    """Bin edges spanning a range SYMMETRIC around zero, sized to the
    largest-magnitude value in `diffs` (positive or negative), clamped to
    at least `floor` (half-width) if given, then rounded UP to the
    nearest whole multiple of bin_width -- guarantees a bin edge falls
    exactly on 0.0 (no single bin straddles both signs, needed since every
    metric is now sign-colored) and every bin has the identical width
    (no partial last bin)."""
    max_abs = max(abs(min(diffs)), abs(max(diffs))) if diffs else 0.0
    if floor is not None:
        max_abs = max(max_abs, floor)
    half_range = max(np.ceil(max_abs / bin_width) * bin_width, bin_width)
    n_bins = int(round(2 * half_range / bin_width))
    return np.linspace(-half_range, half_range, n_bins + 1)


def plot_two_system_diff_histograms(
    system_a: tuple, system_b: tuple, lead_minutes_range: tuple, out_dir: str,
    title_prefix: str | None = None, filename_prefix: str = "",
) -> TwoSystemMatchedCasesReport:
    """system_a/system_b: (system_name, color, dict[case_label, MatchLeadHourClimatology]).

    Writes 4 histogram PNGs to out_dir (`<filename_prefix><metric>_diff_histogram.png`
    for metric in pod/far/fbias/csi) -- one value per matched case
    (ensemble-mean-based), aggregated over lead_minutes_range via
    case_metrics(). Every histogram uses a fixed bin width (_BIN_WIDTH,
    0.025) and a per-metric x-axis range computed dynamically from the
    real data (symmetric around 0, sized to the largest-magnitude
    difference actually present, floored per _MIN_HALF_RANGE -- see
    _symmetric_bin_edges()); bars are colored by the sign of their own bin
    center (positive, system_a scoring higher, in orange; negative,
    system_b scoring higher, in blue) for all 4 metrics, including fbias.
    `filename_prefix` lets a caller writing multiple lead-range periods
    into one flat output directory keep each period's 4 files distinct.
    Returns the same TwoSystemMatchedCasesReport
    plot_two_system_case_diff_performance_diagram() returns, so a caller
    computing both from the same pair only needs to report the match
    count once.
    """
    name_a, _color_a, cases_a = system_a
    name_b, _color_b, cases_b = system_b
    report = matched_cases_report(cases_a, cases_b)
    if not report.matched_cases:
        raise ValueError(
            f"plot_two_system_diff_histograms: no matched cases between "
            f"'{name_a}' ({len(cases_a)} case(s)) and '{name_b}' ({len(cases_b)} case(s))"
        )

    diffs = {metric: [] for metric in _METRICS}
    for case_label in report.matched_cases:
        metrics_a = case_metrics(cases_a[case_label], lead_minutes_range)
        metrics_b = case_metrics(cases_b[case_label], lead_minutes_range)
        for metric in _METRICS:
            diffs[metric].append(metrics_a[metric] - metrics_b[metric])

    os.makedirs(out_dir, exist_ok=True)
    for metric in _METRICS:
        edges = _symmetric_bin_edges(diffs[metric], floor=_MIN_HALF_RANGE.get(metric))
        fig, ax = plt.subplots(figsize=(8, 6))
        _, out_edges, patches = ax.hist(diffs[metric], bins=edges, edgecolor="k", linewidth=0.5)
        bin_width = out_edges[1] - out_edges[0]
        for patch, left_edge in zip(patches, out_edges[:-1]):
            bin_center = left_edge + bin_width / 2.0
            patch.set_facecolor(cb_colors.orange5 if bin_center > 0 else cb_colors.blue5)
        ax.set_xlim(edges[0], edges[-1])
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.set_xlabel(f"{_METRIC_LABELS[metric]} difference ({name_a} - {name_b})")
        ax.set_ylabel("Number of matched cases")
        title = f"{_METRIC_LABELS[metric]} difference distribution, {len(report.matched_cases)} matched cases"
        if title_prefix is not None:
            title = f"{title_prefix}: {title}"
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{filename_prefix}{metric}_diff_histogram.png"), dpi=150)
        plt.close(fig)

    return report
