"""Single-case-vs-baseline mode: isolating ONE case of ONE system against a
baseline of its other available cases (or a user-specified subset) -- not a
comparison between different systems. Two rendering variants, one per plot
family already built in this package:

- plot_single_case_highlight_timeseries: POD/FAR/FBIAS/CSI vs. lead hour
  (the existing time series' data shape, VerificationStats) -- the
  baseline population is shown as a shaded percentile-band distribution
  (10th-90th and 25th-75th, plus a median line), not individual lines.
- plot_single_case_highlight_case_scatter: the case-scatter performance
  diagram (plots.performance_diagram.plot_case_scatter_performance_diagram).

The highlighted case renders in a bold `highlight_color` on top of the
baseline population, full alpha (the time series gives it its own legend
entry; the case-scatter variant has no legend at all -- the highlight
color + label already identify it unambiguously). Baseline population
comes from climatology.split_baseline_and_highlight() -- this module only
renders, it doesn't decide which cases are baseline vs. highlighted.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .. import cb_colors
from ..climatology.aggregate_matches_per_case import case_metrics, case_object_count
from .lead_hour_timeseries import _apply_lead_hour_xticks, _lead_hour_xlim
from .object_stats_timeseries import _ensemble_mean, _member_series
from .performance_diagram import _MARKER_ALPHA, _size_scale, init_performance_diagram

_BASELINE_COLOR = cb_colors.gray5
_BASELINE_ALPHA = 0.25
_BASELINE_MARKER_COLOR = cb_colors.gray7  # darker than _BASELINE_COLOR, for case-scatter markers specifically
_DEFAULT_HIGHLIGHT_COLOR = cb_colors.orange5

_HIGHLIGHT_LABEL_FONTSIZE = 12
_HIGHLIGHT_LABEL_OFFSET_POINTS = (0, 14)

_PERCENTILE_OUTER = (10, 90)
_PERCENTILE_INNER = (25, 75)
_OUTER_BAND_ALPHA = 0.15
_INNER_BAND_ALPHA = 0.30
_MEDIAN_COLOR = "black"
_MEDIAN_LINEWIDTH = 1.5


def plot_single_case_highlight_timeseries(
    baseline_stats: list, highlighted_stats, value_attr: str, out_path: str,
    ylabel: str, title: str, highlight_color=None, highlight_label: str = "Highlighted case",
    ylim: tuple | None = None, xtick_fn=_apply_lead_hour_xticks,
) -> None:
    """baseline_stats: list[VerificationStats] (one per baseline case),
    reduced to its own per-case ensemble-mean value_attr series (e.g.
    "pod", "far", "fbias", "csi") -- NOT the member-spaghetti+CI-band style
    the N-system comparison plots use, since this mode's visual goal is a
    population-vs-one contrast, not member spread. Rather than drawing
    each baseline case as its own line (illegible once there are more
    than a handful of cases), the baseline population is summarized as a
    shaded percentile-band distribution per lead hour: _PERCENTILE_OUTER
    (10th-90th) and _PERCENTILE_INNER (25th-75th) shaded gray bands, plus
    a black median line -- computed across whichever baseline cases have
    data at each lead hour (lead-hour coverage need not be identical
    across all baseline cases). highlighted_stats: VerificationStats for
    the one highlighted case, plotted on top of the background
    distribution in `highlight_color`, full alpha."""
    if highlight_color is None:
        highlight_color = _DEFAULT_HIGHLIGHT_COLOR

    fig, ax = plt.subplots(figsize=(10, 6))
    all_lead_hours: set = set()

    values_by_lead_hour: dict = {}
    for stats in baseline_stats:
        if not stats.lead_hours:
            continue
        values = getattr(stats, value_attr)
        ensemble_mean = _ensemble_mean(_member_series(values, stats.members, stats.lead_hours), stats.members)
        for lh, v in zip(stats.lead_hours, ensemble_mean):
            values_by_lead_hour.setdefault(lh, []).append(v)
        all_lead_hours.update(stats.lead_hours)

    if values_by_lead_hour:
        band_lead_hours = sorted(values_by_lead_hour.keys())
        p_lo_outer, p_hi_outer = _PERCENTILE_OUTER
        p_lo_inner, p_hi_inner = _PERCENTILE_INNER
        outer_lo, outer_hi, inner_lo, inner_hi, median = [], [], [], [], []
        for lh in band_lead_hours:
            vals = values_by_lead_hour[lh]
            outer_lo.append(np.percentile(vals, p_lo_outer))
            outer_hi.append(np.percentile(vals, p_hi_outer))
            inner_lo.append(np.percentile(vals, p_lo_inner))
            inner_hi.append(np.percentile(vals, p_hi_inner))
            median.append(np.percentile(vals, 50))

        ax.fill_between(band_lead_hours, outer_lo, outer_hi, color=_BASELINE_COLOR, alpha=_OUTER_BAND_ALPHA,
                         label=f"All Cases ({p_lo_outer}th-{p_hi_outer}th pct.)", zorder=1)
        ax.fill_between(band_lead_hours, inner_lo, inner_hi, color=_BASELINE_COLOR, alpha=_INNER_BAND_ALPHA,
                         label=f"All Cases ({p_lo_inner}th-{p_hi_inner}th pct.)", zorder=2)
        ax.plot(band_lead_hours, median, color=_MEDIAN_COLOR, linewidth=_MEDIAN_LINEWIDTH,
                label="All Cases (median)", zorder=3)

    if highlighted_stats.lead_hours:
        h_values = getattr(highlighted_stats, value_attr)
        h_mean = _ensemble_mean(_member_series(h_values, highlighted_stats.members, highlighted_stats.lead_hours), highlighted_stats.members)
        ax.plot(highlighted_stats.lead_hours, h_mean, color=highlight_color, linewidth=2.5,
                 alpha=1.0, label=highlight_label, zorder=5)
        all_lead_hours.update(highlighted_stats.lead_hours)

    lead_hours = sorted(all_lead_hours)
    ax.set_xlabel("Forecast lead hour")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(*_lead_hour_xlim(lead_hours))
    if ylim is not None:
        ax.set_ylim(*ylim)
    xtick_fn(ax, lead_hours)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_single_case_highlight_case_scatter(
    baseline_climatologies: list, highlighted_climatology, lead_minutes_range: tuple, out_path: str,
    highlight_color=None, highlight_label: str = "Highlighted case", marker_size_basis: str = "truth",
    title: str | None = None,
) -> None:
    """baseline_climatologies: list[MatchLeadHourClimatology] (one per
    baseline case) and the highlighted case share ONE linear size scale
    based on case_object_count() (see performance_diagram._size_scale) --
    cases with the same object count render at the same marker size
    regardless of whether one of them happens to be the highlighted case.
    Baseline markers render in a darker gray (_BASELINE_MARKER_COLOR,
    low alpha, no edge); the highlighted marker uses `highlight_color` at
    full alpha with a black edge, labeled ABOVE the marker (not centered
    within it). No legend (the highlight color + label already identify
    it unambiguously)."""
    if highlight_color is None:
        highlight_color = _DEFAULT_HIGHLIGHT_COLOR

    fig, ax = init_performance_diagram()

    baseline_points = []  # (sr, pod, count)
    for clim in baseline_climatologies:
        metrics = case_metrics(clim, lead_minutes_range)
        count = case_object_count(clim, lead_minutes_range, basis=marker_size_basis)
        baseline_points.append((1.0 - metrics["far"], metrics["pod"], count))

    h_metrics = case_metrics(highlighted_climatology, lead_minutes_range)
    h_sr, h_pod = 1.0 - h_metrics["far"], h_metrics["pod"]
    h_count = case_object_count(highlighted_climatology, lead_minutes_range, basis=marker_size_basis)

    all_counts = [c for _, _, c in baseline_points] + [h_count]
    min_count, max_count = min(all_counts), max(all_counts)

    for sr, pod, count in baseline_points:
        size = _size_scale(count, min_count, max_count)
        ax.scatter([sr], [pod], s=[size], c=[_BASELINE_MARKER_COLOR], alpha=_BASELINE_ALPHA,
                   edgecolors="none", zorder=2)

    h_size = _size_scale(h_count, min_count, max_count)
    ax.scatter([h_sr], [h_pod], s=[h_size], c=[highlight_color], alpha=_MARKER_ALPHA,
               edgecolors="k", linewidths=1.0, zorder=4)
    ax.annotate(highlight_label, (h_sr, h_pod), textcoords="offset points", xytext=_HIGHLIGHT_LABEL_OFFSET_POINTS,
                ha="center", va="bottom", fontsize=_HIGHLIGHT_LABEL_FONTSIZE, fontweight="bold", zorder=5)

    if title is not None:
        ax.set_title(title, fontsize=13)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
