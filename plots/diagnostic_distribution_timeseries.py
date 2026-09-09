"""Shared low-level drawing helper for matched-pair ("hit" category only)
diagnostic-property-difference distributions over forecast lead hour: area
ratio (forecast/truth), max/mean intensity difference (forecast - truth),
centroid distance, and Total Interest, built on top of
climatology.build_diagnostic_distribution_climatology's pooled per-lead-hour
raw value lists.

Draws a light shaded band between the interquartile range (25th-75th
percentile, see climatology.distribution_stats.DEFAULT_LOWER_PERCENTILE/
DEFAULT_UPPER_PERCENTILE) of the pooled distribution, with the median
plotted as a bold line -- pooled across every ensemble member (no
per-member breakdown, since a per-member split isn't meaningful for a
property-difference distribution).

Factored to take a caller-supplied `ax` and `color` rather than creating its
own figure, so a multi-system comparison plot
(diagnostic_distribution_comparison.py) can call it once per system on one
shared figure/ax.
"""

from .. import cb_colors
from ..climatology import percentile_band_and_median

_MEDIAN_LINEWIDTH = 2.5
_BAND_ALPHA = 0.2
_DEFAULT_COLOR = cb_colors.orange5


def _plot_percentile_band_series(
    ax, lead_hours: list, pooled_values_by_lead: dict,
    color=_DEFAULT_COLOR, series_label: str = "MPAS",
) -> None:
    """Draws one climatology's interquartile-range shaded band + bold
    median line onto an EXISTING ax, in a given color.

    pooled_values_by_lead: dict[lead_hour] -> list[float] of pooled raw
    values (e.g. DiagnosticDistributionClimatology.centroid_dist_km).
    """
    p10s, medians, p90s = [], [], []
    for lh in lead_hours:
        p10, median, p90 = percentile_band_and_median(pooled_values_by_lead.get(lh, []))
        p10s.append(p10)
        medians.append(median)
        p90s.append(p90)

    ax.fill_between(lead_hours, p10s, p90s, color=color, alpha=_BAND_ALPHA, linewidth=0)
    ax.plot(lead_hours, medians, color=color, linestyle="-", linewidth=_MEDIAN_LINEWIDTH,
             label=f"{series_label} median")
