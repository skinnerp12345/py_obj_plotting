"""Multi-threshold comparison plots for matched-pair diagnostic-property-
difference distributions: overlays several matching-threshold results (e.g.
100/200/300/400 km) on ONE figure per diagnostic, each threshold its own
color/series with an interquartile-range shaded band + bold median line.

Same architecture as verification_stats_comparison.py: reuses
diagnostic_distribution_timeseries.py's _plot_percentile_band_series
(caller-supplied ax/color) directly, with zero changes needed there.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..climatology import DiagnosticDistributionClimatology
from .diagnostic_distribution_timeseries import _plot_percentile_band_series
from .lead_hour_timeseries import _apply_lead_hour_xticks, _lead_hour_xlim

# (series_label, color, DiagnosticDistributionClimatology)
ThresholdDiagnosticSeries = tuple[str, tuple, DiagnosticDistributionClimatology]


def _plot_diagnostic_comparison_timeseries(
    series: list[ThresholdDiagnosticSeries], value_attr: str,
    out_path: str, ylabel: str, title: str,
    xtick_fn=_apply_lead_hour_xticks,
) -> None:
    """xtick_fn: defaults to the standard 6h-tick/12h-label convention;
    pass e.g. plots.lead_hour_timeseries._apply_lead_hour_xticks_fine for a
    short-range, fine-cadence lead axis (multi-day MPAS-style callers are
    unaffected by this default)."""
    if not series:
        raise ValueError("_plot_diagnostic_comparison_timeseries: series must be non-empty")

    fig, ax = plt.subplots(figsize=(10, 6))
    all_lead_hours: set = set()
    for series_label, color, clim in series:
        pooled_values_by_lead = getattr(clim, value_attr)
        _plot_percentile_band_series(ax, clim.lead_hours, pooled_values_by_lead, color=color, series_label=series_label)
        all_lead_hours.update(clim.lead_hours)

    lead_hours = sorted(all_lead_hours)
    ax.set_xlabel("Forecast lead hour")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(*_lead_hour_xlim(lead_hours))
    xtick_fn(ax, lead_hours)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_area_ratio_comparison_timeseries(series: list[ThresholdDiagnosticSeries], out_path: str, xtick_fn=_apply_lead_hour_xticks, title: str | None = None) -> None:
    _plot_diagnostic_comparison_timeseries(
        series, "area_ratio", out_path,
        ylabel="Area ratio (forecast / truth)",
        title=title if title is not None else "Matched-pair area ratio vs. lead hour: matching threshold comparison (interquartile range, median)",
        xtick_fn=xtick_fn,
    )


def plot_max_intensity_diff_comparison_timeseries(series: list[ThresholdDiagnosticSeries], out_path: str, xtick_fn=_apply_lead_hour_xticks, title: str | None = None) -> None:
    _plot_diagnostic_comparison_timeseries(
        series, "max_intensity_diff", out_path,
        ylabel="Max intensity difference (forecast - truth, dBZ)",
        title=title if title is not None else "Matched-pair max intensity difference vs. lead hour: matching threshold comparison (interquartile range, median)",
        xtick_fn=xtick_fn,
    )


def plot_mean_intensity_diff_comparison_timeseries(series: list[ThresholdDiagnosticSeries], out_path: str, xtick_fn=_apply_lead_hour_xticks, title: str | None = None) -> None:
    _plot_diagnostic_comparison_timeseries(
        series, "mean_intensity_diff", out_path,
        ylabel="Mean intensity difference (forecast - truth, dBZ)",
        title=title if title is not None else "Matched-pair mean intensity difference vs. lead hour: matching threshold comparison (interquartile range, median)",
        xtick_fn=xtick_fn,
    )


def plot_centroid_dist_comparison_timeseries(series: list[ThresholdDiagnosticSeries], out_path: str, xtick_fn=_apply_lead_hour_xticks, title: str | None = None) -> None:
    _plot_diagnostic_comparison_timeseries(
        series, "centroid_dist_km", out_path,
        ylabel="Centroid distance (km)",
        title=title if title is not None else "Matched-pair centroid distance vs. lead hour: matching threshold comparison (interquartile range, median)",
        xtick_fn=xtick_fn,
    )


def plot_ti_score_comparison_timeseries(series: list[ThresholdDiagnosticSeries], out_path: str, xtick_fn=_apply_lead_hour_xticks, title: str | None = None) -> None:
    _plot_diagnostic_comparison_timeseries(
        series, "ti_score", out_path,
        ylabel="Total Interest score",
        title=title if title is not None else "Matched-pair Total Interest vs. lead hour: matching threshold comparison (interquartile range, median)",
        xtick_fn=xtick_fn,
    )
