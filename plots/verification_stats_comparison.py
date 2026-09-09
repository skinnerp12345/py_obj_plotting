"""Multi-threshold comparison plots: overlays several matching-threshold
results (e.g. 100/200/300/400 km max_boundary_disp_km/max_centroid_disp_km)
on ONE figure per metric (POD, FAR, FBIAS, CSI), each threshold its own
color/series with member lines + ensemble mean + shaded CI band.

This is exactly the use case verification_stats_timeseries.py's
_plot_member_ensemble_series was factored out for: it takes a caller-
supplied ax/color, so this module just creates one figure per metric and
calls it once per threshold, with zero changes needed to that function.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..climatology import VerificationStats
from .lead_hour_timeseries import _apply_lead_hour_xticks, _lead_hour_xlim
from .verification_stats_timeseries import _plot_member_ensemble_series

# (series_label, color, VerificationStats)
ThresholdSeries = tuple[str, tuple, VerificationStats]


def _plot_comparison_timeseries(
    series: list[ThresholdSeries], value_attr: str, ci_attr: str,
    out_path: str, ylabel: str, title: str, ylim: tuple | None = None,
    xtick_fn=_apply_lead_hour_xticks,
) -> None:
    """xtick_fn: defaults to the standard 6h-tick/12h-label convention;
    pass e.g. plots.lead_hour_timeseries._apply_lead_hour_xticks_fine for a
    short-range, fine-cadence lead axis (multi-day MPAS-style callers are
    unaffected by this default)."""
    if not series:
        raise ValueError("_plot_comparison_timeseries: series must be non-empty")

    fig, ax = plt.subplots(figsize=(10, 6))
    all_lead_hours: set = set()
    for series_label, color, stats in series:
        values = getattr(stats, value_attr)
        ci_widths = getattr(stats, ci_attr)
        _plot_member_ensemble_series(ax, stats.lead_hours, stats.members, values, ci_widths,
                                       color=color, series_label=series_label)
        all_lead_hours.update(stats.lead_hours)

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


def plot_pod_comparison_timeseries(series: list[ThresholdSeries], out_path: str, xtick_fn=_apply_lead_hour_xticks, title: str | None = None) -> None:
    _plot_comparison_timeseries(
        series, "pod", "pod_ci_width", out_path,
        ylabel="Probability of Detection",
        title=title if title is not None else "POD vs. lead hour: matching threshold comparison (95% bootstrap CI)",
        ylim=(0.0, 1.0), xtick_fn=xtick_fn,
    )


def plot_far_comparison_timeseries(series: list[ThresholdSeries], out_path: str, xtick_fn=_apply_lead_hour_xticks, title: str | None = None) -> None:
    _plot_comparison_timeseries(
        series, "far", "far_ci_width", out_path,
        ylabel="False Alarm Ratio",
        title=title if title is not None else "FAR vs. lead hour: matching threshold comparison (95% bootstrap CI)",
        ylim=(0.0, 1.0), xtick_fn=xtick_fn,
    )


def plot_fbias_comparison_timeseries(series: list[ThresholdSeries], out_path: str, xtick_fn=_apply_lead_hour_xticks, title: str | None = None) -> None:
    _plot_comparison_timeseries(
        series, "fbias", "fbias_ci_width", out_path,
        ylabel="Frequency Bias",
        title=title if title is not None else "Frequency Bias vs. lead hour: matching threshold comparison (95% bootstrap CI)",
        # fbias is unbounded above 1 -- no fixed ylim, unlike pod/far/csi
        xtick_fn=xtick_fn,
    )


def plot_csi_comparison_timeseries(series: list[ThresholdSeries], out_path: str, xtick_fn=_apply_lead_hour_xticks, title: str | None = None) -> None:
    _plot_comparison_timeseries(
        series, "csi", "csi_ci_width", out_path,
        ylabel="Critical Success Index",
        title=title if title is not None else "CSI vs. lead hour: matching threshold comparison (95% bootstrap CI)",
        ylim=(0.0, 1.0), xtick_fn=xtick_fn,
    )
