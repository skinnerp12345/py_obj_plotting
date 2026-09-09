"""Multi-system comparison plots for "overall" object-level statistics --
total object count and total summed area by forecast lead hour (one line per
system, same thin-member/bold-ensemble-mean styling as every other
comparison plot in this package, plus one black MRMS truth reference line),
and box-and-whisker plots of individual object area and mean intensity
pooled across every included lead hour and member (one box per system, plus
an MRMS truth box).

Same architecture as verification_stats_comparison.py/
diagnostic_distribution_comparison.py: reuses object_stats_timeseries.py's
_plot_member_ensemble_lines (caller-supplied ax/color) for the forecast-side
series, plus its _mrms_series/_MRMS_LINEWIDTH for the truth reference line.

The MRMS series is read from series[0][2] (any system's own
ObjectStatsMultiSystemClimatology) since truth_count/truth_area_km2/
truth_area_values/truth_mean_intensity_values/truth_lead_hours are computed
ONCE and attached identically to every system's climatology object -- see
aggregate_matches_multi_system.build_multi_system_object_climatology's
docstring for why (MRMS truth objects are system-independent).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..climatology import ObjectStatsMultiSystemClimatology
from .lead_hour_timeseries import _apply_lead_hour_xticks, _lead_hour_xlim
from .object_stats_timeseries import _MRMS_LINEWIDTH, _mrms_series, _plot_member_ensemble_lines

# (series_label, color, ObjectStatsMultiSystemClimatology)
ObjectStatsSeries = tuple[str, tuple, ObjectStatsMultiSystemClimatology]


def _plot_object_stats_comparison_timeseries(
    series: list[ObjectStatsSeries], value_attr: str, truth_attr: str | None,
    out_path: str, ylabel: str, title: str,
    xtick_fn=_apply_lead_hour_xticks,
) -> None:
    """truth_attr: pass None to omit the MRMS reference line entirely
    (forecast-side systems only)."""
    if not series:
        raise ValueError("_plot_object_stats_comparison_timeseries: series must be non-empty")

    fig, ax = plt.subplots(figsize=(10, 6))
    all_lead_hours: set = set()
    for series_label, color, clim in series:
        values_by_member_lead = getattr(clim, value_attr)
        _plot_member_ensemble_lines(ax, clim.lead_hours, clim.members, values_by_member_lead,
                                     color=color, series_label=series_label)
        all_lead_hours.update(clim.lead_hours)

    if truth_attr is not None:
        # MRMS truth reference: shared/identical across every system's own
        # climatology object (see module docstring), so any one of them
        # supplies it.
        _, _, any_clim = series[0]
        truth_values_by_lead = getattr(any_clim, truth_attr)
        truth_series = _mrms_series(truth_values_by_lead, any_clim.truth_lead_hours)
        ax.plot(any_clim.truth_lead_hours, truth_series, color="black", linestyle="-",
                 linewidth=_MRMS_LINEWIDTH, label="MRMS")
        all_lead_hours.update(any_clim.truth_lead_hours)

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


def plot_object_count_comparison_timeseries(
    series: list[ObjectStatsSeries], out_path: str, xtick_fn=_apply_lead_hour_xticks, title: str | None = None,
) -> None:
    _plot_object_stats_comparison_timeseries(
        series, "forecast_count", "truth_count", out_path,
        ylabel="Object count", title=title or "Total object count vs. lead hour: system comparison (MRMS in black)",
        xtick_fn=xtick_fn,
    )


def plot_object_area_comparison_timeseries(
    series: list[ObjectStatsSeries], out_path: str, xtick_fn=_apply_lead_hour_xticks, title: str | None = None,
) -> None:
    _plot_object_stats_comparison_timeseries(
        series, "forecast_area_km2", "truth_area_km2", out_path,
        ylabel="Summed object area (km$^2$)", title=title or "Total summed object area vs. lead hour: system comparison (MRMS in black)",
        xtick_fn=xtick_fn,
    )


def _plot_object_stats_boxplot(
    series: list[ObjectStatsSeries], value_attr: str, truth_attr: str,
    out_path: str, ylabel: str, title: str,
) -> None:
    """One box per system (patch_artist-filled in that system's own
    comparison color) plus one MRMS truth box (black, read from series[0][2]
    -- see module docstring for why any one system's climatology object
    supplies it), pooling value_attr/truth_attr (flat list attributes on
    ObjectStatsMultiSystemClimatology) across every included lead hour and
    member -- i.e. every individual object's own area/intensity, not a
    per-lead-hour summary statistic. Outlier fliers are suppressed
    (showfliers=False): with thousands of pooled objects the flier cloud
    otherwise dominates the figure and adds little beyond what the whiskers
    already convey."""
    if not series:
        raise ValueError("_plot_object_stats_boxplot: series must be non-empty")

    labels = [s[0] for s in series]
    colors = [s[1] for s in series]
    data = [getattr(clim, value_attr) for _, _, clim in series]

    _, _, any_clim = series[0]
    labels.append("MRMS")
    colors.append("black")
    data.append(getattr(any_clim, truth_attr))

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for median_line in bp["medians"]:
        median_line.set_color("black")
        median_line.set_linewidth(2.0)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_object_area_boxplot(series: list[ObjectStatsSeries], out_path: str) -> None:
    _plot_object_stats_boxplot(
        series, "forecast_area_values", "truth_area_values", out_path,
        ylabel="Object area (km$^2$)",
        title="Object area distribution, all included lead hours pooled: system comparison",
    )


def plot_object_mean_intensity_boxplot(series: list[ObjectStatsSeries], out_path: str) -> None:
    _plot_object_stats_boxplot(
        series, "forecast_mean_intensity_values", "truth_mean_intensity_values", out_path,
        ylabel="Object mean intensity (dBZ)",
        title="Object mean intensity distribution, all included lead hours pooled: system comparison",
    )
