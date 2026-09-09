"""Performance diagrams (Success Ratio vs. Probability of Detection, with
background Critical Success Index contours and frequency-Bias radial
lines -- Roebber 2009), matching the exact visual style of the legacy
`python_base/wofs_verif_cbook.py`'s `perf_diagram_init()`/
`perf_diagram_cbar_init()`.

`init_performance_diagram()` builds ONLY the axes/contours/labels -- points
are added afterward by the caller (this module's own scatter/line-drawing
functions, or `plots/case_scatter_performance_diagram.py` etc.), exactly
mirroring the legacy's own separation between diagram initialization and
point plotting.
"""

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D

from .. import cb_colors
from ..climatology.aggregate_matches_per_case import case_metrics, case_object_count
from .object_stats_timeseries import _ensemble_mean, _member_series

_CSI_LEVELS = np.arange(0.1, 1.1, 0.1)
_CSI_COLOR = cb_colors.purple5
_BIAS_LEVELS = [0.25, 0.5, 1., 1.5, 2., 3., 5.]
_BIAS_COLORS = [cb_colors.gray5, cb_colors.gray5, cb_colors.gray7, cb_colors.gray5,
                cb_colors.gray5, cb_colors.gray5, cb_colors.gray5]  # bias=1 (index 2) darker
_BIAS_LABEL_POSITIONS = [(.2, 0.95), (.3, .95), (.47, .95), (.57, .95), (.8, .8), (.8, .42), (.8, .2)]
_CSI_LABEL_POSITIONS = [(.92, 0.1), (.92, .2), (.92, 0.3), (.92, .4), (.92, 0.5), (.92, .6),
                         (.92, 0.7), (.95, .85), (.95, 0.95)]
_CONTOUR_LABEL_FONTSIZE = 14  # reduced from 16

_MIN_MARKER_SIZE = 80.0
_MAX_MARKER_SIZE = 650.0
_MARKER_ALPHA = 0.7
_LABEL_FONTSIZE = 9

# Default (no explicit figsize) geometry: the SR-vs-POD plot AXES itself
# is always exactly _AXES_SIZE_INCHES square in physical inches, regardless
# of variant -- reserve_colorbar only adds width beyond the axes' right
# edge (room for a colorbar attached later by the caller), never height.
# _TITLE_MARGIN_INCHES adds fixed headroom above the axes so a caller-
# supplied title (via ax.set_title()) isn't clipped by the figure's top
# edge -- previously the axes' top edge sat only 2% of the figure height
# from the top, leaving no room for a title at all.
_AXES_SIZE_INCHES = 7.04  # 0.88 * 8, the legacy's own axes side length
_LEFT_MARGIN_INCHES = 0.8  # 0.1 * 8
_BOTTOM_MARGIN_INCHES = 0.8  # 0.1 * 8
_RIGHT_MARGIN_INCHES = 0.16  # 0.02 * 8
_TITLE_MARGIN_INCHES = 0.6
_COLORBAR_EXTRA_WIDTH_INCHES = 0.75  # matches the legacy's own 8.75 - 8.0 reserve
_COLORBAR_PAD_INCHES = 0.3  # gap between the main axes' right edge and the colorbar
_COLORBAR_WIDTH_INCHES = 0.3


def _add_colorbar_axes(fig):
    """Adds and returns a colorbar-only axes to the right of the main
    SR-vs-POD axes, positioned/sized from the figure's own actual
    dimensions within the width init_performance_diagram(reserve_colorbar=True)
    already reserved. Callers MUST use this (via fig.colorbar(scat,
    cax=...)) instead of fig.colorbar(scat, ax=ax) -- the latter silently
    auto-shrinks `ax` to make room for the colorbar (confirmed directly:
    it cuts the main axes' own width by ~20%, e.g. 7.04in -> 5.63in while
    leaving height untouched), breaking the square SR-vs-POD aspect ratio
    init_performance_diagram() otherwise guarantees."""
    fig_w_in, fig_h_in = fig.get_size_inches()
    cax_left_in = _LEFT_MARGIN_INCHES + _AXES_SIZE_INCHES + _COLORBAR_PAD_INCHES
    return fig.add_axes([
        cax_left_in / fig_w_in, _BOTTOM_MARGIN_INCHES / fig_h_in,
        _COLORBAR_WIDTH_INCHES / fig_w_in, _AXES_SIZE_INCHES / fig_h_in,
    ])


def init_performance_diagram(figsize: tuple | None = None, reserve_colorbar: bool = False):
    """Returns (fig, ax) with the CSI contour lines + bias radial lines +
    axes/spines/ticks/labels already drawn, per the legacy style exactly
    -- no points plotted yet. With no explicit figsize, the axes box is
    always a square _AXES_SIZE_INCHES x _AXES_SIZE_INCHES (reserve_colorbar
    only widens the figure, never the axes' own shape), with
    _TITLE_MARGIN_INCHES of headroom reserved above it. Pass an explicit
    figsize to opt out of this geometry entirely and use the plain
    0.1/0.88-fraction rect instead (keeping it square, if desired, is then
    the caller's own responsibility)."""
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.1, 0.88, 0.88])
    else:
        fig_width = _LEFT_MARGIN_INCHES + _AXES_SIZE_INCHES + _RIGHT_MARGIN_INCHES
        if reserve_colorbar:
            fig_width += _COLORBAR_EXTRA_WIDTH_INCHES
        fig_height = _BOTTOM_MARGIN_INCHES + _AXES_SIZE_INCHES + _TITLE_MARGIN_INCHES
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_axes([
            _LEFT_MARGIN_INCHES / fig_width, _BOTTOM_MARGIN_INCHES / fig_height,
            _AXES_SIZE_INCHES / fig_width, _AXES_SIZE_INCHES / fig_height,
        ])

    grid = np.arange(0., 1.005, 0.005)
    sr_grid, pod_grid = np.meshgrid(grid, grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        bias_grid = pod_grid / sr_grid
        csi_grid = 1. / (1. / sr_grid + 1. / pod_grid - 1)

    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_alpha(0.7)
        ax.spines[side].set_linewidth(0.5)

    ticks = np.arange(0., 1.1, 0.1)
    tick_labels = [""] + [f"{t:.1f}" for t in ticks[1:-1]] + [""]

    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=16, alpha=0.7)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=16, alpha=0.7)
    ax.tick_params(axis="both", which="both", bottom=True, top=False, labelbottom=True, left=True, right=False, labelleft=True)

    ax.set_xlabel("Success Ratio (1 - FAR)", fontsize=18, alpha=0.7)
    ax.set_ylabel("Probability of Detection", fontsize=18, alpha=0.7)

    csi_con = ax.contour(sr_grid, pod_grid, csi_grid, levels=_CSI_LEVELS,
                          colors=[_CSI_COLOR] * len(_CSI_LEVELS), linewidths=1., alpha=0.8)
    bias_con = ax.contour(sr_grid, pod_grid, bias_grid, levels=_BIAS_LEVELS,
                           colors=_BIAS_COLORS, linewidths=1., alpha=0.4)
    ax.clabel(bias_con, fmt="%1.1f", manual=_BIAS_LABEL_POSITIONS, fontsize=_CONTOUR_LABEL_FONTSIZE)
    ax.clabel(csi_con, fmt="%1.1f", manual=_CSI_LABEL_POSITIONS, fontsize=_CONTOUR_LABEL_FONTSIZE)

    return fig, ax


def _size_scale(count: float, min_count: float, max_count: float) -> float:
    """Linear marker-size interpolation (matplotlib scatter `s`, points^2)
    -- ONE shared scale computed across every point in a figure (never
    per-system/per-case independently), so relative sizes stay directly
    comparable across the whole diagram."""
    if max_count <= min_count:
        return (_MIN_MARKER_SIZE + _MAX_MARKER_SIZE) / 2.0
    frac = (count - min_count) / (max_count - min_count)
    return _MIN_MARKER_SIZE + frac * (_MAX_MARKER_SIZE - _MIN_MARKER_SIZE)


def chronological_case_indices(case_labels) -> dict:
    """1-based integer index, earliest case = 1, assigned by sorting the
    given case_label strings (lexicographic order = chronological order,
    given this project's "YYYY-MM-DD"/"YYYY-MM-DD HHMMZ" case_label
    convention). Used as a compact on-marker label instead of the full
    date string, which takes up too much space once many cases are
    plotted on one diagram -- callers needing to know which real case a
    given number refers to use the returned dict directly (e.g. print it,
    or fold it into a driver's own result reporting)."""
    return {label: i + 1 for i, label in enumerate(sorted(set(case_labels)))}


def plot_case_scatter_performance_diagram(
    series: list, lead_minutes_range: tuple, out_path: str,
    marker_size_basis: str = "truth", title: str | None = None,
) -> dict:
    """series: list[(system_name, color, dict[case_label, MatchLeadHourClimatology])]
    -- same (name, color, ...) tuple convention as every other comparison
    plot in this package (build via aggregate_matches_per_case.build_per_case_match_climatologies()).

    One marker per (system, case), positioned at (SR, POD) = (1-FAR, POD)
    aggregated over lead_minutes_range (a (lo, hi) pair in MINUTES) via
    case_metrics() (ensemble-mean, pooled-then-ratio -- see that function's
    own docstring), colored by the system's own color, sized by
    case_object_count(..., basis=marker_size_basis) using ONE shared
    linear scale across every system/case in the figure, and labeled with
    a compact chronological integer index (see chronological_case_indices())
    drawn at the marker center -- the index is shared across every system
    in `series`, so the same real-world case gets the same number
    regardless of which system's marker it appears on.

    Returns the dict[case_label, index] mapping used for the labels, so
    callers can report which real case each number refers to.
    """
    if not series:
        raise ValueError("plot_case_scatter_performance_diagram: series must be non-empty")

    rows = []
    for system_name, color, cases in series:
        for case_label, clim in cases.items():
            count = case_object_count(clim, lead_minutes_range, basis=marker_size_basis)
            rows.append((system_name, color, case_label, clim, count))
    if not rows:
        raise ValueError("plot_case_scatter_performance_diagram: no cases found across any system")

    case_indices = chronological_case_indices(r[2] for r in rows)
    counts = [r[4] for r in rows]
    min_count, max_count = min(counts), max(counts)

    fig, ax = init_performance_diagram()
    for system_name, color, case_label, clim, count in rows:
        metrics = case_metrics(clim, lead_minutes_range)
        sr, pod = 1.0 - metrics["far"], metrics["pod"]
        size = _size_scale(count, min_count, max_count)
        ax.scatter([sr], [pod], s=[size], c=[color], alpha=_MARKER_ALPHA, edgecolors="k", linewidths=0.5, zorder=3)
        ax.text(sr, pod, str(case_indices[case_label]), ha="center", va="center", fontsize=_LABEL_FONTSIZE, zorder=4)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=color,
               markeredgecolor="k", markersize=8, label=name)
        for name, color, _ in series
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
    if title is not None:
        ax.set_title(title, fontsize=13)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return case_indices


_DEFAULT_DIFF_LEVELS = np.arange(-0.1, 0.11, 0.01)  # matches the legacy's own RdBu_r BoundaryNorm levels


@dataclass
class TwoSystemMatchedCasesReport:
    """Case-level alignment for a two-system diff diagnostic -- a
    genuinely different, narrower question than
    multi_system.check_shared_forecast_cases() (which reports alignment
    across an ENTIRE N-system comparison's forecast cases): here it's
    specifically which of exactly two systems' own per-case dicts overlap,
    since a CSI-difference marker/histogram bin only makes sense for a
    case both systems actually have."""
    matched_cases: list = field(default_factory=list)
    system_a_only: list = field(default_factory=list)
    system_b_only: list = field(default_factory=list)


def matched_cases_report(cases_a: dict, cases_b: dict) -> TwoSystemMatchedCasesReport:
    keys_a, keys_b = set(cases_a.keys()), set(cases_b.keys())
    return TwoSystemMatchedCasesReport(
        matched_cases=sorted(keys_a & keys_b),
        system_a_only=sorted(keys_a - keys_b),
        system_b_only=sorted(keys_b - keys_a),
    )


def plot_two_system_case_diff_performance_diagram(
    system_a: tuple, system_b: tuple, lead_minutes_range: tuple, out_path: str,
    colormap: str = "RdBu_r", diff_levels=None, marker_size_basis: str = "truth", title: str | None = None,
) -> TwoSystemMatchedCasesReport:
    """system_a/system_b: (system_name, color, dict[case_label, MatchLeadHourClimatology])
    -- only meaningful for exactly two systems being compared (validated
    at the config layer, not here, since this function only needs the two
    tuples themselves).

    Restricts to MATCHED cases (the intersection of both systems' own
    case_label sets) -- reported via the returned TwoSystemMatchedCasesReport,
    never silently assumed. For each matched case, one shared
    `csi_diff = CSI(system_a) - CSI(system_b)` (ensemble-mean CSI over
    lead_minutes_range) colors BOTH system_a's point and system_b's point
    for that case (matches the legacy's own "one CSI-difference value
    colors both models' markers for that case-pair" convention). Each
    case is labeled with the same chronological integer index used by
    plot_case_scatter_performance_diagram (shared between both systems
    since only matched cases are plotted), rendered as plain thin black
    numbers for BOTH systems -- the two systems are instead distinguished
    by their marker's own edge linestyle: system_a solid, system_b dashed
    (a thin circle outline around each marker), since colored/outlined
    text would be masked against similarly-colored diff-colormap marker
    fills. Marker fill color is shared (the diff value) and marker size
    still reflects each system's own case_object_count.
    """
    name_a, _color_a, cases_a = system_a
    name_b, _color_b, cases_b = system_b
    report = matched_cases_report(cases_a, cases_b)
    if not report.matched_cases:
        raise ValueError(
            f"plot_two_system_case_diff_performance_diagram: no matched cases between "
            f"'{name_a}' ({len(cases_a)} case(s)) and '{name_b}' ({len(cases_b)} case(s))"
        )

    if diff_levels is None:
        diff_levels = _DEFAULT_DIFF_LEVELS
    cmap = plt.get_cmap(colormap)
    norm = BoundaryNorm(diff_levels, cmap.N)
    case_indices = chronological_case_indices(report.matched_cases)

    rows = []  # (case_label, sr, pod, count, diff, edge_linestyle)
    counts = []
    for case_label in report.matched_cases:
        clim_a, clim_b = cases_a[case_label], cases_b[case_label]
        metrics_a, metrics_b = case_metrics(clim_a, lead_minutes_range), case_metrics(clim_b, lead_minutes_range)
        diff = metrics_a["csi"] - metrics_b["csi"]
        count_a = case_object_count(clim_a, lead_minutes_range, basis=marker_size_basis)
        count_b = case_object_count(clim_b, lead_minutes_range, basis=marker_size_basis)
        rows.append((case_label, 1.0 - metrics_a["far"], metrics_a["pod"], count_a, diff, "solid"))
        rows.append((case_label, 1.0 - metrics_b["far"], metrics_b["pod"], count_b, diff, "dashed"))
        counts.extend((count_a, count_b))

    min_count, max_count = min(counts), max(counts)

    fig, ax = init_performance_diagram(reserve_colorbar=True)
    scat = None
    for case_label, sr, pod, count, diff, edge_linestyle in rows:
        size = _size_scale(count, min_count, max_count)
        scat = ax.scatter([sr], [pod], s=[size], c=[diff], cmap=cmap, norm=norm,
                           alpha=_MARKER_ALPHA, edgecolors="black", linewidths=1.0,
                           linestyle=edge_linestyle, zorder=3)
        ax.text(sr, pod, str(case_indices[case_label]), ha="center", va="center",
                fontsize=_LABEL_FONTSIZE, color="black", zorder=4)

    cax = _add_colorbar_axes(fig)
    cbar = fig.colorbar(scat, cax=cax, extend="both")
    cbar.set_label(f"CSI difference ({name_a} - {name_b})")

    # Legend distinguishes the two systems by marker edge linestyle
    # (solid vs. dashed) rather than a marker color, since marker fill is
    # shared/meaningful as the diff value.
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="black",
               markersize=9, label=f"{name_a} (solid outline)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="black",
               markersize=9, label=f"{name_b} (dashed outline)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)
    if title is not None:
        ax.set_title(title, fontsize=13)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return report


_TRAJECTORY_ALPHA_RANGE = (0.3, 1.0)  # earliest lead -> latest lead
_TRAJECTORY_LINEWIDTH = 2.5
_TRAJECTORY_LABEL_MARKER_SIZE = 40
_TRAJECTORY_LABEL_FONTSIZE = 11
_TRAJECTORY_LABEL_OFFSET_POINTS = (8, 4)


def _format_lead_label(lead_minutes: float) -> str:
    return f"{lead_minutes:.0f}m" if lead_minutes < 60 else f"{lead_minutes / 60:.0f}h"


def _draw_one_trajectory(
    ax, stats, color, label_interval_minutes: float | None, label_lead_hours: list | None = None,
) -> None:
    """One (SR, POD) trajectory for a single VerificationStats, its own
    color, connecting points in lead-hour order -- the per-series drawing
    logic behind plot_lead_time_trajectory_performance_diagram (called
    once per system). Per-segment alpha fades from _TRAJECTORY_ALPHA_RANGE[0]
    (earliest lead) to [1] (latest lead).

    label_lead_hours: when comparing multiple systems with different
    lead-time coverage, pass the INTERSECTION of every system's own
    lead_hours (computed once by the caller) so the exact same set of
    real lead times gets a marker on every system's trajectory -- using
    each system's own (possibly different) lead_hours independently would
    silently misalign the markers between systems. None (the default)
    falls back to this system's own lead_hours (single-system case, or
    when label_interval_minutes is None)."""
    lead_hours = stats.lead_hours
    if not lead_hours:
        return
    ensemble_pod = _ensemble_mean(_member_series(stats.pod, stats.members, lead_hours), stats.members)
    ensemble_far = _ensemble_mean(_member_series(stats.far, stats.members, lead_hours), stats.members)
    ensemble_sr = [1.0 - f for f in ensemble_far]

    points = np.array([ensemble_sr, ensemble_pod]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    if len(segments) > 0:
        alphas = np.linspace(*_TRAJECTORY_ALPHA_RANGE, len(segments))
        rgba_colors = [(*color, a) for a in alphas]
        ax.add_collection(LineCollection(segments, colors=rgba_colors, linewidths=_TRAJECTORY_LINEWIDTH, zorder=3))

    if label_interval_minutes is not None:
        candidate_lead_hours = label_lead_hours if label_lead_hours is not None else lead_hours
        qualifying = sorted(lh for lh in candidate_lead_hours if round(lh * 60.0) % label_interval_minutes == 0)
        lh_to_index = {lh: i for i, lh in enumerate(lead_hours)}
        for lh in qualifying[::2]:  # every other qualifying point -- half as many markers
            idx = lh_to_index.get(lh)
            if idx is None:
                continue  # shared lead hour not present in this system's own series
            sr, pod = ensemble_sr[idx], ensemble_pod[idx]
            lead_min = round(lh * 60.0)
            ax.scatter([sr], [pod], s=_TRAJECTORY_LABEL_MARKER_SIZE, c=[color], edgecolors="k", linewidths=0.5, zorder=4)
            ax.annotate(_format_lead_label(lead_min), (sr, pod), textcoords="offset points",
                        xytext=_TRAJECTORY_LABEL_OFFSET_POINTS, fontsize=_TRAJECTORY_LABEL_FONTSIZE, color=color, zorder=5)


def plot_lead_time_trajectory_performance_diagram(
    series: list, out_path: str, label_interval_minutes: float | None = None, title: str | None = None,
) -> None:
    """series: list[(system_name, color, VerificationStats)] -- reuses the
    EXISTING, already-computed VerificationStats.pod/.far ensemble-mean
    series (climatology.compute_verification_stats), zero new aggregation
    ("essentially the same data as the time series").

    One (SR, POD) = (1-FAR, POD) trajectory per system, its own color,
    connecting points in lead-hour order. Lead-time indication (both
    together, not either/or): per-segment alpha fades from
    _TRAJECTORY_ALPHA_RANGE[0] (earliest lead) to [1] (latest lead) via a
    LineCollection; PLUS markers + text labels at every
    label_interval_minutes (e.g. 30 for WoFS, 720/12h for MPAS) -- pass
    None to skip the interval labels (used by the single-case-highlight
    baseline rendering, to avoid clutter across many overlaid cases).

    When comparing multiple systems whose lead-time coverage differs
    (e.g. WoFSCast has fewer available lead times than WoFS), the
    interval-labeled markers are placed at lead times common to EVERY
    system in `series` (the intersection of their own lead_hours) --
    otherwise each system's markers would land at different real lead
    times and visually fail to line up across trajectories.
    """
    if not series:
        raise ValueError("plot_lead_time_trajectory_performance_diagram: series must be non-empty")

    if label_interval_minutes is not None and len(series) > 1:
        shared_lead_hours = sorted(set.intersection(*(set(stats.lead_hours) for _, _, stats in series)))
    else:
        shared_lead_hours = None

    fig, ax = init_performance_diagram()
    for system_name, color, stats in series:
        _draw_one_trajectory(ax, stats, color, label_interval_minutes, label_lead_hours=shared_lead_hours)

    legend_handles = [Line2D([0], [0], color=color, linewidth=_TRAJECTORY_LINEWIDTH, label=name) for name, color, _ in series]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
    if title is not None:
        ax.set_title(title, fontsize=13)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
