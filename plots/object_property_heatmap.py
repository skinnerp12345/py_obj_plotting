"""Object-property POD heatmaps: Probability of Detection as a function of
BOTH forecast lead time and a truth (MRMS) object's own property (area,
max intensity, or tracked age), styled to match the legacy
`python_base/wofs_verif_cbook.py`'s `heat_map_fig()`/`heat_map_diff_fig()`
(Reds/RdBu_r colormaps, `BoundaryNorm`-stepped `pcolor`, numeric cell
labels, zero-count cells left blank/NaN).

One reusable function per style (single-system, two-system difference),
not one per property -- bins/labels/units are supplied by the caller via
the climatology object itself (see `climatology.aggregate_object_property_pod`),
avoiding per-property duplication.

Y-AXIS uses uniform BIN-INDEX positions (0, 1, 2, ...), not physical bin-
center values, with tick labels showing each bin's real range as text
(e.g. "800-1000", ">2000" for an open-ended final "aggregate everything
above X" bin) -- this is deliberate, not cosmetic: a caller-supplied
overflow bin (fixed-width bins up to a cutoff, then one bin catching
everything larger) can have an effectively infinite upper edge, and
plotting by physical bin-center value would put that one cell's
y-coordinate absurdly far from every other row, badly distorting the whole
plot. Bin-index positioning is correct regardless of whether bins are
equal-width or not.

Each cell's POD text label carries its own object count on a second line
(`0.72\\n(N)`) so a reader can judge how statistically robust a given POD
value/difference is -- for the single-system heatmap N = n_hits+n_misses
for that cell; for the difference heatmap N = the MEAN count between the
two systems.

Label spacing (which lead-time columns get a text label at all) is
computed DYNAMICALLY from how many lead-time columns this particular
heatmap has, calibrated against a real reference case (WoFS's own
heatmaps, ~73 lead-time columns, where a fixed every-5th-column interval
keeps even 5-6 digit object counts readable) so a heatmap with many more columns (e.g. MPAS, ~133) gets
a proportionally wider interval (avoiding label overlap) and one with
fewer columns (e.g. WoFSCast, ~35) gets a proportionally narrower one
(avoiding labels that look too sparse) -- see _dynamic_label_interval().

`plot_object_property_pod_diff_heatmap()` uses a DYNAMIC, symmetric-
around-zero color range with a floor (mirroring
`plots.two_system_diff_histograms._symmetric_bin_edges`'s convention, not
the legacy's hardcoded +-0.4) plus bootstrap-significance alpha shading
(full alpha where a cell's |POD difference| >= its own bootstrap CI width,
0.5 alpha otherwise) -- exactly the legacy's own significant/insignificant
visual convention, just with the newer adaptive range.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm

from ..climatology.contingency_stats import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_CI_INTERVAL,
    bootstrap_pod_diff_ci_width,
    compute_pod,
)

_POD_CMAP = "Reds"
_POD_LEVELS = np.arange(0.0, 1.05, 0.05)
_DIFF_CMAP = "RdBu_r"
_DIFF_BIN_WIDTH = 0.05
_DIFF_MIN_HALF_RANGE = 0.2  # floor -- real magnitude can exceed this, never shrinks below it

# Dynamic label-interval calibration against WoFS's real heatmaps (~73
# lead-time columns at 5-min cadence): the legacy used every 4th column,
# widened here to every 5th (see below) -- scale that same
# (interval / n_columns) ratio to whatever column count a given heatmap
# actually has, rather than reusing it verbatim regardless of column count.
_REFERENCE_N_LEAD_TIMES = 73
_REFERENCE_LABEL_INTERVAL = 5  # was the legacy's 4: with counts >10,000 the "(12345)" line made
                               # neighboring labels nearly touch (min gap 0.009in on real WoFS
                               # heatmaps); 5 gives a >=0.12in gap at those counts
_LABEL_FONTSIZE = 8  # smaller than the legacy's default -- labels are now 2 lines (POD + count)

# Significance-based label styling on the diff heatmap (plot_object_property_
# pod_diff_heatmap only -- the single-system heatmap has no significance
# concept and always uses _SIGNIFICANT_LABEL_ALPHA/normal weight):
# insignificant labels are pushed more transparent, and significant labels
# get more contrast against the colored cells behind them -- bold weight
# reads as "darker" even though the text color itself is already opaque
# black at alpha=1.0.
_SIGNIFICANT_LABEL_ALPHA = 1.0
_INSIGNIFICANT_LABEL_ALPHA = 0.35
_SIGNIFICANT_LABEL_FONTWEIGHT = "bold"


def _style_spines(ax) -> None:
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_alpha(0.7)
        ax.spines[side].set_linewidth(0.5)


def _dynamic_label_interval(n_lead_times: int) -> int:
    """Number of lead-time columns to step between labeled cells, scaled
    from the WoFS-calibrated reference ratio (see module docstring).
    Always at least 1 (never zero, which would be an infinite loop)."""
    if n_lead_times <= 0:
        return 1
    interval = round(_REFERENCE_LABEL_INTERVAL * n_lead_times / _REFERENCE_N_LEAD_TIMES)
    return max(interval, 1)


def _bin_labels(edges: np.ndarray) -> list:
    """'0-200', '1800-2000', '>2000' (open-ended final bin) -- real bin
    ranges as display text, paired with the uniform bin-INDEX y-axis
    positions (see module docstring)."""
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if np.isinf(hi):
            labels.append(f">{lo:.0f}")
        elif np.isinf(lo):
            labels.append(f"<{hi:.0f}")
        else:
            labels.append(f"{lo:.0f}-{hi:.0f}")
    return labels


def _pod_grid(clim, property_name: str) -> tuple:
    """Returns (lead_hours, pod_grid[n_bins, n_lead_hours],
    count_grid[n_bins, n_lead_hours]) -- pod_grid is NaN where
    n_hits+n_misses==0 for that cell (a zero-count cell has an undefined
    POD, matching the legacy's own `var = np.where(var==0, nan, var)`
    treatment, not a misleading real 0.0); count_grid is always a real
    int (0 where empty)."""
    edges = clim.bin_edges[property_name]
    n_bins = len(edges) - 1
    lead_hours = clim.lead_hours
    n_hits, n_misses = clim.n_hits[property_name], clim.n_misses[property_name]

    grid = np.full((n_bins, len(lead_hours)), np.nan)
    count_grid = np.zeros((n_bins, len(lead_hours)), dtype=int)
    for li, lh in enumerate(lead_hours):
        for b in range(n_bins):
            h, m = n_hits.get((b, lh), 0), n_misses.get((b, lh), 0)
            count_grid[b, li] = h + m
            if h + m > 0:
                grid[b, li] = compute_pod(h, m)
    return np.asarray(lead_hours, dtype=float), grid, count_grid


def _symmetric_diff_levels(diff_grid: np.ndarray, floor: float = _DIFF_MIN_HALF_RANGE, bin_width: float = _DIFF_BIN_WIDTH) -> np.ndarray:
    """Bin/color-level edges spanning a range SYMMETRIC around zero, sized
    to the largest-magnitude finite value in `diff_grid`, clamped to at
    least `floor`, then rounded up to the nearest whole multiple of
    bin_width -- mirrors `two_system_diff_histograms._symmetric_bin_edges`,
    adapted to a 2-D grid of diff values instead of a flat list."""
    finite = diff_grid[~np.isnan(diff_grid)]
    max_abs = max(abs(finite.min()), abs(finite.max())) if finite.size else 0.0
    max_abs = max(max_abs, floor)
    half_range = max(np.ceil(max_abs / bin_width) * bin_width, bin_width)
    n_bins = int(round(2 * half_range / bin_width))
    return np.linspace(-half_range, half_range, n_bins + 1)


def _label_cells(ax, lead_x, y_positions, grid, count_grid, alpha_grid=None) -> None:
    n_lead = grid.shape[1]
    interval = _dynamic_label_interval(n_lead)
    offset = max(interval // 2, 1)
    for yi in range(grid.shape[0]):
        for xi in range(offset, max(n_lead - offset, offset + 1), interval):
            if xi >= n_lead or np.isnan(grid[yi, xi]):
                continue
            is_significant = alpha_grid is None or alpha_grid[yi, xi]
            alpha = _SIGNIFICANT_LABEL_ALPHA if is_significant else _INSIGNIFICANT_LABEL_ALPHA
            fontweight = _SIGNIFICANT_LABEL_FONTWEIGHT if (alpha_grid is not None and is_significant) else "normal"
            count = int(count_grid[yi, xi])
            ax.text(lead_x[xi], y_positions[yi], f"{grid[yi, xi]:.2f}\n({count})",
                    ha="center", va="center", alpha=alpha, fontsize=_LABEL_FONTSIZE, linespacing=1.3,
                    fontweight=fontweight)


def plot_object_property_pod_heatmap(
    clim, property_name: str, y_label: str, out_path: str,
    lead_hours_to_minutes: bool = True, title: str | None = None,
) -> None:
    """clim: ObjectPropertyPodClimatology (aggregate_object_property_pod).
    property_name: one of clim.bin_edges' keys ("area"/"max_intensity"/
    "age"). Writes one heatmap PNG: POD (Reds, 0-1 by 0.05) vs. (lead
    time, property bin), each labeled cell showing "POD\\n(count)",
    zero-count cells left blank."""
    if property_name not in clim.bin_edges:
        raise ValueError(f"plot_object_property_pod_heatmap: '{property_name}' has no bin_edges in this climatology")

    edges = clim.bin_edges[property_name]
    lead_hours, grid, count_grid = _pod_grid(clim, property_name)
    if np.all(np.isnan(grid)):
        raise ValueError(f"plot_object_property_pod_heatmap: no hit/miss data at all for property '{property_name}' -- nothing to plot")
    lead_x = lead_hours * 60.0 if lead_hours_to_minutes else lead_hours
    y_positions = np.arange(grid.shape[0])

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(left=0.12, bottom=0.11, right=0.98, top=0.90)
    _style_spines(ax)

    cmap = plt.get_cmap(_POD_CMAP)
    norm = BoundaryNorm(_POD_LEVELS, cmap.N)
    mesh = ax.pcolor(lead_x, y_positions, grid, cmap=cmap, norm=norm)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Probability of Detection", fontsize=16, alpha=0.7)

    _label_cells(ax, lead_x, y_positions, grid, count_grid)

    ax.set_xlabel("Forecast Lead Time (min)" if lead_hours_to_minutes else "Forecast Lead Time (h)", fontsize=18, alpha=0.7)
    ax.set_ylabel(y_label, fontsize=18, alpha=0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(_bin_labels(edges), fontsize=14, alpha=0.7)
    ax.tick_params(axis="x", labelsize=16, colors=(0, 0, 0, 0.7))
    if title is not None:
        ax.set_title(title, fontsize=13)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_object_property_pod_diff_heatmap(
    clim_a, clim_b, property_name: str, y_label: str, out_path: str,
    ci_samples: int = DEFAULT_BOOTSTRAP_SAMPLES, ci_interval: float = DEFAULT_CI_INTERVAL,
    lead_hours_to_minutes: bool = True, difference_label: str = "A - B", title: str | None = None,
) -> None:
    """clim_a/clim_b: ObjectPropertyPodClimatology for the two systems
    being compared -- must share identical bin_edges for `property_name`
    (both built from the same bin_edges dict, the expected normal usage).
    Restricted to lead hours present in BOTH climatologies. Writes one
    diff heatmap PNG: POD_A - POD_B (RdBu_r, dynamic symmetric range with
    a floor), each labeled cell showing "diff\\n(mean count)" (the MEAN of
    the two systems' own object counts for that cell), full alpha where a
    cell's |diff| exceeds its own bootstrap CI width (statistically
    significant), 0.5 alpha otherwise."""
    if property_name not in clim_a.bin_edges or property_name not in clim_b.bin_edges:
        raise ValueError(f"plot_object_property_pod_diff_heatmap: '{property_name}' has no bin_edges in one or both climatologies")
    if not np.array_equal(clim_a.bin_edges[property_name], clim_b.bin_edges[property_name]):
        raise ValueError(
            f"plot_object_property_pod_diff_heatmap: bin_edges for '{property_name}' differ between "
            "the two climatologies -- must be built with identical bin_edges for a diff heatmap"
        )

    edges = clim_a.bin_edges[property_name]
    n_bins = len(edges) - 1
    lead_hours = sorted(set(clim_a.lead_hours) & set(clim_b.lead_hours))
    if not lead_hours:
        raise ValueError("plot_object_property_pod_diff_heatmap: no shared lead hours between the two climatologies")

    n_hits_a, n_misses_a = clim_a.n_hits[property_name], clim_a.n_misses[property_name]
    n_hits_b, n_misses_b = clim_b.n_hits[property_name], clim_b.n_misses[property_name]

    diff_grid = np.full((n_bins, len(lead_hours)), np.nan)
    sig_grid = np.zeros((n_bins, len(lead_hours)), dtype=bool)
    mean_count_grid = np.zeros((n_bins, len(lead_hours)), dtype=int)
    for li, lh in enumerate(lead_hours):
        for b in range(n_bins):
            ha, ma = n_hits_a.get((b, lh), 0), n_misses_a.get((b, lh), 0)
            hb, mb = n_hits_b.get((b, lh), 0), n_misses_b.get((b, lh), 0)
            count_a, count_b = ha + ma, hb + mb
            if count_a == 0 or count_b == 0:
                continue
            diff = compute_pod(ha, ma) - compute_pod(hb, mb)
            diff_grid[b, li] = diff
            mean_count_grid[b, li] = round((count_a + count_b) / 2.0)
            ci_width = bootstrap_pod_diff_ci_width(ha, ma, hb, mb, samples=ci_samples, interval=ci_interval)
            sig_grid[b, li] = abs(diff) >= ci_width

    if np.all(np.isnan(diff_grid)):
        raise ValueError(f"plot_object_property_pod_diff_heatmap: no overlapping hit/miss data for property '{property_name}' -- nothing to plot")

    lead_hours_arr = np.asarray(lead_hours, dtype=float)
    lead_x = lead_hours_arr * 60.0 if lead_hours_to_minutes else lead_hours_arr
    y_positions = np.arange(n_bins)

    levels = _symmetric_diff_levels(diff_grid)
    cmap = plt.get_cmap(_DIFF_CMAP)
    norm = BoundaryNorm(levels, cmap.N)

    sig_masked = np.where(sig_grid, diff_grid, np.nan)
    insig_masked = np.where((~sig_grid) & (~np.isnan(diff_grid)), diff_grid, np.nan)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(left=0.12, bottom=0.11, right=0.98, top=0.90)
    _style_spines(ax)

    mesh = ax.pcolor(lead_x, y_positions, sig_masked, cmap=cmap, norm=norm)
    cbar = fig.colorbar(mesh, ax=ax, extend="both")
    ax.pcolor(lead_x, y_positions, insig_masked, cmap=cmap, norm=norm, alpha=0.5)

    _label_cells(ax, lead_x, y_positions, diff_grid, mean_count_grid, alpha_grid=sig_grid)

    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(f"Probability of Detection ({difference_label})", fontsize=16, alpha=0.7)
    ax.set_xlabel("Forecast Lead Time (min)" if lead_hours_to_minutes else "Forecast Lead Time (h)", fontsize=18, alpha=0.7)
    ax.set_ylabel(y_label, fontsize=18, alpha=0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(_bin_labels(edges), fontsize=14, alpha=0.7)
    ax.tick_params(axis="x", labelsize=16, colors=(0, 0, 0, 0.7))
    if title is not None:
        ax.set_title(title, fontsize=13)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
