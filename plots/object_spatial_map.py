"""Spatial object-footprint maps for a single forecast slice (one member,
one valid time) -- composite-reflectivity-style basemap (`spatial_map.py`),
but shading indicates identified storm OBJECTS rather than the raw
continuous dBZ field. `color_by` selects one mutually-exclusive primary fill
style for the forecast layer:

- "single": one flat color per footprint pixel (plain presence/absence).
- "storm_mode": categorical, keyed by `StormObject.is_linear`
  (0=cellular/1=mixed/2=linear) -- reuses the exact vectorized lookup-table
  pattern already established in `plots/mrms_object_map.py::_category_grid`.
- "age": continuous colormap + colorbar, keyed by `StormObject.age_seconds`.
  Uses the RAW age (converted to minutes), not the
  `aggregate_object_property_pod.py` heatmap feature's lead-relative
  `age_seconds - lead_minutes*60` transform -- a single spatial snapshot
  wants the storm's actual physical maturity, not a cross-lead-time
  normalized quantity. An object with no tracked age (`age_seconds is
  None`, e.g. the truth series wasn't tracked at identification time) maps
  to NaN, so an entirely-untracked file degrades to an all-NaN (empty)
  layer rather than raising -- matches this project's established
  detect-and-report convention for other optional/tracking-gated fields.
- "category": hit/miss/false_alarm/extra, colored per
  `python_test_plot/plot_match_categories.py`'s own established convention
  (blue/red/orange/gray). Requires both `truth` and `match_records`
  (a match file's own records for this slice) -- collapses that script's
  6 separate draw calls into 2 vectorized fill-grids (one per side), same
  visual result. Also triggers an instantaneous POD/FAR/FBIAS/CSI
  annotation box (`compute_instantaneous_stats`/`annotate_stats_box`),
  computed directly from `match_records` via
  `climatology.contingency_stats`'s existing functions -- no new stats math.

MRMS/truth overlay (`truth`) and independent storm-mode hatch toggles
(`show_forecast_hatch`/`show_truth_hatch`) are available regardless of
`color_by`. `plot_paintball_map()` is a separate top-level function for
ensemble "paintball" plots (every selected member's own footprint at one
shared valid time, overlaid on one shared domain) -- its own `color_by`
supports "single" (uniform low-alpha, the real WoFS/SPC convention),
"storm_mode", and "category" (each member's own hit/miss/false_alarm/extra
assignment, with ensemble-mean POD/FAR/FBIAS/CSI annotated).
"""

from dataclasses import dataclass
from datetime import datetime

import cartopy.crs as ccrs
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Patch

from python_obj.obj_core import MatchRecord, StormObject

from .. import cb_colors
from ..climatology.contingency_stats import compute_csi, compute_far, compute_fbias, compute_pod
from . import spatial_map

_STORM_MODE_COLORS = {0: cb_colors.orange5, 1: cb_colors.purple5, 2: cb_colors.blue5}
_STORM_MODE_LABELS = {0: "Cellular", 1: "Mixed", 2: "Linear"}
_VALID_COLOR_BY = ("single", "storm_mode", "age", "category")

# category grid encoding: 0=background, 1=hit, 2=miss(truth-side)/false_alarm
# (forecast-side), 3=extra -- each side rendered with its OWN colormap (same
# 1/2/3 encoding, different color at position 2), since a single side's grid
# never has both miss and false_alarm values simultaneously.
_CATEGORY_TRUTH_COLORS = {1: cb_colors.blue5, 2: cb_colors.red5, 3: cb_colors.gray5}
_CATEGORY_FORECAST_COLORS = {1: cb_colors.blue5, 2: cb_colors.orange5, 3: cb_colors.gray5}
_CATEGORY_LEGEND = [
    ("Hit", cb_colors.blue5), ("Miss", cb_colors.red5),
    ("False alarm", cb_colors.orange5), ("Extra", cb_colors.gray5),
]
_STATS_BOX_POSITIONS = {
    "lower right": (0.98, 0.02, "right", "bottom"),
    "lower left": (0.02, 0.02, "left", "bottom"),
}


@dataclass
class ObjectSlice:
    """One (member, valid_time) slice's worth of object data -- the common
    unit every rendering function in this module consumes. Repackages
    `python_obj.obj_core.iter_object_slices`/`iter_object_slices_lazy`'s
    tuple output for uniform passing between selection/loading code and the
    plotting functions here."""
    lat2d: np.ndarray
    lon2d: np.ndarray
    labels2d: np.ndarray
    objects: list[StormObject]
    member_id: str | None
    valid_time: datetime


def _id_lookup_table(labels2d: np.ndarray, objects: list, value_fn, default=0.0, dtype=float) -> np.ndarray:
    """Vectorized label-id -> per-object-value grid (background/unmatched
    ids map to `default`). Generalizes the lookup-table pattern already
    established in `plots/mrms_object_map.py::_category_grid` to an
    arbitrary per-object value, not just the fixed is_linear+1 case."""
    max_id = int(labels2d.max()) if labels2d.size else 0
    lut = np.full(max_id + 1, default, dtype=dtype)
    for obj in objects:
        if 0 <= obj.id <= max_id:
            v = value_fn(obj)
            if v is not None:
                lut[obj.id] = v
    return lut[labels2d]


def _single_fill_grid(labels2d: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(labels2d == 0, (labels2d > 0).astype(float))


def _storm_mode_fill_grid(labels2d: np.ndarray, objects: list) -> np.ndarray:
    """0=background, 1/2/3 = cellular/mixed/linear (is_linear + 1, keeping 0
    free to mean 'no object')."""
    return _id_lookup_table(labels2d, objects, lambda o: o.is_linear + 1, default=0, dtype=np.int16)


def _age_fill_grid(labels2d: np.ndarray, objects: list) -> np.ndarray:
    """Raw age_seconds (see module docstring), converted to minutes. NaN
    where an object exists but has no tracked age, or where there's no
    object at all."""
    grid = _id_lookup_table(labels2d, objects, lambda o: o.age_seconds, default=np.nan, dtype=float)
    return grid / 60.0


def _category_fill_grid(labels2d: np.ndarray, ids_by_value: dict) -> np.ndarray:
    """`ids_by_value`: {1: hit_ids, 2: miss_or_false_alarm_ids, 3: extra_ids}
    (whichever ids apply to this side -- see module docstring). Vectorized
    lookup table, 0 = background/unmatched."""
    max_id = int(labels2d.max()) if labels2d.size else 0
    lut = np.zeros(max_id + 1, dtype=np.int16)
    for value, ids in ids_by_value.items():
        for i in ids:
            if 0 <= i <= max_id:
                lut[i] = value
    return lut[labels2d]


def _match_category_id_sets(match_records: list) -> tuple:
    """Truth-side and forecast-side {1: hit_ids, 2: miss_or_fa_ids, 3: extra_ids}
    dicts, built from a match file's own records -- exact id-set
    construction ported from `python_test_plot/plot_match_categories.py`."""
    hit_truth_ids = {r.truth_id for r in match_records if r.category == "hit"}
    hit_forecast_ids = {r.forecast_id for r in match_records if r.category == "hit"}
    miss_truth_ids = {r.truth_id for r in match_records if r.category == "miss"}
    fa_forecast_ids = {r.forecast_id for r in match_records if r.category == "false_alarm"}
    extra_truth_ids = {r.truth_id for r in match_records if r.category == "truth_extra"}
    extra_forecast_ids = {r.forecast_id for r in match_records if r.category == "forecast_extra"}
    truth_ids_by_value = {1: hit_truth_ids, 2: miss_truth_ids, 3: extra_truth_ids}
    forecast_ids_by_value = {1: hit_forecast_ids, 2: fa_forecast_ids, 3: extra_forecast_ids}
    return truth_ids_by_value, forecast_ids_by_value


def _category_cmap_norm(colors_by_value: dict) -> tuple:
    cmap = ListedColormap([colors_by_value[1], colors_by_value[2], colors_by_value[3]])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
    return cmap, norm


def _colors_from_id_sets(objects: list, ids_by_value: dict, colors_by_value: dict, default_color) -> list:
    colors = []
    for o in objects:
        color = default_color
        for value, ids in ids_by_value.items():
            if o.id in ids:
                color = colors_by_value[value]
                break
        colors.append(color)
    return colors


def compute_instantaneous_stats(match_records: list) -> dict:
    """POD/FAR/FBIAS/CSI + the 5 underlying category counts for ONE slice's
    match records, via `climatology.contingency_stats`'s existing
    pre-aggregated-count functions -- no new stats math."""
    n_hits = sum(1 for r in match_records if r.category == "hit")
    n_misses = sum(1 for r in match_records if r.category == "miss")
    n_false_alarms = sum(1 for r in match_records if r.category == "false_alarm")
    n_truth_extra = sum(1 for r in match_records if r.category == "truth_extra")
    n_forecast_extra = sum(1 for r in match_records if r.category == "forecast_extra")
    return {
        "pod": compute_pod(n_hits, n_misses),
        "far": compute_far(n_hits, n_false_alarms),
        "fbias": compute_fbias(n_hits, n_misses, n_false_alarms, n_truth_extra, n_forecast_extra),
        "csi": compute_csi(n_hits, n_misses, n_false_alarms),
        "n_hits": n_hits, "n_misses": n_misses, "n_false_alarms": n_false_alarms,
        "n_truth_extra": n_truth_extra, "n_forecast_extra": n_forecast_extra,
    }


def annotate_stats_box(ax, stats: dict, loc: str = "lower right"):
    """Instantaneous POD/FAR/FBIAS/CSI text box in one corner of the plot."""
    if loc not in _STATS_BOX_POSITIONS:
        raise ValueError(f"annotate_stats_box: loc={loc!r} is not one of {list(_STATS_BOX_POSITIONS)}")
    x, y, ha, va = _STATS_BOX_POSITIONS[loc]
    text = f"POD: {stats['pod']:.2f}\nFAR: {stats['far']:.2f}\nBIAS: {stats['fbias']:.2f}\nCSI: {stats['csi']:.2f}"
    return ax.text(x, y, text, transform=ax.transAxes, fontsize=9, ha=ha, va=va,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"), zorder=10)


def _storm_mode_cmap_norm() -> tuple:
    cmap = ListedColormap([_STORM_MODE_COLORS[0], _STORM_MODE_COLORS[1], _STORM_MODE_COLORS[2]])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
    return cmap, norm


def _age_cmap_norm(grid: np.ndarray, vmin: float | None = None, vmax: float | None = None, cmap_name: str = "viridis") -> tuple:
    import matplotlib.pyplot as plt

    finite = grid[np.isfinite(grid)]
    if vmin is None:
        vmin = float(finite.min()) if finite.size else 0.0
    if vmax is None:
        vmax = float(finite.max()) if finite.size else 1.0
    return plt.get_cmap(cmap_name), Normalize(vmin=vmin, vmax=vmax)


def render_fill(ax, lon_plot, lat2d, grid, cmap, norm=None, vmin=None, vmax=None, alpha: float = 0.6):
    """Thin `ax.pcolormesh` wrapper -- returns the mesh (needed for
    `mesh.get_array()` test assertions). Pass EITHER `norm` OR `vmin`/`vmax`,
    never both (matplotlib rejects that combination)."""
    kwargs = dict(transform=ccrs.PlateCarree(), cmap=cmap, alpha=alpha, shading="auto")
    if norm is not None:
        kwargs["norm"] = norm
    else:
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax
    return ax.pcolormesh(lon_plot, lat2d, grid, **kwargs)


def add_colorbar(fig, ax, mesh, label: str):
    return fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02, shrink=0.85, label=label)


def add_centroid_markers(ax, objects: list, colors, marker_size: float = 3.0) -> None:
    if not objects:
        return
    lons = [o.centroid_lon - 360.0 if o.centroid_lon > 180.0 else o.centroid_lon for o in objects]
    lats = [o.centroid_lat for o in objects]
    ax.scatter(lons, lats, s=marker_size, c=colors, marker="o", edgecolors="none", transform=ccrs.PlateCarree(), zorder=5)


def add_storm_mode_hatch(ax, lon_plot, lat2d, labels2d: np.ndarray, objects: list) -> list:
    """Hatches (not colors) mixed/linear objects on top of whatever fill is
    already drawn -- cellular gets no hatch, mixed gets '...', linear gets
    '///'. Reuses the exact convention established in
    `python_test_plot/plot_mrms_vs_mem1_objects.py`. Independent of
    `color_by` -- callable regardless of how the layer's own fill is
    colored. Returns the list of `contourf` artists actually drawn (empty
    if neither mode is present), so callers/tests can inspect them."""
    artists = []
    for is_linear_value, hatch in ((1, "..."), (2, "///")):
        mask = _id_lookup_table(labels2d, objects, lambda o, v=is_linear_value: 1.0 if o.is_linear == v else 0.0, default=0.0, dtype=float)
        if mask.any():
            artists.append(ax.contourf(lon_plot, lat2d, mask, levels=[0.5, 1.5], hatches=[hatch], colors="none", transform=ccrs.PlateCarree()))
    return artists


def legend_handles_storm_mode() -> list:
    return [Patch(facecolor=_STORM_MODE_COLORS[k], label=_STORM_MODE_LABELS[k]) for k in (0, 1, 2)]


def legend_handles_single(color, label: str) -> list:
    return [Patch(facecolor=color, label=label)]


def legend_handles_hatch() -> list:
    return [
        Patch(facecolor="none", edgecolor="black", hatch="...", label="Mixed"),
        Patch(facecolor="none", edgecolor="black", hatch="///", label="Linear"),
    ]


def legend_handles_category() -> list:
    return [Patch(facecolor=color, label=label) for label, color in _CATEGORY_LEGEND]


def _default_title(forecast: ObjectSlice, color_by: str, forecast_label: str, truth: "ObjectSlice | None" = None, truth_label: str = "MRMS") -> tuple:
    """Returns (short_title, valid_time_label) -- two independent title
    anchors (`spatial_map.finalize_and_save` renders them left/right-
    justified) rather than one long combined string, per the convention
    that figure titles should be short with the valid time/date
    right-justified at the top."""
    member_str = f" {forecast.member_id}" if forecast.member_id else ""
    truth_str = f" + {truth_label}" if truth is not None else ""
    short_title = f"{forecast_label}{member_str} -- {color_by}{truth_str}"
    valid_time_label = f"{forecast.valid_time:%Y-%m-%d %H:%MZ}"
    return short_title, valid_time_label


def plot_object_spatial_map(
    forecast: ObjectSlice,
    out_path: str,
    extent: list,
    color_by: str = "single",
    forecast_color: tuple = cb_colors.orange5,
    forecast_label: str = "Forecast",
    age_vmin: float | None = None,
    age_vmax: float | None = None,
    alpha: float = 0.6,
    show_centroids: bool = True,
    centroid_marker_size: float = 3.0,
    truth: "ObjectSlice | None" = None,
    truth_color: tuple = cb_colors.gray8,
    truth_label: str = "MRMS",
    truth_alpha: float = 0.75,
    show_forecast_hatch: bool = False,
    show_truth_hatch: bool = False,
    match_records: list | None = None,
    show_stats: bool | None = None,
    stats_loc: str = "lower right",
    title: str | None = None,
    figsize: tuple = (11, 8.5),
) -> str:
    """truth: an independent, always-available second shading layer (dark,
    per "dark shading as verification") -- plotted as flat/single fill for
    `color_by` in {single,storm_mode,age} (orthogonal to the forecast
    layer's own coloring), or categorically for `color_by="category"` (see
    below). `show_forecast_hatch`/`show_truth_hatch` independently overlay
    the established mixed/linear hatch convention on either layer,
    regardless of `color_by`.

    `color_by="category"` requires BOTH `truth` and `match_records` (a
    match file's own records for this exact slice) -- raises a named
    ValueError otherwise. `show_stats` defaults to True automatically for
    category mode (annotates instantaneous POD/FAR/FBIAS/CSI in `stats_loc`
    corner), False for every other mode; pass explicitly to override
    either way."""
    if color_by not in _VALID_COLOR_BY:
        raise ValueError(f"plot_object_spatial_map: color_by={color_by!r} is not one of {_VALID_COLOR_BY}")
    if color_by == "category" and (truth is None or match_records is None):
        raise ValueError("plot_object_spatial_map: color_by='category' requires both truth and match_records")

    lon_plot = spatial_map.lon_to_plot_range(forecast.lon2d)
    fig, ax = spatial_map.build_base_map(figsize=figsize)

    legend_handles: list = []
    colorbar_mesh = None
    colorbar_label = None

    if color_by == "category":
        truth_ids_by_value, forecast_ids_by_value = _match_category_id_sets(match_records)

        forecast_grid = _category_fill_grid(forecast.labels2d, forecast_ids_by_value)
        forecast_cmap, forecast_norm = _category_cmap_norm(_CATEGORY_FORECAST_COLORS)
        render_fill(ax, lon_plot, forecast.lat2d, np.ma.masked_where(forecast_grid == 0, forecast_grid),
                    forecast_cmap, norm=forecast_norm, alpha=alpha)
        centroid_colors = _colors_from_id_sets(forecast.objects, forecast_ids_by_value, _CATEGORY_FORECAST_COLORS, cb_colors.gray5)

        truth_lon_plot = spatial_map.lon_to_plot_range(truth.lon2d)
        truth_grid = _category_fill_grid(truth.labels2d, truth_ids_by_value)
        truth_cmap, truth_norm = _category_cmap_norm(_CATEGORY_TRUTH_COLORS)
        render_fill(ax, truth_lon_plot, truth.lat2d, np.ma.masked_where(truth_grid == 0, truth_grid),
                    truth_cmap, norm=truth_norm, alpha=truth_alpha)

        legend_handles.extend(legend_handles_category())
        if show_forecast_hatch:
            add_storm_mode_hatch(ax, lon_plot, forecast.lat2d, forecast.labels2d, forecast.objects)
        if show_truth_hatch:
            add_storm_mode_hatch(ax, truth_lon_plot, truth.lat2d, truth.labels2d, truth.objects)
        if show_forecast_hatch or show_truth_hatch:
            legend_handles.extend(legend_handles_hatch())

        if show_stats is None or show_stats:
            stats = compute_instantaneous_stats(match_records)
            annotate_stats_box(ax, stats, loc=stats_loc)
    else:
        if color_by == "single":
            grid = _single_fill_grid(forecast.labels2d)
            render_fill(ax, lon_plot, forecast.lat2d, grid, ListedColormap([forecast_color]), vmin=0, vmax=1, alpha=alpha)
            legend_handles.extend(legend_handles_single(forecast_color, forecast_label))
            centroid_colors = [forecast_color] * len(forecast.objects)
        elif color_by == "storm_mode":
            grid = _storm_mode_fill_grid(forecast.labels2d, forecast.objects)
            grid_masked = np.ma.masked_where(grid == 0, grid)
            cmap, norm = _storm_mode_cmap_norm()
            render_fill(ax, lon_plot, forecast.lat2d, grid_masked, cmap, norm=norm, alpha=alpha)
            legend_handles.extend(legend_handles_storm_mode())
            centroid_colors = [_STORM_MODE_COLORS[o.is_linear] for o in forecast.objects]
        else:  # "age"
            grid = _age_fill_grid(forecast.labels2d, forecast.objects)
            cmap, norm = _age_cmap_norm(grid, vmin=age_vmin, vmax=age_vmax)
            mesh = render_fill(ax, lon_plot, forecast.lat2d, grid, cmap, norm=norm, alpha=alpha)
            colorbar_mesh, colorbar_label = mesh, "Object age (min)"
            centroid_colors = [forecast_color] * len(forecast.objects)

        if show_forecast_hatch:
            add_storm_mode_hatch(ax, lon_plot, forecast.lat2d, forecast.labels2d, forecast.objects)

        if truth is not None:
            truth_lon_plot = spatial_map.lon_to_plot_range(truth.lon2d)
            truth_grid = _single_fill_grid(truth.labels2d)
            render_fill(ax, truth_lon_plot, truth.lat2d, truth_grid, ListedColormap([truth_color]), vmin=0, vmax=1, alpha=truth_alpha)
            legend_handles.extend(legend_handles_single(truth_color, truth_label))
            if show_truth_hatch:
                add_storm_mode_hatch(ax, truth_lon_plot, truth.lat2d, truth.labels2d, truth.objects)

        if show_forecast_hatch or show_truth_hatch:
            legend_handles.extend(legend_handles_hatch())

        if show_stats:
            stats = compute_instantaneous_stats(match_records) if match_records is not None else None
            if stats is not None:
                annotate_stats_box(ax, stats, loc=stats_loc)

    if show_centroids:
        add_centroid_markers(ax, forecast.objects, centroid_colors, marker_size=centroid_marker_size)

    if colorbar_mesh is not None:
        add_colorbar(fig, ax, colorbar_mesh, colorbar_label)

    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower left", fontsize=9)

    default_title, valid_time_label = _default_title(forecast, color_by, forecast_label, truth=truth, truth_label=truth_label)
    final_title = title if title is not None else default_title
    spatial_map.finalize_and_save(fig, ax, extent, out_path, title=final_title, valid_time_label=valid_time_label)
    return out_path


_VALID_PAINTBALL_COLOR_BY = ("single", "storm_mode", "category")


def compute_ensemble_mean_stats(match_records_by_member: dict) -> dict:
    """Ensemble-MEAN POD/FAR/FBIAS/CSI for a paintball panel: each member's
    own instantaneous stats (`compute_instantaneous_stats`, since each
    member is independently matched against the SAME MRMS truth) averaged
    across the selected members -- not one contingency table pooled across
    members, which would double-count each real truth object once per
    member instead of treating it as the single shared observation it is."""
    if not match_records_by_member:
        raise ValueError("compute_ensemble_mean_stats: match_records_by_member must be non-empty")
    per_member = [compute_instantaneous_stats(records) for records in match_records_by_member.values()]
    return {
        "pod": float(np.mean([s["pod"] for s in per_member])),
        "far": float(np.mean([s["far"] for s in per_member])),
        "fbias": float(np.mean([s["fbias"] for s in per_member])),
        "csi": float(np.mean([s["csi"] for s in per_member])),
    }


def plot_paintball_map(
    member_slices: list,
    out_path: str,
    extent: list,
    color_by: str = "single",
    color: tuple = cb_colors.gray8,
    alpha: float = 0.65,
    per_member_colors: list | None = None,
    match_records_by_member: dict | None = None,
    show_stats: bool | None = None,
    stats_loc: str = "lower right",
    truth: "ObjectSlice | None" = None,
    truth_color: tuple = cb_colors.gray8,
    truth_label: str = "MRMS",
    truth_alpha: float = 0.8,
    title: str | None = None,
    figsize: tuple = (11, 8.5),
) -> str:
    """Ensemble "paintball" plot: every selected member's own forecast
    object footprint at ONE shared valid time, drawn on one shared domain.
    `color_by` selects how each member's own footprint is filled:

    - "single" (default): every member shares ONE uniform color (`color`)
      -- the real, actually-used WoFS/SPC paintball convention
      (`python_base/wofs_plotting_cbook.py::wof_plot()`'s paintball block:
      one draw call per member, all sharing the same color). Overlapping
      members' fills stack visually rather than needing to be
      individually distinguishable -- `alpha` defaults high enough
      (0.65) that a single member's own footprint is still clearly
      visible, not a near-invisible wash. `per_member_colors`, if given,
      assigns one distinct color per member instead (an opt-in escape
      hatch -- the never-actually-used legacy 18-color qualitative
      palette had no real precedent to port, so this is new,
      unproven-by-precedent styling, off by default).
    - "storm_mode": each member's own footprint colored by its own
      `is_linear` classification (same colors/legend as
      `plot_object_spatial_map`'s own storm_mode mode).
    - "category": each member's own footprint colored by ITS OWN
      hit/miss/false_alarm/extra match-category assignment -- requires
      `match_records_by_member` (`{member_id: [MatchRecord,...]}`, one
      member's own records at this exact valid time). Also computes and
      annotates the ensemble-MEAN POD/FAR/FBIAS/CSI across the selected
      members (`compute_ensemble_mean_stats`), `show_stats` defaulting to
      True automatically here (matching `plot_object_spatial_map`'s own
      category-mode convention).

    `truth`, if given, draws ONE shared MRMS footprint layer (flat/single
    color, `truth_alpha` -- default 0.8, higher than the member layers so
    the real observed objects read clearly regardless of `color_by`) on
    top of every member's own layer -- independent of `color_by`, same as
    `plot_object_spatial_map`'s own `truth` parameter."""
    if color_by not in _VALID_PAINTBALL_COLOR_BY:
        raise ValueError(f"plot_paintball_map: color_by={color_by!r} is not one of {_VALID_PAINTBALL_COLOR_BY}")
    if not member_slices:
        raise ValueError("plot_paintball_map: member_slices must be non-empty")
    if per_member_colors is not None and len(per_member_colors) != len(member_slices):
        raise ValueError(
            f"plot_paintball_map: per_member_colors has {len(per_member_colors)} entries, "
            f"expected one per member ({len(member_slices)})"
        )
    if color_by == "category" and not match_records_by_member:
        raise ValueError("plot_paintball_map: color_by='category' requires match_records_by_member")

    fig, ax = spatial_map.build_base_map(figsize=figsize)
    legend_handles: list = []

    for i, member in enumerate(member_slices):
        lon_plot = spatial_map.lon_to_plot_range(member.lon2d)
        if color_by == "single":
            grid = _single_fill_grid(member.labels2d)
            member_color = per_member_colors[i] if per_member_colors is not None else color
            render_fill(ax, lon_plot, member.lat2d, grid, ListedColormap([member_color]), vmin=0, vmax=1, alpha=alpha)
        elif color_by == "storm_mode":
            grid = _storm_mode_fill_grid(member.labels2d, member.objects)
            grid_masked = np.ma.masked_where(grid == 0, grid)
            cmap, norm = _storm_mode_cmap_norm()
            render_fill(ax, lon_plot, member.lat2d, grid_masked, cmap, norm=norm, alpha=alpha)
        else:  # "category"
            member_records = match_records_by_member.get(member.member_id, [])
            _, forecast_ids_by_value = _match_category_id_sets(member_records)
            grid = _category_fill_grid(member.labels2d, forecast_ids_by_value)
            grid_masked = np.ma.masked_where(grid == 0, grid)
            cmap, norm = _category_cmap_norm(_CATEGORY_FORECAST_COLORS)
            render_fill(ax, lon_plot, member.lat2d, grid_masked, cmap, norm=norm, alpha=alpha)

    if color_by == "storm_mode":
        legend_handles.extend(legend_handles_storm_mode())
    elif color_by == "category":
        legend_handles.extend(legend_handles_category())
        if show_stats is None or show_stats:
            stats = compute_ensemble_mean_stats(match_records_by_member)
            annotate_stats_box(ax, stats, loc=stats_loc)

    if truth is not None:
        truth_lon_plot = spatial_map.lon_to_plot_range(truth.lon2d)
        truth_grid = _single_fill_grid(truth.labels2d)
        render_fill(ax, truth_lon_plot, truth.lat2d, truth_grid, ListedColormap([truth_color]), vmin=0, vmax=1, alpha=truth_alpha)
        legend_handles.extend(legend_handles_single(truth_color, truth_label))

    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower left", fontsize=9)

    valid_time = member_slices[0].valid_time
    short_title = title if title is not None else f"{len(member_slices)}-member paintball -- {color_by}"
    valid_time_label = f"{valid_time:%Y-%m-%d %H:%MZ}"
    spatial_map.finalize_and_save(fig, ax, extent, out_path, title=short_title, valid_time_label=valid_time_label)
    return out_path
