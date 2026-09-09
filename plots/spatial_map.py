"""Shared cartopy base-map building blocks for spatial object plots --
consolidates the CONUS-boundary/state-lines/extent-cropping/lon-normalization
pattern that `python_test_plot/plot_mrms_vs_mem1_objects.py`,
`python_test_plot/plot_match_categories.py`, and
`py_obj_plotting/plots/mrms_object_map.py` each currently reimplement
near-identically. Those existing scripts are left untouched (out of scope);
this module exists for the new `object_spatial_map.py`/`paintball_map.py`
rendering code (and any future spatial-plotting module) to share instead of
each growing its own copy.

**Aspect ratio**: cartopy's `PlateCarree` axes default to `aspect='equal'`
in the projection's own coordinate system, which for PlateCarree is plain
(lon, lat) degrees -- confirmed empirically (`ax.get_aspect() == 1.0`).
Since a degree of longitude is a physically SMALLER ground distance than a
degree of latitude away from the equator (`111*cos(lat)` km vs `111` km),
an equal-aspect degree-for-degree plot stretches real east-west distances
relative to north-south ones (a real circle renders as a wider ellipse).
`finalize_and_save()` corrects this with the standard regional-PlateCarree
fix: `ax.set_aspect(1/cos(central_latitude))`, where `central_latitude` is
the midpoint of the plotted extent -- exact only at that one latitude, but
this library's domains (WoFS/MPAS/CONUS-regional) are narrow enough in
latitude that the correction is effectively uniform across the plot.
"""

import copy
import math
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm

from python_obj.obj_core import load_conus_boundary

from .. import cb_colors

STATES = cfeature.NaturalEarthFeature(
    category="cultural", name="admin_1_states_provinces_lines", scale="10m", facecolor="none"
)
COUNTRIES = cfeature.NaturalEarthFeature(
    category="cultural", name="admin_0_countries", scale="10m", facecolor="none"
)


def lon_to_plot_range(lon2d: np.ndarray) -> np.ndarray:
    """MRMS/model longitudes are 0-360 convention; cartopy's PlateCarree
    plots correctly in -180-180. Shift only the values that need it."""
    return np.where(lon2d > 180.0, lon2d - 360.0, lon2d)


def crop_extent_from_mask(lat2d: np.ndarray, lon2d: np.ndarray, keep_mask: np.ndarray, margin_deg: float = 0.0) -> list:
    """Bounding box (in [-180,180] lon convention) of the KEPT region of
    `keep_mask` (True = keep). `margin_deg` defaults to 0 -- the plotted
    extent should show only the actual domain, never padding beyond it;
    pass a positive value explicitly if a caller genuinely wants a margin."""
    lon_norm = lon_to_plot_range(lon2d)
    kept_lat = lat2d[keep_mask]
    kept_lon = lon_norm[keep_mask]
    return [
        float(kept_lon.min() - margin_deg), float(kept_lon.max() + margin_deg),
        float(kept_lat.min() - margin_deg), float(kept_lat.max() + margin_deg),
    ]


def build_base_map(
    figsize: tuple = (11, 8.5), draw_conus_outline: bool = True, draw_states: bool = True,
    draw_countries: bool = False, draw_coastlines: bool = True,
):
    """One PlateCarree axes with the CONUS boundary outline (from
    `python_obj.obj_core.load_conus_boundary`, plotted as a line -- not a
    cartopy Feature, matching the existing precedent exactly), physical
    coastlines (10m resolution -- includes Mexico's Gulf/Pacific coast, not
    just the CONUS outline), and optional state/country political-border
    layers, all at Natural Earth's highest (10m) resolution to avoid the
    unrealistically smoothed-out borders 50m produces. Returns (fig, ax);
    callers draw their own data layers on `ax` before calling
    `finalize_and_save`."""
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()})

    if draw_conus_outline:
        boundary = load_conus_boundary()
        polys = list(boundary.geoms) if boundary.geom_type == "MultiPolygon" else [boundary]
        for poly in polys:
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, transform=ccrs.PlateCarree(), color="black", linewidth=0.8)

    if draw_coastlines:
        ax.coastlines(resolution="10m", linewidth=0.8, color="black")
    if draw_states:
        ax.add_feature(STATES, edgecolor="gray", linewidth=0.5)
    if draw_countries:
        ax.add_feature(COUNTRIES, edgecolor="black", linewidth=0.7)

    return fig, ax


def finalize_and_save(
    fig, ax, extent: list, out_path: str,
    title: str | None = None, valid_time_label: str | None = None, dpi: int = 115,
) -> None:
    """Sets the final map extent, corrects the aspect ratio for that
    extent's own central latitude (see module docstring), writes the PNG
    (creating the output directory if needed), and closes the figure.

    `title` (short, left-justified) and `valid_time_label` (right-justified,
    e.g. "2023-04-25 05:00Z") are two independent title anchors -- kept
    separate rather than one long combined string, per the convention that
    figure titles should be short with the valid time/date right-justified
    at the top."""
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    central_lat = (extent[2] + extent[3]) / 2.0
    ax.set_aspect(1.0 / math.cos(math.radians(central_lat)))

    if title is not None:
        ax.set_title(title, loc="left", fontsize=11)
    if valid_time_label is not None:
        ax.set_title(valid_time_label, loc="right", fontsize=10)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def add_reflectivity_layer(
    ax, lon_plot: np.ndarray, lat2d: np.ndarray, data: np.ndarray,
    missing_value: float | None = None, min_dbz: float = 20.0, alpha: float = 1.0,
):
    """Continuous composite-reflectivity field render primitive -- masks
    `data` below `min_dbz` and/or at `missing_value` (if given), renders via
    `pcolormesh` with the standard NWS reflectivity colormap/levels
    (`cb_colors.nws_dz_cmap`/`REFLECTIVITY_LEVELS`, via `BoundaryNorm`).
    Returns the `QuadMesh` (for colorbar attachment / test assertions).

    A FRESH `copy.copy()` of `nws_dz_cmap` is made before calling
    `set_over`/`set_under` -- `nws_dz_cmap` is a shared module-level
    singleton in `cb_colors`; calling those directly on it would mutate
    global state visible to every other caller. Exact defensive pattern
    already established in
    `python_test_plot/plot_regrid_step1.py::_dz_cmap_and_norm()`.

    Deliberately has NO MRMS-specific or model-specific knowledge (no
    sentinel-clipping, no source dispatch) -- that keeps this reusable for
    any `GriddedField`-shaped continuous dBZ field regardless of source."""
    mask = data < min_dbz
    if missing_value is not None:
        mask = mask | (data == missing_value)
    masked = np.ma.masked_where(mask, data)

    cmap = copy.copy(cb_colors.nws_dz_cmap)
    cmap.set_over("black")
    cmap.set_under("none")
    norm = BoundaryNorm(cb_colors.REFLECTIVITY_LEVELS, cmap.N)

    return ax.pcolormesh(lon_plot, lat2d, masked, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, alpha=alpha, shading="auto")


def add_reflectivity_colorbar(fig, ax, mesh, label: str = "Composite Reflectivity (dBZ)"):
    """Horizontal colorbar -- reflectivity plotting's own established
    convention (`python_test_plot/plot_regrid_step1.py`,
    `plot_mpas_mrms_eastern_mask.py`), deliberately distinct from
    `object_spatial_map.py`'s vertical age-mode colorbar; not unified."""
    return fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.04, shrink=0.85, label=label, extend="max")
