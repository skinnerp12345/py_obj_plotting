"""Per-timestep MRMS object footprint + centroid map, "WoFS style": cartopy
PlateCarree, eastern-2/3 CONUS domain, state boundaries + Canada/Mexico
country borders (no county/road layer), objects shaded by is_linear
category (0=cellular=orange, 1=possibly linear=purple, 2=linear=blue, using
python_obj's own three-tier StormObject.is_linear convention), plus a small
circle marking each object's own centroid, colored to match its category.

Reads python_obj's own object-file schema unchanged via
python_obj.obj_core.read_object_file/load_conus_boundary/conus_mask_east --
no reimplementation of object-file I/O or masking logic here.
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from python_obj.obj_core import conus_mask_east, load_conus_boundary, read_object_file

from .. import cb_colors

_CATEGORY_COLORS = {0: cb_colors.orange5, 1: cb_colors.purple5, 2: cb_colors.blue5}
_CATEGORY_LABELS = {0: "Cellular", 1: "Possibly linear", 2: "Linear"}

STATES = cfeature.NaturalEarthFeature(
    category="cultural", name="admin_1_states_provinces_lines", scale="50m", facecolor="none"
)
COUNTRIES = cfeature.NaturalEarthFeature(
    category="cultural", name="admin_0_countries", scale="50m", facecolor="none"
)


def discover_day_files(mrms_objects_dir: str, date_str: str) -> list:
    """date_str: 'YYYYMMDD'. Every obj_obs_<date_str>_*.nc file under
    mrms_objects_dir (a flat directory -- confirmed the real bundled
    ncar_mpas_analysis/mrms_objects/ layout is not per-date subdirectoried)."""
    files = sorted(glob.glob(os.path.join(mrms_objects_dir, f"obj_obs_{date_str}_*.nc")))
    if not files:
        raise FileNotFoundError(f"No MRMS object files found for '{date_str}' under '{mrms_objects_dir}'")
    return files


def grid_extent(lat2d: np.ndarray, lon2d: np.ndarray, margin_deg: float = 2.0) -> list:
    """Eastern-2/3 CONUS domain -- the bounding box of conus_mask_east's own
    KEPT region (same mask already used to identify these objects in the
    first place; confirmed no real object in this dataset sits west of its
    -105 deg cutoff, so this crop is a pure view restriction, not a data
    loss), not the full native MRMS grid's own bounds."""
    keep_mask = ~conus_mask_east(lat2d, lon2d)
    lon_norm = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)
    kept_lat = lat2d[keep_mask]
    kept_lon = lon_norm[keep_mask]
    return [
        float(kept_lon.min() - margin_deg), float(kept_lon.max() + margin_deg),
        float(kept_lat.min() - margin_deg), float(kept_lat.max() + margin_deg),
    ]


def _category_grid(labels2d: np.ndarray, objects: list) -> np.ndarray:
    """Integer grid: 0 = no object, 1/2/3 = cellular/possibly-linear/linear
    (is_linear + 1, keeping 0 free to mean 'background'). Vectorized via a
    lookup table indexed by label id -- avoids a per-pixel Python loop over
    a potentially large (1059x1799) grid."""
    if labels2d.max() == 0:
        return np.zeros_like(labels2d, dtype=np.int16)
    lut = np.zeros(labels2d.max() + 1, dtype=np.int16)
    for obj in objects:
        if 0 <= obj.id <= labels2d.max():
            lut[obj.id] = obj.is_linear + 1
    return lut[labels2d]


def plot_mrms_object_map(
    contents, out_path: str, extent: list,
    centroid_marker_size: float = 3.0,
    draw_conus_outline: bool = True,
) -> None:
    lat2d, lon2d, labels2d, objects = contents.lat2d, contents.lon2d, contents.labels, contents.objects
    lon_plot = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)

    cat_grid = _category_grid(labels2d, objects)
    cat_masked = np.ma.masked_where(cat_grid == 0, cat_grid)

    cmap = ListedColormap([_CATEGORY_COLORS[0], _CATEGORY_COLORS[1], _CATEGORY_COLORS[2]])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(11, 8.5), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.pcolormesh(lon_plot, lat2d, cat_masked, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, shading="auto")

    if draw_conus_outline:
        boundary = load_conus_boundary()
        polys = list(boundary.geoms) if boundary.geom_type == "MultiPolygon" else [boundary]
        for poly in polys:
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, transform=ccrs.PlateCarree(), color="black", linewidth=0.8)

    ax.add_feature(STATES, edgecolor="gray", linewidth=0.5)
    ax.add_feature(COUNTRIES, edgecolor="black", linewidth=0.7)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    centroid_lons = [o.centroid_lon - 360.0 if o.centroid_lon > 180.0 else o.centroid_lon for o in objects]
    centroid_lats = [o.centroid_lat for o in objects]
    centroid_colors = [_CATEGORY_COLORS[o.is_linear] for o in objects]
    ax.scatter(
        centroid_lons, centroid_lats, s=centroid_marker_size, c=centroid_colors, marker="o",
        edgecolors="none", transform=ccrs.PlateCarree(), zorder=5,
    )

    valid_time = contents.valid_times[0]
    n_by_cat = {k: sum(1 for o in objects if o.is_linear == k) for k in (0, 1, 2)}
    ax.set_title(
        f"MRMS objects -- valid {valid_time} UTC\n"
        f"{len(objects)} objects: {n_by_cat[0]} cellular, {n_by_cat[1]} possibly linear, {n_by_cat[2]} linear "
        f"(small dots = object centroids)",
        fontsize=11,
    )
    ax.legend(handles=[
        Patch(facecolor=_CATEGORY_COLORS[0], label=_CATEGORY_LABELS[0]),
        Patch(facecolor=_CATEGORY_COLORS[1], label=_CATEGORY_LABELS[1]),
        Patch(facecolor=_CATEGORY_COLORS[2], label=_CATEGORY_LABELS[2]),
    ], loc="lower left", fontsize=9)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=115)
    plt.close(fig)


def plot_mrms_objects_for_day(mrms_objects_dir: str, date_str: str, out_dir: str) -> list:
    """Reads every available hourly MRMS object file for one day and writes
    one PNG per file, all sharing the same fixed map extent (the eastern-2/3
    CONUS domain, computed once from the first file) so every image in the
    day's series is directly comparable."""
    files = discover_day_files(mrms_objects_dir, date_str)
    print(f"Found {len(files)} MRMS object files for {date_str}")

    out_paths = []
    extent = None
    for f in files:
        contents = read_object_file(f)
        if extent is None:
            extent = grid_extent(contents.lat2d, contents.lon2d)

        valid_time = contents.valid_times[0]
        out_path = os.path.join(os.path.join(out_dir, date_str), f"mrms_objects_{valid_time:%Y%m%d_%H%M%S}.png")
        plot_mrms_object_map(contents, out_path, extent)
        out_paths.append(out_path)
        print(f"wrote {out_path} ({len(contents.objects)} objects)")

    return out_paths
