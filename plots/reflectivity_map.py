"""Standalone composite-reflectivity spatial map -- a continuous dBZ field
plotted geographically, independent of the object-identification pipeline
entirely (unlike `object_spatial_map.py`, which shades identified storm
OBJECT footprints, not the raw field). Applies uniformly to any of 3
sources that share `py_obj.regrid.io_mrms.GriddedField` as their common
return type: forecast model output, native MRMS observations, already-
interpolated MRMS observations -- confirmed the exact same class is
returned by `load_mrms_grib2()`, `load_mrms_netcdf()`, and
`load_model_netcdf()` already.

Reuses the same base-map/aspect-ratio/zero-margin-extent/title conventions
as `object_spatial_map.py`, via `spatial_map.py`'s `build_base_map()`/
`finalize_and_save()`, plus the new `add_reflectivity_layer()`/
`add_reflectivity_colorbar()` render primitives added alongside those.

Deliberately has NO MRMS-specific or model-specific logic (no sentinel
clipping, no source dispatch) -- that's the driver's job
(`drivers/plot_reflectivity_maps.py`), keeping this module source-agnostic.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from . import spatial_map


@dataclass
class ReflectivityFrame:
    """One `GriddedField`'s worth of data + the display label the driver
    attaches (`GriddedField` itself has no notion of "source" or "member")
    -- the common unit this module's rendering function consumes,
    mirroring `object_spatial_map.ObjectSlice`'s role."""
    lat2d: np.ndarray
    lon2d: np.ndarray
    data: np.ndarray
    valid_time: datetime
    missing_value: float | None
    label: str  # e.g. "MRMS (native)", "MRMS (interpolated)", "MPAS mem1", "WoFS mem03"


def _resolve_extent(frame: ReflectivityFrame, extent: list | None) -> list:
    """extent=None means the frame's own full bounding box, zero margin --
    reuses the already-established zero-margin-default convention
    (`spatial_map.crop_extent_from_mask`) rather than inventing a new one."""
    if extent is not None:
        return extent
    keep_mask = np.ones(frame.data.shape, dtype=bool)
    return spatial_map.crop_extent_from_mask(frame.lat2d, frame.lon2d, keep_mask)


def _default_title(frame: ReflectivityFrame) -> tuple:
    """Returns (short_title, valid_time_label) -- same two-anchor split as
    `object_spatial_map._default_title`."""
    valid_time_label = f"{frame.valid_time:%Y-%m-%d %H:%MZ}"
    return frame.label, valid_time_label


def plot_reflectivity_map(
    frame: ReflectivityFrame,
    out_path: str,
    extent: list | None = None,
    min_dbz: float = 20.0,
    alpha: float = 1.0,
    title: str | None = None,
    figsize: tuple = (11, 8.5),
) -> str:
    resolved_extent = _resolve_extent(frame, extent)
    lon_plot = spatial_map.lon_to_plot_range(frame.lon2d)

    fig, ax = spatial_map.build_base_map(figsize=figsize)
    mesh = spatial_map.add_reflectivity_layer(
        ax, lon_plot, frame.lat2d, frame.data,
        missing_value=frame.missing_value, min_dbz=min_dbz, alpha=alpha,
    )
    spatial_map.add_reflectivity_colorbar(fig, ax, mesh)

    default_title, valid_time_label = _default_title(frame)
    final_title = title if title is not None else default_title
    spatial_map.finalize_and_save(fig, ax, resolved_extent, out_path, title=final_title, valid_time_label=valid_time_label)
    return out_path
