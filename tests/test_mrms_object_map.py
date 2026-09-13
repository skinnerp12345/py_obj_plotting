"""Validation of plots.mrms_object_map: numeric checks only (category-grid
values, extent computation, day-file discovery) -- never by viewing the
rendered image, matching this project's convention in test_object_spatial_map.py.
StormObject/ObjectFileContents instances are constructed directly (not via
file I/O) for the synthetic checks.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_mrms_object_map.py -v -s
"""

import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from py_obj.obj_core import ObjectFileContents, StormObject
from py_obj_plotting.plots.mrms_object_map import (
    _category_grid,
    discover_day_files,
    grid_extent,
    plot_mrms_object_map,
)


def _obj(id, is_linear=0, centroid_lat=35.0, centroid_lon=260.0) -> StormObject:
    return StormObject(
        id=id, area_px=10, area_km2=100.0, max_intensity=50.0, mean_intensity=45.0,
        major_axis_length=5.0, minor_axis_length=3.0, eccentricity=0.5, orientation=0.0, solidity=0.9,
        centroid_rowcol=(0.0, 0.0), centroid_lat=centroid_lat, centroid_lon=centroid_lon,
        centroid_x_km=0.0, centroid_y_km=0.0, is_linear=is_linear, age_seconds=None,
    )


def _synthetic_grid():
    """5x5 grid, lat/lon a plain regular mesh, 3 objects: id=1 at (0,0)
    cellular, id=2 at (2,2) mixed, id=3 at (4,4) linear."""
    lat2d = np.tile(np.linspace(30.0, 34.0, 5).reshape(-1, 1), (1, 5))
    lon2d = np.tile(np.linspace(260.0, 264.0, 5).reshape(1, -1), (5, 1))
    labels2d = np.zeros((5, 5), dtype=np.int32)
    labels2d[0, 0] = 1
    labels2d[2, 2] = 2
    labels2d[4, 4] = 3
    objects = [_obj(1, is_linear=0), _obj(2, is_linear=1), _obj(3, is_linear=2)]
    return lat2d, lon2d, labels2d, objects


def test_category_grid_hand_computed():
    lat2d, lon2d, labels2d, objects = _synthetic_grid()
    grid = _category_grid(labels2d, objects)
    print(f"\n[mrms-obj-map-check1] grid=\n{grid}")
    assert grid[0, 0] == 1  # cellular -> is_linear(0)+1
    assert grid[2, 2] == 2  # possibly linear -> 1+1
    assert grid[4, 4] == 3  # linear -> 2+1
    assert grid[1, 1] == 0  # background


def test_category_grid_all_background_when_no_objects():
    labels2d = np.zeros((3, 3), dtype=np.int32)
    grid = _category_grid(labels2d, [])
    assert np.all(grid == 0)


def test_grid_extent_restricted_to_eastern_conus_kept_region():
    """A grid spanning both east (kept) and far-west (masked-out) longitudes
    must produce an extent bounded by the KEPT region only, with the
    requested margin applied -- not the full native grid's own bounds."""
    lat2d = np.array([[35.0, 35.0], [35.0, 35.0]])
    lon2d = np.array([[260.0, 200.0], [260.0, 200.0]])  # 260=east (kept), 200=far west (masked)
    extent = grid_extent(lat2d, lon2d, margin_deg=1.0)
    print(f"\n[mrms-obj-map-check2] extent={extent}")
    lon_min, lon_max, lat_min, lat_max = extent
    # 260 deg -> -100 deg after normalization; the masked-out 200(-160) deg
    # column must not pull the extent westward
    assert lon_min > -150.0
    assert lon_max == pytest.approx(-100.0 + 1.0)


def test_discover_day_files_raises_when_none_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="No MRMS object files"):
        discover_day_files(str(tmp_path), "20230101")


def test_discover_day_files_finds_and_sorts_matching_files(tmp_path):
    for name in ["obj_obs_20230101_020000.nc", "obj_obs_20230101_000000.nc", "obj_obs_20230102_000000.nc"]:
        (tmp_path / name).write_bytes(b"")
    files = discover_day_files(str(tmp_path), "20230101")
    print(f"\n[mrms-obj-map-check3] files={[os.path.basename(f) for f in files]}")
    assert [os.path.basename(f) for f in files] == ["obj_obs_20230101_000000.nc", "obj_obs_20230101_020000.nc"]


def test_plot_mrms_object_map_writes_file_with_all_three_categories(tmp_path):
    lat2d, lon2d, labels2d, objects = _synthetic_grid()
    contents = ObjectFileContents(
        lat2d=lat2d, lon2d=lon2d, labels=labels2d, objects=objects,
        member_index=None, time_index=None, member_ids=None,
        valid_times=[datetime(2023, 5, 13, 0, 0, 15)], init_time=None,
        tracked=False, storm_mode_classification=True, thresh_1=39.8, thresh_2=45.6,
        area_thresh_km2=25.0, track_bound_disp_km=None, n_source_files=1,
    )
    out_path = str(tmp_path / "mrms_objects.png")
    extent = grid_extent(lat2d, lon2d)
    plot_mrms_object_map(contents, out_path, extent, draw_conus_outline=False)
    assert os.path.isfile(out_path)
    assert os.path.getsize(out_path) > 0
