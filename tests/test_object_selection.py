"""Validation of object_selection.py: member/time selection resolution
(pure logic, hand-computable) plus the file-discovery I/O helpers against
real files written via py_obj.obj_core.write_object_file.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_object_selection.py -v -s
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from py_obj.obj_core import IdentificationResult, StormObject, write_object_file
from py_obj_plotting.object_selection import (
    MemberSelectionConfig,
    TimeSelectionConfig,
    discover_object_file_members_and_times,
    find_forecast_file_for_init,
    find_nearest_mrms_object_file,
    resolve_member_selection,
    resolve_time_selection,
)

# ---- resolve_member_selection ----


def test_resolve_member_selection_all():
    cfg = MemberSelectionConfig(mode="all")
    assert resolve_member_selection(cfg, ["mem_1", "mem_2", "mem_3"]) == ["mem_1", "mem_2", "mem_3"]


def test_resolve_member_selection_single():
    cfg = MemberSelectionConfig(mode="single", members=["mem_2"])
    assert resolve_member_selection(cfg, ["mem_1", "mem_2", "mem_3"]) == ["mem_2"]


def test_resolve_member_selection_list_preserves_order():
    cfg = MemberSelectionConfig(mode="list", members=["mem_3", "mem_1"])
    assert resolve_member_selection(cfg, ["mem_1", "mem_2", "mem_3"]) == ["mem_3", "mem_1"]


def test_resolve_member_selection_no_member_dimension_always_returns_none():
    for mode in ("all", "single", "list"):
        cfg = MemberSelectionConfig(mode=mode, members=["mem_1"] if mode != "all" else None)
        assert resolve_member_selection(cfg, None) == [None]


def test_resolve_member_selection_unknown_member_raises():
    cfg = MemberSelectionConfig(mode="single", members=["mem_99"])
    with pytest.raises(ValueError, match="mem_99"):
        resolve_member_selection(cfg, ["mem_1", "mem_2"])


def test_resolve_member_selection_invalid_mode_raises():
    cfg = MemberSelectionConfig(mode="bogus")
    with pytest.raises(ValueError, match="bogus"):
        resolve_member_selection(cfg, ["mem_1"])


# ---- resolve_time_selection ----


def _times(hours):
    base = datetime(2026, 5, 1, 0, 0, 0)
    return [base + timedelta(hours=h) for h in hours]


def test_resolve_time_selection_full():
    init_time = datetime(2026, 5, 1)
    available = _times([0, 1, 2, 3])
    cfg = TimeSelectionConfig(mode="full")
    assert resolve_time_selection(cfg, available, init_time) == available


def test_resolve_time_selection_single():
    init_time = datetime(2026, 5, 1)
    available = _times([0, 1, 2, 3])
    cfg = TimeSelectionConfig(mode="single", lead_minutes=120.0)
    assert resolve_time_selection(cfg, available, init_time) == [available[2]]


def test_resolve_time_selection_single_no_match_raises():
    init_time = datetime(2026, 5, 1)
    available = _times([0, 1, 2, 3])
    cfg = TimeSelectionConfig(mode="single", lead_minutes=90.0)
    with pytest.raises(ValueError, match="90.0"):
        resolve_time_selection(cfg, available, init_time)


def test_resolve_time_selection_range():
    init_time = datetime(2026, 5, 1)
    available = _times([0, 1, 2, 3, 4])
    cfg = TimeSelectionConfig(mode="range", lead_minutes_range=(60.0, 180.0))
    result = resolve_time_selection(cfg, available, init_time)
    assert result == available[1:4]  # hours 1,2,3


def test_resolve_time_selection_range_no_match_raises():
    init_time = datetime(2026, 5, 1)
    available = _times([0, 1])
    cfg = TimeSelectionConfig(mode="range", lead_minutes_range=(500.0, 600.0))
    with pytest.raises(ValueError, match="lead_minutes_range"):
        resolve_time_selection(cfg, available, init_time)


def test_resolve_time_selection_invalid_mode_raises():
    cfg = TimeSelectionConfig(mode="bogus")
    with pytest.raises(ValueError, match="bogus"):
        resolve_time_selection(cfg, _times([0]), datetime(2026, 5, 1))


# ---- I/O helpers, real files written via write_object_file ----


def _make_obj(id) -> StormObject:
    return StormObject(
        id=id, area_px=1, area_km2=1.0, max_intensity=1.0, mean_intensity=1.0,
        major_axis_length=1.0, minor_axis_length=1.0, eccentricity=0.5, orientation=0.0, solidity=0.9,
        centroid_rowcol=(0.0, 0.0), centroid_lat=35.0, centroid_lon=260.0,
        centroid_x_km=0.0, centroid_y_km=0.0, is_linear=0,
    )


def test_discover_object_file_members_and_times_ensemble(tmp_path):
    lat2d = np.tile(np.linspace(30.0, 32.0, 3).reshape(-1, 1), (1, 3))
    lon2d = np.tile(np.linspace(260.0, 262.0, 3).reshape(1, -1), (3, 1))
    labels2d = np.zeros((3, 3), dtype=np.int32)
    init_time = datetime(2026, 5, 1, 0, 0, 0)
    valid_time = datetime(2026, 5, 1, 1, 0, 0)
    results = [
        IdentificationResult(member_id="mem_1", valid_time=valid_time, labels=labels2d, objects=[_make_obj(1)]),
        IdentificationResult(member_id="mem_2", valid_time=valid_time, labels=labels2d, objects=[_make_obj(1)]),
    ]
    path = str(tmp_path / "obj_test.nc")
    write_object_file(path, init_time, lat2d, lon2d, results, n_source_files=1, thresh_1=40.0, thresh_2=45.0, area_thresh_km2=10.0)

    member_ids, valid_times, discovered_init_time = discover_object_file_members_and_times(path)
    print(f"\n[obj-selection-check1] member_ids={member_ids}, valid_times={valid_times}, init_time={discovered_init_time}")
    assert member_ids == ["mem_1", "mem_2"]
    assert valid_times == [valid_time]
    assert discovered_init_time == init_time


def test_discover_object_file_members_and_times_single_shape(tmp_path):
    lat2d = np.tile(np.linspace(30.0, 32.0, 3).reshape(-1, 1), (1, 3))
    lon2d = np.tile(np.linspace(260.0, 262.0, 3).reshape(1, -1), (3, 1))
    labels2d = np.zeros((3, 3), dtype=np.int32)
    valid_time = datetime(2026, 5, 1, 1, 0, 0)
    results = [IdentificationResult(member_id=None, valid_time=valid_time, labels=labels2d, objects=[_make_obj(1)])]
    path = str(tmp_path / "obj_obs_test.nc")
    write_object_file(path, None, lat2d, lon2d, results, n_source_files=1, thresh_1=40.0, thresh_2=45.0, area_thresh_km2=10.0)

    member_ids, valid_times, init_time = discover_object_file_members_and_times(path)
    print(f"\n[obj-selection-check2] member_ids={member_ids}, valid_times={valid_times}, init_time={init_time}")
    assert member_ids is None
    assert valid_times == [valid_time]
    assert init_time is None


def test_find_forecast_file_for_init(tmp_path):
    lat2d = np.tile(np.linspace(30.0, 32.0, 3).reshape(-1, 1), (1, 3))
    lon2d = np.tile(np.linspace(260.0, 262.0, 3).reshape(1, -1), (3, 1))
    labels2d = np.zeros((3, 3), dtype=np.int32)
    d = tmp_path / "forecast_dir"
    d.mkdir()

    for init_hour in (0, 1):
        init_time = datetime(2026, 5, 1, init_hour, 0, 0)
        results = [IdentificationResult(member_id="mem_1", valid_time=init_time, labels=labels2d, objects=[_make_obj(1)]),
                   IdentificationResult(member_id="mem_2", valid_time=init_time, labels=labels2d, objects=[_make_obj(1)])]
        write_object_file(str(d / f"obj_init_{init_hour}.nc"), init_time, lat2d, lon2d, results,
                           n_source_files=1, thresh_1=40.0, thresh_2=45.0, area_thresh_km2=10.0)

    target = datetime(2026, 5, 1, 1, 0, 0)
    found = find_forecast_file_for_init(str(d), target)
    print(f"\n[obj-selection-check3] found={found}")
    assert found == str(d / "obj_init_1.nc")

    with pytest.raises(FileNotFoundError):
        find_forecast_file_for_init(str(d), datetime(2026, 5, 2))


def test_find_forecast_file_for_init_falls_back_to_filename_when_attribute_absent(tmp_path):
    """Regression test for a real gap found running this against
    ncar_mpas_analysis/matched_objects_300km/: those real match files
    predate the stored init_time global attribute entirely (same
    backward-compat situation aggregate_matches_by_lead_hour.py's own
    _parse_init_time_from_filename already handles) -- read_init_time_only
    alone returns None for every one of them, so find_forecast_file_for_init
    must fall back to parsing the filename's own <YYYYMMDD>_<HHMMSS>."""
    import netCDF4
    import numpy as np

    d = tmp_path / "no_attr_dir"
    d.mkdir()
    path = d / "match_init_20230424_000000.nc"
    with netCDF4.Dataset(str(path), "w") as ds:
        ds.createDimension("x", 1)
        ds.createVariable("dummy", "f4", ("x",))[:] = np.array([0.0])
        # deliberately NOT setting ds.init_time

    found = find_forecast_file_for_init(str(d), datetime(2023, 4, 24, 0, 0, 0), pattern="match_init_*.nc")
    print(f"\n[obj-selection-check5] found={found}")
    assert found == str(path)


def test_find_nearest_mrms_object_file(tmp_path):
    d = tmp_path / "mrms_dir"
    d.mkdir()
    for f in ("obj_obs_20260501_000041.nc", "obj_obs_20260501_010038.nc"):
        (d / f).touch()

    found = find_nearest_mrms_object_file(str(d), datetime(2026, 5, 1, 1, 0, 0), tolerance_minutes=20.0)
    print(f"\n[obj-selection-check4] found={found}")
    assert found == str(d / "obj_obs_20260501_010038.nc")

    with pytest.raises(FileNotFoundError):
        find_nearest_mrms_object_file(str(d), datetime(2026, 5, 1, 5, 0, 0), tolerance_minutes=20.0)
