"""Validation of reflectivity_selection.py: absolute-time selection (pure
logic) and MRMS file discovery/valid-time parsing.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_reflectivity_selection.py -v -s
"""

import os
from datetime import datetime, timedelta

import pytest

from py_obj_plotting.reflectivity_selection import (
    AbsoluteTimeSelectionConfig,
    discover_mrms_files_with_time,
    resolve_absolute_time_selection,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _times(minutes):
    base = datetime(2023, 4, 1, 0, 0, 0)
    return [base + timedelta(minutes=m) for m in minutes]


# ---- resolve_absolute_time_selection ----


def test_full_mode_returns_all_sorted():
    available = _times([30, 0, 60])
    cfg = AbsoluteTimeSelectionConfig(mode="full")
    result = resolve_absolute_time_selection(cfg, available)
    assert result == _times([0, 30, 60])


def test_range_mode_is_inclusive_of_bounds():
    available = _times([0, 30, 60, 90])
    lo, hi = _times([30, 60])
    cfg = AbsoluteTimeSelectionConfig(mode="range", valid_time_range=(lo, hi))
    result = resolve_absolute_time_selection(cfg, available)
    assert result == _times([30, 60])


def test_range_mode_no_match_raises():
    available = _times([0, 30])
    lo, hi = _times([500, 600])
    cfg = AbsoluteTimeSelectionConfig(mode="range", valid_time_range=(lo, hi))
    with pytest.raises(ValueError, match="valid_time_range"):
        resolve_absolute_time_selection(cfg, available)


def test_range_mode_missing_range_raises():
    cfg = AbsoluteTimeSelectionConfig(mode="range")
    with pytest.raises(ValueError, match="valid_time_range"):
        resolve_absolute_time_selection(cfg, _times([0]))


def test_single_mode_picks_nearest_within_tolerance():
    available = _times([0, 41, 100])  # jittered, like real MRMS timestamps
    target = _times([38])[0]
    cfg = AbsoluteTimeSelectionConfig(mode="single", valid_time=target, tolerance_minutes=10.0)
    result = resolve_absolute_time_selection(cfg, available)
    print(f"\n[refl-select-check1] result={result}")
    assert result == [_times([41])[0]]


def test_single_mode_outside_tolerance_raises():
    available = _times([0, 100])
    target = _times([50])[0]
    cfg = AbsoluteTimeSelectionConfig(mode="single", valid_time=target, tolerance_minutes=10.0)
    with pytest.raises(ValueError, match="no available time within"):
        resolve_absolute_time_selection(cfg, available)


def test_single_mode_missing_valid_time_raises():
    cfg = AbsoluteTimeSelectionConfig(mode="single")
    with pytest.raises(ValueError, match="valid_time"):
        resolve_absolute_time_selection(cfg, _times([0]))


def test_invalid_mode_raises():
    cfg = AbsoluteTimeSelectionConfig(mode="bogus")
    with pytest.raises(ValueError, match="bogus"):
        resolve_absolute_time_selection(cfg, _times([0]))


# ---- discover_mrms_files_with_time ----


def test_discover_parses_both_native_and_interp_filenames_and_skips_malformed(tmp_path):
    (tmp_path / "MRMS_MergedReflectivityQCComposite_00.50_20230401-010041.grib2.gz").touch()
    (tmp_path / "interp_mrms_20230401_020038.nc").touch()
    (tmp_path / "README.txt").touch()  # malformed -- must be silently skipped, not raise

    discovered = discover_mrms_files_with_time(str(tmp_path), "*")
    print(f"\n[refl-select-check2] discovered={[(d.filepath, d.valid_time) for d in discovered]}")
    valid_times = {d.valid_time for d in discovered}
    assert valid_times == {datetime(2023, 4, 1, 1, 0, 41), datetime(2023, 4, 1, 2, 0, 38)}
    assert len(discovered) == 2


def test_real_discovery_against_bundled_sample_data():
    mrms_dir = os.path.join(REPO_ROOT, "py_obj", "sample_data", "mpas_case", "mrms", "20230501")
    interp_dir = os.path.join(REPO_ROOT, "py_obj", "sample_data", "mpas_case", "interp_mrms", "20230501")
    native = discover_mrms_files_with_time(mrms_dir, "*.grib2*")
    interp = discover_mrms_files_with_time(interp_dir, "*.nc")
    print(f"\n[refl-select-check3] native={[(os.path.basename(d.filepath), d.valid_time) for d in native]}")
    print(f"[refl-select-check3] interp={[(os.path.basename(d.filepath), d.valid_time) for d in interp]}")
    assert len(native) == 3
    assert len(interp) == 3
    assert {d.valid_time for d in native} == {d.valid_time for d in interp}
