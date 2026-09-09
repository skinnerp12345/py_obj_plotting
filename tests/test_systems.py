"""SystemSpec + load_system() validation: synthetic fixtures for the
flexibility axes (ensemble vs. deterministic, opaque member_id formats,
auto color assignment), plus a real-data check against all 4 example
sources cataloged during this redesign's planning phase -- exact
member-count/lead-range numbers below were independently measured directly
from these files before writing these assertions (see
py_obj_plotting/systems.py's module docstring), not guessed.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_systems.py -v -s
"""

import os
from datetime import datetime, timedelta

import pytest

from python_obj.obj_core import MatchRecord, MatchResult, write_match_file
from py_obj_plotting.cb_colors import blue5, green5, orange5
from py_obj_plotting.systems import DEFAULT_COLOR_PALETTE, SystemSpec, assign_colors, load_system

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SFE_MATCHES_DIR = os.path.join(REPO_ROOT, "sfe_poster", "sfe_matches")
NCAR_MPAS_MATCHES_DIR = os.path.join(REPO_ROOT, "ncar_mpas_analysis", "matched_objects_100km")
_SKIP_NO_SFE_POSTER = pytest.mark.skipif(
    not os.path.isdir(SFE_MATCHES_DIR), reason="requires the external sfe_poster/sfe_matches/ dataset (not bundled)",
)
_SKIP_NO_NCAR_MPAS_ANALYSIS = pytest.mark.skipif(
    not os.path.isdir(NCAR_MPAS_MATCHES_DIR), reason="requires the external ncar_mpas_analysis/ dataset (not bundled)",
)


def _record(category: str, rid: int) -> MatchRecord:
    kwargs = dict(category=category, truth_id=rid, forecast_id=rid, ti_score=0.9 if category == "hit" else 0.0)
    if category == "hit":
        kwargs.update(
            centroid_dist_km=5.0, truth_area_km2=100.0, forecast_area_km2=110.0,
            truth_max_intensity=50.0, forecast_max_intensity=52.0,
            truth_mean_intensity=45.0, forecast_mean_intensity=46.0,
        )
    return MatchRecord(**kwargs)


def _write_init_snapshot(path, init_time, member_ids, lead_minutes_list):
    """member_ids=None -> deterministic (no member dim). One 'hit' record
    per (member, lead_minutes)."""
    results = []
    members = member_ids if member_ids is not None else [None]
    for m in members:
        for lm in lead_minutes_list:
            results.append(MatchResult(
                records=[_record("hit", 1)], valid_time=init_time + timedelta(minutes=lm), member_id=m,
            ))
    write_match_file(path, results, len(members), 1, 40.0, 40.0, 0.2, init_time=init_time)


# --- Check 1: assign_colors -------------------------------------------------

def test_assign_colors_fills_in_order_and_skips_explicit():
    systems = [
        SystemSpec("A", "dir_a"),
        SystemSpec("B", "dir_b", color=green5),
        SystemSpec("C", "dir_c"),
    ]
    out = assign_colors(systems)
    print(f"\n[systems-check1] colors: {[s.color for s in out]}")
    assert out[0].color == DEFAULT_COLOR_PALETTE[0] == blue5
    assert out[1].color == green5  # untouched, didn't consume a palette slot
    assert out[2].color == DEFAULT_COLOR_PALETTE[1] == orange5
    # original list untouched
    assert systems[0].color is None


def test_assign_colors_raises_when_too_many_systems_need_auto_color():
    systems = [SystemSpec(f"S{i}", f"dir_{i}") for i in range(len(DEFAULT_COLOR_PALETTE) + 1)]
    with pytest.raises(ValueError, match="DEFAULT_COLOR_PALETTE"):
        assign_colors(systems)


# --- Check 2: load_system, synthetic ensemble vs. deterministic ------------

def test_load_system_deterministic_no_member_dim(tmp_path):
    d = tmp_path / "det_system"
    d.mkdir()
    t0 = datetime(2026, 5, 1, 0, 0, 0)
    _write_init_snapshot(str(d / "match_init_20260501_000000.nc"), t0, None, [10, 20, 30])

    loaded = load_system(SystemSpec("Det", str(d)))
    print(f"\n[systems-check2] member_ids={loaded.member_ids}, lead_minutes={loaded.lead_minutes}")
    assert loaded.member_ids is None
    assert loaded.lead_minutes == [10.0, 20.0, 30.0]
    assert loaded.init_times == [t0]
    assert list(loaded.records_by_key[(t0, 10.0)].keys()) == [None]


def test_load_system_ensemble_opaque_member_ids(tmp_path):
    d = tmp_path / "ens_system"
    d.mkdir()
    t0 = datetime(2026, 5, 1, 0, 0, 0)
    # deliberately using the mem_1-style (not mem00-style) format -- confirms
    # no format is assumed anywhere in the loader
    _write_init_snapshot(str(d / "match_init_20260501_000000.nc"), t0, ["mem_2", "mem_1"], [0, 60])

    loaded = load_system(SystemSpec("Ens", str(d)))
    print(f"\n[systems-check3] member_ids={loaded.member_ids}")
    assert loaded.member_ids == ["mem_1", "mem_2"]  # sorted, opaque
    assert loaded.lead_minutes == [0.0, 60.0]


def test_load_system_mixed_ensemble_and_deterministic_raises(tmp_path):
    """A single-member 'ensemble' collapses to the same on-disk shape as
    deterministic (write_match_file's own use_member_dim = len(distinct) > 1
    rule), so this fixture needs >=2 members to actually produce a real
    member dimension worth conflicting with a deterministic file."""
    d = tmp_path / "mixed_system"
    d.mkdir()
    t0 = datetime(2026, 5, 1, 0, 0, 0)
    _write_init_snapshot(str(d / "match_init_20260501_000000.nc"), t0, ["mem_1", "mem_2"], [0])
    _write_init_snapshot(str(d / "match_init_20260502_000000.nc"), t0 + timedelta(days=1), None, [0])

    with pytest.raises(ValueError, match="consistently ensemble or deterministic"):
        load_system(SystemSpec("Mixed", str(d)))


def test_load_system_single_lead_time_file_no_time_dim(tmp_path):
    """A file with exactly one valid_time has no match_time_index variable
    at all (write_match_file's own use_time_dim rule) -- load_system must
    still handle it (every record implicitly belongs to that one time),
    not just files spanning many lead times."""
    d = tmp_path / "single_time_system"
    d.mkdir()
    t0 = datetime(2026, 5, 1, 0, 0, 0)
    _write_init_snapshot(str(d / "match_init_20260501_000000.nc"), t0, ["mem_1", "mem_2"], [15])

    loaded = load_system(SystemSpec("SingleTime", str(d)))
    print(f"\n[systems-check4] lead_minutes={loaded.lead_minutes}, records={loaded.records_by_key}")
    assert loaded.lead_minutes == [15.0]
    assert set(loaded.records_by_key[(t0, 15.0)].keys()) == {"mem_1", "mem_2"}
    for arrs in loaded.records_by_key[(t0, 15.0)].values():
        assert len(arrs) == 1 and list(arrs[0]) == ["hit"]


# --- Check 3: real data, all 4 example sources ------------------------------

@_SKIP_NO_SFE_POSTER
def test_load_system_real_nowcastnet():
    """Deterministic system, no lead=0 (NowcastNet's first available lead
    is 10 min), 10-minute cadence."""
    loaded = load_system(SystemSpec("NowcastNet", os.path.join(SFE_MATCHES_DIR, "nowcastnet_vs_mrms")))
    print(f"\n[systems-check5] NowcastNet: {len(loaded.files)} files, {len(loaded.init_times)} init_times, "
          f"member_ids={loaded.member_ids}, n_lead_minutes={len(loaded.lead_minutes)}, "
          f"range=({loaded.lead_minutes[0]},{loaded.lead_minutes[-1]})")
    assert len(loaded.files) == len(loaded.init_times) == 169
    assert loaded.member_ids is None
    assert len(loaded.lead_minutes) == 18
    assert loaded.lead_minutes[0] == 10.0
    assert loaded.lead_minutes[-1] == 180.0
    assert loaded.n_files_used_stored_init_time == 169
    assert loaded.n_files_used_filename_init_time == 0


@_SKIP_NO_SFE_POSTER
def test_load_system_real_wofs():
    """18-member ensemble, mem00-format, 5-minute cadence, includes lead=0."""
    loaded = load_system(SystemSpec("WoFS", os.path.join(SFE_MATCHES_DIR, "wofs_vs_mrms")))
    print(f"\n[systems-check6] WoFS: {len(loaded.files)} files, member_ids[:3]={loaded.member_ids[:3]}, "
          f"n_members={len(loaded.member_ids)}, n_lead_minutes={len(loaded.lead_minutes)}, "
          f"range=({loaded.lead_minutes[0]},{loaded.lead_minutes[-1]})")
    assert len(loaded.files) == len(loaded.init_times) == 190
    assert loaded.member_ids == [f"mem{i:02d}" for i in range(18)]
    assert len(loaded.lead_minutes) == 73
    assert loaded.lead_minutes[0] == 0.0
    assert loaded.lead_minutes[-1] == 360.0


@_SKIP_NO_SFE_POSTER
def test_load_system_real_wofscast():
    """18-member ensemble, first lead is 20 min (a real, confirmed startup
    gap), 10-minute cadence."""
    loaded = load_system(SystemSpec("WoFSCast", os.path.join(SFE_MATCHES_DIR, "wofscast_vs_mrms")))
    print(f"\n[systems-check7] WoFSCast: {len(loaded.files)} files, n_members={len(loaded.member_ids)}, "
          f"n_lead_minutes={len(loaded.lead_minutes)}, range=({loaded.lead_minutes[0]},{loaded.lead_minutes[-1]})")
    assert len(loaded.files) == len(loaded.init_times) == 176
    assert loaded.member_ids == [f"mem{i:02d}" for i in range(18)]
    assert len(loaded.lead_minutes) == 35
    assert loaded.lead_minutes[0] == 20.0  # not 0 -- confirmed real startup gap
    assert loaded.lead_minutes[-1] == 360.0


@_SKIP_NO_NCAR_MPAS_ANALYSIS
def test_load_system_real_mpas_100km():
    """5-member ensemble, mem_1-format (opaque, different convention than
    WoFS/WoFSCast), hourly cadence out to 132h, every file's init_time
    falls back to filename parsing (these files predate the stored
    init_time attribute)."""
    loaded = load_system(SystemSpec("MPAS-100km", NCAR_MPAS_MATCHES_DIR))
    print(f"\n[systems-check8] MPAS-100km: {len(loaded.files)} files, member_ids={loaded.member_ids}, "
          f"n_lead_minutes={len(loaded.lead_minutes)}, range=({loaded.lead_minutes[0]},{loaded.lead_minutes[-1]}), "
          f"n_from_filename={loaded.n_files_used_filename_init_time}")
    assert len(loaded.files) == len(loaded.init_times) == 118
    assert loaded.member_ids == [f"mem_{i}" for i in range(1, 6)]
    assert len(loaded.lead_minutes) == 133
    assert loaded.lead_minutes[0] == 0.0
    assert loaded.lead_minutes[-1] == 7920.0  # 132 hours
    assert loaded.n_files_used_filename_init_time == 118
    assert loaded.n_files_used_stored_init_time == 0
