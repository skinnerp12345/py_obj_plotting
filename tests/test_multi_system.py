"""compute_included_keys() validation: synthetic checks for the
intersection/union lead_time_mode policy and the always-reported (never
silent) exclusion/availability bookkeeping, plus a real-data regression
(golden-value snapshot) proving the new binary-mode API can reconstruct
the old, now-retired mixed per-lead-minute (`relaxed_lead_minutes` set)
policy via composition -- the concrete worked example referenced in
multi_system.py's own module docstring. The exact included-key counts
below (2874/2480/2642) were cross-checked against the old private
_system_presence/_compute_included_keys implementation before that
implementation was replaced by this module in the same redesign pass (see
climatology.aggregate_matches_multi_system, which now calls this module's
own compute_included_keys directly) -- kept here as a fixed golden value
since the old implementation no longer exists to compare against live.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_multi_system.py -v -s
"""

import os
from datetime import datetime, timedelta

import pytest

from py_obj.obj_core import MatchRecord, MatchResult, write_match_file
from py_obj_plotting.multi_system import compute_included_keys, load_systems
from py_obj_plotting.systems import SystemSpec

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SFE_MATCHES_DIR = os.path.join(REPO_ROOT, "sfe_poster", "sfe_matches")
_SKIP_NO_SFE_POSTER = pytest.mark.skipif(
    not os.path.isdir(SFE_MATCHES_DIR), reason="requires the external sfe_poster/sfe_matches/ dataset (not bundled)",
)


def _record(category: str, rid: int) -> MatchRecord:
    return MatchRecord(category=category, truth_id=rid, forecast_id=rid, ti_score=0.9 if category == "hit" else 0.0)


def _write_init_snapshot(path, init_time, member_ids, lead_minutes_list, category="hit"):
    results = []
    members = member_ids if member_ids is not None else [None]
    for m in members:
        for lm in lead_minutes_list:
            results.append(MatchResult(
                records=[_record(category, 1)], valid_time=init_time + timedelta(minutes=lm), member_id=m,
            ))
    write_match_file(path, results, len(members), 1, 40.0, 40.0, 0.2, init_time=init_time)


# --- Check 1: synthetic intersection vs. union -------------------------------

def test_intersection_mode_excludes_partial_availability_and_reports_it(tmp_path):
    """System A has lead_minutes {0, 10}; System B has only {0}. Under
    intersection mode, (cycle, 10) must be excluded (not pooled from A
    alone) and reported in excluded_partial_lead_minutes; (cycle, 0) is
    shared and included for both."""
    t0 = datetime(2026, 5, 1, 0, 0, 0)
    da, db = tmp_path / "A", tmp_path / "B"
    da.mkdir(); db.mkdir()
    _write_init_snapshot(str(da / "match_init_20260501_000000.nc"), t0, None, [0, 10])
    _write_init_snapshot(str(db / "match_init_20260501_000000.nc"), t0, None, [0])

    loaded = load_systems([SystemSpec("A", str(da)), SystemSpec("B", str(db))])
    included, report = compute_included_keys(loaded, lead_time_mode="intersection")

    print(f"\n[multisys-check1] included={included}, excluded={report.excluded_partial_lead_minutes}")
    assert included["A"] == {(t0, 0.0)}
    assert included["B"] == {(t0, 0.0)}
    assert report.excluded_partial_lead_minutes == {(t0, 10.0): ["B"]}
    assert report.lead_time_mode == "intersection"
    assert report.shared_cycles == [t0]


def test_union_mode_includes_each_systems_own_presence_independently(tmp_path):
    """Same fixture as check 1, but union mode: (cycle, 10) IS included for
    A (the only system that has it) with no exclusion event at all -- B
    simply doesn't contribute a point there."""
    t0 = datetime(2026, 5, 1, 0, 0, 0)
    da, db = tmp_path / "A", tmp_path / "B"
    da.mkdir(); db.mkdir()
    _write_init_snapshot(str(da / "match_init_20260501_000000.nc"), t0, None, [0, 10])
    _write_init_snapshot(str(db / "match_init_20260501_000000.nc"), t0, None, [0])

    loaded = load_systems([SystemSpec("A", str(da)), SystemSpec("B", str(db))])
    included, report = compute_included_keys(loaded, lead_time_mode="union")

    print(f"\n[multisys-check2] included={included}")
    assert included["A"] == {(t0, 0.0), (t0, 10.0)}
    assert included["B"] == {(t0, 0.0)}
    assert report.excluded_partial_lead_minutes == {}


def test_system_only_cycles_reported(tmp_path):
    """A cycle present in only one system's own files must show up in
    system_only_cycles -- never silently folded into shared_cycles."""
    t0 = datetime(2026, 5, 1, 0, 0, 0)
    t1 = datetime(2026, 5, 2, 0, 0, 0)
    da, db = tmp_path / "A", tmp_path / "B"
    da.mkdir(); db.mkdir()
    _write_init_snapshot(str(da / "match_init_20260501_000000.nc"), t0, None, [0])
    _write_init_snapshot(str(da / "match_init_20260502_000000.nc"), t1, None, [0])  # A-only cycle
    _write_init_snapshot(str(db / "match_init_20260501_000000.nc"), t0, None, [0])

    loaded = load_systems([SystemSpec("A", str(da)), SystemSpec("B", str(db))])
    _, report = compute_included_keys(loaded, lead_time_mode="intersection")

    print(f"\n[multisys-check3] shared={report.shared_cycles}, system_only={report.system_only_cycles}")
    assert report.shared_cycles == [t0]
    assert report.system_only_cycles["A"] == [t1]
    assert report.system_only_cycles["B"] == []


def test_load_systems_duplicate_name_raises(tmp_path):
    da = tmp_path / "A"
    da.mkdir()
    _write_init_snapshot(str(da / "match_init_20260501_000000.nc"), datetime(2026, 5, 1), None, [0])

    with pytest.raises(ValueError, match="duplicate SystemSpec name"):
        load_systems([SystemSpec("Dup", str(da)), SystemSpec("Dup", str(da))])


def test_invalid_lead_time_mode_raises(tmp_path):
    da = tmp_path / "A"
    da.mkdir()
    _write_init_snapshot(str(da / "match_init_20260501_000000.nc"), datetime(2026, 5, 1), None, [0])
    loaded = load_systems([SystemSpec("A", str(da))])
    with pytest.raises(ValueError, match="lead_time_mode"):
        compute_included_keys(loaded, lead_time_mode="not_a_real_mode")


# --- Check 2: real-data golden-value regression, old mixed-mode policy -----

# Measured directly against real sfe_poster data by cross-checking this
# exact composition against the old private _system_presence/
# _compute_included_keys implementation, before that implementation was
# deleted in favor of this module (see aggregate_matches_multi_system.py,
# which now calls compute_included_keys() directly) -- fixed here as a
# golden value.
_OLD_MIXED_MODE_INCLUDED_COUNTS = {"WoFS": 2874, "WoFSCast": 2480, "NowcastNet": 2642}


@_SKIP_NO_SFE_POSTER
def test_real_sfe_poster_reconstructs_old_mixed_mode_via_composition():
    """plot_sfe_multi_system_verification.py's OLD, now-retired settings
    used a MIXED policy: strict (intersection) for most lead minutes, but
    relaxed (union) for the 3 early ones {0, 10, 20} (a real, confirmed
    WoFSCast/NowcastNet startup-availability gap) -- not a pure binary
    mode. Proves compute_included_keys() can reconstruct that policy by
    composition (one 'union' call over the relaxed lead minutes, one
    'intersection' call over the rest, unioned together), against the
    fixed golden-value counts above."""
    match_dirs = {
        "WoFS": os.path.join(SFE_MATCHES_DIR, "wofs_vs_mrms"),
        "WoFSCast": os.path.join(SFE_MATCHES_DIR, "wofscast_vs_mrms"),
        "NowcastNet": os.path.join(SFE_MATCHES_DIR, "nowcastnet_vs_mrms"),
    }
    lead_minutes_full = list(range(0, 190, 10))  # 0..180
    relaxed = {0, 10, 20}
    strict_minutes = [m for m in lead_minutes_full if m not in relaxed]

    systems = [SystemSpec(name, d) for name, d in match_dirs.items()]
    loaded = load_systems(systems)

    included_relaxed, _ = compute_included_keys(loaded, lead_time_mode="union", lead_minutes=sorted(relaxed))
    included_strict, _ = compute_included_keys(loaded, lead_time_mode="intersection", lead_minutes=strict_minutes)

    included_combined = {
        name: included_relaxed[name] | included_strict[name] for name in match_dirs
    }

    for name in match_dirs:
        n = len(included_combined[name])
        print(f"\n[multisys-check4] {name}: new_composed={n} keys (golden={_OLD_MIXED_MODE_INCLUDED_COUNTS[name]})")
        assert n == _OLD_MIXED_MODE_INCLUDED_COUNTS[name]
