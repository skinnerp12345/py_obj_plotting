"""check_shared_forecast_cases() validation: synthetic checks for the
shared/system-only/partial case bookkeeping and the require_shared_cases
raise-vs-report policy, plus real-data checks confirming it correctly
reports the real mismatch across the 3 sfe_poster systems (169/190/176
files -> genuinely different case sets) rather than silently proceeding,
and confirms a clean, fully-shared 118-case match across the 4
ncar_mpas_analysis threshold directories. Real numbers below were measured
directly before being written into these assertions, not guessed.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_case_alignment.py -v -s
"""

import os
from datetime import datetime, timedelta

import pytest

from python_obj.obj_core import MatchRecord, MatchResult, write_match_file
from py_obj_plotting.multi_system import check_shared_forecast_cases, load_systems
from py_obj_plotting.systems import SystemSpec

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SFE_MATCHES_DIR = os.path.join(REPO_ROOT, "sfe_poster", "sfe_matches")
NCAR_MPAS_DIR = os.path.join(REPO_ROOT, "ncar_mpas_analysis")
_SKIP_NO_SFE_POSTER = pytest.mark.skipif(
    not os.path.isdir(SFE_MATCHES_DIR), reason="requires the external sfe_poster/sfe_matches/ dataset (not bundled)",
)
_SKIP_NO_NCAR_MPAS_ANALYSIS = pytest.mark.skipif(
    not os.path.isdir(NCAR_MPAS_DIR), reason="requires the external ncar_mpas_analysis/ dataset (not bundled)",
)


def _write_init_snapshot(path, init_time, lead_minutes_list=(0,)):
    results = [
        MatchResult(records=[MatchRecord(category="hit", truth_id=1, forecast_id=1, ti_score=0.9)],
                    valid_time=init_time + timedelta(minutes=lm), member_id=None)
        for lm in lead_minutes_list
    ]
    write_match_file(path, results, 1, 1, 40.0, 40.0, 0.2, init_time=init_time)


# --- Check 1: synthetic, fully aligned --------------------------------------

def test_fully_aligned_systems_report_no_mismatch(tmp_path):
    t0, t1 = datetime(2026, 5, 1), datetime(2026, 5, 2)
    da, db = tmp_path / "A", tmp_path / "B"
    da.mkdir(); db.mkdir()
    for t in (t0, t1):
        _write_init_snapshot(str(da / f"match_init_{t:%Y%m%d_%H%M%S}.nc"), t)
        _write_init_snapshot(str(db / f"match_init_{t:%Y%m%d_%H%M%S}.nc"), t)

    loaded = load_systems([SystemSpec("A", str(da)), SystemSpec("B", str(db))])
    report = check_shared_forecast_cases(loaded, require_shared_cases=True)  # must not raise
    print(f"\n[casealign-check1] shared={report.shared_cases}, is_fully_aligned={report.is_fully_aligned}")
    assert report.is_fully_aligned
    assert report.shared_cases == [t0, t1]
    assert report.system_only_cases == {"A": [], "B": []}
    assert report.partial_cases == {}


# --- Check 2: synthetic, exclusively-own case mismatch ----------------------

def test_system_only_case_raises_by_default_and_is_reported(tmp_path):
    t0, t1 = datetime(2026, 5, 1), datetime(2026, 5, 2)
    da, db = tmp_path / "A", tmp_path / "B"
    da.mkdir(); db.mkdir()
    _write_init_snapshot(str(da / f"match_init_{t0:%Y%m%d_%H%M%S}.nc"), t0)
    _write_init_snapshot(str(da / f"match_init_{t1:%Y%m%d_%H%M%S}.nc"), t1)  # A-only case
    _write_init_snapshot(str(db / f"match_init_{t0:%Y%m%d_%H%M%S}.nc"), t0)

    loaded = load_systems([SystemSpec("A", str(da)), SystemSpec("B", str(db))])

    with pytest.raises(ValueError, match="do not all cover the same set of forecast cases"):
        check_shared_forecast_cases(loaded, require_shared_cases=True)

    report = check_shared_forecast_cases(loaded, require_shared_cases=False)  # must NOT raise
    print(f"\n[casealign-check2] shared={report.shared_cases}, system_only={report.system_only_cases}")
    assert not report.is_fully_aligned
    assert report.shared_cases == [t0]
    assert report.system_only_cases == {"A": [t1], "B": []}
    assert report.partial_cases == {t1: ["A"]}


# --- Check 3: synthetic, N=3 partial-overlap case NOT caught by system_only -

def test_two_of_three_systems_sharing_a_case_is_flagged_as_partial_not_missed(tmp_path):
    """A case shared by exactly 2 of 3 systems is NOT 'exclusive' to either
    one, so it would never appear in system_only_cases for any system --
    this is exactly why partial_cases exists as a separate, stricter
    check: is_fully_aligned must still correctly report False, and the
    case must show up in partial_cases with its 2 present systems named."""
    t0, t1 = datetime(2026, 5, 1), datetime(2026, 5, 2)
    da, db, dc = tmp_path / "A", tmp_path / "B", tmp_path / "C"
    da.mkdir(); db.mkdir(); dc.mkdir()
    for d in (da, db, dc):
        _write_init_snapshot(str(d / f"match_init_{t0:%Y%m%d_%H%M%S}.nc"), t0)
    # t1 present in A and B only, not C
    _write_init_snapshot(str(da / f"match_init_{t1:%Y%m%d_%H%M%S}.nc"), t1)
    _write_init_snapshot(str(db / f"match_init_{t1:%Y%m%d_%H%M%S}.nc"), t1)

    loaded = load_systems([SystemSpec("A", str(da)), SystemSpec("B", str(db)), SystemSpec("C", str(dc))])
    report = check_shared_forecast_cases(loaded, require_shared_cases=False)
    print(f"\n[casealign-check3] system_only={report.system_only_cases}, partial={report.partial_cases}, "
          f"is_fully_aligned={report.is_fully_aligned}")
    assert report.system_only_cases == {"A": [], "B": [], "C": []}  # confirms the blind spot exists
    assert not report.is_fully_aligned  # but the overall check still correctly catches it
    assert report.partial_cases == {t1: ["A", "B"]}


# --- Check 4: real data -------------------------------------------------------

@_SKIP_NO_SFE_POSTER
def test_real_sfe_poster_reports_real_case_mismatch():
    """169/190/176 real match_init_*.nc files across the 3 sfe_poster
    systems (measured directly, not guessed) -- confirms they do NOT all
    cover the same forecast cases, and that this is correctly raised (not
    silently proceeded past) under the default require_shared_cases=True."""
    match_dirs = {
        "WoFS": os.path.join(SFE_MATCHES_DIR, "wofs_vs_mrms"),
        "WoFSCast": os.path.join(SFE_MATCHES_DIR, "wofscast_vs_mrms"),
        "NowcastNet": os.path.join(SFE_MATCHES_DIR, "nowcastnet_vs_mrms"),
    }
    loaded = load_systems([SystemSpec(name, d) for name, d in match_dirs.items()])

    with pytest.raises(ValueError, match="do not all cover the same set of forecast cases"):
        check_shared_forecast_cases(loaded, require_shared_cases=True)

    report = check_shared_forecast_cases(loaded, require_shared_cases=False)
    print(f"\n[casealign-check4] shared={len(report.shared_cases)}, "
          f"system_only={ {k: len(v) for k, v in report.system_only_cases.items()} }, "
          f"n_partial={len(report.partial_cases)}, is_fully_aligned={report.is_fully_aligned}")
    assert not report.is_fully_aligned
    assert len(report.shared_cases) == 144
    assert len(report.system_only_cases["WoFS"]) == 6
    assert len(report.system_only_cases["WoFSCast"]) == 0
    assert len(report.system_only_cases["NowcastNet"]) == 17
    assert len(report.partial_cases) == 63


@_SKIP_NO_NCAR_MPAS_ANALYSIS
def test_real_ncar_mpas_analysis_reports_clean_shared_match():
    """The 4 real matching-threshold directories (100/200/300/400km) share
    the identical underlying MPAS forecast archive, so they must report a
    clean, fully-shared 118-case match with require_shared_cases=True
    raising nothing."""
    match_dirs = {f"{k}km": os.path.join(NCAR_MPAS_DIR, f"matched_objects_{k}km") for k in (100, 200, 300, 400)}
    loaded = load_systems([SystemSpec(name, d) for name, d in match_dirs.items()])

    report = check_shared_forecast_cases(loaded, require_shared_cases=True)  # must not raise
    print(f"\n[casealign-check5] shared={len(report.shared_cases)}, is_fully_aligned={report.is_fully_aligned}")
    assert report.is_fully_aligned
    assert len(report.shared_cases) == 118
    assert all(v == [] for v in report.system_only_cases.values())
    assert report.partial_cases == {}
