"""Synthetic + real-data validation of climatology.aggregate_matches_per_case:
confirms build_per_case_match_climatologies() preserves per-case granularity
(never flattened, unlike build_multi_system_match_climatology), that "day"
grouping correctly buckets multiple same-calendar-date init_times into one
case, that case labels are unique, and that case_object_count()/case_metrics()
compute the documented aggregation correctly. The core regression proof is
real: summing every per-case count must exactly equal the existing pooled
aggregation's own totals -- checked against real MPAS (init_time grouping)
and real WoFS (day grouping) data.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_aggregate_matches_per_case.py -v -s
"""

import os
from collections import Counter
from datetime import datetime, timedelta

import pytest

from py_obj.obj_core import MatchRecord, MatchResult, write_match_file
from py_obj_plotting.climatology.aggregate_matches_multi_system import build_multi_system_match_climatology
from py_obj_plotting.climatology.aggregate_matches_per_case import (
    build_per_case_match_climatologies,
    case_metrics,
    case_object_count,
)
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

_CATEGORIES = ("hit", "miss", "false_alarm", "truth_extra", "forecast_extra")


def _record(category: str, rid: int) -> MatchRecord:
    kwargs = dict(category=category, truth_id=rid, forecast_id=rid, ti_score=0.9 if category == "hit" else 0.0)
    return MatchRecord(**kwargs)


def _write_case(path: str, init_time: datetime, member_records: dict) -> None:
    """member_records: {(member_id_or_None, lead_minutes): [category_str, ...]}"""
    results = []
    next_id = 1
    for (member_id, lead_min), categories in member_records.items():
        records = []
        for cat in categories:
            records.append(_record(cat, next_id))
            next_id += 1
        results.append(MatchResult(records=records, valid_time=init_time + timedelta(minutes=lead_min), member_id=member_id))
    write_match_file(
        path, results, n_truth_source_files=1, n_forecast_source_files=1,
        max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2, init_time=init_time,
    )


def _totals(cases: dict) -> Counter:
    total = Counter()
    for clim in cases.values():
        for cat, d in zip(_CATEGORIES, (clim.n_hits, clim.n_misses, clim.n_false_alarms, clim.n_truth_extra, clim.n_forecast_extra)):
            total[cat] += sum(d.values())
    return total


# --- Check 1: synthetic init_time grouping -----------------------------------

def test_init_time_grouping_keeps_every_case_separate(tmp_path):
    d = tmp_path / "A"
    d.mkdir()
    t0 = datetime(2026, 5, 1, 0, 0, 0)
    t1 = datetime(2026, 5, 2, 0, 0, 0)
    # member_id=None (deterministic) -- a single NAMED member would collapse
    # to this same on-disk shape anyway (write_match_file's own
    # use_member_dim = len(distinct) > 1 rule), so this is the correct
    # fixture shape for a single-member case, not a simplification.
    _write_case(str(d / f"match_init_{t0:%Y%m%d_%H%M%S}.nc"), t0, {(None, 0): ["hit", "miss"]})
    _write_case(str(d / f"match_init_{t1:%Y%m%d_%H%M%S}.nc"), t1, {(None, 0): ["hit"]})

    per_case = build_per_case_match_climatologies([SystemSpec("A", str(d))], case_grouping="init_time")
    cases = per_case["A"]
    print(f"\n[per-case-check1] labels={sorted(cases.keys())}")
    assert len(cases) == 2
    assert cases["2026-05-01 0000Z"].n_hits[(None, 0.0)] == 1
    assert cases["2026-05-02 0000Z"].n_hits[(None, 0.0)] == 1


# --- Check 2: synthetic day grouping buckets multiple init_times -----------

def test_day_grouping_buckets_same_day_init_times(tmp_path):
    d = tmp_path / "A"
    d.mkdir()
    t0a = datetime(2026, 5, 1, 0, 0, 0)
    t0b = datetime(2026, 5, 1, 12, 0, 0)  # same calendar day, different init_time
    t1 = datetime(2026, 5, 2, 0, 0, 0)
    _write_case(str(d / f"match_init_{t0a:%Y%m%d_%H%M%S}.nc"), t0a, {(None, 0): ["hit", "miss"]})
    _write_case(str(d / f"match_init_{t0b:%Y%m%d_%H%M%S}.nc"), t0b, {(None, 0): ["hit"]})
    _write_case(str(d / f"match_init_{t1:%Y%m%d_%H%M%S}.nc"), t1, {(None, 0): ["false_alarm"]})

    per_case = build_per_case_match_climatologies([SystemSpec("A", str(d))], case_grouping="day")
    cases = per_case["A"]
    print(f"\n[per-case-check2] labels={sorted(cases.keys())}")
    assert len(cases) == 2  # 2026-05-01's two init_times bucketed together, 2026-05-02 separate
    assert cases["2026-05-01"].n_hits[(None, 0.0)] == 2  # 1 (t0a) + 1 (t0b)
    assert cases["2026-05-01"].n_misses[(None, 0.0)] == 1
    assert cases["2026-05-02"].n_false_alarms[(None, 0.0)] == 1


def test_init_time_grouping_of_the_same_fixture_keeps_them_separate(tmp_path):
    """Same fixture as check 2, but init_time grouping must NOT bucket the
    two same-day init_times together -- confirms the two grouping modes
    genuinely produce different results on identical underlying data."""
    d = tmp_path / "A"
    d.mkdir()
    t0a = datetime(2026, 5, 1, 0, 0, 0)
    t0b = datetime(2026, 5, 1, 12, 0, 0)
    _write_case(str(d / f"match_init_{t0a:%Y%m%d_%H%M%S}.nc"), t0a, {("mem_1", 0): ["hit"]})
    _write_case(str(d / f"match_init_{t0b:%Y%m%d_%H%M%S}.nc"), t0b, {("mem_1", 0): ["hit"]})

    per_case = build_per_case_match_climatologies([SystemSpec("A", str(d))], case_grouping="init_time")
    cases = per_case["A"]
    print(f"\n[per-case-check3] labels={sorted(cases.keys())}")
    assert len(cases) == 2


# --- Check 3: invalid case_grouping raises -----------------------------------

def test_invalid_case_grouping_raises(tmp_path):
    d = tmp_path / "A"
    d.mkdir()
    _write_case(str(d / "match_init_20260501_000000.nc"), datetime(2026, 5, 1), {("mem_1", 0): ["hit"]})
    with pytest.raises(ValueError, match="case_grouping"):
        build_per_case_match_climatologies([SystemSpec("A", str(d))], case_grouping="bogus")


# --- Check 4: case_object_count() ---------------------------------------------

def test_case_object_count_truth_basis_uses_one_member_not_summed(tmp_path):
    """basis='truth' must NOT sum across members (truth objects are
    member-independent -- summing would inflate the count by ensemble
    size)."""
    d = tmp_path / "A"
    d.mkdir()
    t0 = datetime(2026, 5, 1)
    _write_case(str(d / "match_init_20260501_000000.nc"), t0, {
        ("mem_1", 0): ["hit", "miss"], ("mem_2", 0): ["hit", "miss"],
    })
    per_case = build_per_case_match_climatologies([SystemSpec("A", str(d))], case_grouping="init_time")
    clim = per_case["A"]["2026-05-01 0000Z"]
    count = case_object_count(clim, basis="truth")
    print(f"\n[per-case-check4] truth count={count}")
    assert count == 2  # ONE member's hit+miss (2), not 4 (both members summed)


def test_case_object_count_forecast_basis_is_ensemble_mean(tmp_path):
    d = tmp_path / "A"
    d.mkdir()
    t0 = datetime(2026, 5, 1)
    _write_case(str(d / "match_init_20260501_000000.nc"), t0, {
        ("mem_1", 0): ["hit", "false_alarm"], ("mem_2", 0): ["hit", "false_alarm", "false_alarm"],
    })
    per_case = build_per_case_match_climatologies([SystemSpec("A", str(d))], case_grouping="init_time")
    clim = per_case["A"]["2026-05-01 0000Z"]
    count = case_object_count(clim, basis="forecast")
    print(f"\n[per-case-check5] forecast count={count}")
    # mem_1: hit+FA = 2, mem_2: hit+FA+FA = 3 -> mean = 2.5 -> round(2.5) == 2 (Python banker's rounding)
    assert count == round((2 + 3) / 2)


def test_case_object_count_respects_lead_minutes_range(tmp_path):
    d = tmp_path / "A"
    d.mkdir()
    t0 = datetime(2026, 5, 1)
    _write_case(str(d / "match_init_20260501_000000.nc"), t0, {
        ("mem_1", 0): ["hit"], ("mem_1", 60): ["hit", "miss"], ("mem_1", 120): ["hit"],
    })
    per_case = build_per_case_match_climatologies([SystemSpec("A", str(d))], case_grouping="init_time")
    clim = per_case["A"]["2026-05-01 0000Z"]
    full = case_object_count(clim, basis="truth")
    restricted = case_object_count(clim, lead_minutes_range=(60, 60), basis="truth")
    print(f"\n[per-case-check6] full={full}, restricted(60-60min)={restricted}")
    assert full == 4  # 1 + 2 + 1
    assert restricted == 2  # just lead=60min


# --- Check 5: case_metrics() ---------------------------------------------------

def test_case_metrics_pools_counts_across_lead_hours_then_averages_members(tmp_path):
    d = tmp_path / "A"
    d.mkdir()
    t0 = datetime(2026, 5, 1)
    _write_case(str(d / "match_init_20260501_000000.nc"), t0, {
        ("mem_1", 0): ["hit", "hit"], ("mem_1", 60): ["miss"],
        ("mem_2", 0): ["hit"], ("mem_2", 60): ["hit"],
    })
    per_case = build_per_case_match_climatologies([SystemSpec("A", str(d))], case_grouping="init_time")
    clim = per_case["A"]["2026-05-01 0000Z"]
    metrics = case_metrics(clim, lead_minutes_range=(0, 60))
    print(f"\n[per-case-check7] metrics={metrics}")
    # mem_1: pooled h=2,m=1 -> pod=2/3; mem_2: pooled h=2,m=0 -> pod=1.0
    # ensemble-mean pod = (2/3 + 1.0)/2 = 0.8333...
    assert abs(metrics["pod"] - (2 / 3 + 1.0) / 2) < 1e-9


def test_case_metrics_empty_range_returns_zeros(tmp_path):
    d = tmp_path / "A"
    d.mkdir()
    t0 = datetime(2026, 5, 1)
    _write_case(str(d / "match_init_20260501_000000.nc"), t0, {("mem_1", 0): ["hit"]})
    per_case = build_per_case_match_climatologies([SystemSpec("A", str(d))], case_grouping="init_time")
    clim = per_case["A"]["2026-05-01 0000Z"]
    metrics = case_metrics(clim, lead_minutes_range=(1000, 2000))  # no data in range
    print(f"\n[per-case-check8] metrics={metrics}")
    assert metrics == {"pod": 0.0, "far": 0.0, "fbias": 0.0, "csi": 0.0}


# --- Check 6: real-data regression -- per-case sums match pooled totals ----

@_SKIP_NO_NCAR_MPAS_ANALYSIS
def test_real_mpas_per_case_sums_match_pooled_totals():
    systems = [SystemSpec("100km", os.path.join(NCAR_MPAS_DIR, "matched_objects_100km"))]
    per_case = build_per_case_match_climatologies(systems, lead_time_mode="intersection", case_grouping="init_time")
    pooled, _ = build_multi_system_match_climatology(systems, lead_time_mode="intersection")

    cases = per_case["100km"]
    print(f"\n[per-case-check9] n_cases={len(cases)}")
    assert len(cases) == 118

    per_case_totals = _totals(cases)
    p = pooled["100km"]
    pooled_totals = Counter({
        cat: sum(d.values()) for cat, d in zip(
            _CATEGORIES, (p.n_hits, p.n_misses, p.n_false_alarms, p.n_truth_extra, p.n_forecast_extra),
        )
    })
    print(f"[per-case-check9] per_case_totals={dict(per_case_totals)}")
    print(f"[per-case-check9] pooled_totals={dict(pooled_totals)}")
    assert per_case_totals == pooled_totals


@_SKIP_NO_SFE_POSTER
def test_real_wofs_day_grouping_reduces_case_count_and_sums_match_pooled():
    systems = [SystemSpec("WoFS", os.path.join(SFE_MATCHES_DIR, "wofs_vs_mrms"))]
    per_case_init = build_per_case_match_climatologies(systems, lead_time_mode="union", case_grouping="init_time")
    per_case_day = build_per_case_match_climatologies(systems, lead_time_mode="union", case_grouping="day")
    pooled, _ = build_multi_system_match_climatology(systems, lead_time_mode="union")

    n_init, n_day = len(per_case_init["WoFS"]), len(per_case_day["WoFS"])
    print(f"\n[per-case-check10] n_init_time_cases={n_init}, n_day_cases={n_day}")
    assert n_init == 190
    assert n_day == 24
    assert n_day < n_init  # real, confirmed reduction: many initializations per WoFS day

    p = pooled["WoFS"]
    pooled_totals = Counter({
        cat: sum(d.values()) for cat, d in zip(
            _CATEGORIES, (p.n_hits, p.n_misses, p.n_false_alarms, p.n_truth_extra, p.n_forecast_extra),
        )
    })
    assert _totals(per_case_init["WoFS"]) == pooled_totals
    assert _totals(per_case_day["WoFS"]) == pooled_totals
