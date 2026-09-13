"""Synthetic-fixture validation of
climatology.aggregate_matches_by_lead_hour, with hand-computed expected
per-(member, lead_hour) contingency-table counts -- confirms cross-case
pooling, both init_time sources (stored attribute vs. filename fallback),
and that the aggregator correctly rejects a non-init_snapshot-shaped file.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_aggregate_matches_by_lead_hour.py -v -s
"""

from datetime import datetime, timedelta

import netCDF4
import pytest

from py_obj.obj_core import MatchRecord, MatchResult, write_match_file
from py_obj_plotting.climatology import (
    build_match_lead_hour_climatology,
    build_match_lead_hour_climatology_by_storm_mode,
)


def _record(category: str, rid: int) -> MatchRecord:
    return MatchRecord(category=category, truth_id=rid, forecast_id=rid, ti_score=0.9 if category == "hit" else 0.0)


def _record_with_mode(category: str, rid: int, truth_is_linear=None, forecast_is_linear=None) -> MatchRecord:
    return MatchRecord(
        category=category, truth_id=rid, forecast_id=rid, ti_score=0.9 if category == "hit" else 0.0,
        truth_is_linear=truth_is_linear, forecast_is_linear=forecast_is_linear,
    )


def _write_case(path: str, init_time: datetime, member_records: dict, write_init_time: bool = True) -> None:
    """member_records: {(member_id, lead_hour): [category_str, ...]} -- one
    MatchRecord per category string, ids auto-assigned."""
    results = []
    next_id = 1
    for (member_id, lead_hour), categories in member_records.items():
        records = []
        for cat in categories:
            records.append(_record(cat, next_id))
            next_id += 1
        results.append(MatchResult(records=records, valid_time=init_time + timedelta(hours=lead_hour), member_id=member_id))
    write_match_file(
        path, results, n_truth_source_files=1, n_forecast_source_files=1,
        max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2,
        init_time=init_time if write_init_time else None,
    )


def test_cross_case_pooling_and_hand_computed_counts(tmp_path):
    init_a = datetime(2023, 5, 1, 0, 0, 0)
    init_b = datetime(2023, 5, 2, 0, 0, 0)

    # NOTE: every case fixture below includes a lead_hour=1 entry too (even
    # though the assertions only check lead_hour=0) -- write_match_file only
    # creates a `time` dimension when >1 distinct valid_time is present; a
    # single-lead_hour fixture would silently collapse to the "ensemble_
    # snapshot"-equivalent shape (member dim only, no time dim), which the
    # aggregator correctly rejects (see test_non_init_snapshot_shape_raises)
    # since real match files always have both dims.

    # Case A: mem_1 @ lead=0 has 2 hits + 1 miss; mem_2 @ lead=0 has 1 false_alarm
    _write_case(str(tmp_path / "match_init_a.nc"), init_a, {
        ("mem_1", 0): ["hit", "hit", "miss"],
        ("mem_1", 1): ["hit"],
        ("mem_2", 0): ["false_alarm"],
        ("mem_2", 1): ["hit"],
    })
    # Case B: mem_1 @ lead=0 (a DIFFERENT case, same lead_hour) has 1 hit + 1 truth_extra + 1 forecast_extra
    _write_case(str(tmp_path / "match_init_b.nc"), init_b, {
        ("mem_1", 0): ["hit", "truth_extra", "forecast_extra"],
        ("mem_1", 1): ["hit"],
        ("mem_2", 0): ["false_alarm"],
        ("mem_2", 1): ["hit"],
    })

    clim = build_match_lead_hour_climatology(str(tmp_path))
    print(f"\n[match-clim-check1] lead_hours={clim.lead_hours}, members={clim.members}")
    assert clim.lead_hours == [0.0, 1.0]
    assert clim.members == ["mem_1", "mem_2"]

    # mem_1 @ lead=0: pooled across both cases -> 3 hits, 1 miss, 1 truth_extra, 1 forecast_extra
    assert clim.n_hits[("mem_1", 0.0)] == 3
    assert clim.n_misses[("mem_1", 0.0)] == 1
    assert clim.n_truth_extra[("mem_1", 0.0)] == 1
    assert clim.n_forecast_extra[("mem_1", 0.0)] == 1
    assert clim.n_false_alarms.get(("mem_1", 0.0), 0) == 0

    # mem_2 @ lead=0: pooled across both cases -> 2 false_alarms, everything else 0
    assert clim.n_false_alarms[("mem_2", 0.0)] == 2
    assert clim.n_hits.get(("mem_2", 0.0), 0) == 0

    assert clim.n_cases_per_lead_hour[0.0] == 2  # 2 cases, not 4 (not double-counted per member)
    print(f"[match-clim-check1] mem_1 hits={clim.n_hits[('mem_1', 0.0)]}, mem_2 false_alarms={clim.n_false_alarms[('mem_2', 0.0)]}")


def test_init_time_source_diagnostics_stored_vs_filename(tmp_path):
    init_a = datetime(2023, 5, 1, 0, 0, 0)
    init_b = datetime(2023, 5, 2, 0, 0, 0)

    # written WITH the stored init_time attribute
    _write_case(str(tmp_path / "match_init_20230501_000000.nc"), init_a, {
        ("mem_1", 0): ["hit"], ("mem_1", 1): ["hit"],
        ("mem_2", 0): ["miss"], ("mem_2", 1): ["hit"],
    }, write_init_time=True)
    # written WITHOUT it (simulates a real legacy file) -- must fall back to
    # parsing init_time from the filename itself
    path_b = str(tmp_path / "match_init_20230502_000000.nc")
    _write_case(path_b, init_b, {
        ("mem_1", 0): ["hit"], ("mem_1", 1): ["hit"],
        ("mem_2", 0): ["miss"], ("mem_2", 1): ["hit"],
    }, write_init_time=False)
    with netCDF4.Dataset(path_b) as ds:
        assert not hasattr(ds, "init_time")  # confirm the fixture really omitted it

    clim = build_match_lead_hour_climatology(str(tmp_path))
    print(f"\n[match-clim-check2] stored={clim.n_files_used_stored_init_time}, filename={clim.n_files_used_filename_init_time}")
    assert clim.n_files_used_stored_init_time == 1
    assert clim.n_files_used_filename_init_time == 1
    # both files' data must still be pooled correctly regardless of which
    # init_time source was used
    assert clim.n_hits[("mem_1", 0.0)] == 2
    assert clim.n_misses[("mem_2", 0.0)] == 2


def test_non_init_snapshot_shape_raises(tmp_path):
    """A single-member, single-time match file has no member_id/valid_time
    DIMENSION at all (write_match_file collapses it to global attributes) --
    the lead-hour aggregator requires the full ensemble/lead-time shape and
    must raise a clear error, not silently produce an empty/wrong result."""
    t0 = datetime(2023, 5, 1, 0, 0, 0)
    result = MatchResult(records=[_record("hit", 1)], valid_time=t0, member_id=None)
    path = str(tmp_path / "match_init_20230501_000000.nc")
    write_match_file(
        path, [result], n_truth_source_files=1, n_forecast_source_files=1,
        max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2, init_time=t0,
    )
    with pytest.raises(ValueError, match="member_id/match_member_index"):
        build_match_lead_hour_climatology(str(tmp_path))


# --- Storm-mode-filtered pooling ------------------------------------------------

def _write_case_with_records(path: str, init_time: datetime, member_records: dict) -> None:
    """member_records: {(member_id, lead_hour): [MatchRecord, ...]}"""
    results = []
    for (member_id, lead_hour), records in member_records.items():
        results.append(MatchResult(records=records, valid_time=init_time + timedelta(hours=lead_hour), member_id=member_id))
    write_match_file(
        path, results, n_truth_source_files=1, n_forecast_source_files=1,
        max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2, init_time=init_time,
    )


def test_storm_mode_hand_computed_split_including_false_alarm_asymmetry(tmp_path):
    """hit/miss/truth_extra rows are keyed by truth_is_linear; false_alarm/
    forecast_extra rows (no truth object at all) are keyed by
    forecast_is_linear instead -- a real, necessary asymmetry, not an
    oversight."""
    init_a = datetime(2023, 5, 1, 0, 0, 0)
    _write_case_with_records(str(tmp_path / "match_init_a.nc"), init_a, {
        ("mem_1", 0.0): [
            _record_with_mode("hit", 1, truth_is_linear=0),      # cellular hit
            _record_with_mode("hit", 2, truth_is_linear=2),      # linear hit
            _record_with_mode("miss", 3, truth_is_linear=0),     # cellular miss
            _record_with_mode("false_alarm", 4, forecast_is_linear=1),  # mixed false alarm (no truth object)
        ],
        ("mem_1", 1.0): [_record_with_mode("hit", 5, truth_is_linear=0)],
        ("mem_2", 0.0): [_record_with_mode("hit", 6, truth_is_linear=0)],
        ("mem_2", 1.0): [_record_with_mode("hit", 7, truth_is_linear=0)],
    })

    cellular = build_match_lead_hour_climatology_by_storm_mode(str(tmp_path), storm_mode=0)
    mixed = build_match_lead_hour_climatology_by_storm_mode(str(tmp_path), storm_mode=1)
    linear = build_match_lead_hour_climatology_by_storm_mode(str(tmp_path), storm_mode=2)

    print(f"\n[storm-mode-check1] cellular_hits={cellular.n_hits}, cellular_misses={cellular.n_misses}")
    assert cellular.n_hits[("mem_1", 0.0)] == 1
    assert cellular.n_misses[("mem_1", 0.0)] == 1
    assert cellular.n_false_alarms.get(("mem_1", 0.0), 0) == 0  # the false_alarm is mixed, not cellular

    print(f"[storm-mode-check1] mixed_false_alarms={mixed.n_false_alarms}")
    assert mixed.n_false_alarms[("mem_1", 0.0)] == 1
    assert mixed.n_hits.get(("mem_1", 0.0), 0) == 0

    print(f"[storm-mode-check1] linear_hits={linear.n_hits}")
    assert linear.n_hits[("mem_1", 0.0)] == 1
    assert linear.n_misses.get(("mem_1", 0.0), 0) == 0


def test_storm_modes_sum_to_unfiltered_pooled_totals(tmp_path):
    """The strongest regression proof: summing cellular+mixed+linear counts
    at every (member, lead_hour) key must exactly reproduce
    build_match_lead_hour_climatology()'s own unfiltered totals for the
    same real data -- no record double-counted or silently dropped."""
    init_a = datetime(2023, 5, 1, 0, 0, 0)
    init_b = datetime(2023, 5, 2, 0, 0, 0)
    modes = [0, 1, 2, 0, 1, 2, 0]
    for path, init_time in ((tmp_path / "match_init_a.nc", init_a), (tmp_path / "match_init_b.nc", init_b)):
        _write_case_with_records(str(path), init_time, {
            ("mem_1", 0.0): [_record_with_mode("hit", i, truth_is_linear=m) for i, m in enumerate(modes)]
                            + [_record_with_mode("miss", 100 + i, truth_is_linear=m) for i, m in enumerate(modes)]
                            + [_record_with_mode("false_alarm", 200 + i, forecast_is_linear=m) for i, m in enumerate(modes)],
            ("mem_1", 1.0): [_record_with_mode("hit", 300, truth_is_linear=0)],
            ("mem_2", 0.0): [_record_with_mode("hit", 400, truth_is_linear=1)],
            ("mem_2", 1.0): [_record_with_mode("hit", 500, truth_is_linear=2)],
        })

    unfiltered = build_match_lead_hour_climatology(str(tmp_path))
    by_mode = [build_match_lead_hour_climatology_by_storm_mode(str(tmp_path), storm_mode=m) for m in (0, 1, 2)]

    for key in unfiltered.n_hits:
        summed = sum(c.n_hits.get(key, 0) for c in by_mode)
        assert summed == unfiltered.n_hits[key], f"n_hits mismatch at {key}: {summed} != {unfiltered.n_hits[key]}"
    for key in unfiltered.n_misses:
        summed = sum(c.n_misses.get(key, 0) for c in by_mode)
        assert summed == unfiltered.n_misses[key], f"n_misses mismatch at {key}: {summed} != {unfiltered.n_misses[key]}"
    for key in unfiltered.n_false_alarms:
        summed = sum(c.n_false_alarms.get(key, 0) for c in by_mode)
        assert summed == unfiltered.n_false_alarms[key], f"n_false_alarms mismatch at {key}: {summed} != {unfiltered.n_false_alarms[key]}"

    total_hits_unfiltered = sum(unfiltered.n_hits.values())
    total_hits_by_mode = sum(sum(c.n_hits.values()) for c in by_mode)
    print(f"\n[storm-mode-check2] total_hits_unfiltered={total_hits_unfiltered}, total_hits_by_mode={total_hits_by_mode}")
    assert total_hits_unfiltered == total_hits_by_mode
    assert total_hits_unfiltered > 0  # a real, non-trivial check


def test_storm_mode_missing_data_degrades_to_all_zero_not_error(tmp_path):
    """A match file predating storm_mode_classification has no
    truth_is_linear/forecast_is_linear variables at all -- every record
    must be excluded from every mode (all-zero climatology), not raise."""
    init_a = datetime(2023, 5, 1, 0, 0, 0)
    _write_case(str(tmp_path / "match_init_a.nc"), init_a, {
        ("mem_1", 0.0): ["hit", "hit", "miss"],
        ("mem_1", 1.0): ["hit"],
        ("mem_2", 0.0): ["hit"],
        ("mem_2", 1.0): ["hit"],
    })
    cellular = build_match_lead_hour_climatology_by_storm_mode(str(tmp_path), storm_mode=0)
    print(f"\n[storm-mode-check3] cellular_hits={cellular.n_hits}")
    assert sum(cellular.n_hits.values()) == 0
    assert sum(cellular.n_misses.values()) == 0


def test_storm_mode_deterministic_single_member_shape_tolerated(tmp_path):
    """Unlike build_match_lead_hour_climatology (which strictly requires a
    real ensemble/init_snapshot shape), this function tolerates a
    deterministic (no member dimension) match file too -- pooled under a
    single member=None key -- since drivers.plot_matching_timeseries must
    support both deterministic and ensemble match data for this feature."""
    t0 = datetime(2023, 5, 1, 0, 0, 0)
    result = MatchResult(
        records=[_record_with_mode("hit", 1, truth_is_linear=0), _record_with_mode("miss", 2, truth_is_linear=0)],
        valid_time=t0, member_id=None,
    )
    path = str(tmp_path / "match_init_20230501_000000.nc")
    write_match_file(
        path, [result], n_truth_source_files=1, n_forecast_source_files=1,
        max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2, init_time=t0,
    )
    cellular = build_match_lead_hour_climatology_by_storm_mode(str(tmp_path), storm_mode=0)
    print(f"\n[storm-mode-check4] lead_hours={cellular.lead_hours}, members={cellular.members}, n_hits={cellular.n_hits}")
    assert cellular.lead_hours == [0.0]
    assert cellular.members == [None]
    assert cellular.n_hits[(None, 0.0)] == 1
    assert cellular.n_misses[(None, 0.0)] == 1
