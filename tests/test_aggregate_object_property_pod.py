"""Synthetic-fixture validation of
climatology.aggregate_object_property_pod, with hand-computed expected
per-(bin_index, lead_hour) hit/miss counts -- confirms area/max_intensity/
age binning, the age "relative to init" transform, exclusion of non-hit/
miss categories, and independent per-property NaN exclusion (an
untracked-age record still counts under area/max_intensity).

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_aggregate_object_property_pod.py -v -s
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from python_obj.obj_core import MatchRecord, MatchResult, write_match_file
from py_obj_plotting.climatology.aggregate_object_property_pod import build_object_property_pod_climatology


def _record(category: str, rid: int, area_km2=None, max_intensity=None, age_seconds=None) -> MatchRecord:
    kwargs = dict(category=category, truth_id=rid, forecast_id=rid, ti_score=0.9 if category == "hit" else 0.0)
    if category in ("hit", "miss", "truth_extra"):
        kwargs["truth_area_km2"] = area_km2
        kwargs["truth_max_intensity"] = max_intensity
        kwargs["truth_age_seconds"] = age_seconds
    return MatchRecord(**kwargs)


def _write_case(path: str, init_time: datetime, member_records: dict) -> None:
    """member_records: {(member_id, lead_hour): [MatchRecord, ...]}"""
    results = []
    for (member_id, lead_hour), records in member_records.items():
        results.append(MatchResult(records=records, valid_time=init_time + timedelta(hours=lead_hour), member_id=member_id))
    write_match_file(
        path, results, n_truth_source_files=1, n_forecast_source_files=1,
        max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2, init_time=init_time,
    )


def test_hand_computed_pod_by_area_bin(tmp_path):
    init_a = datetime(2023, 5, 1, 0, 0, 0)
    _write_case(str(tmp_path / "match_init_a.nc"), init_a, {
        ("mem_1", 0.0): [
            _record("hit", 1, area_km2=5.0), _record("hit", 2, area_km2=5.0), _record("miss", 3, area_km2=5.0),
            _record("hit", 4, area_km2=15.0),
        ],
        ("mem_1", 1.0): [_record("hit", 5, area_km2=5.0)],  # a distinct lead_hour, needed so write_match_file keeps a time dim
        # a second, distinct member is required or write_match_file collapses
        # to the deterministic (no member dimension) shape this aggregator
        # rejects (see test_non_init_snapshot_shape_raises) -- area=999 keeps
        # these records safely outside every bin under test here.
        ("mem_2", 0.0): [_record("hit", 6, area_km2=999.0)],
        ("mem_2", 1.0): [_record("hit", 7, area_km2=999.0)],
    })

    clim = build_object_property_pod_climatology(str(tmp_path), bin_edges={"area": np.array([0.0, 10.0, 20.0, 30.0])})
    print(f"\n[obj-prop-pod-check1] lead_hours={clim.lead_hours}, members={clim.members}")
    assert clim.lead_hours == [0.0, 1.0]
    assert clim.members == ["mem_1", "mem_2"]

    # bin 0 = [0,10): 2 hits, 1 miss at lead_hour=0.0
    assert clim.n_hits["area"][(0, 0.0)] == 2
    assert clim.n_misses["area"][(0, 0.0)] == 1
    # bin 1 = [10,20): 1 hit at lead_hour=0.0
    assert clim.n_hits["area"][(1, 0.0)] == 1
    assert (1, 0.0) not in clim.n_misses["area"]
    # lead_hour=1.0: 1 hit in bin 0
    assert clim.n_hits["area"][(0, 1.0)] == 1


def test_age_relative_to_init_transform(tmp_path):
    """A hit at lead_hour=1.0 with an absolute tracked age of 3900s (65min)
    has a relative-to-init age of 3900 - 1*3600 = 300s -- must land in the
    [0,600) bin, not wherever the raw 3900s value would fall."""
    init_a = datetime(2023, 5, 1, 0, 0, 0)
    _write_case(str(tmp_path / "match_init_a.nc"), init_a, {
        ("mem_1", 0.0): [_record("hit", 1, area_km2=1.0, age_seconds=100.0)],
        ("mem_1", 1.0): [_record("hit", 2, area_km2=1.0, age_seconds=3900.0)],
        ("mem_2", 0.0): [_record("hit", 3, area_km2=1.0, age_seconds=99999.0)],
        ("mem_2", 1.0): [_record("hit", 4, area_km2=1.0, age_seconds=99999.0)],
    })

    clim = build_object_property_pod_climatology(str(tmp_path), bin_edges={"age": np.array([0.0, 600.0, 1200.0])})
    print(f"\n[obj-prop-pod-check2] n_hits[age]={clim.n_hits['age']}")
    # lead_hour=0.0: raw age 100s, relative = 100 - 0 = 100 -> bin 0
    assert clim.n_hits["age"][(0, 0.0)] == 1
    # lead_hour=1.0: raw age 3900s, relative = 3900 - 3600 = 300 -> bin 0 (NOT bin 6, where raw 3900 would fall)
    assert clim.n_hits["age"][(0, 1.0)] == 1
    assert (6, 1.0) not in clim.n_hits["age"]


def test_false_alarm_and_truth_extra_excluded_from_hit_miss_bins(tmp_path):
    """POD is conditioned on hit/miss rows only -- a truth_extra row has a
    real truth-side object (with a real area) but must never contribute to
    n_hits/n_misses, matching the legacy's own scope for this diagnostic."""
    init_a = datetime(2023, 5, 1, 0, 0, 0)
    _write_case(str(tmp_path / "match_init_a.nc"), init_a, {
        ("mem_1", 0.0): [
            _record("hit", 1, area_km2=5.0),
            _record("truth_extra", 2, area_km2=5.0),
            _record("false_alarm", 3),
        ],
        ("mem_1", 1.0): [_record("hit", 4, area_km2=5.0)],
        ("mem_2", 0.0): [_record("hit", 5, area_km2=999.0)],
        ("mem_2", 1.0): [_record("hit", 6, area_km2=999.0)],
    })

    clim = build_object_property_pod_climatology(str(tmp_path), bin_edges={"area": np.array([0.0, 10.0])})
    print(f"\n[obj-prop-pod-check3] n_hits={clim.n_hits['area']}, n_misses={clim.n_misses['area']}")
    assert clim.n_hits["area"][(0, 0.0)] == 1  # only the real hit counted
    assert (0, 0.0) not in clim.n_misses["area"]  # truth_extra/false_alarm never counted as a miss either


def test_untracked_age_excluded_but_area_still_counts(tmp_path):
    """A hit with a real area but no tracked age (truth_age_seconds=None,
    written as NaN) must be excluded from the 'age' property's bins while
    still counting normally under 'area' -- independent per-property
    exclusion, not an all-or-nothing record filter."""
    init_a = datetime(2023, 5, 1, 0, 0, 0)
    _write_case(str(tmp_path / "match_init_a.nc"), init_a, {
        ("mem_1", 0.0): [_record("hit", 1, area_km2=5.0, age_seconds=None)],
        ("mem_1", 1.0): [_record("hit", 2, area_km2=5.0, age_seconds=None)],
        ("mem_2", 0.0): [_record("hit", 3, area_km2=999.0, age_seconds=None)],
        ("mem_2", 1.0): [_record("hit", 4, area_km2=999.0, age_seconds=None)],
    })

    clim = build_object_property_pod_climatology(
        str(tmp_path), bin_edges={"area": np.array([0.0, 10.0]), "age": np.array([0.0, 600.0])},
    )
    print(f"\n[obj-prop-pod-check4] n_hits[area]={clim.n_hits['area']}, n_hits[age]={clim.n_hits['age']}")
    assert clim.n_hits["area"][(0, 0.0)] == 1
    assert clim.n_hits["age"] == {}  # no real age data anywhere -- climatology built successfully, just empty


def test_unknown_property_raises(tmp_path):
    init_a = datetime(2023, 5, 1, 0, 0, 0)
    _write_case(str(tmp_path / "match_init_a.nc"), init_a, {
        ("mem_1", 0.0): [_record("hit", 1, area_km2=5.0)],
        ("mem_1", 1.0): [_record("hit", 2, area_km2=5.0)],
    })
    with pytest.raises(ValueError, match="unknown propert"):
        build_object_property_pod_climatology(str(tmp_path), bin_edges={"bogus": np.array([0.0, 1.0])})


def test_empty_bin_edges_raises(tmp_path):
    with pytest.raises(ValueError, match="at least one property"):
        build_object_property_pod_climatology(str(tmp_path), bin_edges={})


def test_deterministic_single_member_shape_tolerated(tmp_path):
    """Unlike aggregate_matches_by_lead_hour.build_match_lead_hour_climatology
    (which strictly requires a real ensemble/init_snapshot shape), this
    aggregator pools across every member regardless (never keys its output
    by member), so a deterministic match file -- no member_id/
    match_member_index/match_time_index DIMENSION at all, the shape
    write_match_file produces for a single member and a single lead time
    -- must be tolerated, not raise. Needed since
    drivers.plot_matching_timeseries must support both deterministic and
    ensemble match data for this feature."""
    t0 = datetime(2023, 5, 1, 0, 0, 0)
    result = MatchResult(records=[_record("hit", 1, area_km2=5.0), _record("miss", 2, area_km2=5.0)], valid_time=t0, member_id=None)
    path = str(tmp_path / "match_init_20230501_000000.nc")
    write_match_file(
        path, [result], n_truth_source_files=1, n_forecast_source_files=1,
        max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2, init_time=t0,
    )
    clim = build_object_property_pod_climatology(str(tmp_path), bin_edges={"area": np.array([0.0, 10.0])})
    print(f"\n[obj-prop-pod-check6] lead_hours={clim.lead_hours}, members={clim.members}, n_hits={clim.n_hits['area']}")
    assert clim.lead_hours == [0.0]
    assert clim.members == [None]
    assert clim.n_hits["area"][(0, 0.0)] == 1
    assert clim.n_misses["area"][(0, 0.0)] == 1


def test_cross_case_pooling(tmp_path):
    """Two different forecast cases, same (member, lead_hour, bin) --
    counts must pool/sum across cases, not overwrite."""
    init_a = datetime(2023, 5, 1, 0, 0, 0)
    init_b = datetime(2023, 5, 2, 0, 0, 0)
    _write_case(str(tmp_path / "match_init_a.nc"), init_a, {
        ("mem_1", 0.0): [_record("hit", 1, area_km2=5.0)],
        ("mem_1", 1.0): [_record("hit", 2, area_km2=5.0)],
        ("mem_2", 0.0): [_record("hit", 3, area_km2=999.0)],
        ("mem_2", 1.0): [_record("hit", 4, area_km2=999.0)],
    })
    _write_case(str(tmp_path / "match_init_b.nc"), init_b, {
        ("mem_1", 0.0): [_record("hit", 1, area_km2=5.0), _record("miss", 2, area_km2=5.0)],
        ("mem_1", 1.0): [_record("hit", 3, area_km2=5.0)],
        ("mem_2", 0.0): [_record("hit", 5, area_km2=999.0)],
        ("mem_2", 1.0): [_record("hit", 6, area_km2=999.0)],
    })

    clim = build_object_property_pod_climatology(str(tmp_path), bin_edges={"area": np.array([0.0, 10.0])})
    print(f"\n[obj-prop-pod-check5] n_hits={clim.n_hits['area']}, n_misses={clim.n_misses['area']}")
    assert clim.n_hits["area"][(0, 0.0)] == 2  # 1 from each case
    assert clim.n_misses["area"][(0, 0.0)] == 1
