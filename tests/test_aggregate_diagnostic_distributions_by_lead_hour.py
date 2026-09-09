"""Synthetic-fixture validation of
climatology.aggregate_diagnostic_distributions_by_lead_hour -- hand-computed
expected area_ratio/intensity-diff/centroid-dist/ti_score values, confirms
only "hit" category records are pooled (miss/false_alarm/truth_extra/
forecast_extra excluded), and confirms cross-case pooling.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_aggregate_diagnostic_distributions_by_lead_hour.py -v -s
"""

from datetime import datetime, timedelta

from python_obj.obj_core import MatchRecord, MatchResult, write_match_file
from py_obj_plotting.climatology import (
    build_diagnostic_distribution_climatology,
    build_diagnostic_distribution_climatology_by_storm_mode,
)
from py_obj_plotting.climatology.aggregate_diagnostic_distributions_by_lead_hour import _haversine_km


def _hit_record(rid, truth_area, forecast_area, truth_maxint, forecast_maxint,
                 truth_meanint, forecast_meanint, truth_lat, truth_lon, forecast_lat, forecast_lon, ti,
                 truth_is_linear=None) -> MatchRecord:
    return MatchRecord(
        category="hit", truth_id=rid, forecast_id=rid, ti_score=ti,
        truth_area_km2=truth_area, forecast_area_km2=forecast_area,
        truth_max_intensity=truth_maxint, forecast_max_intensity=forecast_maxint,
        truth_mean_intensity=truth_meanint, forecast_mean_intensity=forecast_meanint,
        truth_centroid_lat=truth_lat, truth_centroid_lon=truth_lon,
        forecast_centroid_lat=forecast_lat, forecast_centroid_lon=forecast_lon,
        truth_is_linear=truth_is_linear,
    )


def _non_hit_record(category: str, rid: int) -> MatchRecord:
    kwargs = {"category": category, "ti_score": 0.0}
    if category in ("miss", "truth_extra"):
        kwargs["truth_id"], kwargs["forecast_id"] = rid, -1
    else:
        kwargs["truth_id"], kwargs["forecast_id"] = -1, rid
    return MatchRecord(**kwargs)


def _write_case(path: str, init_time: datetime, member_records: dict) -> None:
    """member_records: {(member_id, lead_hour): [MatchRecord, ...]}"""
    results = [
        MatchResult(records=records, valid_time=init_time + timedelta(hours=lh), member_id=member_id)
        for (member_id, lh), records in member_records.items()
    ]
    write_match_file(
        path, results, n_truth_source_files=1, n_forecast_source_files=1,
        max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2, init_time=init_time,
    )


def test_only_hit_category_pooled_hand_computed_values(tmp_path):
    init_time = datetime(2023, 5, 1, 0, 0, 0)

    # mem_1 @ lead=0: one real hit (known area/intensity/centroid values) +
    # one miss/false_alarm/truth_extra/forecast_extra that must be excluded
    hit = _hit_record(
        1, truth_area=10.0, forecast_area=15.0,
        truth_maxint=50.0, forecast_maxint=55.0,
        truth_meanint=40.0, forecast_meanint=38.0,
        truth_lat=35.0, truth_lon=-97.0, forecast_lat=35.0, forecast_lon=-96.9,
        ti=0.75,
    )
    excluded = [
        _non_hit_record("miss", 2), _non_hit_record("false_alarm", 3),
        _non_hit_record("truth_extra", 4), _non_hit_record("forecast_extra", 5),
    ]
    # mem_1 @ lead=1: needed so write_match_file gets a real time dimension (>1 distinct valid_time)
    hit_lead1 = _hit_record(
        6, truth_area=20.0, forecast_area=20.0,
        truth_maxint=60.0, forecast_maxint=60.0,
        truth_meanint=45.0, forecast_meanint=45.0,
        truth_lat=36.0, truth_lon=-98.0, forecast_lat=36.0, forecast_lon=-98.0,
        ti=1.0,
    )

    _write_case(str(tmp_path / "match_init_a.nc"), init_time, {
        ("mem_1", 0): [hit] + excluded,
        ("mem_1", 1): [hit_lead1],
        ("mem_2", 0): [],
        ("mem_2", 1): [],
    })

    clim = build_diagnostic_distribution_climatology(str(tmp_path))
    print(f"\n[diag-clim-check1] lead_hours={clim.lead_hours}")
    assert clim.lead_hours == [0.0, 1.0]

    # only the 1 real hit at lead=0 should be pooled -- excluded records must not appear
    assert clim.n_hits_per_lead_hour[0.0] == 1
    assert len(clim.area_ratio[0.0]) == 1

    expected_area_ratio = 15.0 / 10.0
    expected_max_intensity_diff = 55.0 - 50.0
    expected_mean_intensity_diff = 38.0 - 40.0
    expected_centroid_dist = _haversine_km(35.0, -97.0, 35.0, -96.9)

    assert abs(clim.area_ratio[0.0][0] - expected_area_ratio) < 1e-9
    assert abs(clim.max_intensity_diff[0.0][0] - expected_max_intensity_diff) < 1e-9
    assert abs(clim.mean_intensity_diff[0.0][0] - expected_mean_intensity_diff) < 1e-9
    assert abs(clim.centroid_dist_km[0.0][0] - expected_centroid_dist) < 1e-6
    assert abs(clim.ti_score[0.0][0] - 0.75) < 1e-9
    print(f"[diag-clim-check1] area_ratio={clim.area_ratio[0.0][0]:.3f} (expected {expected_area_ratio:.3f}), "
          f"centroid_dist_km={clim.centroid_dist_km[0.0][0]:.3f}")


def test_cross_case_pooling(tmp_path):
    init_a = datetime(2023, 5, 1, 0, 0, 0)
    init_b = datetime(2023, 5, 2, 0, 0, 0)

    def _simple_hit(rid, ti):
        return _hit_record(rid, 10.0, 10.0, 50.0, 50.0, 40.0, 40.0, 35.0, -97.0, 35.0, -97.0, ti)

    _write_case(str(tmp_path / "match_init_a.nc"), init_a, {
        ("mem_1", 0): [_simple_hit(1, 0.5)],
        ("mem_1", 1): [_simple_hit(2, 0.5)],
        ("mem_2", 0): [_simple_hit(3, 0.5)],
        ("mem_2", 1): [_simple_hit(4, 0.5)],
    })
    _write_case(str(tmp_path / "match_init_b.nc"), init_b, {
        ("mem_1", 0): [_simple_hit(5, 0.9), _simple_hit(6, 0.9)],
        ("mem_1", 1): [_simple_hit(7, 0.9)],
        ("mem_2", 0): [_simple_hit(8, 0.9)],
        ("mem_2", 1): [_simple_hit(9, 0.9)],
    })

    clim = build_diagnostic_distribution_climatology(str(tmp_path))
    # lead=0: case A contributes 2 (mem_1 + mem_2), case B contributes 3 (2 + 1) -> 5 total, pooled across members
    print(f"\n[diag-clim-check2] ti_score@lead0={clim.ti_score[0.0]}")
    assert clim.n_hits_per_lead_hour[0.0] == 5
    assert sorted(clim.ti_score[0.0]) == [0.5, 0.5, 0.9, 0.9, 0.9]


def test_storm_mode_hand_computed_split(tmp_path):
    init_time = datetime(2023, 5, 1, 0, 0, 0)

    def _hit(rid, ti, mode):
        return _hit_record(rid, 10.0, 12.0, 50.0, 52.0, 40.0, 41.0, 35.0, -97.0, 35.0, -97.0, ti, truth_is_linear=mode)

    _write_case(str(tmp_path / "match_init_a.nc"), init_time, {
        ("mem_1", 0): [_hit(1, 0.5, 0), _hit(2, 0.6, 0), _hit(3, 0.9, 2)],
        ("mem_1", 1): [_hit(4, 0.7, 1)],
    })

    cellular = build_diagnostic_distribution_climatology_by_storm_mode(str(tmp_path), storm_mode=0)
    mixed = build_diagnostic_distribution_climatology_by_storm_mode(str(tmp_path), storm_mode=1)
    linear = build_diagnostic_distribution_climatology_by_storm_mode(str(tmp_path), storm_mode=2)

    print(f"\n[diag-storm-mode-check1] cellular_ti={cellular.ti_score}, mixed_ti={mixed.ti_score}, linear_ti={linear.ti_score}")
    assert sorted(cellular.ti_score[0.0]) == [0.5, 0.6]
    assert cellular.n_hits_per_lead_hour[0.0] == 2
    assert linear.ti_score[0.0] == [0.9]
    assert mixed.ti_score[1.0] == [0.7]
    assert 1.0 not in cellular.ti_score  # no cellular hits at lead=1.0


def test_storm_mode_diagnostics_sum_to_unfiltered_pooled_totals(tmp_path):
    """Regression proof mirroring the contingency-table storm-mode split:
    cellular+mixed+linear hit counts at every lead hour must sum exactly
    to the unfiltered build_diagnostic_distribution_climatology() totals."""
    init_time = datetime(2023, 5, 1, 0, 0, 0)
    modes = [0, 1, 2, 0, 1, 2, 0]

    def _hit(rid, mode):
        return _hit_record(rid, 10.0, 12.0, 50.0, 52.0, 40.0, 41.0, 35.0, -97.0, 35.0, -97.0, 0.8, truth_is_linear=mode)

    _write_case(str(tmp_path / "match_init_a.nc"), init_time, {
        ("mem_1", 0): [_hit(i, m) for i, m in enumerate(modes)],
        ("mem_1", 1): [_hit(100, 0)],
    })

    unfiltered = build_diagnostic_distribution_climatology(str(tmp_path))
    by_mode = [build_diagnostic_distribution_climatology_by_storm_mode(str(tmp_path), storm_mode=m) for m in (0, 1, 2)]

    for lh in unfiltered.lead_hours:
        summed = sum(c.n_hits_per_lead_hour.get(lh, 0) for c in by_mode)
        assert summed == unfiltered.n_hits_per_lead_hour[lh]

    total_unfiltered = sum(unfiltered.n_hits_per_lead_hour.values())
    total_by_mode = sum(sum(c.n_hits_per_lead_hour.values()) for c in by_mode)
    print(f"\n[diag-storm-mode-check2] total_unfiltered={total_unfiltered}, total_by_mode={total_by_mode}")
    assert total_unfiltered == total_by_mode
    assert total_unfiltered > 0


def test_storm_mode_missing_data_degrades_to_empty_not_error(tmp_path):
    init_time = datetime(2023, 5, 1, 0, 0, 0)

    def _simple_hit(rid, ti):
        return _hit_record(rid, 10.0, 10.0, 50.0, 50.0, 40.0, 40.0, 35.0, -97.0, 35.0, -97.0, ti)  # no truth_is_linear at all

    _write_case(str(tmp_path / "match_init_a.nc"), init_time, {
        ("mem_1", 0): [_simple_hit(1, 0.5)],
        ("mem_1", 1): [_simple_hit(2, 0.5)],
    })
    cellular = build_diagnostic_distribution_climatology_by_storm_mode(str(tmp_path), storm_mode=0)
    print(f"\n[diag-storm-mode-check3] cellular_ti={cellular.ti_score}")
    assert cellular.n_hits_per_lead_hour == {}
