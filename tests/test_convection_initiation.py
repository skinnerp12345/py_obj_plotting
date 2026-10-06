"""Synthetic-fixture validation of climatology.aggregate_ci_skill, with
hand-computed expected CI hit/miss/false_alarm counts -- confirms the CI
candidate masks (age_seconds < lead_seconds on each side independently), the
age-difference-threshold reclassification of "hit" rows into CI miss +
CI false alarm, exclusion of non-CI-candidate objects, and that a record
missing forecast_age_seconds (forecast series untracked) degrades to a
truth-side CI miss rather than a silently-wrong classification.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_convection_initiation.py -v -s
"""

from datetime import datetime, timedelta

import numpy as np

from py_obj.obj_core import MatchRecord, MatchResult, write_match_file
from py_obj_plotting.climatology.aggregate_ci_skill import (
    build_ci_age_difference_distribution,
    build_ci_skill_climatology,
)
from py_obj_plotting.climatology.contingency_stats import compute_pod

_INIT = datetime(2023, 5, 1, 0, 0, 0)


def _record(category: str, rid: int, truth_age_seconds=None, forecast_age_seconds=None) -> MatchRecord:
    """truth_age_seconds is only ever meaningful (both in this fixture and
    in real py_obj output) for a row with a real truth object
    (hit/miss/truth_extra); forecast_age_seconds only for a row with a real
    forecast object (hit/false_alarm/forecast_extra) -- matches
    py_obj.obj_core.matching's own MatchRecord population rules exactly."""
    kwargs = dict(category=category, truth_id=rid, forecast_id=rid, ti_score=0.9 if category == "hit" else 0.0)
    if category in ("hit", "miss", "truth_extra"):
        kwargs["truth_age_seconds"] = truth_age_seconds
    if category in ("hit", "false_alarm", "forecast_extra"):
        kwargs["forecast_age_seconds"] = forecast_age_seconds
    return MatchRecord(**kwargs)


def _write_case(path: str, member_records: dict, init_time: datetime = _INIT) -> None:
    """member_records: {(member_id, lead_hour): [MatchRecord, ...]}"""
    results = [
        MatchResult(records=records, valid_time=init_time + timedelta(hours=lead_hour), member_id=member_id)
        for (member_id, lead_hour), records in member_records.items()
    ]
    write_match_file(
        path, results, n_truth_source_files=1, n_forecast_source_files=1,
        max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2, init_time=init_time,
    )


def test_hit_within_threshold_is_a_ci_hit(tmp_path):
    """lead_hour=1.0 (lead_seconds=3600): truth_age=300s (< 3600, CI
    candidate), forecast_age=900s (< 3600, CI candidate too), age diff
    600s <= a 1200s (20 min) threshold -> CI hit."""
    _write_case(str(tmp_path / "match_init_a.nc"), {
        ("mem_1", 1.0): [_record("hit", 1, truth_age_seconds=300.0, forecast_age_seconds=900.0)],
        ("mem_2", 1.0): [_record("hit", 2, truth_age_seconds=300.0, forecast_age_seconds=900.0)],
    })
    clim = build_ci_skill_climatology(str(tmp_path), age_diff_threshold_minutes=20.0)
    print(f"\n[ci-check1] n_hits={clim.n_hits}, n_misses={clim.n_misses}, n_false_alarms={clim.n_false_alarms}")
    assert clim.n_hits[("mem_1", 1.0)] == 1
    assert clim.n_misses.get(("mem_1", 1.0), 0) == 0
    assert clim.n_false_alarms.get(("mem_1", 1.0), 0) == 0


def test_hit_outside_threshold_demoted_to_ci_miss_and_ci_false_alarm(tmp_path):
    """Same CI-candidate setup as above, but age diff 2000s exceeds the
    1200s threshold -- the spatial hit is real, but temporally wrong, so it
    counts as BOTH a truth-side CI miss AND a forecast-side CI false alarm
    (not a single shared label) -- see module docstring."""
    _write_case(str(tmp_path / "match_init_a.nc"), {
        ("mem_1", 1.0): [_record("hit", 1, truth_age_seconds=300.0, forecast_age_seconds=2300.0)],
        ("mem_2", 1.0): [_record("hit", 2, truth_age_seconds=300.0, forecast_age_seconds=2300.0)],
    })
    clim = build_ci_skill_climatology(str(tmp_path), age_diff_threshold_minutes=20.0)
    print(f"\n[ci-check2] n_hits={clim.n_hits}, n_misses={clim.n_misses}, n_false_alarms={clim.n_false_alarms}")
    assert clim.n_hits.get(("mem_1", 1.0), 0) == 0
    assert clim.n_misses[("mem_1", 1.0)] == 1
    assert clim.n_false_alarms[("mem_1", 1.0)] == 1


def test_plain_miss_of_a_young_truth_object_is_a_ci_miss(tmp_path):
    _write_case(str(tmp_path / "match_init_a.nc"), {
        ("mem_1", 1.0): [_record("miss", 1, truth_age_seconds=100.0)],
        ("mem_2", 1.0): [_record("miss", 2, truth_age_seconds=100.0)],
    })
    clim = build_ci_skill_climatology(str(tmp_path), age_diff_threshold_minutes=20.0)
    print(f"\n[ci-check3] n_misses={clim.n_misses}")
    assert clim.n_misses[("mem_1", 1.0)] == 1
    assert clim.n_hits.get(("mem_1", 1.0), 0) == 0
    assert clim.n_false_alarms.get(("mem_1", 1.0), 0) == 0


def test_plain_false_alarm_of_a_young_forecast_object_is_a_ci_false_alarm(tmp_path):
    _write_case(str(tmp_path / "match_init_a.nc"), {
        ("mem_1", 1.0): [_record("false_alarm", 1, forecast_age_seconds=200.0)],
        ("mem_2", 1.0): [_record("false_alarm", 2, forecast_age_seconds=200.0)],
    })
    clim = build_ci_skill_climatology(str(tmp_path), age_diff_threshold_minutes=20.0)
    print(f"\n[ci-check4] n_false_alarms={clim.n_false_alarms}")
    assert clim.n_false_alarms[("mem_1", 1.0)] == 1
    assert clim.n_hits.get(("mem_1", 1.0), 0) == 0
    assert clim.n_misses.get(("mem_1", 1.0), 0) == 0


def test_non_ci_object_excluded_entirely(tmp_path):
    """Both sides' age (5000s) exceeds lead_seconds (3600s at lead_hour=1.0)
    -- this is ordinary, pre-existing convection on both sides, not a CI
    event at all -- must not appear in n_hits/n_misses/n_false_alarms."""
    _write_case(str(tmp_path / "match_init_a.nc"), {
        ("mem_1", 1.0): [_record("hit", 1, truth_age_seconds=5000.0, forecast_age_seconds=5000.0)],
        ("mem_2", 1.0): [_record("hit", 2, truth_age_seconds=5000.0, forecast_age_seconds=5000.0)],
    })
    clim = build_ci_skill_climatology(str(tmp_path), age_diff_threshold_minutes=20.0)
    print(f"\n[ci-check5] n_hits={clim.n_hits}, n_misses={clim.n_misses}, n_false_alarms={clim.n_false_alarms}")
    assert clim.n_hits.get(("mem_1", 1.0), 0) == 0
    assert clim.n_misses.get(("mem_1", 1.0), 0) == 0
    assert clim.n_false_alarms.get(("mem_1", 1.0), 0) == 0


def test_untracked_forecast_side_degrades_to_truth_side_ci_miss_only(tmp_path):
    """truth_age=300s is a genuine CI candidate, but forecast_age_seconds is
    None (forecast series wasn't tracked at identification time) -- must
    NOT silently count as a CI hit (we can't verify timing) and must NOT be
    double-counted as a CI false alarm on the forecast-CI population either
    (we don't know the forecast object's own age, so it can't be judged a
    CI candidate at all) -- only a truth-side CI miss is recorded."""
    _write_case(str(tmp_path / "match_init_a.nc"), {
        ("mem_1", 1.0): [_record("hit", 1, truth_age_seconds=300.0, forecast_age_seconds=None)],
        ("mem_2", 1.0): [_record("hit", 2, truth_age_seconds=300.0, forecast_age_seconds=None)],
    })
    clim = build_ci_skill_climatology(str(tmp_path), age_diff_threshold_minutes=20.0)
    print(f"\n[ci-check6] n_hits={clim.n_hits}, n_misses={clim.n_misses}, n_false_alarms={clim.n_false_alarms}")
    assert clim.n_hits.get(("mem_1", 1.0), 0) == 0
    assert clim.n_misses[("mem_1", 1.0)] == 1
    assert clim.n_false_alarms.get(("mem_1", 1.0), 0) == 0


def test_include_near_miss_candidates_folds_truth_extra_and_forecast_extra(tmp_path):
    """With include_near_miss_candidates=True, a young truth_extra/
    forecast_extra row (a contested near-miss, per the standard matching
    algorithm) is folded into n_truth_extra/n_forecast_extra -- never into
    hits/misses/false_alarms, mirroring the standard schema's own treatment
    -- and excluded entirely (climatology unaffected) when the flag is left
    at its default False."""
    _write_case(str(tmp_path / "match_init_a.nc"), {
        ("mem_1", 1.0): [
            _record("truth_extra", 1, truth_age_seconds=100.0),
            _record("forecast_extra", 2, forecast_age_seconds=100.0),
        ],
        ("mem_2", 1.0): [
            _record("truth_extra", 3, truth_age_seconds=100.0),
            _record("forecast_extra", 4, forecast_age_seconds=100.0),
        ],
    })
    clim_excluded = build_ci_skill_climatology(str(tmp_path), age_diff_threshold_minutes=20.0)
    print(f"\n[ci-check7] excluded: n_truth_extra={clim_excluded.n_truth_extra}, n_forecast_extra={clim_excluded.n_forecast_extra}")
    assert clim_excluded.n_truth_extra.get(("mem_1", 1.0), 0) == 0
    assert clim_excluded.n_forecast_extra.get(("mem_1", 1.0), 0) == 0

    clim_included = build_ci_skill_climatology(str(tmp_path), age_diff_threshold_minutes=20.0, include_near_miss_candidates=True)
    print(f"[ci-check7] included: n_truth_extra={clim_included.n_truth_extra}, n_forecast_extra={clim_included.n_forecast_extra}")
    assert clim_included.n_truth_extra[("mem_1", 1.0)] == 1
    assert clim_included.n_forecast_extra[("mem_1", 1.0)] == 1
    assert clim_included.n_hits.get(("mem_1", 1.0), 0) == 0
    assert clim_included.n_misses.get(("mem_1", 1.0), 0) == 0
    assert clim_included.n_false_alarms.get(("mem_1", 1.0), 0) == 0


def test_pooled_pod_across_two_lead_hours(tmp_path):
    """Structural check: lead_hours/members discovered correctly, and the
    resulting climatology flows into compute_pod() (the SAME pure function
    every other POD in this codebase uses) with a hand-verified value."""
    _write_case(str(tmp_path / "match_init_a.nc"), {
        ("mem_1", 1.0): [
            _record("hit", 1, truth_age_seconds=300.0, forecast_age_seconds=400.0),
            _record("miss", 2, truth_age_seconds=100.0),
        ],
        ("mem_1", 2.0): [_record("hit", 3, truth_age_seconds=300.0, forecast_age_seconds=400.0)],
        ("mem_2", 1.0): [_record("hit", 4, truth_age_seconds=300.0, forecast_age_seconds=400.0)],
        ("mem_2", 2.0): [_record("hit", 5, truth_age_seconds=300.0, forecast_age_seconds=400.0)],
    })
    clim = build_ci_skill_climatology(str(tmp_path), age_diff_threshold_minutes=20.0)
    print(f"\n[ci-check8] lead_hours={clim.lead_hours}, members={clim.members}")
    assert clim.lead_hours == [1.0, 2.0]
    assert clim.members == ["mem_1", "mem_2"]
    # lead_hour=1.0: 1 hit + 1 miss for mem_1 -> POD 0.5
    assert compute_pod(clim.n_hits.get(("mem_1", 1.0), 0), clim.n_misses.get(("mem_1", 1.0), 0)) == 0.5
    # lead_hour=2.0: 1 hit, 0 misses -> POD 1.0
    assert compute_pod(clim.n_hits.get(("mem_1", 2.0), 0), clim.n_misses.get(("mem_1", 2.0), 0)) == 1.0


def test_age_difference_distribution_hand_computed(tmp_path):
    """build_ci_age_difference_distribution pools truth_age - forecast_age
    (observed - forecast; positive = model late) over every BOTH-sides-CI-candidate hit, at any magnitude (not gated by
    a threshold) -- confirms the non-CI object and the threshold-failing
    pair from earlier checks are both still included here (unlike
    build_ci_skill_climatology, which excludes/reclassifies them), since
    this function's whole purpose is showing the FULL early/late
    distribution, not the post-threshold contingency counts."""
    _write_case(str(tmp_path / "match_init_a.nc"), {
        ("mem_1", 1.0): [
            _record("hit", 1, truth_age_seconds=300.0, forecast_age_seconds=900.0),  # diff=-600 (model early), within CI test1
            _record("hit", 2, truth_age_seconds=300.0, forecast_age_seconds=2300.0),  # diff=-2000, still both-CI (test2)
            _record("hit", 3, truth_age_seconds=5000.0, forecast_age_seconds=5000.0),  # neither side a CI candidate
            _record("miss", 4, truth_age_seconds=100.0),  # no forecast object at all -- excluded (not a "hit" row)
        ],
        ("mem_2", 1.0): [_record("hit", 5, truth_age_seconds=999.0, forecast_age_seconds=999.0)],
    })
    dist = build_ci_age_difference_distribution(str(tmp_path))
    print(f"\n[ci-check9] age_diff_seconds={sorted(dist.age_diff_seconds.tolist())}")
    assert sorted(dist.age_diff_seconds.tolist()) == [-2000.0, -600.0, 0.0]


def test_age_difference_sign_positive_means_model_late(tmp_path):
    """Ages count up from first detection: a forecast object YOUNGER than
    its observed match formed later, so observed - forecast > 0 = model late."""
    _write_case(str(tmp_path / "match_init_a.nc"), {
        ("mem_1", 1.0): [_record("hit", 1, truth_age_seconds=1800.0, forecast_age_seconds=600.0)],  # model 20 min late
        ("mem_2", 1.0): [_record("hit", 2, truth_age_seconds=600.0, forecast_age_seconds=1800.0)],  # model 20 min early
    })
    dist = build_ci_age_difference_distribution(str(tmp_path))
    assert sorted(dist.age_diff_seconds.tolist()) == [-1200.0, 1200.0]


# --- Float-boundary regression: lead_hour * 3600 overshoots at some leads ---
# (e.g. 125 min: (7500/3600)*3600 == 7500.000000000001), which used to let an
# object present since init (age == lead exactly) pass "age < lead".
_LEAD_125_MIN_H = 125.0 / 60.0


def test_object_present_since_init_is_not_ci_at_overshooting_lead(tmp_path):
    assert (7500.0 / 3600.0) * 3600.0 > 7500.0  # precondition: this lead really overshoots in float math
    _write_case(str(tmp_path / "match_init_a.nc"), {
        # forecast object carried over from init (age == lead) matched to a young observed storm
        ("mem_1", _LEAD_125_MIN_H): [_record("hit", 1, truth_age_seconds=300.0, forecast_age_seconds=7500.0)],
        ("mem_2", _LEAD_125_MIN_H): [_record("hit", 2, truth_age_seconds=300.0, forecast_age_seconds=7500.0)],
    })
    dist = build_ci_age_difference_distribution(str(tmp_path))
    assert len(dist.age_diff_seconds) == 0  # forecast side is not a CI candidate -> no both-CI pair

    clim = build_ci_skill_climatology(str(tmp_path), age_diff_threshold_minutes=1000.0)
    lh = clim.lead_hours[0]
    # truth side IS a CI candidate and its hit fails the joint test -> CI miss; forecast side contributes nothing
    assert clim.n_hits.get(("mem_1", lh), 0) == 0
    assert clim.n_misses.get(("mem_1", lh), 0) == 1
    assert clim.n_false_alarms.get(("mem_1", lh), 0) == 0


def test_no_files_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        build_ci_skill_climatology(str(tmp_path), age_diff_threshold_minutes=20.0)


def test_count_tracking_restarts_flags_all_age_zero_slices_only(tmp_path):
    """A post-init slice where every truth object (>= 3) has age 0 is a likely
    truth-tracking restart; a mixed-age slice, a slice with < 3 objects, and
    the init-time slice are not flagged. Report only -- counts, no correction."""
    from py_obj_plotting.climatology import count_tracking_restarts
    _write_case(str(tmp_path / "match_init_a.nc"), {
        ("mem_1", 0.0): [_record("miss", i, truth_age_seconds=0.0) for i in range(1, 5)],  # init slice: not flagged
        ("mem_1", 1.0): [_record("miss", i, truth_age_seconds=0.0) for i in range(1, 5)],  # restart: 4 objects all age 0
        ("mem_1", 2.0): [_record("miss", 1, truth_age_seconds=0.0), _record("miss", 2, truth_age_seconds=3600.0),
                         _record("miss", 3, truth_age_seconds=0.0)],                       # mixed ages: not flagged
        ("mem_1", 3.0): [_record("miss", 1, truth_age_seconds=0.0), _record("miss", 2, truth_age_seconds=0.0)],  # < 3 objects
        ("mem_2", 1.0): [_record("miss", i, truth_age_seconds=0.0) for i in range(1, 5)],  # same truth, other member
    })
    r = count_tracking_restarts(str(tmp_path))
    print(f"\n[ci-restart] {r}")
    assert r.n_flagged_slices == 1
    assert r.n_truth_objects_reset == 4  # one member's rows only, not 4 x 2 members
    assert r.flagged_valid_times == [_INIT + timedelta(hours=1)]
