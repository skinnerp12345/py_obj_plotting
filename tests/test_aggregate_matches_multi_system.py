"""Synthetic-fixture validation of climatology.aggregate_matches_multi_system,
with hand-computed expected results -- confirms the strict "intersection"
joint-availability rule (a (cycle, lead) combination is only accumulated if
every system has it), the "union" mode (each system's own presence suffices
independently), that each system's own `lead_hours` list only ever contains
lead times it actually has data for (never a spurious zero-count entry for
an excluded lead time), and that a system with NO member dimension at all
(a real, confirmed case: NowcastNet's real match files have no member_id
variable or attribute whatsoever) is handled as one implicit member rather
than raising.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_aggregate_matches_multi_system.py -v -s
"""

from datetime import datetime, timedelta

import pytest

from python_obj.obj_core import MatchRecord, MatchResult, write_match_file
from py_obj_plotting.climatology.aggregate_matches_multi_system import (
    build_multi_system_diagnostic_climatology,
    build_multi_system_match_climatology,
    build_multi_system_object_climatology,
    build_multi_system_object_climatology_by_storm_mode,
)
from py_obj_plotting.systems import SystemSpec


def _record(category: str, rid: int) -> MatchRecord:
    kwargs = dict(category=category, truth_id=rid, forecast_id=rid, ti_score=0.9 if category == "hit" else 0.0)
    if category == "hit":
        kwargs.update(
            centroid_dist_km=5.0, truth_area_km2=100.0, forecast_area_km2=110.0,
            truth_max_intensity=50.0, forecast_max_intensity=52.0,
            truth_mean_intensity=45.0, forecast_mean_intensity=46.0,
            truth_centroid_lat=35.0, truth_centroid_lon=-97.0,
            forecast_centroid_lat=35.05, forecast_centroid_lon=-97.05,
        )
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
        results.append(MatchResult(
            records=records, valid_time=init_time + timedelta(minutes=lead_min), member_id=member_id,
        ))
    write_match_file(
        path, results, n_truth_source_files=1, n_forecast_source_files=1,
        max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2, init_time=init_time,
    )


@pytest.fixture
def three_system_dirs(tmp_path):
    a_dir, b_dir, c_dir = tmp_path / "A", tmp_path / "B", tmp_path / "C"
    for d in (a_dir, b_dir, c_dir):
        d.mkdir()
    init_x = datetime(2026, 5, 1, 17, 0, 0)

    # System A (ensemble, like WoFS): full coverage at every requested lead minute
    _write_case(str(a_dir / "match_init_x.nc"), init_x, {
        ("mem_1", 0): ["hit", "hit"], ("mem_1", 10): ["hit"], ("mem_1", 20): ["hit"], ("mem_1", 30): ["hit", "miss"],
        ("mem_2", 0): ["false_alarm"], ("mem_2", 10): ["hit"], ("mem_2", 20): ["hit"], ("mem_2", 30): ["hit"],
    })
    # System B (ensemble, like WoFSCast): missing lead=0 and lead=10 entirely
    # (a real, confirmed startup-availability gap) -- present at 20, 30.
    _write_case(str(b_dir / "match_init_x.nc"), init_x, {
        ("mem_1", 20): ["hit"], ("mem_1", 30): ["hit", "hit"],
        ("mem_2", 20): ["miss"], ("mem_2", 30): ["hit"],
    })
    # System C (deterministic, like NowcastNet): NO member dimension at all
    # (member_id=None for every record) -- present at 0, 10, 20, but NOT 30
    # (simulating a real per-cycle gap distinct from the relaxed-lead-minute case).
    _write_case(str(c_dir / "match_init_x.nc"), init_x, {
        (None, 0): ["hit"], (None, 10): ["hit", "miss"], (None, 20): ["hit"],
    })

    return [SystemSpec("A", str(a_dir)), SystemSpec("B", str(b_dir)), SystemSpec("C", str(c_dir))]


def test_strict_lead_requires_all_three_systems(three_system_dirs):
    clim, report = build_multi_system_match_climatology(
        three_system_dirs, lead_time_mode="intersection", lead_minutes=[20, 30],
    )
    # lead=20: all 3 systems present -> included for all 3
    # lead=30: A and B present, C absent -> excluded for ALL (strict rule), even for A/B
    print(f"\n[multi-match-check1] A.lead_hours={clim['A'].lead_hours}, "
          f"B.lead_hours={clim['B'].lead_hours}, C.lead_hours={clim['C'].lead_hours}")
    assert 20 / 60.0 in clim["A"].lead_hours
    assert 30 / 60.0 not in clim["A"].lead_hours  # excluded despite A having real data there
    assert 30 / 60.0 not in clim["B"].lead_hours
    assert (init_time_key := list(report.excluded_partial_lead_minutes.keys())[0])
    print(f"[multi-match-check1] excluded_partial_lead_minutes: {report.excluded_partial_lead_minutes}")
    assert report.excluded_partial_lead_minutes[init_time_key] == ["C"]


def test_union_mode_uses_each_systems_own_presence(three_system_dirs):
    clim, _ = build_multi_system_match_climatology(
        three_system_dirs, lead_time_mode="union", lead_minutes=[0, 10],
    )
    # Under union mode, A and C both have real data at lead=0/10 and should
    # be included independently, even though B has none at all.
    assert 0.0 in clim["A"].lead_hours
    assert (10 / 60.0) in clim["A"].lead_hours
    assert 0.0 in clim["C"].lead_hours
    assert (10 / 60.0) in clim["C"].lead_hours
    # B genuinely has no data at lead=0/10 -- correctly absent, not zero-filled
    assert 0.0 not in clim["B"].lead_hours
    assert (10 / 60.0) not in clim["B"].lead_hours
    print(f"\n[multi-match-check2] B.lead_hours={clim['B'].lead_hours} (correctly excludes 0, 10 min)")


def test_hand_computed_counts_at_shared_lead(three_system_dirs):
    clim, _ = build_multi_system_match_climatology(
        three_system_dirs, lead_time_mode="intersection", lead_minutes=[20],
    )
    lh20 = 20 / 60.0
    assert clim["A"].n_hits[("mem_1", lh20)] == 1
    assert clim["B"].n_hits[("mem_1", lh20)] == 1
    assert clim["B"].n_misses[("mem_2", lh20)] == 1
    assert clim["C"].n_hits[(None, lh20)] == 1
    print(f"\n[multi-match-check3] hand-computed counts at lead=20min confirmed for all 3 systems")


def test_deterministic_system_no_member_dimension_handled(three_system_dirs):
    """System C has no member_id at all in its real match file (confirmed
    real behavior for NowcastNet) -- must not raise, and must be treated as
    one implicit member (member_id=None)."""
    clim, _ = build_multi_system_match_climatology(
        three_system_dirs, lead_time_mode="intersection", lead_minutes=[20],
    )
    assert clim["C"].members == [None]
    print(f"\n[multi-match-check4] deterministic system C members={clim['C'].members}, no exception raised")


def test_diagnostic_climatology_respects_same_filtering(three_system_dirs):
    clim, _ = build_multi_system_diagnostic_climatology(
        three_system_dirs, lead_time_mode="intersection", lead_minutes=[20, 30],
    )
    lh20 = 20 / 60.0
    # lead=30 excluded for A/B (C absent, strict rule) -- must not appear
    assert 30 / 60.0 not in clim["A"].lead_hours
    # lead=20 included for all three, with real hit-pair diagnostic values pooled
    assert lh20 in clim["A"].ti_score
    assert lh20 in clim["C"].ti_score
    # A has 2 members, both with 1 hit at lead=20min -- diagnostics pool
    # across members (member-agnostic, matching this project's established
    # convention), so the correct count is 2, not 1.
    assert clim["A"].n_hits_per_lead_hour[lh20] == 2
    assert clim["C"].n_hits_per_lead_hour[lh20] == 1
    print(f"\n[multi-match-check5] diagnostic climatology correctly filtered; "
          f"A ti_score@20min={clim['A'].ti_score[lh20]}")


def test_object_climatology_respects_same_filtering_and_dedupes_truth_across_members(three_system_dirs):
    clim, _ = build_multi_system_object_climatology(
        three_system_dirs, lead_time_mode="intersection", lead_minutes=[20, 30],
    )
    lh20 = 20 / 60.0
    lh30 = 30 / 60.0
    # Same shared-availability filtering as every other multi-system
    # climatology: lead=30 excluded for A/B (C lacks it, strict rule).
    assert lh30 not in clim["A"].lead_hours
    assert lh30 not in clim["B"].lead_hours
    assert lh20 in clim["A"].lead_hours
    assert lh20 in clim["C"].lead_hours

    # A has 2 members at lead=20min, each with 1 "hit" -- forecast_count is
    # genuinely per-member (2 distinct forecast objects, one per member).
    assert clim["A"].forecast_count[("mem_1", lh20)] == 1
    assert clim["A"].forecast_count[("mem_2", lh20)] == 1
    # Truth-side count is de-duplicated across members (uses only one
    # representative member's own canonical rows), NOT summed across both
    # members' hit counts -- otherwise the same shared MRMS truth field would
    # be double-counted once per ensemble member.
    assert clim["A"].truth_count[lh20] == 1
    print(f"\n[object-clim-check1] A truth_count@20min={clim['A'].truth_count[lh20]} (deduplicated, not 2)")


def test_object_climatology_avoids_double_counting_candidate_rows(tmp_path):
    """Reproduces the real risk this module's masking exists to avoid: a
    truth_extra/forecast_extra row carries its OWN best near-miss
    candidate's full field set on the OTHER side (see
    python_obj/obj_core/matching.py's _record_from_objects), so a naive
    np.isfinite(...)-based mask would double-count an object that appears
    both on its own canonical row and as a candidate on a different object's
    extra row. One system, one case, 3 truth objects / 3 forecast objects:
      - truth #1 <-> forecast #1: "hit" (canonical for both)
      - truth #2 "wins" against forecast #3 becomes forecast #3's hit
        instead, so forecast #2 becomes "forecast_extra" -- its own row
        carries forecast #2's real data AND truth #2's data as the
        near-miss candidate (duplicating truth #2's own field values, which
        also appear on truth #2's own canonical "hit" row with forecast #3).
      - truth #2's canonical row: "hit" with forecast #3.
    """
    d = tmp_path / "D"
    d.mkdir()
    init_x = datetime(2026, 5, 1, 18, 0, 0)

    records = [
        MatchRecord(category="hit", truth_id=1, forecast_id=1, ti_score=0.9,
                    truth_area_km2=100.0, truth_mean_intensity=40.0,
                    forecast_area_km2=110.0, forecast_mean_intensity=42.0),
        # forecast #2's own canonical row -- ALSO carries truth #2's data as
        # its near-miss candidate (a real, confirmed match_io.py behavior).
        MatchRecord(category="forecast_extra", truth_id=2, forecast_id=2, ti_score=0.3,
                    truth_area_km2=200.0, truth_mean_intensity=45.0,
                    forecast_area_km2=50.0, forecast_mean_intensity=30.0),
        # truth #2's own canonical row, matched to a DIFFERENT forecast object.
        MatchRecord(category="hit", truth_id=2, forecast_id=3, ti_score=0.8,
                    truth_area_km2=200.0, truth_mean_intensity=45.0,
                    forecast_area_km2=80.0, forecast_mean_intensity=35.0),
    ]
    # A second, unrelated valid_time is included solely so write_match_file
    # emits a real dimensioned match_time_index variable (a file with only
    # one distinct valid_time uses a scalar attribute instead, which
    # _read_match_metadata/_read_object_stats_metadata don't accept) --
    # assertions below only ever look at lead=0.
    other_time_records = [
        MatchRecord(category="hit", truth_id=4, forecast_id=4, ti_score=0.9,
                    truth_area_km2=999.0, truth_mean_intensity=99.0,
                    forecast_area_km2=999.0, forecast_mean_intensity=99.0),
    ]
    write_match_file(
        str(d / "match_init_x.nc"), [
            MatchResult(records=records, valid_time=init_x, member_id=None),
            MatchResult(records=other_time_records, valid_time=init_x + timedelta(minutes=10), member_id=None),
        ],
        n_truth_source_files=1, n_forecast_source_files=1,
        max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2, init_time=init_x,
    )

    clim, _ = build_multi_system_object_climatology(
        [SystemSpec("D", str(d))], lead_time_mode="intersection", lead_minutes=[0],
    )
    lh0 = 0.0

    # Exactly 2 distinct truth objects (areas 100, 200) -- truth #2's area
    # must NOT appear twice (once from its own hit row, once from forecast
    # #2's forecast_extra row candidate slot).
    assert clim["D"].truth_count[lh0] == 2
    assert sorted(clim["D"].truth_area_values) == [100.0, 200.0]

    # Exactly 3 distinct forecast objects (areas 110, 50, 80) -- forecast #2
    # must NOT appear twice, and its own forecast_extra row IS counted once.
    assert clim["D"].forecast_count[(None, lh0)] == 3
    assert sorted(clim["D"].forecast_area_values) == [50.0, 80.0, 110.0]
    print(f"\n[object-clim-check2] no double-counting: "
          f"truth_area_values={sorted(clim['D'].truth_area_values)}, "
          f"forecast_area_values={sorted(clim['D'].forecast_area_values)}")


def test_object_climatology_pools_intensity_values_for_boxplot(three_system_dirs):
    clim, _ = build_multi_system_object_climatology(
        three_system_dirs, lead_time_mode="intersection", lead_minutes=[20],
    )
    # System C (deterministic) has 1 hit at lead=20min with truth/forecast
    # mean_intensity=45.0/46.0 (see _record()) -- confirms raw per-object
    # values (not just counts) are pooled into the flat lists the box-whisker
    # plots consume.
    assert 46.0 in clim["C"].forecast_mean_intensity_values
    assert 45.0 in clim["C"].truth_mean_intensity_values
    print(f"\n[object-clim-check3] C forecast_mean_intensity_values={clim['C'].forecast_mean_intensity_values}")


def test_object_climatology_falls_back_to_filename_init_time(tmp_path):
    """Regression test for a real bug found running build_multi_system_object_climatology
    against ncar_mpas_analysis/ for the first time: _read_object_stats_metadata
    had NO fallback to filename-parsed init_time at all (unlike its sibling
    _read_match_metadata), so it raised outright on every real file lacking
    the stored init_time attribute -- exactly what every ncar_mpas_analysis
    match file looks like (they predate that attribute). Reproduced here with
    a synthetic file written with init_time=None but a parseable filename."""
    d = tmp_path / "E"
    d.mkdir()
    init_x = datetime(2023, 4, 24, 0, 0, 0)
    records = [MatchRecord(category="hit", truth_id=1, forecast_id=1, ti_score=0.9,
                            truth_area_km2=100.0, truth_mean_intensity=40.0,
                            forecast_area_km2=110.0, forecast_mean_intensity=42.0)]
    other_time_records = [MatchRecord(category="hit", truth_id=2, forecast_id=2, ti_score=0.9,
                                       truth_area_km2=999.0, truth_mean_intensity=99.0,
                                       forecast_area_km2=999.0, forecast_mean_intensity=99.0)]
    write_match_file(
        str(d / f"match_init_{init_x:%Y%m%d_%H%M%S}.nc"), [
            MatchResult(records=records, valid_time=init_x, member_id=None),
            MatchResult(records=other_time_records, valid_time=init_x + timedelta(minutes=60), member_id=None),
        ],
        n_truth_source_files=1, n_forecast_source_files=1,
        max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2,
        init_time=None,  # deliberately absent -- must fall back to the filename
    )

    clim, _ = build_multi_system_object_climatology(
        [SystemSpec("E", str(d))], lead_time_mode="intersection", lead_minutes=[0],
    )
    print(f"\n[object-clim-check4] E.truth_count={clim['E'].truth_count}")
    assert clim["E"].truth_count[0.0] == 1


def test_diagnostic_and_object_climatology_handle_single_lead_time_files(tmp_path):
    """Regression test for a real gap found via the new driver's end-to-end
    tests: a match file with exactly ONE distinct valid_time has no
    match_time_index variable at all (write_match_file's own use_time_dim
    rule), but build_multi_system_diagnostic_climatology/
    build_multi_system_object_climatology both hard-required it (raising
    ValueError) unlike systems.py's load_system(), which already handles
    this case correctly. Every real sfe_poster/ncar_mpas_analysis file has
    many lead times, so this was never triggered by real data -- only by a
    deliberately minimal synthetic config (exactly the kind of small/quick
    comparison this redesign is meant to support robustly)."""
    d = tmp_path / "F"
    d.mkdir()
    t0 = datetime(2026, 5, 1, 19, 0, 0)
    records = [MatchRecord(category="hit", truth_id=1, forecast_id=1, ti_score=0.9,
                            truth_area_km2=100.0, truth_mean_intensity=40.0,
                            forecast_area_km2=110.0, forecast_mean_intensity=42.0,
                            truth_centroid_lat=35.0, truth_centroid_lon=-97.0,
                            forecast_centroid_lat=35.05, forecast_centroid_lon=-97.05)]
    write_match_file(
        str(d / "match_init_x.nc"),
        [MatchResult(records=records, valid_time=t0, member_id=None)],
        n_truth_source_files=1, n_forecast_source_files=1,
        max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2, init_time=t0,
    )

    diag_clim, _ = build_multi_system_diagnostic_climatology(
        [SystemSpec("F", str(d))], lead_time_mode="intersection", lead_minutes=[0],
    )
    print(f"\n[object-clim-check5] diag n_hits_per_lead_hour={diag_clim['F'].n_hits_per_lead_hour}")
    assert diag_clim["F"].n_hits_per_lead_hour[0.0] == 1

    obj_clim, _ = build_multi_system_object_climatology(
        [SystemSpec("F", str(d))], lead_time_mode="intersection", lead_minutes=[0],
    )
    print(f"[object-clim-check5] obj truth_count={obj_clim['F'].truth_count}")
    assert obj_clim["F"].truth_count[0.0] == 1


def test_object_climatology_by_storm_mode_hand_computed_and_sums_to_unfiltered_total(tmp_path):
    """One system, one case, 4 records deliberately spanning all 3 storm
    modes on BOTH the truth and forecast sides, plus one miss (truth object,
    no forecast object) and one false_alarm (forecast object, no truth
    object) -- confirms build_multi_system_object_climatology_by_storm_mode
    independently masks each side by its OWN is_linear field (no
    truth/forecast merge needed here, unlike the match-climatology sibling,
    since forecast-canonical rows always carry a real forecast_is_linear and
    truth-canonical rows always carry a real truth_is_linear -- see the
    function's own docstring), and that summing forecast_count/truth_count
    across all 3 modes reproduces the already-validated unfiltered totals
    from build_multi_system_object_climatology exactly (a regression proof,
    not just a standalone hand-computed check)."""
    g = tmp_path / "G"
    g.mkdir()
    init_x = datetime(2026, 5, 1, 20, 0, 0)

    records = [
        # hit, both sides cellular (mode 0)
        MatchRecord(category="hit", truth_id=1, forecast_id=1, ti_score=0.9,
                    truth_area_km2=100.0, truth_mean_intensity=40.0, truth_is_linear=0,
                    forecast_area_km2=110.0, forecast_mean_intensity=42.0, forecast_is_linear=0),
        # miss: truth-only, truth mode=1 (mixed)
        MatchRecord(category="miss", truth_id=2, forecast_id=-1, ti_score=0.0,
                    truth_area_km2=200.0, truth_mean_intensity=45.0, truth_is_linear=1),
        # false_alarm: forecast-only, forecast mode=2 (linear)
        MatchRecord(category="false_alarm", truth_id=-1, forecast_id=3, ti_score=0.0,
                    forecast_area_km2=50.0, forecast_mean_intensity=20.0, forecast_is_linear=2),
        # hit, truth mode=2 (linear), forecast mode=1 (mixed) -- confirms each
        # side is masked by its OWN field, independently.
        MatchRecord(category="hit", truth_id=4, forecast_id=4, ti_score=0.8,
                    truth_area_km2=300.0, truth_mean_intensity=48.0, truth_is_linear=2,
                    forecast_area_km2=310.0, forecast_mean_intensity=49.0, forecast_is_linear=1),
    ]
    other_time_records = [
        MatchRecord(category="hit", truth_id=5, forecast_id=5, ti_score=0.9,
                    truth_area_km2=999.0, truth_mean_intensity=99.0, truth_is_linear=0,
                    forecast_area_km2=999.0, forecast_mean_intensity=99.0, forecast_is_linear=0),
    ]
    write_match_file(
        str(g / "match_init_x.nc"), [
            MatchResult(records=records, valid_time=init_x, member_id=None),
            MatchResult(records=other_time_records, valid_time=init_x + timedelta(minutes=10), member_id=None),
        ],
        n_truth_source_files=1, n_forecast_source_files=1,
        max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2, init_time=init_x,
    )

    spec = [SystemSpec("G", str(g))]
    lh0 = 0.0

    unfiltered, _ = build_multi_system_object_climatology(spec, lead_time_mode="intersection", lead_minutes=[0])
    total_forecast = unfiltered["G"].forecast_count[(None, lh0)]
    total_truth = unfiltered["G"].truth_count[lh0]
    print(f"\n[object-clim-storm-mode-check1] unfiltered forecast={total_forecast}, truth={total_truth}")

    summed_forecast, summed_truth = 0, 0
    for mode, (exp_fc, exp_fc_area, exp_truth, exp_truth_area) in {
        0: (1, 110.0, 1, 100.0),   # cellular: forecast=hit#1 only; truth=hit#1 only
        1: (1, 310.0, 1, 200.0),   # mixed: forecast=hit#4; truth=miss#2
        2: (1, 50.0, 1, 300.0),    # linear: forecast=false_alarm#3; truth=hit#4
    }.items():
        mode_clim, _ = build_multi_system_object_climatology_by_storm_mode(
            spec, storm_mode=mode, lead_time_mode="intersection", lead_minutes=[0],
        )
        clim = mode_clim["G"]
        fc = clim.forecast_count.get((None, lh0), 0)
        fc_area = clim.forecast_area_km2.get((None, lh0), 0.0)
        truth = clim.truth_count.get(lh0, 0)
        truth_area = clim.truth_area_km2.get(lh0, 0.0)
        print(f"[object-clim-storm-mode-check1] mode={mode}: forecast={fc}({fc_area}), truth={truth}({truth_area})")
        assert fc == exp_fc
        assert fc_area == pytest.approx(exp_fc_area)
        assert truth == exp_truth
        assert truth_area == pytest.approx(exp_truth_area)
        summed_forecast += fc
        summed_truth += truth

    assert summed_forecast == total_forecast
    assert summed_truth == total_truth
    print(f"[object-clim-storm-mode-check1] sums-to-unfiltered-total confirmed: "
          f"forecast {summed_forecast}=={total_forecast}, truth {summed_truth}=={total_truth}")
