"""Synthetic-fixture validation of climatology.aggregate_by_lead_hour, with
hand-computed expected sums -- confirms per-member aggregation, the
once-per-case-per-lead-hour MRMS de-duplication rule, cross-MRMS-day-file
matching, and the +/- tolerance boundary, before trusting this logic against
the real (much larger) ncar_mpas_analysis/ dataset.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_aggregate_by_lead_hour.py -v -s
"""

from datetime import datetime

import numpy as np

from python_obj.histogram import HistogramSlice, default_bin_edges, write_histogram_file
from py_obj_plotting.climatology import build_lead_hour_climatology

BINS = default_bin_edges(bin_min=0.0, bin_max=10.0, bin_width=1.0)


def _hist_with(counts: dict) -> np.ndarray:
    h = np.zeros(len(BINS) - 1, dtype=np.int64)
    for idx, val in counts.items():
        h[idx] = val
    return h


def _write_mpas_case(path: str, init_time: datetime, members_leads: dict, n_source_files: int = 1) -> None:
    """members_leads: {(member_id, lead_hour): counts_dict}"""
    from datetime import timedelta
    slices = []
    for (member_id, lead_hour), counts in members_leads.items():
        valid_time = init_time + timedelta(hours=lead_hour)
        slices.append(HistogramSlice(
            valid_time=valid_time, hist=_hist_with(counts), lead_hours=lead_hour, member_id=member_id,
        ))
    write_histogram_file(path, BINS, slices, "refl10cm_max", 7, False, n_source_files)


def _write_mrms_day(path: str, day: datetime, hour_counts: dict, n_source_files: int | None = None) -> None:
    """hour_counts: {hour_of_day: counts_dict} -- one slice per given hour, rest omitted."""
    from datetime import timedelta
    slices = [
        HistogramSlice(valid_time=day + timedelta(hours=h), hist=_hist_with(counts))
        for h, counts in hour_counts.items()
    ]
    write_histogram_file(path, BINS, slices, "refl_consv", 7, False, n_source_files or len(slices))


def test_real_synthetic_climatology_matches_hand_computed_expectations(tmp_path):
    mpas_dir = tmp_path / "mpas"
    mrms_dir = tmp_path / "mrms"
    mpas_dir.mkdir()
    mrms_dir.mkdir()

    # --- MRMS fixture: two consecutive days, sparse hourly slices ---------
    # day 1 (2023-05-01): only hour 12 populated with a known, distinguishable value
    _write_mrms_day(str(mrms_dir / "hist_mrms_20230501.nc"), datetime(2023, 5, 1), {
        12: {5: 7},
        11: {5: 999},  # distractor hour -- confirms the nearest match, not just any match, is chosen
    })
    # day 2 (2023-05-02): hours 0 and 1 populated
    _write_mrms_day(str(mrms_dir / "hist_mrms_20230502.nc"), datetime(2023, 5, 2), {
        0: {5: 3},
        1: {5: 9},
    })

    # --- MPAS case 1: init 2023-05-01 12:00Z, members mem_1/mem_2 --------
    # lead 0  -> valid 2023-05-01 12:00 (exact MRMS match, day 1)
    # lead 13 -> valid 2023-05-02 01:00 (exact MRMS match, day 2 -- crosses the day boundary)
    # lead 50 -> valid 2023-05-03 14:00 (no MRMS data exists for this day at all -- expect unmatched)
    _write_mpas_case(str(mpas_dir / "hist_model_20230501_120000.nc"), datetime(2023, 5, 1, 12, 0), {
        ("mem_1", 0.0): {1: 10}, ("mem_2", 0.0): {1: 100},
        ("mem_1", 13.0): {2: 20}, ("mem_2", 13.0): {2: 200},
        ("mem_1", 50.0): {3: 5}, ("mem_2", 50.0): {3: 50},
    })

    # --- MPAS case 2: init 2023-05-02 00:00Z, members mem_1/mem_2 --------
    # lead 0 -> valid 2023-05-02 00:00 (exact MRMS match, day 2) -- same lead_hour as
    # case 1's lead=0, so mpas_hist at (member, 0.0) must SUM across both cases.
    _write_mpas_case(str(mpas_dir / "hist_model_20230502_000000.nc"), datetime(2023, 5, 2, 0, 0), {
        ("mem_1", 0.0): {1: 1}, ("mem_2", 0.0): {1: 2},
    })

    result = build_lead_hour_climatology(str(mpas_dir), str(mrms_dir), valid_time_tolerance_minutes=10.0)

    print(f"\n[climatology-check1] lead_hours={result.lead_hours}, members={result.members}")
    assert result.members == ["mem_1", "mem_2"]
    assert result.lead_hours == [0.0, 13.0, 50.0]

    # --- MPAS aggregation: per-member sums, summed ACROSS cases at lead=0 --
    np.testing.assert_array_equal(result.mpas_hist[("mem_1", 0.0)], _hist_with({1: 11}))   # 10 + 1
    np.testing.assert_array_equal(result.mpas_hist[("mem_2", 0.0)], _hist_with({1: 102}))  # 100 + 2
    np.testing.assert_array_equal(result.mpas_hist[("mem_1", 13.0)], _hist_with({2: 20}))
    np.testing.assert_array_equal(result.mpas_hist[("mem_2", 13.0)], _hist_with({2: 200}))
    np.testing.assert_array_equal(result.mpas_hist[("mem_1", 50.0)], _hist_with({3: 5}))
    np.testing.assert_array_equal(result.mpas_hist[("mem_2", 50.0)], _hist_with({3: 50}))
    print("[climatology-check1] MPAS per-member aggregation OK (including cross-case summation at lead=0)")

    # --- MRMS aggregation: matched by real valid time, counted ONCE per --
    # --- case+lead_hour regardless of the 2 members in each case --------
    np.testing.assert_array_equal(result.mrms_hist[0.0], _hist_with({5: 10}))  # case1's 7 (day1 h12) + case2's 3 (day2 h0)
    np.testing.assert_array_equal(result.mrms_hist[13.0], _hist_with({5: 9}))  # case1's match, day2 h1 -- crosses day boundary correctly
    assert 50.0 not in result.mrms_hist, "lead 50 has no real MRMS match -- must be absent, not zero-filled"
    print("[climatology-check1] MRMS aggregation OK (cross-day matching, once-per-case-per-lead-hour dedup)")

    # --- bookkeeping: contributing case counts + unmatched reporting -----
    assert result.n_cases_per_lead_hour == {0.0: 2, 13.0: 1, 50.0: 1}
    assert result.n_mrms_matched_per_lead_hour == {0.0: 2, 13.0: 1}  # 50.0 absent -- 0 matches
    assert result.n_unmatched == 1
    unmatched_lead_hours = [u[1] for u in result.unmatched]
    assert unmatched_lead_hours == [50.0]
    print(f"[climatology-check1] n_cases_per_lead_hour={result.n_cases_per_lead_hour}, "
          f"n_unmatched={result.n_unmatched} (expect exactly 1, lead=50)")


def test_valid_time_tolerance_boundary(tmp_path):
    """+/-10 minutes: a 9-minute offset must match, an 11-minute offset must not."""
    mpas_dir = tmp_path / "mpas"
    mrms_dir = tmp_path / "mrms"
    mpas_dir.mkdir()
    mrms_dir.mkdir()

    _write_mrms_day(str(mrms_dir / "hist_mrms_20230601.nc"), datetime(2023, 6, 1), {12: {5: 42}})

    # lead 0 -> valid 12:09 (9 min after the 12:00 MRMS slice -- within 10 min tolerance)
    # lead 1 -> valid 12:11+... constructed directly below via a non-integer-hour init time
    from datetime import timedelta
    init_time = datetime(2023, 6, 1, 11, 51)  # + 0h = 11:51 (9 min before 12:00) -- within tolerance
    _write_mpas_case(str(mpas_dir / "hist_model_20230601_115100.nc"), init_time, {
        ("mem_1", 0.0): {1: 1},
    })
    init_time_far = datetime(2023, 6, 1, 11, 48)  # + 0h = 11:48 (12 min before 12:00) -- outside tolerance
    _write_mpas_case(str(mpas_dir / "hist_model_20230601_114800.nc"), init_time_far, {
        ("mem_1", 5.0): {1: 1},  # distinct lead_hour so it doesn't collide with the case above
    })

    result = build_lead_hour_climatology(str(mpas_dir), str(mrms_dir), valid_time_tolerance_minutes=10.0)

    print(f"\n[climatology-check2] mrms_hist keys={list(result.mrms_hist.keys())}, unmatched leads={[u[1] for u in result.unmatched]}")
    assert 0.0 in result.mrms_hist, "9-minute offset should match within +/-10 min tolerance"
    assert 5.0 not in result.mrms_hist, "12-minute offset should NOT match within +/-10 min tolerance"
    assert [u[1] for u in result.unmatched] == [5.0]


def test_mismatched_bins_raises(tmp_path):
    import pytest
    mpas_dir = tmp_path / "mpas"
    mrms_dir = tmp_path / "mrms"
    mpas_dir.mkdir()
    mrms_dir.mkdir()

    other_bins = default_bin_edges(bin_min=0.0, bin_max=20.0, bin_width=1.0)
    write_histogram_file(
        str(mpas_dir / "hist_model_20230501_000000.nc"), other_bins,
        [HistogramSlice(valid_time=datetime(2023, 5, 1), hist=np.zeros(len(other_bins) - 1, dtype=np.int64), lead_hours=0.0, member_id="mem_1")],
        "refl10cm_max", 7, False, 1,
    )
    write_histogram_file(
        str(mrms_dir / "hist_mrms_20230501.nc"), BINS,
        [HistogramSlice(valid_time=datetime(2023, 5, 1), hist=np.zeros(len(BINS) - 1, dtype=np.int64))],
        "refl_consv", 7, False, 1,
    )

    with pytest.raises(ValueError, match="bins"):
        build_lead_hour_climatology(str(mpas_dir), str(mrms_dir))
