"""Synthetic-fixture validation of climatology.aggregate_shared_valid_times,
with hand-computed expected results -- confirms the shared-valid-time
intersection (across 3+ sources, not just 2), the ensemble-mean-of-PDFs
method (deliberately NOT sum-then-normalize -- these give different results
whenever members have unequal total counts, exercised here concretely), and
the shared-times summary table's per-date/TOTAL-row file counting (including
the cross-midnight double-counting fix), before trusting this logic against
the real (much larger) sfe_poster/ dataset.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_aggregate_shared_valid_times.py -v -s
"""

from datetime import datetime

import numpy as np
import pytest

from python_obj.histogram import HistogramSlice, default_bin_edges, write_histogram_file
from py_obj_plotting.climatology import build_shared_times_summary_table, build_shared_valid_times_climatology

BINS = default_bin_edges(bin_min=0.0, bin_max=10.0, bin_width=1.0)

T1 = datetime(2023, 5, 1, 23, 0)  # shared by all 3 sources
T2 = datetime(2023, 5, 2, 0, 0)   # shared by all 3 sources -- crosses midnight from T1
T3 = datetime(2023, 5, 2, 1, 0)   # MRMS has it, NowcastNet does not -> must be excluded
T4 = datetime(2023, 5, 2, 2, 0)   # NowcastNet has it, MRMS does not -> must be excluded


def _hist_with(counts: dict) -> np.ndarray:
    h = np.zeros(len(BINS) - 1, dtype=np.int64)
    for idx, val in counts.items():
        h[idx] = val
    return h


def _write(path: str, slices: list, varname: str = "refl_consv", n_source_files: int = 1) -> None:
    write_histogram_file(path, BINS, slices, varname, 7, False, n_source_files)


@pytest.fixture
def fixture_dirs(tmp_path):
    mrms_dir = tmp_path / "mrms"
    nowcastnet_dir = tmp_path / "nowcastnet"
    wofs_dir = tmp_path / "wofs"
    for d in (mrms_dir, nowcastnet_dir, wofs_dir):
        d.mkdir()

    # --- MRMS: two day-files (T1 on day 1, T2/T3 on day 2) -----------------
    _write(str(mrms_dir / "hist_mrms_20230501.nc"), [HistogramSlice(valid_time=T1, hist=_hist_with({0: 5}))])
    _write(str(mrms_dir / "hist_mrms_20230502.nc"), [
        HistogramSlice(valid_time=T2, hist=_hist_with({1: 7})),
        HistogramSlice(valid_time=T3, hist=_hist_with({2: 999})),  # not shared (NowcastNet lacks T3)
    ])

    # --- NowcastNet: one cycle file, no member_id -------------------------
    _write(str(nowcastnet_dir / "hist_model_20230501_2250.nc"), [
        HistogramSlice(valid_time=T1, hist=_hist_with({0: 3})),
        HistogramSlice(valid_time=T2, hist=_hist_with({1: 2})),
        HistogramSlice(valid_time=T4, hist=_hist_with({3: 999})),  # not shared (MRMS lacks T4)
    ])

    # --- WoFS: one cycle file, 2 members with DELIBERATELY unequal totals --
    # mem_1: all weight at T1 (bin 0), zero at T2 -> total 10, PDF = [1.0, 0.0, ...]
    # mem_2: zero at T1, all weight at T2 (bin 1) -> total 90, PDF = [0.0, 1.0, ...]
    # Average-of-PDFs (this module's method) -> [0.5, 0.5, ...]
    # Sum-then-normalize (the OTHER, explicitly-rejected method) would give
    # sum={0:10,1:90}, total=100 -> [0.1, 0.9, ...] -- a genuinely different
    # result, so this fixture actually discriminates between the two methods.
    _write(str(wofs_dir / "hist_model_20230501_2250.nc"), [
        HistogramSlice(valid_time=T1, hist=_hist_with({0: 10}), member_id="mem_1"),
        HistogramSlice(valid_time=T2, hist=_hist_with({}), member_id="mem_1"),
        HistogramSlice(valid_time=T1, hist=_hist_with({}), member_id="mem_2"),
        HistogramSlice(valid_time=T2, hist=_hist_with({1: 90}), member_id="mem_2"),
        HistogramSlice(valid_time=T3, hist=_hist_with({2: 500}), member_id="mem_1"),  # not shared
        HistogramSlice(valid_time=T3, hist=_hist_with({2: 500}), member_id="mem_2"),  # not shared
    ])

    return {"mrms_dir": str(mrms_dir), "nowcastnet_dir": str(nowcastnet_dir), "wofs_dir": str(wofs_dir)}


def test_shared_valid_time_intersection_across_three_sources(fixture_dirs):
    clim = build_shared_valid_times_climatology(
        model_dirs={"NowcastNet": fixture_dirs["nowcastnet_dir"], "WoFS": fixture_dirs["wofs_dir"]},
        ensemble_systems={"WoFS"},
        mrms_dir=fixture_dirs["mrms_dir"],
    )
    print(f"\n[shared-times-check1] shared_valid_times={clim.shared_valid_times}")
    assert clim.shared_valid_times == [T1, T2]  # T3/T4 correctly excluded


def test_non_ensemble_aggregation_restricted_to_shared_times(fixture_dirs):
    clim = build_shared_valid_times_climatology(
        model_dirs={"NowcastNet": fixture_dirs["nowcastnet_dir"], "WoFS": fixture_dirs["wofs_dir"]},
        ensemble_systems={"WoFS"},
        mrms_dir=fixture_dirs["mrms_dir"],
    )
    mrms_pdf = clim.pdf["MRMS"]
    nowcastnet_pdf = clim.pdf["NowcastNet"]
    np.testing.assert_allclose(mrms_pdf, _hist_with({0: 5, 1: 7}) / 12.0)
    np.testing.assert_allclose(nowcastnet_pdf, _hist_with({0: 3, 1: 2}) / 5.0)
    print("[shared-times-check2] MRMS/NowcastNet aggregation restricted to shared times only, T3/T4 excluded")


def test_ensemble_mean_is_average_of_member_pdfs_not_sum_then_normalize(fixture_dirs):
    clim = build_shared_valid_times_climatology(
        model_dirs={"NowcastNet": fixture_dirs["nowcastnet_dir"], "WoFS": fixture_dirs["wofs_dir"]},
        ensemble_systems={"WoFS"},
        mrms_dir=fixture_dirs["mrms_dir"],
    )
    np.testing.assert_allclose(clim.member_pdf["WoFS"]["mem_1"], [1.0, 0.0] + [0.0] * 8)
    np.testing.assert_allclose(clim.member_pdf["WoFS"]["mem_2"], [0.0, 1.0] + [0.0] * 8)

    expected_ensemble_mean = [0.5, 0.5] + [0.0] * 8  # average of PDFs, NOT [0.1, 0.9] (sum-then-normalize)
    np.testing.assert_allclose(clim.pdf["WoFS"], expected_ensemble_mean)
    np.testing.assert_allclose(clim.cdf["WoFS"], np.cumsum(expected_ensemble_mean))
    print(f"\n[shared-times-check3] WoFS ensemble-mean PDF={clim.pdf['WoFS'][:2]} "
          "(confirmed average-of-PDFs, not sum-then-normalize which would give [0.1, 0.9])")


def test_bin_mismatch_raises(tmp_path, fixture_dirs):
    other_bins = default_bin_edges(bin_min=0.0, bin_max=5.0, bin_width=1.0)
    bad_dir = tmp_path / "bad_bins"
    bad_dir.mkdir()
    write_histogram_file(
        str(bad_dir / "hist_model_bad.nc"), other_bins,
        [HistogramSlice(valid_time=T1, hist=np.zeros(5, dtype=np.int64))],
        "refl_consv", 7, False, 1,
    )
    with pytest.raises(ValueError, match="different bin edges"):
        build_shared_valid_times_climatology(
            model_dirs={"Bad": str(bad_dir)}, ensemble_systems=set(), mrms_dir=fixture_dirs["mrms_dir"],
        )


def test_no_shared_valid_times_raises(tmp_path):
    mrms_dir = tmp_path / "mrms2"
    other_dir = tmp_path / "other2"
    mrms_dir.mkdir()
    other_dir.mkdir()
    _write(str(mrms_dir / "hist_mrms_x.nc"), [HistogramSlice(valid_time=T1, hist=_hist_with({0: 1}))])
    _write(str(other_dir / "hist_model_x.nc"), [HistogramSlice(valid_time=T4, hist=_hist_with({0: 1}))])

    with pytest.raises(ValueError, match="No valid_time is shared"):
        build_shared_valid_times_climatology(
            model_dirs={"Other": str(other_dir)}, ensemble_systems=set(), mrms_dir=str(mrms_dir),
        )


def test_shared_times_summary_table_per_date_and_total(fixture_dirs):
    clim = build_shared_valid_times_climatology(
        model_dirs={"NowcastNet": fixture_dirs["nowcastnet_dir"], "WoFS": fixture_dirs["wofs_dir"]},
        ensemble_systems={"WoFS"},
        mrms_dir=fixture_dirs["mrms_dir"],
    )
    rows = build_shared_times_summary_table(clim)
    by_date = {r["date"]: r for r in rows}

    print(f"\n[shared-times-check4] summary rows: {rows}")

    day1 = by_date["2023-05-01"]
    assert day1["n_shared_valid_times"] == 1  # T1 only
    assert day1["MRMS_n_initializations"] == 1
    assert day1["NowcastNet_n_initializations"] == 1
    assert day1["WoFS_n_initializations"] == 1

    day2 = by_date["2023-05-02"]
    assert day2["n_shared_valid_times"] == 1  # T2 only
    assert day2["MRMS_n_initializations"] == 1  # MRMS's SECOND day-file
    assert day2["NowcastNet_n_initializations"] == 1  # the SAME NowcastNet file as day 1
    assert day2["WoFS_n_initializations"] == 1  # the SAME WoFS file as day 1

    total = by_date["TOTAL"]
    assert total["n_shared_valid_times"] == 2
    assert total["MRMS_n_initializations"] == 2  # 2 genuinely distinct MRMS files
    # NowcastNet/WoFS: the SAME single cross-midnight file contributed to
    # both dates -- the TOTAL row must count it once, not twice (the bug
    # this fixture was specifically designed to catch: summing per-date
    # counts would double it to 2).
    assert total["NowcastNet_n_initializations"] == 1
    assert total["WoFS_n_initializations"] == 1
    print("[shared-times-check4] TOTAL row correctly deduplicates the cross-midnight NowcastNet/WoFS file")
