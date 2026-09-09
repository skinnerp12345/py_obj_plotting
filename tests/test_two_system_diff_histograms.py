"""Validation of plots.two_system_diff_histograms.plot_two_system_diff_histograms:
synthetic checks confirming exactly one histogram value per matched case
and correct diff sign/direction, plus a real end-to-end check against
MPAS 100km-vs-400km (32-37h) confirming the CSI-diff histogram's
underlying values reproduce the already-known real mean (-0.377, from the
two-system CSI-diff performance diagram's own validation).

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_two_system_diff_histograms.py -v -s
"""

import os
from datetime import datetime, timedelta

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from python_obj.obj_core import MatchRecord, MatchResult, write_match_file
from py_obj_plotting import cb_colors
from py_obj_plotting.climatology import build_per_case_match_climatologies, case_metrics
from py_obj_plotting.plots.two_system_diff_histograms import plot_two_system_diff_histograms
from py_obj_plotting.systems import SystemSpec, assign_colors

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NCAR_MPAS_DIR = os.path.join(REPO_ROOT, "ncar_mpas_analysis")
_SKIP_NO_NCAR_MPAS_ANALYSIS = pytest.mark.skipif(
    not os.path.isdir(NCAR_MPAS_DIR), reason="requires the external ncar_mpas_analysis/ dataset (not bundled)",
)


def _record(category: str, rid: int) -> MatchRecord:
    return MatchRecord(category=category, truth_id=rid, forecast_id=rid, ti_score=0.9 if category == "hit" else 0.0)


def _write_case(path: str, init_time: datetime, member_records: dict) -> None:
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


def test_writes_four_histograms_one_value_per_matched_case(tmp_path):
    da, db = tmp_path / "A", tmp_path / "B"
    da.mkdir(); db.mkdir()
    t0, t1 = datetime(2026, 5, 1), datetime(2026, 5, 2)
    # A: perfect (CSI=1.0) both days; B: CSI=0.5 both days -> diff always +0.5
    for t in (t0, t1):
        _write_case(str(da / f"match_init_{t:%Y%m%d_%H%M%S}.nc"), t, {(None, 0): ["hit", "hit"]})
        _write_case(str(db / f"match_init_{t:%Y%m%d_%H%M%S}.nc"), t, {(None, 0): ["hit", "miss"]})

    per_case_a = build_per_case_match_climatologies([SystemSpec("A", str(da))], case_grouping="init_time")
    per_case_b = build_per_case_match_climatologies([SystemSpec("B", str(db))], case_grouping="init_time")
    system_a = ("A", cb_colors.blue5, per_case_a["A"])
    system_b = ("B", cb_colors.orange5, per_case_b["B"])

    out_dir = str(tmp_path / "hist_out")
    report = plot_two_system_diff_histograms(system_a, system_b, lead_minutes_range=(0, 0), out_dir=out_dir)
    print(f"\n[diff-hist-check1] matched={report.matched_cases}, files={sorted(os.listdir(out_dir))}")
    assert len(report.matched_cases) == 2
    assert sorted(os.listdir(out_dir)) == [
        "csi_diff_histogram.png", "far_diff_histogram.png", "fbias_diff_histogram.png", "pod_diff_histogram.png",
    ]


def test_dynamic_symmetric_xlim_with_floors_and_sign_colored_bars_all_metrics(tmp_path):
    """X-axis limits are computed dynamically per metric as symmetric-
    around-zero, sized to the largest-magnitude diff actually present,
    but never smaller than the metric's floor (0.4 for POD/FAR, 0.2 for
    CSI, no floor for FBIAS). All 4 metrics (including FBIAS, which
    previously kept an uncolored/auto-range rendering) are now sign-
    colored: positive (system_a higher) in orange, negative (system_b
    higher) in blue."""
    da, db = tmp_path / "A", tmp_path / "B"
    da.mkdir(); db.mkdir()
    t0, t1 = datetime(2026, 5, 1), datetime(2026, 5, 2)
    # A: perfect (CSI=1.0, POD=1.0, FAR=0.0, FBIAS=1.0) both days
    # vs. B: CSI=0.5, POD=0.5, FAR=0.0, FBIAS=0.5 both days.
    # -> pod/csi/fbias diff = +0.5 (exceeds their floors); far diff = 0.0
    # (the 0.4 floor applies since the real range is degenerate).
    for t in (t0, t1):
        _write_case(str(da / f"match_init_{t:%Y%m%d_%H%M%S}.nc"), t, {(None, 0): ["hit", "hit"]})
        _write_case(str(db / f"match_init_{t:%Y%m%d_%H%M%S}.nc"), t, {(None, 0): ["hit", "miss"]})

    per_case_a = build_per_case_match_climatologies([SystemSpec("A", str(da))], case_grouping="init_time")
    per_case_b = build_per_case_match_climatologies([SystemSpec("B", str(db))], case_grouping="init_time")
    system_a = ("A", cb_colors.blue5, per_case_a["A"])
    system_b = ("B", cb_colors.orange5, per_case_b["B"])

    out_dir = str(tmp_path / "hist_out")

    # plt.get_fignums() is process-global, not scoped to this test -- other
    # tests in a full-suite run may leave figures open, so diff against the
    # pre-call figure set rather than asserting an absolute count.
    pre_existing_figs = set(plt.get_fignums())
    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_two_system_diff_histograms(system_a, system_b, lead_minutes_range=(0, 0), out_dir=out_dir)
    open_figs = sorted(set(plt.get_fignums()) - pre_existing_figs)
    plt.close = orig_close
    print(f"\n[diff-hist-dynamic] n_new_figs={len(open_figs)}")
    # 4 figures were created (one per metric, in _METRICS order: pod, far, fbias, csi)
    assert len(open_figs) == 4
    pod_ax, far_ax, fbias_ax, csi_ax = [plt.figure(n).axes[0] for n in open_figs]

    print(f"[diff-hist-dynamic] pod_xlim={pod_ax.get_xlim()}, far_xlim={far_ax.get_xlim()}, "
          f"fbias_xlim={fbias_ax.get_xlim()}, csi_xlim={csi_ax.get_xlim()}")
    # POD/CSI/FBIAS: real diff (0.5) exceeds their floor (or has none) ->
    # xlim reflects the real value.
    assert pod_ax.get_xlim() == pytest.approx((-0.5, 0.5))
    assert csi_ax.get_xlim() == pytest.approx((-0.5, 0.5))
    assert fbias_ax.get_xlim() == pytest.approx((-0.5, 0.5))
    # FAR: real diff is 0 -> the 0.4 floor applies.
    assert far_ax.get_xlim() == pytest.approx((-0.4, 0.4))

    # every metric's populated bar(s) must be orange (all fixture diffs >= 0).
    for ax, metric in ((pod_ax, "pod"), (far_ax, "far"), (fbias_ax, "fbias"), (csi_ax, "csi")):
        colors = [tuple(p.get_facecolor()[:3]) for p in ax.patches if p.get_height() > 0]
        print(f"[diff-hist-dynamic] {metric} populated_bar_colors={colors}")
        assert len(colors) >= 1
        for c in colors:
            assert c == pytest.approx(cb_colors.orange5)

    for n in open_figs:
        plt.close(plt.figure(n))


def test_fbias_floor_applies_when_real_range_is_smaller(tmp_path):
    """FBIAS has the same 0.4 floor as POD/FAR -- confirmed with a
    fixture whose real fbias diff is exactly 0 (both systems' fbias
    identical), which never exercises this in the other fixtures above
    (their fbias diff is +0.5, already exceeding the floor)."""
    da, db = tmp_path / "A", tmp_path / "B"
    da.mkdir(); db.mkdir()
    t0 = datetime(2026, 5, 1)
    # A: 1 hit, 1 miss -> fbias = 1/2 = 0.5; B: 1 hit, 1 false_alarm ->
    # fbias = 2/1 = 2.0 -- deliberately NOT matched fbias, so pick values
    # giving an identical fbias instead: both 1 hit, 1 miss -> fbias=0.5 each.
    _write_case(str(da / f"match_init_{t0:%Y%m%d_%H%M%S}.nc"), t0, {(None, 0): ["hit", "miss"]})
    _write_case(str(db / f"match_init_{t0:%Y%m%d_%H%M%S}.nc"), t0, {(None, 0): ["hit", "miss"]})

    per_case_a = build_per_case_match_climatologies([SystemSpec("A", str(da))], case_grouping="init_time")
    per_case_b = build_per_case_match_climatologies([SystemSpec("B", str(db))], case_grouping="init_time")
    system_a = ("A", cb_colors.blue5, per_case_a["A"])
    system_b = ("B", cb_colors.orange5, per_case_b["B"])

    out_dir = str(tmp_path / "hist_out")
    pre_existing_figs = set(plt.get_fignums())
    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_two_system_diff_histograms(system_a, system_b, lead_minutes_range=(0, 0), out_dir=out_dir)
    open_figs = sorted(set(plt.get_fignums()) - pre_existing_figs)
    plt.close = orig_close
    fbias_ax = plt.figure(open_figs[2]).axes[0]  # fbias is 3rd in _METRICS order
    print(f"\n[diff-hist-fbias-floor] fbias_xlim={fbias_ax.get_xlim()}")
    assert fbias_ax.get_xlim() == pytest.approx((-0.4, 0.4))

    for n in open_figs:
        plt.close(plt.figure(n))


def test_negative_diff_bars_colored_blue(tmp_path):
    """When system_b scores higher (negative diff), bars must be blue --
    complements the all-positive fixture above, which never exercises
    this branch."""
    da, db = tmp_path / "A", tmp_path / "B"
    da.mkdir(); db.mkdir()
    t0 = datetime(2026, 5, 1)
    # A: CSI=0.5; B: CSI=1.0 (perfect) -> diff = 0.5 - 1.0 = -0.5 (negative)
    _write_case(str(da / f"match_init_{t0:%Y%m%d_%H%M%S}.nc"), t0, {(None, 0): ["hit", "miss"]})
    _write_case(str(db / f"match_init_{t0:%Y%m%d_%H%M%S}.nc"), t0, {(None, 0): ["hit", "hit"]})

    per_case_a = build_per_case_match_climatologies([SystemSpec("A", str(da))], case_grouping="init_time")
    per_case_b = build_per_case_match_climatologies([SystemSpec("B", str(db))], case_grouping="init_time")
    system_a = ("A", cb_colors.blue5, per_case_a["A"])
    system_b = ("B", cb_colors.orange5, per_case_b["B"])

    out_dir = str(tmp_path / "hist_out")
    pre_existing_figs = set(plt.get_fignums())
    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_two_system_diff_histograms(system_a, system_b, lead_minutes_range=(0, 0), out_dir=out_dir)
    open_figs = sorted(set(plt.get_fignums()) - pre_existing_figs)
    plt.close = orig_close
    csi_ax = plt.figure(open_figs[-1]).axes[0]  # csi is last in _METRICS order
    colors = [tuple(p.get_facecolor()[:3]) for p in csi_ax.patches if p.get_height() > 0]
    print(f"\n[diff-hist-negative] csi populated_bar_colors={colors}")
    assert len(colors) >= 1
    for c in colors:
        assert c == pytest.approx(cb_colors.blue5)

    for n in open_figs:
        plt.close(plt.figure(n))


def test_no_matched_cases_raises(tmp_path):
    da, db = tmp_path / "A", tmp_path / "B"
    da.mkdir(); db.mkdir()
    _write_case(str(da / "match_init_20260501_000000.nc"), datetime(2026, 5, 1), {(None, 0): ["hit"]})
    _write_case(str(db / "match_init_20260502_000000.nc"), datetime(2026, 5, 2), {(None, 0): ["hit"]})

    per_case_a = build_per_case_match_climatologies([SystemSpec("A", str(da))], case_grouping="init_time")
    per_case_b = build_per_case_match_climatologies([SystemSpec("B", str(db))], case_grouping="init_time")
    system_a = ("A", cb_colors.blue5, per_case_a["A"])
    system_b = ("B", cb_colors.orange5, per_case_b["B"])

    with pytest.raises(ValueError, match="no matched cases"):
        plot_two_system_diff_histograms(system_a, system_b, lead_minutes_range=(0, 0), out_dir=str(tmp_path / "out"))


@_SKIP_NO_NCAR_MPAS_ANALYSIS
def test_real_mpas_csi_diff_histogram_matches_known_mean(tmp_path):
    systems = assign_colors([
        SystemSpec("100km", os.path.join(NCAR_MPAS_DIR, "matched_objects_100km")),
        SystemSpec("400km", os.path.join(NCAR_MPAS_DIR, "matched_objects_400km")),
    ])
    per_case = build_per_case_match_climatologies(systems, lead_time_mode="intersection", case_grouping="init_time")
    system_a = (systems[0].name, systems[0].color, per_case[systems[0].name])
    system_b = (systems[1].name, systems[1].color, per_case[systems[1].name])

    out_dir = str(tmp_path / "hist_out")
    report = plot_two_system_diff_histograms(system_a, system_b, lead_minutes_range=(1920, 2220), out_dir=out_dir, title_prefix="MPAS 32-37h")
    print(f"\n[diff-hist-check3] matched={len(report.matched_cases)}")
    assert len(report.matched_cases) == 118
    assert len(os.listdir(out_dir)) == 4

    diffs = [
        case_metrics(per_case["100km"][label], (1920, 2220))["csi"] - case_metrics(per_case["400km"][label], (1920, 2220))["csi"]
        for label in report.matched_cases
    ]
    print(f"[diff-hist-check3] csi diff mean={np.mean(diffs):.4f}")
    assert np.mean(diffs) == pytest.approx(-0.37664909988825884, abs=1e-6)
