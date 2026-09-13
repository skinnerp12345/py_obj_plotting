"""Validation of plots.performance_diagram.plot_case_scatter_performance_diagram:
synthetic checks for marker positioning/sizing/coloring/labeling, plus real
end-to-end tests over two representative lead-time periods (MPAS 32-37h,
WoFS 60-90min) confirming real marker counts match known real case counts.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_case_scatter_performance_diagram.py -v -s
"""

import os
from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.collections
import matplotlib.pyplot as plt
import pytest

from py_obj.obj_core import MatchRecord, MatchResult, write_match_file
from py_obj_plotting import cb_colors
from py_obj_plotting.climatology import build_per_case_match_climatologies
from py_obj_plotting.plots.performance_diagram import plot_case_scatter_performance_diagram
from py_obj_plotting.systems import SystemSpec, assign_colors

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SFE_MATCHES_DIR = os.path.join(REPO_ROOT, "sfe_poster", "sfe_matches")
NCAR_MPAS_DIR = os.path.join(REPO_ROOT, "ncar_mpas_analysis")
_SKIP_NO_SFE_POSTER = pytest.mark.skipif(
    not os.path.isdir(SFE_MATCHES_DIR), reason="requires the external sfe_poster/sfe_matches/ dataset (not bundled)",
)
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


# --- Check 1: synthetic, one system, two cases -------------------------------

def test_one_marker_per_case_with_correct_label_and_color(tmp_path):
    d = tmp_path / "A"
    d.mkdir()
    t0 = datetime(2026, 5, 1)
    t1 = datetime(2026, 5, 2)
    _write_case(str(d / f"match_init_{t0:%Y%m%d_%H%M%S}.nc"), t0, {(None, 0): ["hit", "hit", "miss"]})
    _write_case(str(d / f"match_init_{t1:%Y%m%d_%H%M%S}.nc"), t1, {(None, 0): ["hit", "false_alarm"]})

    per_case = build_per_case_match_climatologies([SystemSpec("A", str(d))], case_grouping="init_time")
    series = [("A", cb_colors.blue5, per_case["A"])]

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_case_scatter_performance_diagram(series, lead_minutes_range=(0, 0), out_path=str(tmp_path / "out.png"))
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    print(f"\n[case-scatter-check1] texts={texts}")
    # labels are now compact chronological integers (earliest case = "1"),
    # not the full case_label date string -- ax.texts also includes the
    # diagram's own CSI/bias contour clabels (e.g. "1.0", "0.5"), which
    # always carry a decimal point and so never collide with a bare
    # integer index.
    assert set(texts) >= {"1", "2"}

    collections = [c for c in ax.collections if isinstance(c, matplotlib.collections.PathCollection)]
    scatter_colors = [tuple(c.get_facecolor()[0][:3]) for c in collections]
    print(f"[case-scatter-check1] n_scatter_points={len(collections)}, colors={scatter_colors}")
    assert len(collections) == 2  # one marker per case
    for c in scatter_colors:
        assert c == pytest.approx(cb_colors.blue5)
    plt.close(fig)


def test_marker_size_scales_with_object_count(tmp_path):
    """A case with more truth objects must get a strictly larger marker
    than a case with fewer, under the shared figure-wide size scale."""
    d = tmp_path / "A"
    d.mkdir()
    t0 = datetime(2026, 5, 1)
    t1 = datetime(2026, 5, 2)
    _write_case(str(d / f"match_init_{t0:%Y%m%d_%H%M%S}.nc"), t0, {(None, 0): ["hit"] * 20})
    _write_case(str(d / f"match_init_{t1:%Y%m%d_%H%M%S}.nc"), t1, {(None, 0): ["hit"] * 2})

    per_case = build_per_case_match_climatologies([SystemSpec("A", str(d))], case_grouping="init_time")
    series = [("A", cb_colors.blue5, per_case["A"])]

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_case_scatter_performance_diagram(series, lead_minutes_range=(0, 0), out_path=str(tmp_path / "out.png"))
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    collections = [c for c in ax.collections if isinstance(c, matplotlib.collections.PathCollection)]
    sizes = sorted(c.get_sizes()[0] for c in collections)
    print(f"\n[case-scatter-check2] sizes={sizes}")
    assert sizes[0] < sizes[1]
    plt.close(fig)


def test_legend_has_one_entry_per_system(tmp_path):
    d1, d2 = tmp_path / "A", tmp_path / "B"
    d1.mkdir(); d2.mkdir()
    t0 = datetime(2026, 5, 1)
    _write_case(str(d1 / f"match_init_{t0:%Y%m%d_%H%M%S}.nc"), t0, {(None, 0): ["hit"]})
    _write_case(str(d2 / f"match_init_{t0:%Y%m%d_%H%M%S}.nc"), t0, {(None, 0): ["hit"]})

    per_case_a = build_per_case_match_climatologies([SystemSpec("A", str(d1))], case_grouping="init_time")
    per_case_b = build_per_case_match_climatologies([SystemSpec("B", str(d2))], case_grouping="init_time")
    series = [("A", cb_colors.blue5, per_case_a["A"]), ("B", cb_colors.orange5, per_case_b["B"])]

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_case_scatter_performance_diagram(series, lead_minutes_range=(0, 0), out_path=str(tmp_path / "out.png"))
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    print(f"\n[case-scatter-check3] legend_labels={legend_labels}")
    assert legend_labels == ["A", "B"]
    plt.close(fig)


def test_empty_series_raises(tmp_path):
    with pytest.raises(ValueError, match="series must be non-empty"):
        plot_case_scatter_performance_diagram([], lead_minutes_range=(0, 60), out_path=str(tmp_path / "out.png"))


# --- Check 2: real data, two representative example periods -----------------

@_SKIP_NO_NCAR_MPAS_ANALYSIS
def test_real_mpas_32_37h_period(tmp_path):
    systems = assign_colors([
        SystemSpec("100km", os.path.join(NCAR_MPAS_DIR, "matched_objects_100km")),
        SystemSpec("400km", os.path.join(NCAR_MPAS_DIR, "matched_objects_400km")),
    ])
    per_case = build_per_case_match_climatologies(systems, lead_time_mode="intersection", case_grouping="init_time")
    series = [(s.name, s.color, per_case[s.name]) for s in systems]

    out_path = str(tmp_path / "mpas_32_37h.png")
    plot_case_scatter_performance_diagram(series, lead_minutes_range=(1920, 2220), out_path=out_path, title="MPAS 32-37h")
    print(f"\n[case-scatter-check4] n_cases per system={ {n: len(c) for n, _, c in series} }")
    assert os.path.isfile(out_path)
    for name, _, cases in series:
        assert len(cases) == 118


@_SKIP_NO_SFE_POSTER
def test_real_wofs_60_90min_period_day_grouped(tmp_path):
    systems = assign_colors([SystemSpec("WoFS", os.path.join(SFE_MATCHES_DIR, "wofs_vs_mrms"))])
    per_case = build_per_case_match_climatologies(systems, lead_time_mode="union", case_grouping="day")
    series = [(s.name, s.color, per_case[s.name]) for s in systems]

    out_path = str(tmp_path / "wofs_60_90min.png")
    plot_case_scatter_performance_diagram(series, lead_minutes_range=(60, 90), out_path=out_path, title="WoFS 60-90min")
    print(f"\n[case-scatter-check5] n_days={len(series[0][2])}")
    assert os.path.isfile(out_path)
    assert len(series[0][2]) == 24  # real, confirmed unique WoFS days
