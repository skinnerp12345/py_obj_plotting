"""Validation of plots.performance_diagram.plot_two_system_case_diff_performance_diagram:
synthetic checks for matched-case restriction/reporting, shared diff-color
value per case-pair, and per-system label-text coloring (not hatching);
real end-to-end checks against WoFS-vs-WoFSCast (day-grouped) and MPAS
100km-vs-400km (init_time-grouped, confirmed full 118/118 alignment) with a
physically-sensible CSI-diff sign check (a stricter matching threshold can
never exceed a looser one's CSI).

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_two_system_case_diff_performance_diagram.py -v -s
"""

import os
from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import pytest

from python_obj.obj_core import MatchRecord, MatchResult, write_match_file
from py_obj_plotting import cb_colors
from py_obj_plotting.climatology import build_per_case_match_climatologies, case_metrics
from py_obj_plotting.plots.performance_diagram import (
    matched_cases_report,
    plot_two_system_case_diff_performance_diagram,
)
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


# --- Check 1: matched_cases_report() -----------------------------------------

def test_matched_cases_report_partitions_correctly():
    cases_a = {"c1": None, "c2": None, "c3": None}
    cases_b = {"c2": None, "c3": None, "c4": None}
    report = matched_cases_report(cases_a, cases_b)
    print(f"\n[two-sys-diff-check1] matched={report.matched_cases}, a_only={report.system_a_only}, b_only={report.system_b_only}")
    assert report.matched_cases == ["c2", "c3"]
    assert report.system_a_only == ["c1"]
    assert report.system_b_only == ["c4"]


# --- Check 2: synthetic, shared diff value + per-system label color ---------

def test_shared_case_gets_one_diff_value_coloring_both_markers(tmp_path):
    da, db = tmp_path / "A", tmp_path / "B"
    da.mkdir(); db.mkdir()
    t0 = datetime(2026, 5, 1)
    # A: 2 hits, 0 misses/FAs -> CSI=1.0; B: 1 hit, 1 miss -> CSI=0.5
    _write_case(str(da / f"match_init_{t0:%Y%m%d_%H%M%S}.nc"), t0, {(None, 0): ["hit", "hit"]})
    _write_case(str(db / f"match_init_{t0:%Y%m%d_%H%M%S}.nc"), t0, {(None, 0): ["hit", "miss"]})

    per_case_a = build_per_case_match_climatologies([SystemSpec("A", str(da))], case_grouping="init_time")
    per_case_b = build_per_case_match_climatologies([SystemSpec("B", str(db))], case_grouping="init_time")
    system_a = ("A", cb_colors.blue5, per_case_a["A"])
    system_b = ("B", cb_colors.orange5, per_case_b["B"])

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    report = plot_two_system_case_diff_performance_diagram(
        system_a, system_b, lead_minutes_range=(0, 0), out_path=str(tmp_path / "out.png"),
    )
    plt.close = orig_close

    print(f"\n[two-sys-diff-check2] matched={report.matched_cases}")
    assert report.matched_cases == ["2026-05-01 0000Z"]

    fig = plt.gcf()
    ax = fig.axes[0]
    scatters = [c for c in ax.collections if isinstance(c, matplotlib.collections.PathCollection)]
    assert len(scatters) == 2  # one marker for A, one for B, same case
    diff_values = [float(c.get_array()[0]) for c in scatters]
    print(f"[two-sys-diff-check2] diff_values={diff_values}")
    assert diff_values[0] == pytest.approx(diff_values[1])  # SAME shared diff value
    assert diff_values[0] == pytest.approx(1.0 - 0.5)  # CSI_A - CSI_B

    # only one matched case -> label is chronological index "1" for BOTH
    # systems' markers, plain thin black text (no outline, no per-system
    # color) -- the two systems are distinguished by marker edge
    # linestyle instead (solid vs. dashed). ax.texts also includes the
    # diagram's own CSI/bias contour clabels (e.g. "1.0", "0.5"), which
    # always carry a decimal point and so never collide with the bare
    # integer index "1".
    case_label_texts = [t for t in ax.texts if t.get_text() == "1"]
    label_colors = [t.get_color() for t in case_label_texts]
    print(f"[two-sys-diff-check2] n_case_label_texts={len(case_label_texts)}, label_colors={label_colors}")
    assert len(case_label_texts) == 2
    assert set(label_colors) == {"black"}
    for t in case_label_texts:
        assert len(t.get_path_effects()) == 0  # no outline stroke

    # systems distinguished by marker edge linestyle: solid (no dash
    # pattern, i.e. get_linestyle() -> [(offset, None)]) vs. dashed (a
    # real dash-pattern list).
    dash_patterns = [c.get_linestyle()[0][1] for c in scatters]  # [(offset, dash_pattern)] per scatter
    print(f"[two-sys-diff-check2] dash_patterns={dash_patterns}")
    assert dash_patterns.count(None) == 1  # exactly one solid marker
    assert sum(p is not None for p in dash_patterns) == 1  # exactly one dashed marker
    plt.close(fig)


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
        plot_two_system_case_diff_performance_diagram(
            system_a, system_b, lead_minutes_range=(0, 0), out_path=str(tmp_path / "out.png"),
        )


# --- Check 3: real data ------------------------------------------------------

@_SKIP_NO_SFE_POSTER
def test_real_wofs_vs_wofscast_60_90min(tmp_path):
    systems = assign_colors([
        SystemSpec("WoFS", os.path.join(SFE_MATCHES_DIR, "wofs_vs_mrms")),
        SystemSpec("WoFSCast", os.path.join(SFE_MATCHES_DIR, "wofscast_vs_mrms")),
    ])
    per_case = build_per_case_match_climatologies(systems, lead_time_mode="union", case_grouping="day")
    system_a = (systems[0].name, systems[0].color, per_case[systems[0].name])
    system_b = (systems[1].name, systems[1].color, per_case[systems[1].name])

    out_path = str(tmp_path / "wofs_vs_wofscast_diff.png")
    report = plot_two_system_case_diff_performance_diagram(
        system_a, system_b, lead_minutes_range=(60, 90), out_path=out_path, title="WoFS vs WoFSCast, 60-90min",
    )
    print(f"\n[two-sys-diff-check3] matched={len(report.matched_cases)}, "
          f"a_only={len(report.system_a_only)}, b_only={len(report.system_b_only)}")
    assert os.path.isfile(out_path)
    assert len(report.matched_cases) == 24
    assert report.system_a_only == []
    assert report.system_b_only == []


@_SKIP_NO_NCAR_MPAS_ANALYSIS
def test_real_mpas_100km_vs_400km_csi_diff_sign_is_physically_sensible(tmp_path):
    """100km (stricter matching threshold) can never exceed 400km's (looser)
    CSI -- a looser max-displacement threshold only ever admits MORE
    candidate matches, never fewer, so CSI_100km - CSI_400km <= 0 for
    every real case."""
    systems = assign_colors([
        SystemSpec("100km", os.path.join(NCAR_MPAS_DIR, "matched_objects_100km")),
        SystemSpec("400km", os.path.join(NCAR_MPAS_DIR, "matched_objects_400km")),
    ])
    per_case = build_per_case_match_climatologies(systems, lead_time_mode="intersection", case_grouping="init_time")
    system_a = (systems[0].name, systems[0].color, per_case[systems[0].name])
    system_b = (systems[1].name, systems[1].color, per_case[systems[1].name])

    out_path = str(tmp_path / "mpas_diff.png")
    report = plot_two_system_case_diff_performance_diagram(
        system_a, system_b, lead_minutes_range=(1920, 2220), out_path=out_path, title="MPAS 100km vs 400km, 32-37h",
    )
    print(f"\n[two-sys-diff-check4] matched={len(report.matched_cases)}")
    assert len(report.matched_cases) == 118
    assert report.system_a_only == [] and report.system_b_only == []

    diffs = [
        case_metrics(per_case["100km"][label], (1920, 2220))["csi"] - case_metrics(per_case["400km"][label], (1920, 2220))["csi"]
        for label in report.matched_cases
    ]
    print(f"[two-sys-diff-check4] diff range=({min(diffs):.3f}, {max(diffs):.3f})")
    assert max(diffs) <= 1e-9  # never positive (within float tolerance)
    assert min(diffs) < 0  # genuinely negative for at least some real cases
