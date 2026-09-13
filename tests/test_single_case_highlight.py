"""Validation of plots.single_case_highlight's 2 rendering variants:
synthetic checks confirming the baseline population renders as a shaded
percentile-band distribution (time series) or gray/low-alpha uniform
markers (case scatter) while the highlighted case renders bold/full-alpha
with its own real size/label; real end-to-end checks against a real MPAS
case highlighted against its 117 real baseline cases (118 total, matching
the already-known real case count).

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_single_case_highlight.py -v -s
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
from py_obj_plotting.climatology import (
    build_per_case_match_climatologies,
    compute_verification_stats,
    split_baseline_and_highlight,
)
from py_obj_plotting.plots.single_case_highlight import (
    plot_single_case_highlight_case_scatter,
    plot_single_case_highlight_timeseries,
)
from py_obj_plotting.systems import SystemSpec

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


def _three_case_fixture(tmp_path):
    d = tmp_path / "A"
    d.mkdir()
    times = [datetime(2026, 5, i) for i in (1, 2, 3)]
    for t in times:
        _write_case(str(d / f"match_init_{t:%Y%m%d_%H%M%S}.nc"), t, {
            (None, 0): ["hit", "miss"], (None, 60): ["hit"], (None, 120): ["hit", "hit"],
        })
    per_case = build_per_case_match_climatologies([SystemSpec("A", str(d))], case_grouping="init_time")
    return per_case["A"]


# --- Time series --------------------------------------------------------------

def test_timeseries_percentile_band_background_highlight_bold(tmp_path):
    cases = _three_case_fixture(tmp_path)
    highlight = sorted(cases.keys())[0]
    baseline_clims, highlighted_clim = split_baseline_and_highlight(cases, highlight)
    baseline_stats = [compute_verification_stats(c) for c in baseline_clims]
    highlighted_stats = compute_verification_stats(highlighted_clim)

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_single_case_highlight_timeseries(baseline_stats, highlighted_stats, "pod", str(tmp_path / "out.png"),
                                            ylabel="POD", title="test", highlight_label="H")
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    lines = ax.get_lines()
    fills = [c for c in ax.collections if isinstance(c, matplotlib.collections.PolyCollection)]
    print(f"\n[highlight-ts-check1] n_lines={len(lines)}, n_fills={len(fills)}, "
          f"alphas={[l.get_alpha() for l in lines]}, colors={[l.get_color() for l in lines]}")

    # 1 black median line (baseline) + 1 bold highlighted line -- baseline
    # is no longer rendered as individual per-case lines.
    assert len(lines) == 2
    highlight_lines = [l for l in lines if l.get_label() == "H"]
    median_lines = [l for l in lines if l.get_label() != "H"]
    assert len(highlight_lines) == 1
    assert highlight_lines[0].get_alpha() == 1.0
    assert median_lines[0].get_color() == "black"

    # 2 shaded percentile bands (10th-90th outer, 25th-75th inner)
    assert len(fills) == 2

    # legend uses "All Cases" wording, not "Other cases"
    legend_labels = [f.get_label() for f in fills] + [median_lines[0].get_label()]
    print(f"[highlight-ts-check1] legend_labels={legend_labels}")
    assert all(label.startswith("All Cases") for label in legend_labels)
    assert not any("Other" in label for label in legend_labels)
    plt.close(fig)


def test_empty_baseline_timeseries_renders_highlighted_only_no_bands(tmp_path):
    cases = _three_case_fixture(tmp_path)
    highlight = sorted(cases.keys())[0]
    highlighted_clim = cases[highlight]
    highlighted_stats = compute_verification_stats(highlighted_clim)

    out_path = str(tmp_path / "out.png")
    plot_single_case_highlight_timeseries([], highlighted_stats, "pod", out_path, ylabel="POD", title="t", highlight_label="H")
    assert os.path.isfile(out_path)


# --- Case scatter ---------------------------------------------------------------

def test_case_scatter_shared_size_scale_equal_counts_equal_size(tmp_path):
    """Baseline and highlighted markers share ONE size scale based on
    object count -- cases with the same count must render at the same
    size regardless of which one is highlighted; a darker-gray baseline
    marker; no legend; the highlighted label drawn above the marker."""
    d = tmp_path / "A"
    d.mkdir()
    t1, t2, t3 = datetime(2026, 5, 1), datetime(2026, 5, 2), datetime(2026, 5, 3)
    _write_case(str(d / f"match_init_{t1:%Y%m%d_%H%M%S}.nc"), t1, {(None, 0): ["hit", "hit"]})    # 2 objects
    _write_case(str(d / f"match_init_{t2:%Y%m%d_%H%M%S}.nc"), t2, {(None, 0): ["hit", "hit"]})    # 2 objects (same count)
    _write_case(str(d / f"match_init_{t3:%Y%m%d_%H%M%S}.nc"), t3, {(None, 0): ["hit"] * 10})       # 10 objects

    per_case = build_per_case_match_climatologies([SystemSpec("A", str(d))], case_grouping="init_time")
    cases = per_case["A"]
    highlight = sorted(cases.keys())[0]  # t1 (2 objects)
    baseline_clims, highlighted_clim = split_baseline_and_highlight(cases, highlight)  # t2 (2), t3 (10)

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_single_case_highlight_case_scatter(baseline_clims, highlighted_clim, lead_minutes_range=(0, 0),
                                              out_path=str(tmp_path / "out.png"), highlight_label="H")
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    scatters = [c for c in ax.collections if isinstance(c, matplotlib.collections.PathCollection)]
    print(f"\n[highlight-scatter-check1] n_scatters={len(scatters)}")
    assert len(scatters) == 3  # 2 baseline + 1 highlighted
    baseline_scatters = [c for c in scatters if tuple(c.get_facecolor()[0][:3]) == pytest.approx(cb_colors.gray7)]
    assert len(baseline_scatters) == 2  # darker gray, not gray5
    highlighted_scatter = [c for c in scatters if c not in baseline_scatters][0]
    baseline_sizes = sorted(c.get_sizes()[0] for c in baseline_scatters)
    highlighted_size = highlighted_scatter.get_sizes()[0]
    print(f"[highlight-scatter-check1] baseline_sizes={baseline_sizes}, highlighted_size={highlighted_size}")
    # highlighted case (2 objects) matches the equal-count baseline case exactly
    assert highlighted_size == pytest.approx(baseline_sizes[0])
    # the 10-object baseline case is strictly larger than both
    assert baseline_sizes[1] > highlighted_size

    assert ax.get_legend() is None  # no legend in this plot
    texts = [t.get_text() for t in ax.texts if t.get_text() == "H"]
    assert texts == ["H"]
    plt.close(fig)


def test_empty_baseline_case_scatter_still_renders_highlighted_only(tmp_path):
    cases = _three_case_fixture(tmp_path)
    highlight = sorted(cases.keys())[0]
    highlighted_clim = cases[highlight]

    out_path = str(tmp_path / "out.png")
    plot_single_case_highlight_case_scatter([], highlighted_clim, lead_minutes_range=(0, 60),
                                              out_path=out_path, highlight_label="H")
    assert os.path.isfile(out_path)


# --- Real data ------------------------------------------------------------------

@_SKIP_NO_NCAR_MPAS_ANALYSIS
def test_real_mpas_single_case_highlight_both_plot_families(tmp_path):
    systems = [SystemSpec("100km", os.path.join(NCAR_MPAS_DIR, "matched_objects_100km"))]
    per_case = build_per_case_match_climatologies(systems, lead_time_mode="intersection", case_grouping="init_time")
    cases = per_case["100km"]
    highlight = sorted(cases.keys())[0]
    baseline_clims, highlighted_clim = split_baseline_and_highlight(cases, highlight)
    print(f"\n[highlight-real-check1] n_baseline={len(baseline_clims)}, highlight={highlight}")
    assert len(baseline_clims) == 117  # 118 real cases total, minus the highlighted one

    baseline_stats = [compute_verification_stats(c) for c in baseline_clims]
    highlighted_stats = compute_verification_stats(highlighted_clim)

    ts_path = str(tmp_path / "ts.png")
    scatter_path = str(tmp_path / "scatter.png")
    plot_single_case_highlight_timeseries(baseline_stats, highlighted_stats, "pod", ts_path, ylabel="POD", title="t", highlight_label=highlight)
    plot_single_case_highlight_case_scatter(baseline_clims, highlighted_clim, (1920, 2220), scatter_path, highlight_label=highlight)

    assert os.path.isfile(ts_path) and os.path.isfile(scatter_path)
