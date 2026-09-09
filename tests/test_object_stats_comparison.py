"""Validation of plots.object_stats_comparison -- confirms the multi-system
object count/area comparison timeseries plots include a black MRMS truth
reference line, and that the pre-existing box-whisker plots' MRMS inclusion
is unaffected.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_object_stats_comparison.py -v -s
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from py_obj_plotting import cb_colors
from py_obj_plotting.climatology import ObjectStatsMultiSystemClimatology
from py_obj_plotting.plots.object_stats_comparison import (
    plot_object_area_boxplot,
    plot_object_area_comparison_timeseries,
    plot_object_count_comparison_timeseries,
    plot_object_mean_intensity_boxplot,
)


def _clim_fixture(members, forecast_count_base) -> ObjectStatsMultiSystemClimatology:
    lead_hours = [0.0, 1.0, 2.0]
    forecast_count = {(m, lh): forecast_count_base + i for i, (m, lh) in
                       enumerate((m, lh) for m in members for lh in lead_hours)}
    forecast_area_km2 = {k: v * 10.0 for k, v in forecast_count.items()}
    # truth_* is shared/identical across every system's own climatology
    # object (see aggregate_matches_multi_system.build_multi_system_object_climatology)
    truth_count = {lh: 5 + i for i, lh in enumerate(lead_hours)}
    truth_area_km2 = {lh: v * 20.0 for lh, v in truth_count.items()}
    return ObjectStatsMultiSystemClimatology(
        lead_hours=lead_hours, members=members,
        forecast_count=forecast_count, forecast_area_km2=forecast_area_km2,
        forecast_area_values=[100.0, 110.0], forecast_mean_intensity_values=[40.0, 42.0],
        truth_lead_hours=lead_hours, truth_count=truth_count, truth_area_km2=truth_area_km2,
        truth_area_values=[90.0, 95.0], truth_mean_intensity_values=[38.0, 39.0],
    )


def _series_fixture():
    return [
        ("SystemA", cb_colors.blue5, _clim_fixture(["mem_1", "mem_2"], 1)),
        ("SystemB", cb_colors.orange5, _clim_fixture(["mem_1"], 5)),
    ]


def test_object_count_comparison_includes_black_mrms_line(tmp_path):
    series = _series_fixture()
    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_object_count_comparison_timeseries(series, str(tmp_path / "count.png"))
    plt.close = orig_close

    ax = plt.gcf().axes[0]
    lines = ax.get_lines()
    black_lines = [l for l in lines if l.get_color() == "black"]
    print(f"\n[objstats-comp-check1] n_lines={len(lines)}, n_black={len(black_lines)}, "
          f"labels={[l.get_label() for l in lines]}")
    assert len(black_lines) == 1
    assert black_lines[0].get_label() == "MRMS"
    plt.close(plt.gcf())


def test_object_area_comparison_includes_black_mrms_line(tmp_path):
    series = _series_fixture()
    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_object_area_comparison_timeseries(series, str(tmp_path / "area.png"))
    plt.close = orig_close

    ax = plt.gcf().axes[0]
    black_lines = [l for l in ax.get_lines() if l.get_color() == "black"]
    print(f"\n[objstats-comp-check2] n_black={len(black_lines)}")
    assert len(black_lines) == 1
    assert black_lines[0].get_label() == "MRMS"
    plt.close(plt.gcf())


def test_object_count_comparison_mrms_values_match_truth_count(tmp_path):
    """Confirms the black line's actual plotted y-values are truth_count
    (not, say, accidentally re-plotting a forecast series) -- reads the
    climatology's own truth_count values as the source of truth."""
    series = _series_fixture()
    _, _, clim = series[0]
    expected = [clim.truth_count[lh] for lh in clim.truth_lead_hours]

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_object_count_comparison_timeseries(series, str(tmp_path / "count.png"))
    plt.close = orig_close

    ax = plt.gcf().axes[0]
    black_line = next(l for l in ax.get_lines() if l.get_color() == "black")
    actual = list(black_line.get_ydata())
    print(f"\n[objstats-comp-check3] expected={expected}, actual={actual}")
    assert actual == expected
    plt.close(plt.gcf())


def test_title_override_applies_to_both_comparison_timeseries(tmp_path):
    series = _series_fixture()
    orig_close = plt.close
    plt.close = lambda *a, **k: None
    for plot_fn, out_name in [
        (plot_object_count_comparison_timeseries, "count_title.png"),
        (plot_object_area_comparison_timeseries, "area_title.png"),
    ]:
        plot_fn(series, str(tmp_path / out_name), title="Custom title (Linear storm mode)")
        ax = plt.gcf().axes[0]
        print(f"\n[objstats-comp-check5] {plot_fn.__name__} title={ax.get_title()!r}")
        assert ax.get_title() == "Custom title (Linear storm mode)"
        plt.close(plt.gcf())
    plt.close = orig_close


def test_boxplots_still_include_mrms_unaffected(tmp_path):
    series = _series_fixture()
    orig_close = plt.close
    plt.close = lambda *a, **k: None
    for plot_fn, out_name in [
        (plot_object_area_boxplot, "area_box.png"),
        (plot_object_mean_intensity_boxplot, "intensity_box.png"),
    ]:
        plot_fn(series, str(tmp_path / out_name))
        ax = plt.gcf().axes[0]
        labels = [t.get_text() for t in ax.get_xticklabels()]
        print(f"\n[objstats-comp-check4] {plot_fn.__name__} labels={labels}")
        assert labels == ["SystemA", "SystemB", "MRMS"]
        plt.close(plt.gcf())
    plt.close = orig_close
