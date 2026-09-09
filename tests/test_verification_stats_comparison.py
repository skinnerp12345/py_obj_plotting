"""Validation of plots.verification_stats_comparison -- confirms overlaying
multiple threshold series on one figure produces the expected per-series
line/color/legend structure, and that mismatched lead_hour coverage across
series is handled (x-axis spans the union, not just the first series').

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_verification_stats_comparison.py -v -s
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from py_obj_plotting import cb_colors
from py_obj_plotting.climatology import VerificationStats
from py_obj_plotting.plots.verification_stats_comparison import (
    _plot_comparison_timeseries,
    plot_csi_comparison_timeseries,
    plot_fbias_comparison_timeseries,
    plot_far_comparison_timeseries,
    plot_pod_comparison_timeseries,
)


def _stats(lead_hours, members, value) -> VerificationStats:
    d = {(m, lh): value for m in members for lh in lead_hours}
    ci = {(m, lh): 0.05 for m in members for lh in lead_hours}
    return VerificationStats(
        lead_hours=lead_hours, members=members,
        pod=d, pod_ci_width=ci, far=d, far_ci_width=ci, csi=d, csi_ci_width=ci, fbias=d, fbias_ci_width=ci,
    )


def test_plot_pod_comparison_draws_one_series_per_threshold(tmp_path):
    members = ["mem_1", "mem_2"]
    series = [
        ("100km", cb_colors.orange5, _stats([0.0, 1.0, 2.0], members, 0.5)),
        ("200km", cb_colors.green5, _stats([0.0, 1.0, 2.0], members, 0.4)),
        ("300km", cb_colors.blue5, _stats([0.0, 1.0, 2.0], members, 0.3)),
        ("400km", cb_colors.purple5, _stats([0.0, 1.0, 2.0], members, 0.2)),
    ]
    out_path = str(tmp_path / "pod_comparison.png")

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_pod_comparison_timeseries(series, out_path)
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    lines = ax.get_lines()
    print(f"\n[compare-check1] n_lines={len(lines)}, n_shaded={len(ax.collections)}")
    assert len(lines) == 4 * 3  # 4 thresholds x (2 members + 1 mean)
    assert len(ax.collections) == 4  # one CI band per threshold

    expected_colors = {cb_colors.orange5, cb_colors.green5, cb_colors.blue5, cb_colors.purple5}
    actual_colors = {line.get_color() for line in lines}
    assert actual_colors == expected_colors

    legend_labels = {l.get_label() for l in lines if l.get_label() and not l.get_label().startswith("_")}
    assert legend_labels == {"100km (member)", "200km (member)", "300km (member)", "400km (member)"}
    plt.close(fig)


def test_plot_comparison_handles_mismatched_lead_hour_coverage(tmp_path):
    """One series covering fewer lead hours than another must not truncate
    the x-axis to the shorter series -- the union of all series' lead hours
    should be used."""
    members = ["mem_1"]
    series = [
        ("short", cb_colors.orange5, _stats([0.0, 1.0], members, 0.5)),
        ("long", cb_colors.blue5, _stats([0.0, 1.0, 2.0, 3.0], members, 0.3)),
    ]
    out_path = str(tmp_path / "pod_mismatched.png")

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_pod_comparison_timeseries(series, out_path)
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    print(f"\n[compare-check2] xlim={ax.get_xlim()}")
    # xlim is the union's [min, max] padded by +/-5 minutes (1/12 hour) on
    # each side, not the raw union bounds -- see lead_hour_timeseries._lead_hour_xlim.
    pad = 5.0 / 60.0
    lo, hi = ax.get_xlim()
    assert abs(lo - (0.0 - pad)) < 1e-9
    assert abs(hi - (3.0 + pad)) < 1e-9
    plt.close(fig)


def test_plot_comparison_timeseries_requires_nonempty_series(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        _plot_comparison_timeseries([], "pod", "pod_ci_width", str(tmp_path / "x.png"), "y", "t")


def test_pod_far_csi_comparison_have_fixed_zero_one_ylim_fbias_does_not(tmp_path):
    members = ["mem_1"]
    series = [
        ("100km", cb_colors.orange5, _stats([0.0, 1.0], members, 0.6)),
        ("200km", cb_colors.green5, _stats([0.0, 1.0], members, 0.4)),
    ]
    orig_close = plt.close
    plt.close = lambda *a, **k: None

    for plot_fn, expected_ylim in [
        (plot_pod_comparison_timeseries, (0.0, 1.0)),
        (plot_far_comparison_timeseries, (0.0, 1.0)),
        (plot_csi_comparison_timeseries, (0.0, 1.0)),
    ]:
        plot_fn(series, str(tmp_path / "x.png"))
        ax = plt.gcf().axes[0]
        print(f"\n[compare-check3] {plot_fn.__name__} ylim={ax.get_ylim()}")
        assert ax.get_ylim() == expected_ylim
        plt.close(plt.gcf())

    plot_fbias_comparison_timeseries(series, str(tmp_path / "fbias.png"))
    ax = plt.gcf().axes[0]
    print(f"[compare-check3] plot_fbias_comparison_timeseries ylim={ax.get_ylim()}")
    assert ax.get_ylim() != (0.0, 1.0)
    plt.close(plt.gcf())

    plt.close = orig_close
