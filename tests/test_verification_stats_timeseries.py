"""Validation of plots.verification_stats_timeseries's shared
_plot_member_ensemble_series helper: genuinely reusable to overlay multiple
series (e.g. different systems or matching thresholds) on one shared
ax/figure, which is the point of factoring it out separately from figure
creation (see the module's own docstring).

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_verification_stats_timeseries.py -v -s
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from py_obj_plotting import cb_colors
from py_obj_plotting.plots.verification_stats_timeseries import _plot_member_ensemble_series


def test_plot_member_ensemble_series_reusable_for_multi_threshold_overlay():
    """The core claim this design is built around: calling the low-level
    helper twice on ONE shared ax, with different colors/labels, must
    overlay both series distinctly -- exactly what a future multi-threshold
    comparison plot needs, with zero changes to this function."""
    lead_hours = [0.0, 1.0, 2.0]
    members = ["mem_1", "mem_2"]
    values_a = {(m, lh): 0.5 for m in members for lh in lead_hours}
    ci_a = {(m, lh): 0.1 for m in members for lh in lead_hours}
    values_b = {(m, lh): 0.3 for m in members for lh in lead_hours}
    ci_b = {(m, lh): 0.05 for m in members for lh in lead_hours}

    fig, ax = plt.subplots()
    _plot_member_ensemble_series(ax, lead_hours, members, values_a, ci_a, color=cb_colors.orange5, series_label="100km")
    _plot_member_ensemble_series(ax, lead_hours, members, values_b, ci_b, color=cb_colors.blue5, series_label="200km")

    lines = ax.get_lines()
    print(f"\n[verif-plot-check2] n_lines={len(lines)}, n_shaded={len(ax.collections)}")
    assert len(lines) == 6  # (2 members + 1 mean) x 2 series
    assert len(ax.collections) == 2  # one fill_between per series

    orange_lines = [l for l in lines if l.get_color() == cb_colors.orange5]
    blue_lines = [l for l in lines if l.get_color() == cb_colors.blue5]
    assert len(orange_lines) == 3
    assert len(blue_lines) == 3

    labels = {l.get_label() for l in lines if l.get_label() and not l.get_label().startswith("_")}
    assert labels == {"100km (member)", "200km (member)"}
    plt.close(fig)


def test_ci_band_spans_half_the_full_width_on_each_side():
    """A CI width is the FULL interval width (upper - lower percentile), so the
    band must be value +/- width/2: total band height == width, not 2 x width
    (regression: the band used to be drawn at +/- the full width)."""
    import numpy as np

    lead_hours = [0.0, 1.0, 2.0]
    members = [None]
    values = {(None, lh): 0.4 for lh in lead_hours}
    widths = {(None, lh): 0.1 for lh in lead_hours}

    fig, ax = plt.subplots()
    _plot_member_ensemble_series(ax, lead_hours, members, values, widths)
    y = np.concatenate([p.vertices[:, 1] for p in ax.collections[0].get_paths()])
    assert np.isclose(y.min(), 0.35) and np.isclose(y.max(), 0.45)
    plt.close(fig)


def test_deterministic_system_renders_one_bold_line_not_member_spaghetti():
    """A system with members == [None] (this project's established
    sentinel for "no member dimension at all," e.g. a deterministic
    forecast with no ensemble members) must render as ONE bold, fully-opaque line
    carrying its own legend label -- not a thin, semi-transparent "member"
    line duplicated by an on-top ensemble-mean line (the previous,
    unintentional "1-member ensemble" rendering)."""
    lead_hours = [0.0, 1.0, 2.0]
    values = {(None, lh): 0.6 for lh in lead_hours}
    ci = {(None, lh): 0.1 for lh in lead_hours}

    fig, ax = plt.subplots()
    _plot_member_ensemble_series(ax, lead_hours, [None], values, ci, color=cb_colors.green5, series_label="NowcastNet")

    lines = ax.get_lines()
    print(f"\n[verif-plot-check4] n_lines={len(lines)}, alphas={[l.get_alpha() for l in lines]}, "
          f"labels={[l.get_label() for l in lines]}")
    assert len(lines) == 1  # not 2 (one thin "member" line + one thick "mean" line)
    assert lines[0].get_alpha() == 1.0  # fully opaque, not the 0.35 member-alpha
    assert lines[0].get_label() == "NowcastNet"  # carries the legend entry directly, no "(member)" suffix
    assert len(ax.collections) == 1  # still gets its own CI-shaded band
    plt.close(fig)


def test_real_ensemble_with_exactly_one_named_member_still_gets_member_styling():
    """A genuine 1-member ENSEMBLE (a real member name, not the None
    sentinel) is structurally different from a deterministic system and
    must still render as an ensemble (thin member line + thick mean line,
    even though they're numerically identical for a single member) -- the
    distinguishing signal is members == [None] specifically, not len==1."""
    lead_hours = [0.0, 1.0]
    values = {("mem_1", lh): 0.6 for lh in lead_hours}
    ci = {("mem_1", lh): 0.1 for lh in lead_hours}

    fig, ax = plt.subplots()
    _plot_member_ensemble_series(ax, lead_hours, ["mem_1"], values, ci, color=cb_colors.blue5, series_label="Solo")

    lines = ax.get_lines()
    print(f"\n[verif-plot-check5] n_lines={len(lines)}, labels={[l.get_label() for l in lines]}")
    assert len(lines) == 2
    assert lines[0].get_label() == "Solo (member)"
    plt.close(fig)
