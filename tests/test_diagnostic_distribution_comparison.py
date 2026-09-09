"""Validation of plots.diagnostic_distribution_comparison -- confirms
overlaying multiple threshold series on one figure produces one band+median
per threshold, correctly colored.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_diagnostic_distribution_comparison.py -v -s
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from py_obj_plotting import cb_colors
from py_obj_plotting.climatology import DiagnosticDistributionClimatology
from py_obj_plotting.plots.diagnostic_distribution_comparison import (
    _plot_diagnostic_comparison_timeseries,
    plot_centroid_dist_comparison_timeseries,
)


def _clim(lead_hours, base) -> DiagnosticDistributionClimatology:
    centroid_dist_km = {lh: [base, base + 10, base + 20] for lh in lead_hours}
    return DiagnosticDistributionClimatology(lead_hours=lead_hours, centroid_dist_km=centroid_dist_km)


def test_plot_centroid_dist_comparison_draws_one_series_per_threshold(tmp_path):
    series = [
        ("100km", cb_colors.orange5, _clim([0.0, 1.0], 10.0)),
        ("200km", cb_colors.green5, _clim([0.0, 1.0], 20.0)),
        ("300km", cb_colors.blue5, _clim([0.0, 1.0], 30.0)),
        ("400km", cb_colors.purple5, _clim([0.0, 1.0], 40.0)),
    ]
    out_path = str(tmp_path / "centroid_dist_comparison.png")

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_centroid_dist_comparison_timeseries(series, out_path)
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    lines = ax.get_lines()
    print(f"\n[diag-compare-check1] n_lines={len(lines)}, n_shaded={len(ax.collections)}")
    assert len(lines) == 4
    assert len(ax.collections) == 4
    colors = {l.get_color() for l in lines}
    assert colors == {cb_colors.orange5, cb_colors.green5, cb_colors.blue5, cb_colors.purple5}
    labels = {l.get_label() for l in lines}
    assert labels == {"100km median", "200km median", "300km median", "400km median"}
    plt.close(fig)


def test_plot_diagnostic_comparison_timeseries_requires_nonempty_series(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        _plot_diagnostic_comparison_timeseries([], "centroid_dist_km", str(tmp_path / "x.png"), "y", "t")
