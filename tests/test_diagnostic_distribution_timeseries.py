"""Validation of plots.diagnostic_distribution_timeseries's shared
_plot_percentile_band_series helper: reusable to overlay multiple series on
one shared ax (the same caller-supplied-ax architecture
verification_stats_timeseries.py uses, applied to the distribution-band
convention here).

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_diagnostic_distribution_timeseries.py -v -s
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from py_obj_plotting import cb_colors
from py_obj_plotting.plots.diagnostic_distribution_timeseries import _plot_percentile_band_series


def test_plot_percentile_band_series_reusable_for_multi_threshold_overlay():
    lead_hours = [0.0, 1.0, 2.0]
    values_a = {lh: [10.0, 20.0, 30.0] for lh in lead_hours}
    values_b = {lh: [5.0, 15.0, 25.0] for lh in lead_hours}

    fig, ax = plt.subplots()
    _plot_percentile_band_series(ax, lead_hours, values_a, color=cb_colors.orange5, series_label="100km")
    _plot_percentile_band_series(ax, lead_hours, values_b, color=cb_colors.green5, series_label="200km")

    lines = ax.get_lines()
    print(f"\n[diag-plot-check2] n_lines={len(lines)}, n_shaded={len(ax.collections)}")
    assert len(lines) == 2
    assert len(ax.collections) == 2
    labels = {l.get_label() for l in lines}
    assert labels == {"100km median", "200km median"}
    plt.close(fig)
