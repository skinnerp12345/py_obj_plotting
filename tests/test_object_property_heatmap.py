"""Validation of plots.object_property_heatmap: numeric checks only (mesh
array values, colormap/norm boundaries, NaN-masking of zero-count cells,
significant-vs-insignificant alpha split) -- never by viewing the
rendered image, per this project's established convention. Climatology
objects are constructed directly (not via file I/O) since this is a
plotting-layer test, mirroring test_performance_diagram.py's own approach.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_object_property_heatmap.py -v -s
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from py_obj_plotting.climatology.aggregate_object_property_pod import ObjectPropertyPodClimatology
from py_obj_plotting.plots.object_property_heatmap import (
    _REFERENCE_LABEL_INTERVAL,
    _REFERENCE_N_LEAD_TIMES,
    _bin_labels,
    _dynamic_label_interval,
    _symmetric_diff_levels,
    plot_object_property_pod_diff_heatmap,
    plot_object_property_pod_heatmap,
)


def _clim(bin_edges, n_hits, n_misses, lead_hours) -> ObjectPropertyPodClimatology:
    return ObjectPropertyPodClimatology(
        lead_hours=lead_hours, members=["mem_1"], bin_edges=bin_edges, n_hits=n_hits, n_misses=n_misses,
    )


def test_pod_heatmap_hand_computed_grid_values(tmp_path):
    # bin 0 @ lead 0.0: 8 hits, 2 misses -> POD=0.8; bin 1 @ lead 1.0: 1 hit, 1 miss -> POD=0.5
    clim = _clim(
        bin_edges={"area": np.array([0.0, 10.0, 20.0])},
        n_hits={"area": {(0, 0.0): 8, (1, 1.0): 1}},
        n_misses={"area": {(0, 0.0): 2, (1, 1.0): 1}},
        lead_hours=[0.0, 1.0],
    )
    out_path = str(tmp_path / "out.png")

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_object_property_pod_heatmap(clim, "area", "Object Area (km2)", out_path)
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    mesh = [c for c in ax.collections if hasattr(c, "get_array")][0]
    # pcolor's mesh stores NaN cells as a masked array, not literal NaN --
    # .filled(nan) recovers plain NaN for a normal np.isnan() check.
    grid = np.ma.filled(mesh.get_array(), np.nan).reshape(2, 2)  # (n_bins, n_lead_hours)
    print(f"\n[heatmap-check1] grid={grid}")
    assert grid[0, 0] == pytest.approx(0.8)
    assert grid[1, 1] == pytest.approx(0.5)
    assert np.isnan(grid[1, 0])  # zero-count cell -- undefined POD, not 0.0
    assert np.isnan(grid[0, 1])
    plt.close(fig)


def test_bin_labels_formats_overflow_bin_and_normal_bins():
    edges = np.array([0.0, 200.0, 400.0, np.inf])
    labels = _bin_labels(edges)
    print(f"\n[heatmap-check-bins] labels={labels}")
    assert labels == ["0-200", "200-400", ">400"]


def test_yaxis_uses_bin_index_positions_not_physical_values(tmp_path):
    """A caller-supplied overflow bin (edges=[...,2000,inf]) must not
    distort the y-axis -- confirmed the mesh's own y-coordinates are plain
    bin-index integers (0,1,2,...), never physical bin values (which would
    include an effectively-infinite value for the overflow bin's own
    'center')."""
    edges = np.array([0.0, 200.0, 400.0, np.inf])
    clim = _clim(
        bin_edges={"area": edges},
        n_hits={"area": {(0, 0.0): 5, (1, 0.0): 5, (2, 0.0): 5}},
        n_misses={"area": {(0, 0.0): 5, (1, 0.0): 5, (2, 0.0): 5}},
        lead_hours=[0.0],
    )
    out_path = str(tmp_path / "out.png")
    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_object_property_pod_heatmap(clim, "area", "Object Area (km2)", out_path)
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    yticks = list(ax.get_yticks())
    yticklabels = [t.get_text() for t in ax.get_yticklabels()]
    print(f"\n[heatmap-check-yaxis] yticks={yticks}, yticklabels={yticklabels}")
    assert yticks == [0.0, 1.0, 2.0]  # plain bin-index positions, not [100, 300, inf]
    assert yticklabels == ["0-200", "200-400", ">400"]
    assert np.isfinite(ax.get_ylim()).all()  # axis limits stay finite/sane despite the inf bin edge
    plt.close(fig)


def test_object_count_shown_under_pod_label(tmp_path):
    clim = _clim(
        bin_edges={"area": np.array([0.0, 10.0])},
        n_hits={"area": {(0, 2.0): 8}}, n_misses={"area": {(0, 2.0): 2}},
        lead_hours=[0.0, 1.0, 2.0, 3.0, 4.0],
    )
    out_path = str(tmp_path / "out.png")
    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_object_property_pod_heatmap(clim, "area", "Object Area (km2)", out_path)
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts if t.get_text()]
    print(f"\n[heatmap-check-count] texts={texts}")
    assert "0.80\n(10)" in texts  # 8 hits + 2 misses = 10 total
    plt.close(fig)


def test_diff_heatmap_count_is_mean_of_both_systems(tmp_path):
    lead_hours = [0.0, 1.0, 2.0, 3.0, 4.0]
    clim_a = _clim(
        bin_edges={"area": np.array([0.0, 10.0])},
        n_hits={"area": {(0, 2.0): 8}}, n_misses={"area": {(0, 2.0): 2}},  # count_a=10
        lead_hours=lead_hours,
    )
    clim_b = _clim(
        bin_edges={"area": np.array([0.0, 10.0])},
        n_hits={"area": {(0, 2.0): 15}}, n_misses={"area": {(0, 2.0): 5}},  # count_b=20
        lead_hours=lead_hours,
    )
    out_path = str(tmp_path / "out.png")
    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_object_property_pod_diff_heatmap(clim_a, clim_b, "area", "Object Area (km2)", out_path, ci_samples=200)
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts if t.get_text()]
    print(f"\n[heatmap-check-diff-count] texts={texts}")
    assert any(t.endswith("(15)") for t in texts)  # mean(10, 20) = 15
    plt.close(fig)


def test_dynamic_label_interval_scales_with_column_count():
    # at the reference column count, the interval must equal the reference interval exactly
    assert _dynamic_label_interval(_REFERENCE_N_LEAD_TIMES) == _REFERENCE_LABEL_INTERVAL
    # far more columns (MPAS-like, ~133) -> a proportionally LARGER interval
    interval_many = _dynamic_label_interval(133)
    print(f"\n[heatmap-check-interval] many={interval_many}, reference={_REFERENCE_LABEL_INTERVAL}")
    assert interval_many > _REFERENCE_LABEL_INTERVAL
    # far fewer columns (WoFSCast-like, ~35) -> a proportionally SMALLER interval
    interval_few = _dynamic_label_interval(35)
    print(f"[heatmap-check-interval] few={interval_few}")
    assert interval_few < _REFERENCE_LABEL_INTERVAL
    # never zero (would infinite-loop range())
    assert _dynamic_label_interval(1) >= 1
    assert _dynamic_label_interval(0) >= 1


def test_pod_heatmap_zero_count_cells_get_no_text_label(tmp_path):
    clim = _clim(
        bin_edges={"area": np.array([0.0, 10.0])},
        n_hits={"area": {(0, 0.0): 5}}, n_misses={"area": {}},
        lead_hours=[0.0],
    )
    out_path = str(tmp_path / "out.png")
    plot_object_property_pod_heatmap(clim, "area", "Object Area (km2)", out_path)
    assert __import__("os").path.isfile(out_path)


def test_pod_heatmap_no_data_raises(tmp_path):
    clim = _clim(bin_edges={"area": np.array([0.0, 10.0])}, n_hits={"area": {}}, n_misses={"area": {}}, lead_hours=[0.0])
    with pytest.raises(ValueError, match="no hit/miss data"):
        plot_object_property_pod_heatmap(clim, "area", "y", str(tmp_path / "out.png"))


def test_pod_heatmap_unknown_property_raises(tmp_path):
    clim = _clim(bin_edges={"area": np.array([0.0, 10.0])}, n_hits={"area": {(0, 0.0): 1}}, n_misses={"area": {}}, lead_hours=[0.0])
    with pytest.raises(ValueError, match="no bin_edges"):
        plot_object_property_pod_heatmap(clim, "max_intensity", "y", str(tmp_path / "out.png"))


def test_symmetric_diff_levels_floor_applies_when_range_is_small():
    grid = np.array([[0.01, np.nan], [np.nan, -0.02]])
    levels = _symmetric_diff_levels(grid, floor=0.2, bin_width=0.05)
    print(f"\n[heatmap-check2] levels_range=({levels[0]}, {levels[-1]})")
    assert levels[0] == pytest.approx(-0.2)
    assert levels[-1] == pytest.approx(0.2)


def test_symmetric_diff_levels_reflects_real_range_when_it_exceeds_floor():
    grid = np.array([[0.5, np.nan], [np.nan, -0.3]])
    levels = _symmetric_diff_levels(grid, floor=0.2, bin_width=0.05)
    print(f"\n[heatmap-check3] levels_range=({levels[0]}, {levels[-1]})")
    assert levels[-1] == pytest.approx(0.5)
    assert levels[0] == pytest.approx(-0.5)


def test_diff_heatmap_significant_vs_insignificant_alpha(tmp_path):
    """One cell with a huge, lopsided sample on both sides (POD estimates
    essentially certain -> the real diff exceeds its own tiny bootstrap CI
    -> significant); one cell with a small, ambiguous sample (large CI ->
    the diff does NOT exceed it -> insignificant). Data only exists at
    lead_hour=2.0 (index 2 of 5 total columns) -- the OTHER candidate
    label columns _dynamic_label_interval picks for a 5-column grid have
    no data there and are skipped by _label_cells' own NaN check, so
    exactly one label per bin/row still lands, regardless of interval."""
    np.random.seed(0)
    lead_hours = [0.0, 1.0, 2.0, 3.0, 4.0]
    clim_a = _clim(
        bin_edges={"area": np.array([0.0, 10.0, 20.0])},
        n_hits={"area": {(0, 2.0): 9900, (1, 2.0): 6}},
        n_misses={"area": {(0, 2.0): 100, (1, 2.0): 4}},
        lead_hours=lead_hours,
    )
    clim_b = _clim(
        bin_edges={"area": np.array([0.0, 10.0, 20.0])},
        n_hits={"area": {(0, 2.0): 100, (1, 2.0): 5}},
        n_misses={"area": {(0, 2.0): 9900, (1, 2.0): 5}},
        lead_hours=lead_hours,
    )
    out_path = str(tmp_path / "out.png")

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_object_property_pod_diff_heatmap(clim_a, clim_b, "area", "Object Area (km2)", out_path, ci_samples=300)
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    texts = [t for t in ax.texts if t.get_text() not in ("",)]
    # y-position is now the plain bin INDEX (0, 1, ...), not a physical
    # bin-center value -- see plot_object_property_pod_heatmap's y-axis
    # redesign (module docstring).
    alphas_by_bin = {round(t.get_position()[1]): t.get_alpha() for t in texts}
    fontweights_by_bin = {round(t.get_position()[1]): t.get_fontweight() for t in texts}
    print(f"\n[heatmap-check4] alphas_by_bin={alphas_by_bin} fontweights_by_bin={fontweights_by_bin}")
    # bin 0 = [0,10) (huge, lopsided sample -> significant) should be full
    # alpha + bold (higher contrast); bin 1 = [10,20) (small, ambiguous
    # sample -> not significant) should be more transparent than the old
    # flat 0.5 + normal weight.
    assert alphas_by_bin[0] == pytest.approx(1.0)
    assert fontweights_by_bin[0] == "bold"
    assert alphas_by_bin[1] == pytest.approx(0.35)
    assert fontweights_by_bin[1] == "normal"
    plt.close(fig)


def test_diff_heatmap_mismatched_bin_edges_raises(tmp_path):
    clim_a = _clim(bin_edges={"area": np.array([0.0, 10.0])}, n_hits={"area": {(0, 0.0): 5}}, n_misses={"area": {}}, lead_hours=[0.0])
    clim_b = _clim(bin_edges={"area": np.array([0.0, 20.0])}, n_hits={"area": {(0, 0.0): 5}}, n_misses={"area": {}}, lead_hours=[0.0])
    with pytest.raises(ValueError, match="bin_edges.*differ"):
        plot_object_property_pod_diff_heatmap(clim_a, clim_b, "area", "y", str(tmp_path / "out.png"))


def test_diff_heatmap_no_shared_lead_hours_raises(tmp_path):
    clim_a = _clim(bin_edges={"area": np.array([0.0, 10.0])}, n_hits={"area": {(0, 0.0): 5}}, n_misses={"area": {}}, lead_hours=[0.0])
    clim_b = _clim(bin_edges={"area": np.array([0.0, 10.0])}, n_hits={"area": {(0, 1.0): 5}}, n_misses={"area": {}}, lead_hours=[1.0])
    with pytest.raises(ValueError, match="no shared lead hours"):
        plot_object_property_pod_diff_heatmap(clim_a, clim_b, "area", "y", str(tmp_path / "out.png"))
