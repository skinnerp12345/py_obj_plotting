"""Validation of plots.spatial_map: numeric checks only (extent values, lon
normalization, projection type, output file existence) -- never by viewing
the rendered image, per this project's established convention.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_spatial_map.py -v -s
"""

import math
import os

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from py_obj_plotting import cb_colors
from py_obj_plotting.plots.spatial_map import (
    add_reflectivity_colorbar,
    add_reflectivity_layer,
    build_base_map,
    crop_extent_from_mask,
    finalize_and_save,
    lon_to_plot_range,
)


def test_lon_to_plot_range_shifts_only_values_above_180():
    lon2d = np.array([[10.0, 200.0], [359.0, 180.0]])
    out = lon_to_plot_range(lon2d)
    print(f"\n[spatial-map-check1] in={lon2d.tolist()} out={out.tolist()}")
    assert out[0, 0] == pytest.approx(10.0)
    assert out[0, 1] == pytest.approx(200.0 - 360.0)
    assert out[1, 0] == pytest.approx(359.0 - 360.0)
    assert out[1, 1] == pytest.approx(180.0)  # boundary case: not > 180, unchanged


def test_crop_extent_from_mask_hand_computed():
    lat2d = np.array([[30.0, 30.0, 30.0], [35.0, 35.0, 35.0], [40.0, 40.0, 40.0]])
    lon2d = np.array([[260.0, 265.0, 270.0], [260.0, 265.0, 270.0], [260.0, 265.0, 270.0]])
    keep_mask = np.zeros_like(lat2d, dtype=bool)
    keep_mask[0, 0] = True   # (30, 260 -> -100)
    keep_mask[2, 2] = True   # (40, 270 -> -90)

    extent = crop_extent_from_mask(lat2d, lon2d, keep_mask, margin_deg=1.0)
    print(f"\n[spatial-map-check2] extent={extent}")
    # kept lon range: [-100, -90] normalized; kept lat range: [30, 40]
    assert extent == pytest.approx([-101.0, -89.0, 29.0, 41.0])


def test_crop_extent_from_mask_default_margin_is_zero():
    """The plotted extent should show only the actual domain, no padding
    beyond it, unless a caller explicitly asks for a margin."""
    lat2d = np.array([[30.0, 30.0], [40.0, 40.0]])
    lon2d = np.array([[260.0, 270.0], [260.0, 270.0]])
    keep_mask = np.ones_like(lat2d, dtype=bool)
    extent = crop_extent_from_mask(lat2d, lon2d, keep_mask)
    print(f"\n[spatial-map-check2b] default-margin extent={extent}")
    assert extent == pytest.approx([-100.0, -90.0, 30.0, 40.0])


def test_build_base_map_returns_platecarree_axes():
    fig, ax = build_base_map()
    print(f"\n[spatial-map-check3] projection={type(ax.projection).__name__}, n_axes={len(fig.axes)}")
    assert isinstance(ax.projection, ccrs.PlateCarree)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_build_base_map_conus_outline_toggle_controls_lines():
    """The CONUS boundary outline is drawn as plain ax.plot() Line2D
    artists (not a cartopy Feature) -- directly verifiable via ax.lines.
    States/countries are added as cartopy Features instead (no stable,
    version-independent introspection point), so those two toggles are
    only smoke-tested here (no exception across every combination)."""
    fig_with, ax_with = build_base_map(draw_conus_outline=True)
    fig_without, ax_without = build_base_map(draw_conus_outline=False)
    print(f"\n[spatial-map-check4] n_lines_with={len(ax_with.lines)}, n_lines_without={len(ax_without.lines)}")
    assert len(ax_with.lines) > 0
    assert len(ax_without.lines) == 0
    plt.close(fig_with)
    plt.close(fig_without)

    for states, countries in [(True, True), (True, False), (False, True), (False, False)]:
        fig, ax = build_base_map(draw_states=states, draw_countries=countries)
        plt.close(fig)


def test_finalize_and_save_sets_extent_and_writes_file(tmp_path):
    fig, ax = build_base_map()
    extent = [-100.0, -90.0, 30.0, 40.0]
    out_path = str(tmp_path / "sub" / "out.png")
    finalize_and_save(fig, ax, extent, out_path, title="Test title")

    print(f"\n[spatial-map-check5] out_path exists={os.path.isfile(out_path)}, size={os.path.getsize(out_path) if os.path.isfile(out_path) else 0}")
    assert os.path.isfile(out_path)
    assert os.path.getsize(out_path) > 0


def test_finalize_and_save_corrects_aspect_for_central_latitude(tmp_path):
    """PlateCarree's default equal aspect (1.0, confirmed empirically --
    see spatial_map.py's own module docstring) stretches real east-west
    distances away from the equator; finalize_and_save must correct this
    to 1/cos(central_latitude)."""
    fig, ax = build_base_map()
    extent = [-100.0, -90.0, 30.0, 40.0]  # central_lat = 35.0
    finalize_and_save(fig, ax, extent, str(tmp_path / "out.png"))
    expected_aspect = 1.0 / math.cos(math.radians(35.0))
    print(f"\n[spatial-map-check6] aspect={ax.get_aspect()}, expected={expected_aspect}")
    assert ax.get_aspect() == pytest.approx(expected_aspect)


def test_finalize_and_save_left_and_right_titles_are_independent(tmp_path):
    fig, ax = build_base_map()
    extent = [-100.0, -90.0, 30.0, 40.0]
    finalize_and_save(fig, ax, extent, str(tmp_path / "out.png"), title="Short title", valid_time_label="2023-04-25 05:00Z")
    print(f"\n[spatial-map-check7] left={ax.get_title(loc='left')!r}, right={ax.get_title(loc='right')!r}")
    assert ax.get_title(loc="left") == "Short title"
    assert ax.get_title(loc="right") == "2023-04-25 05:00Z"


def test_build_base_map_coastlines_toggle_no_exception():
    for coastlines in (True, False):
        fig, ax = build_base_map(draw_coastlines=coastlines)
        plt.close(fig)


# ---- add_reflectivity_layer / add_reflectivity_colorbar ----


def test_add_reflectivity_layer_masks_below_threshold_and_missing_value():
    lat2d = np.array([[30.0, 30.0, 30.0]])
    lon_plot = np.array([[260.0, 261.0, 262.0]])
    # cell0: below min_dbz -> masked; cell1: at missing_value but ABOVE
    # min_dbz -> still masked (proves the OR-condition, not just threshold);
    # cell2: real valid reflectivity -> unmasked
    data = np.array([[15.0, 30.0, 45.0]])
    fig, ax = build_base_map()
    mesh = add_reflectivity_layer(ax, lon_plot, lat2d, data, missing_value=30.0, min_dbz=20.0)
    arr = mesh.get_array()
    mask = np.ma.getmaskarray(arr).reshape(data.shape)
    print(f"\n[spatial-map-check8] mask={mask}")
    assert mask[0, 0] == True
    assert mask[0, 1] == True
    assert mask[0, 2] == False
    plt.close(fig)


def test_add_reflectivity_layer_uses_standard_cmap_and_levels():
    lat2d = np.array([[30.0, 30.0]])
    lon_plot = np.array([[260.0, 261.0]])
    data = np.array([[25.0, 45.0]])
    fig, ax = build_base_map()
    mesh = add_reflectivity_layer(ax, lon_plot, lat2d, data)
    print(f"\n[spatial-map-check9] cmap.N={mesh.cmap.N}, norm.boundaries={list(mesh.norm.boundaries)}")
    assert mesh.cmap.N == 11
    assert np.array_equal(mesh.norm.boundaries, cb_colors.REFLECTIVITY_LEVELS)
    plt.close(fig)


def test_add_reflectivity_layer_does_not_mutate_shared_cmap_singleton():
    lat2d = np.array([[30.0]])
    lon_plot = np.array([[260.0]])
    data = np.array([[45.0]])
    fig, ax = build_base_map()
    add_reflectivity_layer(ax, lon_plot, lat2d, data)
    add_reflectivity_layer(ax, lon_plot, lat2d, data)
    # the shared cb_colors.nws_dz_cmap singleton itself must never have
    # set_over/set_under applied directly -- confirmed by checking its
    # "over" color is still the colormap's own top color, not "black"
    # (matplotlib defaults _rgba_over to None until set_over is called)
    print(f"\n[spatial-map-check10] _rgba_over={cb_colors.nws_dz_cmap._rgba_over}")
    assert cb_colors.nws_dz_cmap._rgba_over is None
    plt.close(fig)


def test_add_reflectivity_colorbar_is_horizontal(tmp_path):
    lat2d = np.array([[30.0, 30.0]])
    lon_plot = np.array([[260.0, 261.0]])
    data = np.array([[25.0, 45.0]])
    fig, ax = build_base_map()
    mesh = add_reflectivity_layer(ax, lon_plot, lat2d, data)
    cbar = add_reflectivity_colorbar(fig, ax, mesh)
    print(f"\n[spatial-map-check11] orientation={cbar.orientation}")
    assert cbar.orientation == "horizontal"
    plt.close(fig)
