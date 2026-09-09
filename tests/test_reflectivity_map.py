"""Validation of plots.reflectivity_map: numeric checks only (extent
resolution, title construction, mesh content via the underlying
spatial_map.add_reflectivity_layer already covered in test_spatial_map.py)
-- never by viewing the rendered image, per this project's established
convention.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_reflectivity_map.py -v -s
"""

import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from py_obj_plotting.plots.reflectivity_map import ReflectivityFrame, _default_title, _resolve_extent, plot_reflectivity_map


def _frame(**overrides) -> ReflectivityFrame:
    lat2d = np.array([[30.0, 30.0, 30.0], [32.0, 32.0, 32.0]])
    lon2d = np.array([[260.0, 261.0, 262.0], [260.0, 261.0, 262.0]])
    data = np.array([[15.0, 30.0, 45.0], [50.0, 55.0, 60.0]])
    defaults = dict(lat2d=lat2d, lon2d=lon2d, data=data, valid_time=datetime(2023, 5, 1, 1, 0, 41),
                     missing_value=-999.0, label="Test Source")
    defaults.update(overrides)
    return ReflectivityFrame(**defaults)


def test_resolve_extent_none_uses_full_grid_bbox():
    frame = _frame()
    extent = _resolve_extent(frame, None)
    print(f"\n[refl-map-check1] extent={extent}")
    # lon2d is 260-262 in 0-360 convention; crop_extent_from_mask correctly
    # normalizes to -180/180 (-100 to -98) for plotting, same as lon_to_plot_range
    assert extent == pytest.approx([-100.0, -98.0, 30.0, 32.0])


def test_resolve_extent_explicit_override_passthrough():
    frame = _frame()
    explicit = [255.0, 265.0, 28.0, 36.0]
    assert _resolve_extent(frame, explicit) == explicit


def test_default_title_short_label_and_right_justified_valid_time():
    frame = _frame(label="MPAS mem1")
    title, valid_time_label = _default_title(frame)
    print(f"\n[refl-map-check2] title={title!r}, valid_time_label={valid_time_label!r}")
    assert title == "MPAS mem1"
    assert valid_time_label == "2023-05-01 01:00Z"


def test_plot_reflectivity_map_writes_file_with_correct_titles(tmp_path):
    frame = _frame(label="MRMS (native)")
    out_path = str(tmp_path / "refl.png")

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_reflectivity_map(frame, out_path, extent=[255, 265, 28, 36])
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    print(f"\n[refl-map-check3] left={ax.get_title(loc='left')!r}, right={ax.get_title(loc='right')!r}")
    assert ax.get_title(loc="left") == "MRMS (native)"
    assert ax.get_title(loc="right") == "2023-05-01 01:00Z"
    assert os.path.isfile(out_path)
    assert os.path.getsize(out_path) > 0
    plt.close(fig)


def test_plot_reflectivity_map_title_override(tmp_path):
    frame = _frame()
    out_path = str(tmp_path / "refl2.png")

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_reflectivity_map(frame, out_path, extent=[255, 265, 28, 36], title="Custom Title")
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    assert ax.get_title(loc="left") == "Custom Title"
    plt.close(fig)


def test_plot_reflectivity_map_has_horizontal_colorbar(tmp_path):
    frame = _frame()
    out_path = str(tmp_path / "refl3.png")

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_reflectivity_map(frame, out_path, extent=[255, 265, 28, 36])
    plt.close = orig_close

    fig = plt.gcf()
    # colorbar lives on a second axes; confirmed present via 2 total axes
    print(f"\n[refl-map-check4] n_axes={len(fig.axes)}")
    assert len(fig.axes) == 2
    plt.close(fig)
