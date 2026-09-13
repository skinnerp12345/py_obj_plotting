"""Validation of plots.object_spatial_map: numeric checks only (fill-grid
values, mesh array/colormap/norm content, colorbar vmin/vmax, invalid input
raises) -- never by viewing the rendered image, per this project's
established convention. StormObject instances are constructed directly (not
via file I/O) for the synthetic checks, mirroring test_object_property_heatmap.py's
own approach; real-data smoke tests use already-present external local data,
skip-marked when absent (matching test_driver_new_features.py's convention).

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_object_spatial_map.py -v -s
"""

import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from py_obj.obj_core import MatchRecord, StormObject, iter_object_slices, iter_object_slices_lazy, read_match_file, read_object_file
from py_obj_plotting import cb_colors
from py_obj_plotting.plots.object_spatial_map import (
    ObjectSlice,
    _age_fill_grid,
    _category_fill_grid,
    _match_category_id_sets,
    _storm_mode_fill_grid,
    compute_ensemble_mean_stats,
    compute_instantaneous_stats,
    plot_object_spatial_map,
    plot_paintball_map,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NCAR_MPAS_DIR = os.path.join(REPO_ROOT, "ncar_mpas_analysis")
DRYLINE_DIR = os.path.join(REPO_ROOT, "dryline_testing")
_SKIP_NO_NCAR_MPAS_ANALYSIS = pytest.mark.skipif(
    not os.path.isdir(NCAR_MPAS_DIR), reason="requires the external ncar_mpas_analysis/ dataset (not bundled)",
)
_SKIP_NO_DRYLINE_TESTING = pytest.mark.skipif(
    not os.path.isdir(DRYLINE_DIR), reason="requires the external dryline_testing/ dataset (not bundled)",
)


def _obj(id, is_linear=0, age_seconds=None, centroid_lat=35.0, centroid_lon=260.0) -> StormObject:
    return StormObject(
        id=id, area_px=10, area_km2=100.0, max_intensity=50.0, mean_intensity=45.0,
        major_axis_length=5.0, minor_axis_length=3.0, eccentricity=0.5, orientation=0.0, solidity=0.9,
        centroid_rowcol=(0.0, 0.0), centroid_lat=centroid_lat, centroid_lon=centroid_lon,
        centroid_x_km=0.0, centroid_y_km=0.0, is_linear=is_linear, age_seconds=age_seconds,
    )


def _synthetic_grid():
    """5x5 grid, lat/lon a plain regular mesh, 3 objects: id=1 at (0,0)
    cellular, id=2 at (2,2) mixed, id=3 at (4,4) linear."""
    lat2d = np.tile(np.linspace(30.0, 34.0, 5).reshape(-1, 1), (1, 5))
    lon2d = np.tile(np.linspace(260.0, 264.0, 5).reshape(1, -1), (5, 1))
    labels2d = np.zeros((5, 5), dtype=np.int32)
    labels2d[0, 0] = 1
    labels2d[2, 2] = 2
    labels2d[4, 4] = 3
    objects = [_obj(1, is_linear=0), _obj(2, is_linear=1), _obj(3, is_linear=2)]
    return lat2d, lon2d, labels2d, objects


def test_storm_mode_fill_grid_hand_computed():
    lat2d, lon2d, labels2d, objects = _synthetic_grid()
    grid = _storm_mode_fill_grid(labels2d, objects)
    print(f"\n[obj-spatial-check1] grid=\n{grid}")
    assert grid[0, 0] == 1  # cellular -> is_linear(0)+1
    assert grid[2, 2] == 2  # mixed -> 1+1
    assert grid[4, 4] == 3  # linear -> 2+1
    assert grid[1, 1] == 0  # background


def test_age_fill_grid_hand_computed_and_untracked_is_nan():
    lat2d, lon2d, labels2d, objects = _synthetic_grid()
    objects = [_obj(1, age_seconds=600.0), _obj(2, age_seconds=None), _obj(3, age_seconds=1800.0)]
    grid = _age_fill_grid(labels2d, objects)
    print(f"\n[obj-spatial-check2] grid=\n{grid}")
    assert grid[0, 0] == pytest.approx(10.0)   # 600s -> 10min
    assert np.isnan(grid[2, 2])                 # untracked -> NaN
    assert grid[4, 4] == pytest.approx(30.0)    # 1800s -> 30min
    assert np.isnan(grid[1, 1])                 # background -> NaN


def _find_mesh(ax):
    """ax.collections also holds the STATES feature artist and any centroid
    scatter (PathCollection) -- both also expose get_array(), so filter to
    the actual pcolormesh QuadMesh specifically."""
    from matplotlib.collections import QuadMesh

    for c in ax.collections:
        if isinstance(c, QuadMesh):
            return c
    raise AssertionError("no QuadMesh found in ax.collections")


def test_plot_object_spatial_map_storm_mode_mesh_and_norm(tmp_path):
    lat2d, lon2d, labels2d, objects = _synthetic_grid()
    forecast = ObjectSlice(lat2d, lon2d, labels2d, objects, member_id="mem1", valid_time=datetime(2026, 5, 1, 0, 0, 0))
    out_path = str(tmp_path / "storm_mode.png")

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_object_spatial_map(forecast, out_path, extent=[255, 265, 28, 36], color_by="storm_mode")
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    mesh = _find_mesh(ax)
    arr = mesh.get_array()
    expected_plain = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 3]])
    filled = np.ma.filled(arr.reshape(expected_plain.shape), 0)
    print(f"\n[obj-spatial-check3] mesh_array=\n{filled}")
    assert np.array_equal(filled, expected_plain)
    assert list(mesh.norm.boundaries) == [0.5, 1.5, 2.5, 3.5]
    assert os.path.isfile(out_path)
    plt.close(fig)


def test_plot_object_spatial_map_age_colorbar_vmin_vmax(tmp_path):
    lat2d, lon2d, labels2d, objects = _synthetic_grid()
    objects = [_obj(1, age_seconds=600.0), _obj(2, age_seconds=1200.0), _obj(3, age_seconds=1800.0)]
    forecast = ObjectSlice(lat2d, lon2d, labels2d, objects, member_id=None, valid_time=datetime(2026, 5, 1, 0, 0, 0))
    out_path = str(tmp_path / "age.png")

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_object_spatial_map(forecast, out_path, extent=[255, 265, 28, 36], color_by="age")
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    mesh = _find_mesh(ax)
    print(f"\n[obj-spatial-check4] auto vmin={mesh.norm.vmin}, vmax={mesh.norm.vmax}")
    assert mesh.norm.vmin == pytest.approx(10.0)   # 600s -> 10min
    assert mesh.norm.vmax == pytest.approx(30.0)   # 1800s -> 30min
    # exactly one extra axes for the colorbar
    assert len(fig.axes) == 2
    plt.close(fig)

    out_path2 = str(tmp_path / "age_explicit.png")
    plt.close = lambda *a, **k: None
    plot_object_spatial_map(forecast, out_path2, extent=[255, 265, 28, 36], color_by="age", age_vmin=0.0, age_vmax=60.0)
    plt.close = orig_close
    fig2 = plt.gcf()
    mesh2 = _find_mesh(fig2.axes[0])
    print(f"[obj-spatial-check4] explicit vmin={mesh2.norm.vmin}, vmax={mesh2.norm.vmax}")
    assert mesh2.norm.vmin == pytest.approx(0.0)
    assert mesh2.norm.vmax == pytest.approx(60.0)
    plt.close(fig2)


def test_plot_object_spatial_map_invalid_color_by_raises(tmp_path):
    lat2d, lon2d, labels2d, objects = _synthetic_grid()
    forecast = ObjectSlice(lat2d, lon2d, labels2d, objects, member_id=None, valid_time=datetime(2026, 5, 1))
    with pytest.raises(ValueError, match="bogus"):
        plot_object_spatial_map(forecast, str(tmp_path / "x.png"), extent=[255, 265, 28, 36], color_by="bogus")


@_SKIP_NO_NCAR_MPAS_ANALYSIS
def test_real_mrms_single_and_storm_mode(tmp_path):
    contents = read_object_file(os.path.join(NCAR_MPAS_DIR, "mrms_objects", "obj_obs_20230424_000041.nc"))
    _, valid_time, labels2d, objects = next(iter(iter_object_slices(contents)))
    forecast = ObjectSlice(contents.lat2d, contents.lon2d, labels2d, objects, member_id=None, valid_time=valid_time)

    keep_mask = np.ones(contents.lat2d.shape, dtype=bool)  # full-domain extent for this smoke test
    from py_obj_plotting.plots.spatial_map import crop_extent_from_mask
    extent = crop_extent_from_mask(contents.lat2d, contents.lon2d, keep_mask)

    for color_by in ("single", "storm_mode"):
        out_path = str(tmp_path / f"real_{color_by}.png")
        orig_close = plt.close
        plt.close = lambda *a, **k: None
        plot_object_spatial_map(forecast, out_path, extent=extent, color_by=color_by)
        plt.close = orig_close
        print(f"\n[obj-spatial-check5] color_by={color_by} n_objects={len(objects)} out_size={os.path.getsize(out_path)}")
        assert os.path.isfile(out_path)
        assert os.path.getsize(out_path) > 0
        plt.close(plt.gcf())


@_SKIP_NO_DRYLINE_TESTING
def test_real_tracked_age_mode(tmp_path):
    contents = read_object_file(os.path.join(DRYLINE_DIR, "obj_full_20230511.nc"))
    assert contents.tracked
    _, valid_time, labels2d, objects = next(iter(iter_object_slices(contents)))
    forecast = ObjectSlice(contents.lat2d, contents.lon2d, labels2d, objects, member_id=None, valid_time=valid_time)

    keep_mask = np.ones(contents.lat2d.shape, dtype=bool)
    from py_obj_plotting.plots.spatial_map import crop_extent_from_mask
    extent = crop_extent_from_mask(contents.lat2d, contents.lon2d, keep_mask)

    out_path = str(tmp_path / "real_age.png")
    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_object_spatial_map(forecast, out_path, extent=extent, color_by="age")
    plt.close = orig_close

    fig = plt.gcf()
    mesh = _find_mesh(fig.axes[0])
    arr = mesh.get_array()
    filled = np.ma.filled(arr.astype(float), np.nan)
    n_finite = int(np.isfinite(filled).sum())
    print(f"\n[obj-spatial-check6] n_objects={len(objects)}, n_finite_age_pixels={n_finite}")
    assert n_finite > 0, "expected real, non-NaN tracked age data to flow through, not silent degradation"
    assert os.path.isfile(out_path)
    plt.close(fig)


# ---- MRMS/truth overlay + independent storm-mode hatch toggles ----


def _find_meshes(ax):
    from matplotlib.collections import QuadMesh

    return [c for c in ax.collections if isinstance(c, QuadMesh)]


def _find_contour_sets(ax):
    from matplotlib.contour import QuadContourSet

    return [a for a in ax.get_children() if isinstance(a, QuadContourSet)]


def test_truth_overlay_produces_two_independent_mesh_layers(tmp_path):
    lat2d, lon2d, labels2d, objects = _synthetic_grid()
    forecast = ObjectSlice(lat2d, lon2d, labels2d, objects, member_id=None, valid_time=datetime(2026, 5, 1))

    # disjoint truth footprint (different pixel than any forecast object)
    truth_labels2d = np.zeros((5, 5), dtype=np.int32)
    truth_labels2d[1, 3] = 1
    truth_objects = [_obj(1, is_linear=0, centroid_lat=31.0, centroid_lon=263.0)]
    truth = ObjectSlice(lat2d, lon2d, truth_labels2d, truth_objects, member_id=None, valid_time=datetime(2026, 5, 1))

    out_path = str(tmp_path / "overlay.png")
    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_object_spatial_map(forecast, out_path, extent=[255, 265, 28, 36], color_by="single", truth=truth)
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    meshes = _find_meshes(ax)
    print(f"\n[obj-spatial-check7] n_meshes={len(meshes)}, facecolors={[m.get_alpha() for m in meshes]}")
    assert len(meshes) == 2
    # forecast layer uses the default forecast alpha (0.6), truth layer its own (0.75)
    alphas = sorted(m.get_alpha() for m in meshes)
    assert alphas == pytest.approx([0.6, 0.75])
    plt.close(fig)


def test_storm_mode_hatch_toggles_are_independent_and_correctly_patterned(tmp_path):
    lat2d, lon2d, labels2d, objects = _synthetic_grid()  # forecast: id2=mixed, id3=linear
    forecast = ObjectSlice(lat2d, lon2d, labels2d, objects, member_id=None, valid_time=datetime(2026, 5, 1))

    truth_labels2d = np.zeros((5, 5), dtype=np.int32)
    truth_labels2d[1, 3] = 1
    truth_objects = [_obj(1, is_linear=2, centroid_lat=31.0, centroid_lon=263.0)]  # linear
    truth = ObjectSlice(lat2d, lon2d, truth_labels2d, truth_objects, member_id=None, valid_time=datetime(2026, 5, 1))

    out_path = str(tmp_path / "hatch.png")
    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_object_spatial_map(
        forecast, out_path, extent=[255, 265, 28, 36], color_by="single",
        truth=truth, show_forecast_hatch=True, show_truth_hatch=True,
    )
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    contour_sets = _find_contour_sets(ax)
    all_hatches = sorted(cs.hatches[0] for cs in contour_sets)
    print(f"\n[obj-spatial-check8] all_hatches={all_hatches}")
    # forecast contributes mixed ('...') + linear ('///'); truth contributes linear ('///') only
    assert all_hatches == ["...", "///", "///"]
    plt.close(fig)

    # toggled off entirely -> no contour sets at all
    out_path2 = str(tmp_path / "no_hatch.png")
    plt.close = lambda *a, **k: None
    plot_object_spatial_map(forecast, out_path2, extent=[255, 265, 28, 36], color_by="single", truth=truth)
    plt.close = orig_close
    fig2 = plt.gcf()
    assert _find_contour_sets(fig2.axes[0]) == []
    plt.close(fig2)


@_SKIP_NO_NCAR_MPAS_ANALYSIS
def test_real_forecast_and_truth_overlay_shapes_match(tmp_path):
    """Real MPAS mem_1 forecast slice (lead=0) + the matching real MRMS
    truth slice -- confirms real forecast+truth footprints actually render
    together on one shared grid with no shape mismatch (both live on the
    same interpolated MPAS 3km grid per this project's own pipeline)."""
    forecast_path = os.path.join(NCAR_MPAS_DIR, "mpas_objects", "obj_init_20230424_000000.nc")
    forecast_slice = None
    for member_id, valid_time, labels2d, objects in iter_object_slices_lazy(forecast_path):
        if member_id == "mem_1":
            forecast_slice = (member_id, valid_time, labels2d, objects)
            break
    assert forecast_slice is not None
    member_id, valid_time, labels2d, objects = forecast_slice

    # lat2d/lon2d only -- NOT read_object_file(forecast_path), which would
    # eagerly load the full multi-GB `labels` array this test is
    # specifically avoiding via iter_object_slices_lazy above.
    import netCDF4
    with netCDF4.Dataset(forecast_path, "r") as ds:
        lat2d = np.asarray(ds.variables["lat"][:])
        lon2d = np.asarray(ds.variables["lon"][:])
    forecast = ObjectSlice(lat2d, lon2d, labels2d, objects, member_id=member_id, valid_time=valid_time)

    truth_contents = read_object_file(os.path.join(NCAR_MPAS_DIR, "mrms_objects", "obj_obs_20230424_000041.nc"))
    _, truth_valid_time, truth_labels2d, truth_objects = next(iter(iter_object_slices(truth_contents)))
    truth = ObjectSlice(truth_contents.lat2d, truth_contents.lon2d, truth_labels2d, truth_objects, member_id=None, valid_time=truth_valid_time)

    print(f"\n[obj-spatial-check9] forecast shape={forecast.labels2d.shape}, truth shape={truth.labels2d.shape}, "
          f"n_forecast_objs={len(objects)}, n_truth_objs={len(truth_objects)}")
    assert forecast.lat2d.shape == truth.lat2d.shape

    keep_mask = np.ones(forecast.lat2d.shape, dtype=bool)
    from py_obj_plotting.plots.spatial_map import crop_extent_from_mask
    extent = crop_extent_from_mask(forecast.lat2d, forecast.lon2d, keep_mask)

    out_path = str(tmp_path / "real_overlay.png")
    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_object_spatial_map(forecast, out_path, extent=extent, color_by="single", truth=truth, show_truth_hatch=True)
    plt.close = orig_close
    assert os.path.isfile(out_path)
    assert os.path.getsize(out_path) > 0
    plt.close(plt.gcf())


# ---- category mode + instantaneous POD/FAR/FBIAS/CSI annotation ----


def _category_scenario():
    """1 hit, 1 miss, 1 false_alarm, 1 truth_extra, 1 forecast_extra --
    hand-computable: pod=0.5, far=0.5, fbias=1.0, csi=1/3."""
    match_records = [
        MatchRecord(category="hit", truth_id=1, forecast_id=1, ti_score=0.9),
        MatchRecord(category="miss", truth_id=4, forecast_id=-1, ti_score=0.0),
        MatchRecord(category="false_alarm", truth_id=-1, forecast_id=2, ti_score=0.0),
        MatchRecord(category="truth_extra", truth_id=5, forecast_id=2, ti_score=0.3),
        MatchRecord(category="forecast_extra", truth_id=1, forecast_id=3, ti_score=0.3),
    ]
    forecast_labels2d = np.zeros((5, 5), dtype=np.int32)
    forecast_labels2d[0, 0] = 1  # hit
    forecast_labels2d[1, 1] = 2  # false_alarm
    forecast_labels2d[2, 2] = 3  # forecast_extra
    forecast_objects = [_obj(1), _obj(2), _obj(3)]

    truth_labels2d = np.zeros((5, 5), dtype=np.int32)
    truth_labels2d[0, 0] = 1  # hit
    truth_labels2d[3, 3] = 4  # miss
    truth_labels2d[4, 4] = 5  # truth_extra
    truth_objects = [_obj(1), _obj(4), _obj(5)]

    forecast = ObjectSlice(*_synthetic_grid()[:2], forecast_labels2d, forecast_objects, member_id=None, valid_time=datetime(2026, 5, 1))
    truth = ObjectSlice(*_synthetic_grid()[:2], truth_labels2d, truth_objects, member_id=None, valid_time=datetime(2026, 5, 1))
    return forecast, truth, match_records


def test_compute_instantaneous_stats_hand_computed():
    _, _, match_records = _category_scenario()
    stats = compute_instantaneous_stats(match_records)
    print(f"\n[obj-spatial-check10] stats={stats}")
    assert stats["n_hits"] == 1
    assert stats["n_misses"] == 1
    assert stats["n_false_alarms"] == 1
    assert stats["n_truth_extra"] == 1
    assert stats["n_forecast_extra"] == 1
    assert stats["pod"] == pytest.approx(0.5)
    assert stats["far"] == pytest.approx(0.5)
    assert stats["fbias"] == pytest.approx(1.0)
    assert stats["csi"] == pytest.approx(1.0 / 3.0)


def test_category_fill_grid_hand_computed():
    forecast, truth, match_records = _category_scenario()
    truth_ids_by_value, forecast_ids_by_value = _match_category_id_sets(match_records)

    forecast_grid = _category_fill_grid(forecast.labels2d, forecast_ids_by_value)
    truth_grid = _category_fill_grid(truth.labels2d, truth_ids_by_value)
    print(f"\n[obj-spatial-check11] forecast_grid=\n{forecast_grid}\ntruth_grid=\n{truth_grid}")

    assert forecast_grid[0, 0] == 1  # hit
    assert forecast_grid[1, 1] == 2  # false_alarm
    assert forecast_grid[2, 2] == 3  # forecast_extra (extra)
    assert truth_grid[0, 0] == 1     # hit
    assert truth_grid[3, 3] == 2     # miss
    assert truth_grid[4, 4] == 3     # truth_extra (extra)


def test_plot_object_spatial_map_category_mode_stats_text_and_toggle(tmp_path):
    forecast, truth, match_records = _category_scenario()

    out_path = str(tmp_path / "category.png")
    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_object_spatial_map(forecast, out_path, extent=[255, 265, 28, 36], color_by="category", truth=truth, match_records=match_records)
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    print(f"\n[obj-spatial-check12] texts={texts}")
    stats_text = next(t for t in texts if "POD" in t)
    assert "POD: 0.50" in stats_text
    assert "FAR: 0.50" in stats_text
    assert "BIAS: 1.00" in stats_text
    assert "CSI: 0.33" in stats_text
    plt.close(fig)

    # show_stats=False explicitly suppresses the annotation even in category mode
    out_path2 = str(tmp_path / "category_no_stats.png")
    plt.close = lambda *a, **k: None
    plot_object_spatial_map(forecast, out_path2, extent=[255, 265, 28, 36], color_by="category",
                             truth=truth, match_records=match_records, show_stats=False)
    plt.close = orig_close
    fig2 = plt.gcf()
    texts2 = [t.get_text() for t in fig2.axes[0].texts]
    print(f"[obj-spatial-check12] texts2(show_stats=False)={texts2}")
    assert not any("POD" in t for t in texts2)
    plt.close(fig2)


def test_category_mode_missing_truth_or_match_records_raises(tmp_path):
    forecast, truth, match_records = _category_scenario()
    with pytest.raises(ValueError, match="category"):
        plot_object_spatial_map(forecast, str(tmp_path / "x.png"), extent=[255, 265, 28, 36], color_by="category", truth=truth)
    with pytest.raises(ValueError, match="category"):
        plot_object_spatial_map(forecast, str(tmp_path / "y.png"), extent=[255, 265, 28, 36], color_by="category", match_records=match_records)


@_SKIP_NO_NCAR_MPAS_ANALYSIS
def test_real_category_mode_stats_match_filtered_record_count(tmp_path):
    """Real MPAS mem_1 match records at a real lead time confirmed to
    contain hit/miss/false_alarm/truth_extra (lead index 29, valid
    2023-04-25 05:00Z) -- assert the rendered stats dict's counts sum to
    the filtered real record count."""
    match_contents = read_match_file(os.path.join(NCAR_MPAS_DIR, "matched_objects_300km", "match_init_20230424_000000.nc"))
    mi, ti = 0, 29  # mem_1, the confirmed-rich real lead time
    mask = (match_contents.member_index == mi) & (match_contents.time_index == ti)
    records = [r for r, m in zip(match_contents.records, mask) if m]
    assert len(records) > 0

    forecast_path = os.path.join(NCAR_MPAS_DIR, "mpas_objects", "obj_init_20230424_000000.nc")
    forecast_slice = None
    for member_id, valid_time, labels2d, objects in iter_object_slices_lazy(forecast_path):
        if member_id == "mem_1" and valid_time == match_contents.valid_times[ti]:
            forecast_slice = (member_id, valid_time, labels2d, objects)
            break
    assert forecast_slice is not None
    member_id, valid_time, labels2d, objects = forecast_slice

    import netCDF4
    with netCDF4.Dataset(forecast_path, "r") as ds:
        lat2d = np.asarray(ds.variables["lat"][:])
        lon2d = np.asarray(ds.variables["lon"][:])
    forecast = ObjectSlice(lat2d, lon2d, labels2d, objects, member_id=member_id, valid_time=valid_time)

    mrms_path = os.path.join(NCAR_MPAS_DIR, "mrms_objects", "obj_obs_20230425_050041.nc")
    truth_contents = read_object_file(mrms_path)
    _, truth_valid_time, truth_labels2d, truth_objects = next(iter(iter_object_slices(truth_contents)))
    truth = ObjectSlice(truth_contents.lat2d, truth_contents.lon2d, truth_labels2d, truth_objects, member_id=None, valid_time=truth_valid_time)

    stats = compute_instantaneous_stats(records)
    n_total = stats["n_hits"] + stats["n_misses"] + stats["n_false_alarms"] + stats["n_truth_extra"] + stats["n_forecast_extra"]
    print(f"\n[obj-spatial-check13] n_records={len(records)}, stats={stats}")
    assert n_total == len(records)

    keep_mask = np.ones(forecast.lat2d.shape, dtype=bool)
    from py_obj_plotting.plots.spatial_map import crop_extent_from_mask
    extent = crop_extent_from_mask(forecast.lat2d, forecast.lon2d, keep_mask)

    out_path = str(tmp_path / "real_category.png")
    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_object_spatial_map(forecast, out_path, extent=extent, color_by="category", truth=truth, match_records=records)
    plt.close = orig_close
    assert os.path.isfile(out_path)
    assert os.path.getsize(out_path) > 0
    plt.close(plt.gcf())


# ---- ensemble paintball mode ----


def _paintball_members():
    lat2d, lon2d, _, _ = _synthetic_grid()
    members = []
    for i, (r, c) in enumerate([(0, 0), (0, 1), (2, 2)]):  # partially overlapping footprints
        labels2d = np.zeros((5, 5), dtype=np.int32)
        labels2d[r, c] = 1
        members.append(ObjectSlice(lat2d, lon2d, labels2d, [_obj(1)], member_id=f"mem{i+1}", valid_time=datetime(2026, 5, 1)))
    return members


def test_paintball_default_uniform_gray_alpha(tmp_path):
    members = _paintball_members()
    out_path = str(tmp_path / "paintball.png")

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_paintball_map(members, out_path, extent=[255, 265, 28, 36])
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    from matplotlib.collections import QuadMesh
    meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
    print(f"\n[obj-spatial-check14] n_meshes={len(meshes)}, alphas={[m.get_alpha() for m in meshes]}")
    assert len(meshes) == 3
    # default alpha raised to 0.65 (was 0.08) -- objects were barely
    # visible at 0.08; it's fine for overlapping members' fills to stack
    # rather than needing individual distinguishability
    assert all(m.get_alpha() == pytest.approx(0.65) for m in meshes)
    # all 3 share the same uniform gray8-only colormap
    cmaps = {tuple(m.cmap(0.999)) for m in meshes}
    assert len(cmaps) == 1
    plt.close(fig)


def test_paintball_per_member_colors_produces_distinct_cmaps(tmp_path):
    members = _paintball_members()
    colors = [cb_colors.blue5, cb_colors.orange5, cb_colors.purple5]
    out_path = str(tmp_path / "paintball_colors.png")

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_paintball_map(members, out_path, extent=[255, 265, 28, 36], per_member_colors=colors)
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    from matplotlib.collections import QuadMesh
    meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
    cmap_colors = [tuple(m.cmap(0.999))[:3] for m in meshes]
    print(f"\n[obj-spatial-check15] cmap_colors={cmap_colors}")
    assert len(set(cmap_colors)) == 3
    plt.close(fig)


def test_paintball_empty_members_raises(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        plot_paintball_map([], str(tmp_path / "x.png"), extent=[255, 265, 28, 36])


def test_paintball_mismatched_per_member_colors_raises(tmp_path):
    members = _paintball_members()
    with pytest.raises(ValueError, match="per_member_colors"):
        plot_paintball_map(members, str(tmp_path / "x.png"), extent=[255, 265, 28, 36], per_member_colors=[cb_colors.blue5])


@_SKIP_NO_NCAR_MPAS_ANALYSIS
def test_real_paintball_all_five_members(tmp_path):
    forecast_path = os.path.join(NCAR_MPAS_DIR, "mpas_objects", "obj_init_20230424_000000.nc")
    import netCDF4
    with netCDF4.Dataset(forecast_path, "r") as ds:
        lat2d = np.asarray(ds.variables["lat"][:])
        lon2d = np.asarray(ds.variables["lon"][:])

    target_valid_time = None
    slices_by_member = {}
    for member_id, valid_time, labels2d, objects in iter_object_slices_lazy(forecast_path):
        if target_valid_time is None:
            target_valid_time = valid_time
        if valid_time == target_valid_time:
            slices_by_member[member_id] = ObjectSlice(lat2d, lon2d, labels2d, objects, member_id=member_id, valid_time=valid_time)
        if len(slices_by_member) == 5:
            break

    print(f"\n[obj-spatial-check16] members captured: {sorted(slices_by_member)}")
    assert len(slices_by_member) == 5
    members = [slices_by_member[m] for m in sorted(slices_by_member)]

    keep_mask = np.ones(lat2d.shape, dtype=bool)
    from py_obj_plotting.plots.spatial_map import crop_extent_from_mask
    extent = crop_extent_from_mask(lat2d, lon2d, keep_mask)

    out_path = str(tmp_path / "real_paintball.png")
    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_paintball_map(members, out_path, extent=extent)
    plt.close = orig_close
    assert os.path.isfile(out_path)
    assert os.path.getsize(out_path) > 0
    plt.close(plt.gcf())


# ---- Paintball color_by extension: storm_mode, category + ensemble-mean stats ----


def _paintball_members_by_storm_mode():
    """3 members, each with ONE object at a distinct pixel and a distinct
    is_linear value (cellular/mixed/linear) -- lets a test assert each
    member's own mesh carries the correct per-member storm-mode value."""
    lat2d, lon2d, _, _ = _synthetic_grid()
    members = []
    for i, ((r, c), mode) in enumerate(zip([(0, 0), (0, 1), (2, 2)], [0, 1, 2])):
        labels2d = np.zeros((5, 5), dtype=np.int32)
        labels2d[r, c] = 1
        members.append(ObjectSlice(lat2d, lon2d, labels2d, [_obj(1, is_linear=mode)], member_id=f"mem{i+1}", valid_time=datetime(2026, 5, 1)))
    return members


def test_paintball_storm_mode_colors_each_member_by_its_own_mode(tmp_path):
    members = _paintball_members_by_storm_mode()
    out_path = str(tmp_path / "paintball_storm_mode.png")

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_paintball_map(members, out_path, extent=[255, 265, 28, 36], color_by="storm_mode")
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    from matplotlib.collections import QuadMesh
    meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
    print(f"\n[obj-spatial-check17] n_meshes={len(meshes)}")
    assert len(meshes) == 3
    # a legend with the 3 storm-mode categories should be present
    legend_labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert legend_labels == {"Cellular", "Mixed", "Linear"}
    plt.close(fig)


def _paintball_category_scenario():
    """3 members, each with its own single forecast object and its own
    match record for that same valid time -- member 'mem1' hits, 'mem2'
    misses (no forecast object contributes, but still counts in the
    ensemble-mean denominator), 'mem3' has a false alarm."""
    lat2d, lon2d, _, _ = _synthetic_grid()
    member_slices = []
    match_records_by_member = {}

    labels2d_1 = np.zeros((5, 5), dtype=np.int32)
    labels2d_1[0, 0] = 1
    member_slices.append(ObjectSlice(lat2d, lon2d, labels2d_1, [_obj(1)], member_id="mem1", valid_time=datetime(2026, 5, 1)))
    match_records_by_member["mem1"] = [MatchRecord(category="hit", truth_id=1, forecast_id=1, ti_score=0.9)]

    labels2d_2 = np.zeros((5, 5), dtype=np.int32)  # mem2 forecast an empty domain -- a real miss
    member_slices.append(ObjectSlice(lat2d, lon2d, labels2d_2, [], member_id="mem2", valid_time=datetime(2026, 5, 1)))
    match_records_by_member["mem2"] = [MatchRecord(category="miss", truth_id=1, forecast_id=-1, ti_score=0.0)]

    labels2d_3 = np.zeros((5, 5), dtype=np.int32)
    labels2d_3[2, 2] = 2
    member_slices.append(ObjectSlice(lat2d, lon2d, labels2d_3, [_obj(2)], member_id="mem3", valid_time=datetime(2026, 5, 1)))
    match_records_by_member["mem3"] = [MatchRecord(category="false_alarm", truth_id=-1, forecast_id=2, ti_score=0.0)]

    return member_slices, match_records_by_member


def test_compute_ensemble_mean_stats_hand_computed():
    _, match_records_by_member = _paintball_category_scenario()
    stats = compute_ensemble_mean_stats(match_records_by_member)
    print(f"\n[obj-spatial-check18] ensemble_mean_stats={stats}")
    # mem1: pod=1.0,far=0.0,fbias=1.0,csi=1.0 (1 hit only)
    # mem2: pod=0.0,far=0.0,fbias=0.0,csi=0.0 (1 miss only; fbias denom hits+misses+truth_extra=1 -> 0/1=0.0)
    # mem3: pod=0.0,far=1.0,fbias=0.0,csi=0.0 (1 false_alarm only; fbias denom hits+misses+truth_extra=0 -> 0.0 by convention)
    assert stats["pod"] == pytest.approx((1.0 + 0.0 + 0.0) / 3.0)
    assert stats["far"] == pytest.approx((0.0 + 0.0 + 1.0) / 3.0)
    assert stats["fbias"] == pytest.approx((1.0 + 0.0 + 0.0) / 3.0)
    assert stats["csi"] == pytest.approx((1.0 + 0.0 + 0.0) / 3.0)


def test_compute_ensemble_mean_stats_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        compute_ensemble_mean_stats({})


def test_paintball_category_mode_renders_and_annotates_ensemble_mean(tmp_path):
    member_slices, match_records_by_member = _paintball_category_scenario()
    out_path = str(tmp_path / "paintball_category.png")

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_paintball_map(member_slices, out_path, extent=[255, 265, 28, 36], color_by="category", match_records_by_member=match_records_by_member)
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    from matplotlib.collections import QuadMesh
    meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
    print(f"\n[obj-spatial-check19] n_meshes={len(meshes)}")
    assert len(meshes) == 3  # mem1 hit + mem3 false_alarm both render; mem2 has no footprint pixels but is still one draw call
    legend_labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert legend_labels == {"Hit", "Miss", "False alarm", "Extra"}

    texts = [t.get_text() for t in ax.texts]
    stats_text = next(t for t in texts if "POD" in t)
    print(f"[obj-spatial-check19] stats_text={stats_text!r}")
    assert "POD: 0.33" in stats_text  # (1.0+0.0+0.0)/3
    assert "FAR: 0.33" in stats_text  # (0.0+0.0+1.0)/3
    plt.close(fig)

    # show_stats=False suppresses the annotation even in category mode
    out_path2 = str(tmp_path / "paintball_category_no_stats.png")
    plt.close = lambda *a, **k: None
    plot_paintball_map(member_slices, out_path2, extent=[255, 265, 28, 36], color_by="category",
                        match_records_by_member=match_records_by_member, show_stats=False)
    plt.close = orig_close
    fig2 = plt.gcf()
    texts2 = [t.get_text() for t in fig2.axes[0].texts]
    assert not any("POD" in t for t in texts2)
    plt.close(fig2)


def test_paintball_invalid_color_by_raises(tmp_path):
    members = _paintball_members()
    with pytest.raises(ValueError, match="bogus"):
        plot_paintball_map(members, str(tmp_path / "x.png"), extent=[255, 265, 28, 36], color_by="bogus")


def test_paintball_category_mode_missing_match_records_raises(tmp_path):
    members = _paintball_members()
    with pytest.raises(ValueError, match="match_records_by_member"):
        plot_paintball_map(members, str(tmp_path / "x.png"), extent=[255, 265, 28, 36], color_by="category")


def test_paintball_left_and_right_titles(tmp_path):
    members = _paintball_members()
    out_path = str(tmp_path / "paintball_titles.png")
    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_paintball_map(members, out_path, extent=[255, 265, 28, 36])
    plt.close = orig_close
    fig = plt.gcf()
    ax = fig.axes[0]
    print(f"\n[obj-spatial-check20] left={ax.get_title(loc='left')!r}, right={ax.get_title(loc='right')!r}")
    assert ax.get_title(loc="left") == "3-member paintball -- single"
    assert ax.get_title(loc="right") == "2026-05-01 00:00Z"
    plt.close(fig)


def test_paintball_truth_overlay_uses_higher_alpha(tmp_path):
    members = _paintball_members()
    lat2d, lon2d, _, _ = _synthetic_grid()
    truth_labels2d = np.zeros((5, 5), dtype=np.int32)
    truth_labels2d[3, 3] = 1
    truth = ObjectSlice(lat2d, lon2d, truth_labels2d, [_obj(1, centroid_lat=32.0, centroid_lon=263.0)],
                         member_id=None, valid_time=datetime(2026, 5, 1))
    out_path = str(tmp_path / "paintball_truth.png")

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_paintball_map(members, out_path, extent=[255, 265, 28, 36], truth=truth)
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    from matplotlib.collections import QuadMesh
    meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
    print(f"\n[obj-spatial-check21] n_meshes={len(meshes)}, alphas={[m.get_alpha() for m in meshes]}")
    assert len(meshes) == 4  # 3 members + 1 truth layer
    alphas = sorted(m.get_alpha() for m in meshes)
    assert alphas == pytest.approx([0.65, 0.65, 0.65, 0.8])
    legend_labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert "MRMS" in legend_labels
    plt.close(fig)
