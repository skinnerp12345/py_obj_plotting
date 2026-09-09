"""Validation of config.ObjectSpatialMapConfig/load_object_spatial_map_config
-- the standalone config shape for a single forecast initialization's object
spatial maps (see object_selection.py, drivers/plot_object_spatial_maps.py).

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_object_spatial_map_config.py -v -s
"""

import os
from datetime import datetime

import pytest
import yaml

from py_obj_plotting.config import load_object_spatial_map_config


def _write_config(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def test_minimal_valid_config(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {
        "forecast_object_dir": "forecast_objs",
        "init_time": "2023-04-24T00:00:00",
    })
    cfg = load_object_spatial_map_config(str(cfg_path))
    print(f"\n[obj-spatial-cfg-check1] cfg={cfg}")
    assert cfg.forecast_object_dir == str(tmp_path / "forecast_objs")
    assert cfg.init_time == datetime(2023, 4, 24, 0, 0, 0)
    assert cfg.color_by == "single"
    assert cfg.members.mode == "all"
    assert cfg.times.mode == "full"
    assert cfg.mrms_object_dir is None
    assert cfg.match_dir is None
    assert cfg.output_dir == str(tmp_path / "output")


def test_paths_resolve_relative_to_config_file_directory(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    cfg_path = sub / "cfg.yaml"
    _write_config(str(cfg_path), {
        "forecast_object_dir": "../forecast_objs",
        "init_time": "2023-04-24T00:00:00",
        "output_dir": "out",
    })
    cfg = load_object_spatial_map_config(str(cfg_path))
    assert cfg.forecast_object_dir == os.path.normpath(str(tmp_path / "forecast_objs"))
    assert cfg.output_dir == str(sub / "out")


def test_missing_required_fields_raises(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {"forecast_object_dir": "x"})
    with pytest.raises(ValueError, match="init_time"):
        load_object_spatial_map_config(str(cfg_path))


def test_invalid_color_by_raises(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {
        "forecast_object_dir": "x", "init_time": "2023-04-24T00:00:00", "color_by": "bogus",
    })
    with pytest.raises(ValueError, match="bogus"):
        load_object_spatial_map_config(str(cfg_path))


def test_category_mode_requires_mrms_and_match_dir(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {
        "forecast_object_dir": "x", "init_time": "2023-04-24T00:00:00", "color_by": "category",
    })
    with pytest.raises(ValueError, match="category"):
        load_object_spatial_map_config(str(cfg_path))

    _write_config(str(cfg_path), {
        "forecast_object_dir": "x", "init_time": "2023-04-24T00:00:00", "color_by": "category",
        "mrms_object_dir": "mrms",
    })
    with pytest.raises(ValueError, match="category"):
        load_object_spatial_map_config(str(cfg_path))


def test_paintball_category_mode_requires_only_match_dir(tmp_path):
    """Paintball category mode never renders a truth layer (see
    plot_paintball_map's own docstring) -- unlike single-panel category
    mode, mrms_object_dir is genuinely not required there."""
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {
        "forecast_object_dir": "x", "init_time": "2023-04-24T00:00:00", "color_by": "category",
        "paintball": True,
    })
    with pytest.raises(ValueError, match="match_dir"):
        load_object_spatial_map_config(str(cfg_path))

    _write_config(str(cfg_path), {
        "forecast_object_dir": "x", "init_time": "2023-04-24T00:00:00", "color_by": "category",
        "paintball": True, "match_dir": "matches",
    })
    cfg = load_object_spatial_map_config(str(cfg_path))
    print(f"\n[obj-spatial-cfg-check3] paintball category cfg with no mrms_object_dir: {cfg.match_dir}")
    assert cfg.mrms_object_dir is None
    assert cfg.match_dir is not None


def test_show_mrms_overlay_requires_mrms_object_dir(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {
        "forecast_object_dir": "x", "init_time": "2023-04-24T00:00:00", "show_mrms_overlay": True,
    })
    with pytest.raises(ValueError, match="mrms_object_dir"):
        load_object_spatial_map_config(str(cfg_path))


def test_full_config_with_member_and_time_selection(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {
        "forecast_object_dir": "forecast_objs",
        "mrms_object_dir": "mrms_objs",
        "match_dir": "matches",
        "init_time": "2023-04-24T00:00:00",
        "color_by": "category",
        "members": {"mode": "list", "members": ["mem_1", "mem_2"]},
        "times": {"mode": "range", "lead_minutes_range": [1680, 1740]},
        "show_forecast_hatch": True,
        "output_dir": "out",
    })
    cfg = load_object_spatial_map_config(str(cfg_path))
    print(f"\n[obj-spatial-cfg-check2] members={cfg.members}, times={cfg.times}")
    assert cfg.members.mode == "list"
    assert cfg.members.members == ["mem_1", "mem_2"]
    assert cfg.times.mode == "range"
    assert cfg.times.lead_minutes_range == (1680, 1740)
    assert cfg.show_forecast_hatch is True


def test_member_selection_single_without_members_raises(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {
        "forecast_object_dir": "x", "init_time": "2023-04-24T00:00:00",
        "members": {"mode": "single"},
    })
    with pytest.raises(ValueError, match="single"):
        load_object_spatial_map_config(str(cfg_path))


def test_time_selection_range_without_range_raises(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {
        "forecast_object_dir": "x", "init_time": "2023-04-24T00:00:00",
        "times": {"mode": "range"},
    })
    with pytest.raises(ValueError, match="range"):
        load_object_spatial_map_config(str(cfg_path))
