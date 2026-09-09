"""Validation of config.ReflectivityMapConfig/load_reflectivity_map_config
-- the standalone config shape for the composite-reflectivity spatial map.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_reflectivity_map_config.py -v -s
"""

import os

import pytest
import yaml

from py_obj_plotting.config import load_reflectivity_map_config


def _write_config(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def test_minimal_valid_mrms_native_config(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {"source": "mrms_native", "input_dir": "mrms"})
    cfg = load_reflectivity_map_config(str(cfg_path))
    print(f"\n[refl-cfg-check1] cfg={cfg}")
    assert cfg.source == "mrms_native"
    assert cfg.input_dir == str(tmp_path / "mrms")
    assert cfg.times.mode == "full"
    assert cfg.min_dbz == 20.0
    assert cfg.output_dir == str(tmp_path / "output")


def test_paths_resolve_relative_to_config_file_directory(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    cfg_path = sub / "cfg.yaml"
    _write_config(str(cfg_path), {"source": "mrms_interp", "input_dir": "../interp_mrms", "output_dir": "out"})
    cfg = load_reflectivity_map_config(str(cfg_path))
    assert cfg.input_dir == os.path.normpath(str(tmp_path / "interp_mrms"))
    assert cfg.output_dir == str(sub / "out")


def test_missing_required_fields_raises(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {"source": "mrms_native"})
    with pytest.raises(ValueError, match="input_dir"):
        load_reflectivity_map_config(str(cfg_path))


def test_invalid_source_raises(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {"source": "bogus", "input_dir": "x"})
    with pytest.raises(ValueError, match="bogus"):
        load_reflectivity_map_config(str(cfg_path))


def test_model_source_requires_var_name(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {"source": "model", "input_dir": "x"})
    with pytest.raises(ValueError, match="var_name"):
        load_reflectivity_map_config(str(cfg_path))


def test_model_source_stacked_members_requires_member_id(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {
        "source": "model", "input_dir": "x", "var_name": "comp_dz", "stacked_members": True,
    })
    with pytest.raises(ValueError, match="member_id"):
        load_reflectivity_map_config(str(cfg_path))

    _write_config(str(cfg_path), {
        "source": "model", "input_dir": "x", "var_name": "comp_dz", "stacked_members": True, "member_id": "mem03",
    })
    cfg = load_reflectivity_map_config(str(cfg_path))
    assert cfg.member_id == "mem03"


def test_model_source_member_subdirs_requires_member_id(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {
        "source": "model", "input_dir": "x", "var_name": "refl10cm_max", "member_subdirs": True,
    })
    with pytest.raises(ValueError, match="member_id"):
        load_reflectivity_map_config(str(cfg_path))


def test_model_source_deterministic_no_member_id_needed(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {"source": "model", "input_dir": "x", "var_name": "refl10cm_max"})
    cfg = load_reflectivity_map_config(str(cfg_path))
    assert cfg.member_id is None


def test_times_single_mode_requires_valid_time(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {
        "source": "mrms_native", "input_dir": "x", "times": {"mode": "single"},
    })
    with pytest.raises(ValueError, match="valid_time"):
        load_reflectivity_map_config(str(cfg_path))


def test_times_range_mode_requires_valid_time_range(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {
        "source": "mrms_native", "input_dir": "x", "times": {"mode": "range"},
    })
    with pytest.raises(ValueError, match="valid_time_range"):
        load_reflectivity_map_config(str(cfg_path))


def test_full_config_with_range_selection(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {
        "source": "mrms_interp",
        "input_dir": "interp_mrms",
        "times": {"mode": "range", "valid_time_range": ["2023-05-01T00:00:00", "2023-05-01T06:00:00"]},
        "min_dbz": 15.0,
        "label": "My MRMS",
        "output_dir": "out",
    })
    cfg = load_reflectivity_map_config(str(cfg_path))
    print(f"\n[refl-cfg-check2] times={cfg.times}")
    assert cfg.times.mode == "range"
    assert cfg.min_dbz == 15.0
    assert cfg.label == "My MRMS"
