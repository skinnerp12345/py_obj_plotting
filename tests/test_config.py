"""load_config() validation: synthetic schema/validation checks, plus a
real-parse check of the two shipped example configs (config_sfe_3system.yaml,
config_mpas_thresholds.yaml) -- confirming they parse and resolve paths
correctly regardless of whether the external sfe_poster/ncar_mpas_analysis
datasets those paths point at happen to be present locally (load_config()
itself never touches the filesystem beyond the YAML file itself).

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_config.py -v -s
"""

import os

import pytest
import yaml

from py_obj_plotting.config import BootstrapConfig, LeadTimeConfig, load_config
from py_obj_plotting.systems import SystemSpec

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PY_OBJ_PLOTTING_DIR = os.path.dirname(_THIS_DIR)
CONFIGS_DIR = os.path.join(PY_OBJ_PLOTTING_DIR, "configs")


def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


# --- Check 1: minimal valid config, defaults ---------------------------------

def test_minimal_config_uses_defaults(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {"systems": [{"name": "A", "match_dir": "matches_a"}]})

    cfg = load_config(str(p))
    print(f"\n[config-check1] systems={cfg.systems}, lead_time={cfg.lead_time}, "
          f"metrics={cfg.metrics}, bootstrap={cfg.bootstrap}, output_dir={cfg.output_dir}")
    assert len(cfg.systems) == 1
    assert cfg.systems[0].name == "A"
    assert cfg.systems[0].match_dir == os.path.normpath(str(tmp_path / "matches_a"))
    assert cfg.systems[0].color is None
    assert cfg.lead_time == LeadTimeConfig(mode="intersection", lead_minutes=None)
    assert cfg.metrics == ["pod", "far", "fbias", "csi"]
    assert cfg.bootstrap == BootstrapConfig(samples=1000, ci_interval=95)
    assert cfg.require_shared_cases is True


# --- Check 2: path resolution relative to the config file's own directory ---

def test_paths_resolve_relative_to_config_dir_not_cwd(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    p = sub / "cfg.yaml"
    _write_yaml(str(p), {
        "systems": [{"name": "A", "match_dir": "../matches_a"}],
        "output_dir": "out",
    })
    cfg = load_config(str(p))
    print(f"\n[config-check2] match_dir={cfg.systems[0].match_dir}, output_dir={cfg.output_dir}")
    assert cfg.systems[0].match_dir == os.path.normpath(str(tmp_path / "matches_a"))
    assert cfg.output_dir == os.path.normpath(str(sub / "out"))


def test_absolute_paths_left_untouched(tmp_path):
    p = tmp_path / "cfg.yaml"
    abs_dir = str(tmp_path / "abs_matches")
    _write_yaml(str(p), {"systems": [{"name": "A", "match_dir": abs_dir}]})
    cfg = load_config(str(p))
    assert cfg.systems[0].match_dir == os.path.normpath(abs_dir)


# --- Check 3: full field population, including per-system color/pattern ----

def test_full_config_all_fields(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {
        "systems": [
            {"name": "A", "match_dir": "a", "color": [0.1, 0.2, 0.3], "match_pattern": "match_*.nc"},
            {"name": "B", "match_dir": "b"},
        ],
        "lead_time": {"mode": "union", "lead_minutes": [0, 10, 20]},
        "metrics": ["pod", "csi"],
        "bootstrap": {"samples": 500, "ci_interval": 90},
        "output_dir": "out",
        "require_shared_cases": False,
    })
    cfg = load_config(str(p))
    print(f"\n[config-check3] {cfg}")
    assert cfg.systems[0].color == (0.1, 0.2, 0.3)
    assert cfg.systems[0].match_pattern == "match_*.nc"
    assert cfg.systems[1].color is None
    assert cfg.lead_time.mode == "union"
    assert cfg.lead_time.lead_minutes == [0, 10, 20]
    assert cfg.metrics == ["pod", "csi"]
    assert cfg.bootstrap.samples == 500
    assert cfg.bootstrap.ci_interval == 90
    assert cfg.require_shared_cases is False


# --- Check 4: validation errors ---------------------------------------------

def test_missing_systems_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {"metrics": ["pod"]})
    with pytest.raises(ValueError, match="systems"):
        load_config(str(p))


def test_empty_systems_list_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {"systems": []})
    with pytest.raises(ValueError, match="systems"):
        load_config(str(p))


def test_system_missing_required_field_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {"systems": [{"name": "A"}]})  # no match_dir
    with pytest.raises(ValueError, match="match_dir"):
        load_config(str(p))


def test_duplicate_system_name_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {"systems": [{"name": "A", "match_dir": "a"}, {"name": "A", "match_dir": "b"}]})
    with pytest.raises(ValueError, match="duplicate system name"):
        load_config(str(p))


def test_invalid_lead_time_mode_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {"systems": [{"name": "A", "match_dir": "a"}], "lead_time": {"mode": "bogus"}})
    with pytest.raises(ValueError, match="lead_time.mode"):
        load_config(str(p))


def test_invalid_metric_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {"systems": [{"name": "A", "match_dir": "a"}], "metrics": ["pod", "not_a_metric"]})
    with pytest.raises(ValueError, match="metrics"):
        load_config(str(p))


def test_non_mapping_top_level_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(["not", "a", "mapping"], f)
    with pytest.raises(ValueError, match="mapping"):
        load_config(str(p))


# --- Check 5: the two real, shipped example configs -------------------------

def test_real_sfe_3system_config_parses():
    """Confirms this config parses correctly regardless of whether
    sfe_poster/ (an external, non-bundled dataset) is present locally --
    load_config() itself never touches the filesystem beyond the YAML."""
    cfg = load_config(os.path.join(CONFIGS_DIR, "config_sfe_3system.yaml"))
    print(f"\n[config-check4] sfe_3system: {[s.name for s in cfg.systems]}, mode={cfg.lead_time.mode}")
    assert [s.name for s in cfg.systems] == ["WoFS", "WoFSCast", "NowcastNet"]
    for s in cfg.systems:
        assert os.path.isabs(s.match_dir)
        assert s.match_dir.endswith(os.path.join("sfe_matches", {
            "WoFS": "wofs_vs_mrms", "WoFSCast": "wofscast_vs_mrms", "NowcastNet": "nowcastnet_vs_mrms",
        }[s.name]))
    assert cfg.lead_time.mode == "union"
    assert cfg.lead_time.lead_minutes is None


def test_real_mpas_thresholds_config_parses():
    cfg = load_config(os.path.join(CONFIGS_DIR, "config_mpas_thresholds.yaml"))
    print(f"\n[config-check5] mpas_thresholds: {[s.name for s in cfg.systems]}, mode={cfg.lead_time.mode}")
    assert [s.name for s in cfg.systems] == ["100km", "200km", "300km", "400km"]
    for s in cfg.systems:
        assert os.path.isabs(s.match_dir)
        assert s.match_dir.endswith(f"matched_objects_{s.name}")
    assert cfg.lead_time.mode == "intersection"
