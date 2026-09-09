"""Validation of config.py's performance_diagrams:/
two_system_comparison:/single_case_highlight: sections on the flat config
shape, and the new suite-style parent config (load_suite_config()) --
a shared systems: registry + an analyses: list, each entry resolving its
own systems by name (or defaulting to every registered system).

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_config_suite_and_new_sections.py -v -s
"""

import os

import pytest
import yaml

from py_obj_plotting.config import load_config, load_suite_config

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PY_OBJ_PLOTTING_DIR = os.path.dirname(_THIS_DIR)
CONFIGS_DIR = os.path.join(PY_OBJ_PLOTTING_DIR, "configs")


def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


# --- performance_diagrams: ----------------------------------------------------

def test_performance_diagrams_case_scatter_and_trajectory_parse(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {
        "systems": [{"name": "A", "match_dir": "a"}],
        "performance_diagrams": {
            "case_scatter": {
                "case_grouping": "day",
                "marker_size_basis": "forecast",
                "periods": [{"name": "60-90min", "lead_minutes_range": [60, 90]}, {"name": "32-37h", "lead_minutes_range": [1920, 2220]}],
            },
            "lead_time_trajectory": {"label_interval_minutes": 30},
        },
    })
    cfg = load_config(str(p))
    pd = cfg.performance_diagrams
    print(f"\n[cfg-pd-check1] case_scatter={pd.case_scatter}, trajectory={pd.lead_time_trajectory}")
    assert pd.case_scatter.case_grouping == "day"
    assert pd.case_scatter.marker_size_basis == "forecast"
    assert len(pd.case_scatter.periods) == 2
    assert pd.case_scatter.periods[0].name == "60-90min"
    assert pd.case_scatter.periods[0].lead_minutes_range == (60, 90)
    assert pd.lead_time_trajectory.label_interval_minutes == 30


def test_case_scatter_invalid_case_grouping_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {
        "systems": [{"name": "A", "match_dir": "a"}],
        "performance_diagrams": {"case_scatter": {"case_grouping": "bogus", "periods": [{"name": "x", "lead_minutes_range": [0, 60]}]}},
    })
    with pytest.raises(ValueError, match="case_grouping"):
        load_config(str(p))


def test_case_scatter_empty_periods_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {"systems": [{"name": "A", "match_dir": "a"}], "performance_diagrams": {"case_scatter": {"periods": []}}})
    with pytest.raises(ValueError, match="periods"):
        load_config(str(p))


def test_lead_minutes_range_requires_lo_not_greater_than_hi(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {
        "systems": [{"name": "A", "match_dir": "a"}],
        "performance_diagrams": {"case_scatter": {"periods": [{"name": "bad", "lead_minutes_range": [90, 60]}]}},
    })
    with pytest.raises(ValueError, match="lo <= hi"):
        load_config(str(p))


def test_lead_minutes_range_allows_lo_equal_hi_single_instant_period(tmp_path):
    """lo == hi is a legitimate single-lead-time 'period' (e.g. exactly
    24h), not just a genuine range -- confirmed real via
    config_mpas_performance_diagrams.yaml's own 24h/120h periods."""
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {
        "systems": [{"name": "A", "match_dir": "a"}],
        "performance_diagrams": {"case_scatter": {"periods": [{"name": "24h", "lead_minutes_range": [1440, 1440]}]}},
    })
    cfg = load_config(str(p))
    print(f"\n[cfg-pd-check2] period={cfg.performance_diagrams.case_scatter.periods[0]}")
    assert cfg.performance_diagrams.case_scatter.periods[0].lead_minutes_range == (1440, 1440)


# --- two_system_comparison: ----------------------------------------------------

def test_two_system_comparison_requires_exactly_two_systems(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {
        "systems": [{"name": "A", "match_dir": "a"}, {"name": "B", "match_dir": "b"}, {"name": "C", "match_dir": "c"}],
        "two_system_comparison": {},
    })
    with pytest.raises(ValueError, match="exactly 2 systems"):
        load_config(str(p))


def test_two_system_comparison_parses_with_two_systems(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {
        "systems": [{"name": "A", "match_dir": "a"}, {"name": "B", "match_dir": "b"}],
        "two_system_comparison": {"colormap": "coolwarm", "diff_levels": [-0.2, -0.1, 0.0, 0.1, 0.2]},
    })
    cfg = load_config(str(p))
    tsc = cfg.two_system_comparison
    print(f"\n[cfg-tsc-check1] {tsc}")
    assert tsc.colormap == "coolwarm"
    assert tsc.diff_levels == [-0.2, -0.1, 0.0, 0.1, 0.2]
    assert tsc.case_scatter_periods is None
    assert tsc.histogram_periods is None


def test_two_system_comparison_defaults_with_two_systems(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {
        "systems": [{"name": "A", "match_dir": "a"}, {"name": "B", "match_dir": "b"}],
        "two_system_comparison": {},
    })
    cfg = load_config(str(p))
    assert cfg.two_system_comparison.colormap == "RdBu_r"
    assert cfg.two_system_comparison.diff_levels is None


# --- single_case_highlight: ----------------------------------------------------

def test_single_case_highlight_parses(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {
        "systems": [{"name": "A", "match_dir": "a"}],
        "single_case_highlight": {"system": "A", "case": "2023-04-24", "baseline_cases": ["2023-04-25"], "highlight_color": [1.0, 0.5, 0.0]},
    })
    cfg = load_config(str(p))
    sch = cfg.single_case_highlight
    print(f"\n[cfg-sch-check1] {sch}")
    assert sch.system == "A"
    assert sch.case == "2023-04-24"
    assert sch.baseline_cases == ["2023-04-25"]
    assert sch.highlight_color == (1.0, 0.5, 0.0)


def test_single_case_highlight_unknown_system_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {
        "systems": [{"name": "A", "match_dir": "a"}],
        "single_case_highlight": {"system": "NOT_A", "case": "2023-04-24"},
    })
    with pytest.raises(ValueError, match="not among this analysis's own"):
        load_config(str(p))


def test_single_case_highlight_missing_required_field_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {"systems": [{"name": "A", "match_dir": "a"}], "single_case_highlight": {"system": "A"}})
    with pytest.raises(ValueError, match="case"):
        load_config(str(p))


# --- suite (analyses:) shape ----------------------------------------------------

def test_suite_config_registry_and_named_system_subset(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {
        "systems": [
            {"name": "A", "match_dir": "a"}, {"name": "B", "match_dir": "b"}, {"name": "C", "match_dir": "c"},
        ],
        "analyses": [
            {"name": "ab_comparison", "systems": ["A", "B"], "output_dir": "out_ab"},
            {"name": "c_solo", "systems": ["C"], "output_dir": "out_c"},
        ],
    })
    suite = load_suite_config(str(p))
    print(f"\n[cfg-suite-check1] analyses={list(suite.keys())}")
    assert set(suite.keys()) == {"ab_comparison", "c_solo"}
    assert [s.name for s in suite["ab_comparison"].systems] == ["A", "B"]
    assert [s.name for s in suite["c_solo"].systems] == ["C"]


def test_suite_config_analysis_without_systems_uses_full_registry(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {
        "systems": [{"name": "A", "match_dir": "a"}, {"name": "B", "match_dir": "b"}],
        "analyses": [{"name": "full", "output_dir": "out"}],
    })
    suite = load_suite_config(str(p))
    print(f"\n[cfg-suite-check2] systems={[s.name for s in suite['full'].systems]}")
    assert [s.name for s in suite["full"].systems] == ["A", "B"]


def test_suite_config_unknown_system_name_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {
        "systems": [{"name": "A", "match_dir": "a"}],
        "analyses": [{"name": "bad", "systems": ["A", "ZZZ"]}],
    })
    with pytest.raises(ValueError, match="unknown system name"):
        load_suite_config(str(p))


def test_suite_config_duplicate_analysis_name_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {
        "systems": [{"name": "A", "match_dir": "a"}],
        "analyses": [{"name": "dup"}, {"name": "dup"}],
    })
    with pytest.raises(ValueError, match="duplicate analysis name"):
        load_suite_config(str(p))


def test_suite_config_each_analysis_gets_independent_settings(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {
        "systems": [{"name": "A", "match_dir": "a"}, {"name": "B", "match_dir": "b"}],
        "analyses": [
            {"name": "union_analysis", "systems": ["A", "B"], "lead_time": {"mode": "union"}, "metrics": ["pod"]},
            {"name": "intersection_analysis", "systems": ["A", "B"], "lead_time": {"mode": "intersection"}, "metrics": ["csi", "far"]},
        ],
    })
    suite = load_suite_config(str(p))
    print(f"\n[cfg-suite-check3] union={suite['union_analysis'].lead_time.mode}, "
          f"intersection={suite['intersection_analysis'].lead_time.mode}")
    assert suite["union_analysis"].lead_time.mode == "union"
    assert suite["union_analysis"].metrics == ["pod"]
    assert suite["intersection_analysis"].lead_time.mode == "intersection"
    assert suite["intersection_analysis"].metrics == ["csi", "far"]


def test_suite_config_paths_resolve_relative_to_config_dir(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    p = sub / "cfg.yaml"
    _write_yaml(str(p), {
        "systems": [{"name": "A", "match_dir": "../matches_a"}],
        "analyses": [{"name": "solo", "output_dir": "out"}],
    })
    suite = load_suite_config(str(p))
    print(f"\n[cfg-suite-check4] match_dir={suite['solo'].systems[0].match_dir}, output_dir={suite['solo'].output_dir}")
    assert suite["solo"].systems[0].match_dir == os.path.normpath(str(tmp_path / "matches_a"))
    assert suite["solo"].output_dir == os.path.normpath(str(sub / "out"))


def test_load_config_rejects_file_with_analyses_key(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {"systems": [{"name": "A", "match_dir": "a"}], "analyses": [{"name": "x"}]})
    with pytest.raises(ValueError, match="load_suite_config"):
        load_config(str(p))


def test_load_suite_config_rejects_flat_file(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {"systems": [{"name": "A", "match_dir": "a"}]})
    with pytest.raises(ValueError, match="load_config"):
        load_suite_config(str(p))


def test_suite_config_two_system_comparison_and_single_case_highlight_per_analysis(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {
        "systems": [{"name": "A", "match_dir": "a"}, {"name": "B", "match_dir": "b"}],
        "analyses": [
            {"name": "diff", "systems": ["A", "B"], "two_system_comparison": {}},
            {"name": "highlight_a", "systems": ["A"], "single_case_highlight": {"system": "A", "case": "c1"}},
        ],
    })
    suite = load_suite_config(str(p))
    print(f"\n[cfg-suite-check5] diff.tsc={suite['diff'].two_system_comparison}, "
          f"highlight.sch={suite['highlight_a'].single_case_highlight}")
    assert suite["diff"].two_system_comparison is not None
    assert suite["highlight_a"].single_case_highlight.case == "c1"


# --- heatmaps: ------------------------------------------------------------------

def test_heatmaps_parses_all_three_properties(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {
        "systems": [{"name": "A", "match_dir": "a"}],
        "heatmaps": {
            "area_bins": [0, 50, 100, 150],
            "max_intensity_bins": [40, 50, 60, 70],
            "age_bins": [-3600, 0, 3600, 7200],
        },
    })
    cfg = load_config(str(p))
    hm = cfg.heatmaps
    print(f"\n[cfg-heatmap-check1] area={hm.area_bins}, max_intensity={hm.max_intensity_bins}, age={hm.age_bins}")
    assert hm.area_bins == [0, 50, 100, 150]
    assert hm.max_intensity_bins == [40, 50, 60, 70]
    assert hm.age_bins == [-3600, 0, 3600, 7200]
    assert set(hm.bin_edges()) == {"area", "max_intensity", "age"}


def test_heatmaps_one_property_only(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {
        "systems": [{"name": "A", "match_dir": "a"}],
        "heatmaps": {"area_bins": [0, 50, 100]},
    })
    cfg = load_config(str(p))
    print(f"\n[cfg-heatmap-check2] bin_edges={cfg.heatmaps.bin_edges()}")
    assert set(cfg.heatmaps.bin_edges()) == {"area"}
    assert cfg.heatmaps.max_intensity_bins is None
    assert cfg.heatmaps.age_bins is None


def test_heatmaps_no_properties_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {"systems": [{"name": "A", "match_dir": "a"}], "heatmaps": {}})
    with pytest.raises(ValueError, match="at least one is required"):
        load_config(str(p))


# --- storm_mode_timeseries: ------------------------------------------------------

def test_storm_mode_timeseries_defaults_to_all_three_modes(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {"systems": [{"name": "A", "match_dir": "a"}], "storm_mode_timeseries": {}})
    cfg = load_config(str(p))
    print(f"\n[cfg-storm-mode-check1] modes={cfg.storm_mode_timeseries.modes}")
    assert cfg.storm_mode_timeseries.modes == ["cellular", "mixed", "linear"]


def test_storm_mode_timeseries_explicit_subset(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {"systems": [{"name": "A", "match_dir": "a"}], "storm_mode_timeseries": {"modes": ["cellular", "linear"]}})
    cfg = load_config(str(p))
    assert cfg.storm_mode_timeseries.modes == ["cellular", "linear"]


def test_storm_mode_timeseries_invalid_mode_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    _write_yaml(str(p), {"systems": [{"name": "A", "match_dir": "a"}], "storm_mode_timeseries": {"modes": ["bogus"]}})
    with pytest.raises(ValueError, match="invalid entries"):
        load_config(str(p))
