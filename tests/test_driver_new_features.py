"""Validation of drivers.plot_matching_timeseries's
performance_diagrams:/two_system_comparison: wired into run_one_case()'s
normal comparison path, single_case_highlight: branching to the dedicated
highlight path (never combined with the N-system comparison path), and
run_suite() executing multiple analyses from one parent config, each with
independent settings and its own output directory.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_driver_new_features.py -v -s
"""

import os
from datetime import datetime, timedelta

import pytest
import yaml

from python_obj.obj_core import MatchRecord, MatchResult, write_match_file
from py_obj_plotting.drivers.plot_matching_timeseries import run_one_case, run_suite

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY_OBJ_PLOTTING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(PY_OBJ_PLOTTING_DIR, "configs")
NCAR_MPAS_DIR = os.path.join(REPO_ROOT, "ncar_mpas_analysis")
_SKIP_NO_NCAR_MPAS_ANALYSIS = pytest.mark.skipif(
    not os.path.isdir(NCAR_MPAS_DIR), reason="requires the external ncar_mpas_analysis/ dataset (not bundled)",
)


def _record(category: str, rid: int) -> MatchRecord:
    kwargs = dict(category=category, truth_id=rid, forecast_id=rid, ti_score=0.9 if category == "hit" else 0.0)
    if category == "hit":
        kwargs.update(
            centroid_dist_km=5.0, truth_area_km2=100.0, forecast_area_km2=110.0,
            truth_max_intensity=50.0, forecast_max_intensity=52.0,
            truth_mean_intensity=45.0, forecast_mean_intensity=46.0,
        )
    return MatchRecord(**kwargs)


def _write_case(path, init_time, member_records):
    results = []
    next_id = 1
    for (member_id, lead_min), categories in member_records.items():
        records = []
        for cat in categories:
            records.append(_record(cat, next_id))
            next_id += 1
        results.append(MatchResult(records=records, valid_time=init_time + timedelta(minutes=lead_min), member_id=member_id))
    write_match_file(
        path, results, n_truth_source_files=1, n_forecast_source_files=1,
        max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2, init_time=init_time,
    )


def _two_system_dirs(tmp_path, n_cases=3):
    da, db = tmp_path / "A", tmp_path / "B"
    da.mkdir(); db.mkdir()
    for i in range(n_cases):
        t = datetime(2026, 5, 1 + i)
        _write_case(str(da / f"match_init_{t:%Y%m%d_%H%M%S}.nc"), t, {
            (None, 0): ["hit", "miss"], (None, 60): ["hit"],
        })
        _write_case(str(db / f"match_init_{t:%Y%m%d_%H%M%S}.nc"), t, {
            (None, 0): ["hit", "hit"], (None, 60): ["hit", "miss"],
        })
    return str(da), str(db)


def _write_config(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


# --- performance_diagrams: wired into the normal comparison path ------------

def test_performance_diagrams_produce_case_scatter_and_trajectory(tmp_path):
    da, db = _two_system_dirs(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {
        "systems": [{"name": "A", "match_dir": da}, {"name": "B", "match_dir": db}],
        "performance_diagrams": {
            "case_scatter": {"periods": [{"name": "0-60min", "lead_minutes_range": [0, 60]}]},
            "lead_time_trajectory": {"label_interval_minutes": 60},
        },
        "output_dir": str(tmp_path / "out"),
    })

    result = run_one_case(str(cfg_path))
    basenames = {os.path.basename(p) for p in result["written_paths"]}
    print(f"\n[driver-new-check1] basenames_extra={[b for b in basenames if 'case_scatter' in b or 'trajectory' in b]}")
    assert "case_scatter_0-60min.png" in basenames
    assert "lead_time_trajectory.png" in basenames
    assert result["per_case_climatologies"] is not None
    assert len(result["per_case_climatologies"]["A"]) == 3


# --- two_system_comparison: wired into the normal comparison path -----------

def test_two_system_comparison_produces_diff_diagram_and_histograms(tmp_path):
    da, db = _two_system_dirs(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {
        "systems": [{"name": "A", "match_dir": da}, {"name": "B", "match_dir": db}],
        "performance_diagrams": {"case_scatter": {"periods": [{"name": "0-60min", "lead_minutes_range": [0, 60]}]}},
        "two_system_comparison": {},
        "output_dir": str(tmp_path / "out"),
    })

    result = run_one_case(str(cfg_path))
    basenames = {os.path.basename(p) for p in result["written_paths"]}
    print(f"\n[driver-new-check2] basenames_extra={[b for b in basenames if 'diff' in b]}")
    assert "two_system_diff_0-60min.png" in basenames
    assert "diff_histogram_0-60min_pod_diff_histogram.png" in basenames
    assert "diff_histogram_0-60min_csi_diff_histogram.png" in basenames


def test_two_system_comparison_without_any_periods_raises(tmp_path):
    da, db = _two_system_dirs(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {
        "systems": [{"name": "A", "match_dir": da}, {"name": "B", "match_dir": db}],
        "two_system_comparison": {},  # no periods of its own, no performance_diagrams to fall back to
        "output_dir": str(tmp_path / "out"),
    })
    with pytest.raises(ValueError, match="no periods"):
        run_one_case(str(cfg_path))


# --- single_case_highlight: dedicated path, never combined with comparison --

def test_single_case_highlight_uses_dedicated_path_not_comparison(tmp_path):
    da, _db = _two_system_dirs(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {
        "systems": [{"name": "A", "match_dir": da}],
        "single_case_highlight": {"system": "A", "case": "2026-05-01 0000Z"},
        "performance_diagrams": {
            "case_scatter": {"periods": [{"name": "0-60min", "lead_minutes_range": [0, 60]}]},
            "lead_time_trajectory": {"label_interval_minutes": 60},
        },
        "output_dir": str(tmp_path / "out"),
    })

    result = run_one_case(str(cfg_path))
    basenames = {os.path.basename(p) for p in result["written_paths"]}
    print(f"\n[driver-new-check3] basenames={sorted(basenames)}")
    # highlight-mode output, not the N-system comparison bundle
    assert "pod_highlight.png" in basenames
    assert "case_scatter_highlight_0-60min.png" in basenames
    # lead_time_trajectory has no single-case-highlight equivalent (removed
    # as unnecessary) -- configuring it is silently ignored in this mode.
    assert "lead_time_trajectory_highlight.png" not in basenames
    assert "pod_comparison.png" not in basenames  # confirms the comparison path did NOT also run
    assert len(result["baseline_climatologies"]) == 2  # 3 cases total, minus the highlighted one


# --- run_suite(): multiple analyses from one parent config -------------------

def test_run_suite_executes_multiple_independent_analyses(tmp_path):
    da, db = _two_system_dirs(tmp_path)
    cfg_path = tmp_path / "suite.yaml"
    _write_config(str(cfg_path), {
        "systems": [{"name": "A", "match_dir": da}, {"name": "B", "match_dir": db}],
        "analyses": [
            {"name": "ab_comparison", "systems": ["A", "B"], "output_dir": str(tmp_path / "out_ab")},
            {"name": "a_solo_highlight", "systems": ["A"],
             "single_case_highlight": {"system": "A", "case": "2026-05-01 0000Z"},
             "output_dir": str(tmp_path / "out_a_highlight")},
        ],
    })

    results = run_suite(str(cfg_path))
    print(f"\n[driver-new-check4] analyses={list(results.keys())}")
    assert set(results.keys()) == {"ab_comparison", "a_solo_highlight"}

    ab_basenames = {os.path.basename(p) for p in results["ab_comparison"]["written_paths"]}
    assert "pod_comparison.png" in ab_basenames
    assert os.path.isfile(os.path.join(str(tmp_path / "out_ab"), "pod_comparison.png"))

    highlight_basenames = {os.path.basename(p) for p in results["a_solo_highlight"]["written_paths"]}
    assert "pod_highlight.png" in highlight_basenames
    assert os.path.isfile(os.path.join(str(tmp_path / "out_a_highlight"), "pod_highlight.png"))


# --- Real data ------------------------------------------------------------------

@_SKIP_NO_NCAR_MPAS_ANALYSIS
def test_real_mpas_two_system_comparison_with_performance_diagrams(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {
        "systems": [
            {"name": "100km", "match_dir": os.path.join(NCAR_MPAS_DIR, "matched_objects_100km")},
            {"name": "400km", "match_dir": os.path.join(NCAR_MPAS_DIR, "matched_objects_400km")},
        ],
        "performance_diagrams": {
            "case_scatter": {"periods": [{"name": "32-37h", "lead_minutes_range": [1920, 2220]}]},
            "lead_time_trajectory": {"label_interval_minutes": 720},
        },
        "two_system_comparison": {},
        "output_dir": str(tmp_path / "out"),
    })

    result = run_one_case(str(cfg_path))
    basenames = {os.path.basename(p) for p in result["written_paths"]}
    print(f"\n[driver-new-check5] n_written={len(result['written_paths'])}")
    assert "case_scatter_32-37h.png" in basenames
    assert "lead_time_trajectory.png" in basenames
    assert "two_system_diff_32-37h.png" in basenames
    assert "diff_histogram_32-37h_csi_diff_histogram.png" in basenames
    assert len(result["per_case_climatologies"]["100km"]) == 118
