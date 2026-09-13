"""End-to-end driver validation for drivers.plot_matching_timeseries.run_one_case:
synthetic fixtures targeting the flexibility axes this whole redesign was
built for (1/2/3+ systems, ensemble/deterministic/mixed, mismatched forecast
lengths, mismatched case sets under both require_shared_cases settings,
intersection vs. union lead-time modes), plus real-data smoke tests against
both shipped example configs (skip-marked if the external dataset isn't
present locally, this project's established convention).

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_plot_matching_timeseries.py -v -s
"""

import os
from datetime import datetime, timedelta

import pytest
import yaml

from py_obj.obj_core import MatchRecord, MatchResult, write_match_file
from py_obj_plotting.drivers.plot_matching_timeseries import run_one_case

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY_OBJ_PLOTTING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(PY_OBJ_PLOTTING_DIR, "configs")
SFE_MATCHES_DIR = os.path.join(REPO_ROOT, "sfe_poster", "sfe_matches")
NCAR_MPAS_DIR = os.path.join(REPO_ROOT, "ncar_mpas_analysis")
_SKIP_NO_SFE_POSTER = pytest.mark.skipif(
    not os.path.isdir(SFE_MATCHES_DIR), reason="requires the external sfe_poster/sfe_matches/ dataset (not bundled)",
)
_SKIP_NO_NCAR_MPAS_ANALYSIS = pytest.mark.skipif(
    not os.path.isdir(NCAR_MPAS_DIR), reason="requires the external ncar_mpas_analysis/ dataset (not bundled)",
)

_EXPECTED_N_PLOTS = 4 + 5 + 2 + 2  # 4 metrics + 5 diagnostics + 2 object timeseries + 2 boxplots


def _record(category: str, rid: int) -> MatchRecord:
    kwargs = dict(category=category, truth_id=rid, forecast_id=rid, ti_score=0.9 if category == "hit" else 0.0)
    if category == "hit":
        kwargs.update(
            centroid_dist_km=5.0, truth_area_km2=100.0, forecast_area_km2=110.0,
            truth_max_intensity=50.0, forecast_max_intensity=52.0,
            truth_mean_intensity=45.0, forecast_mean_intensity=46.0,
            truth_centroid_lat=35.0, truth_centroid_lon=-97.0,
            forecast_centroid_lat=35.05, forecast_centroid_lon=-97.05,
        )
    return MatchRecord(**kwargs)


def _write_case(path: str, init_time: datetime, member_records: dict) -> None:
    """member_records: {(member_id_or_None, lead_minutes): [category_str, ...]}"""
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


def _write_config(path: str, systems: list, **overrides) -> None:
    data = {"systems": systems, **overrides}
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


# --- Check 1: N=1 system (minimum viable) -----------------------------------

def test_single_system_runs_end_to_end(tmp_path):
    d = tmp_path / "solo"
    d.mkdir()
    t0 = datetime(2026, 5, 1, 0, 0, 0)
    _write_case(str(d / "match_init_x.nc"), t0, {
        ("mem_1", 0): ["hit", "miss"], ("mem_1", 60): ["hit"],
    })
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), [{"name": "Solo", "match_dir": str(d)}], output_dir=str(tmp_path / "out"))

    result = run_one_case(str(cfg_path))
    print(f"\n[driver-check1] written={len(result['written_paths'])}")
    assert len(result["written_paths"]) == _EXPECTED_N_PLOTS
    assert all(os.path.isfile(p) for p in result["written_paths"])
    assert result["case_report"].is_fully_aligned


# --- Check 2: N=2, mixed ensemble/deterministic -----------------------------

def test_two_systems_mixed_ensemble_and_deterministic(tmp_path):
    ens_dir, det_dir = tmp_path / "ens", tmp_path / "det"
    ens_dir.mkdir(); det_dir.mkdir()
    t0 = datetime(2026, 5, 1, 0, 0, 0)
    _write_case(str(ens_dir / "match_init_x.nc"), t0, {
        ("mem_1", 0): ["hit"], ("mem_1", 60): ["hit", "miss"],
        ("mem_2", 0): ["false_alarm"], ("mem_2", 60): ["hit"],
    })
    _write_case(str(det_dir / "match_init_x.nc"), t0, {
        (None, 0): ["hit"], (None, 60): ["hit", "miss"],
    })
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), [
        {"name": "Ensemble", "match_dir": str(ens_dir)},
        {"name": "Deterministic", "match_dir": str(det_dir)},
    ], output_dir=str(tmp_path / "out"))

    result = run_one_case(str(cfg_path))
    print(f"\n[driver-check2] members: Ensemble={result['match_climatologies']['Ensemble'].members}, "
          f"Deterministic={result['match_climatologies']['Deterministic'].members}")
    assert result["match_climatologies"]["Ensemble"].members == ["mem_1", "mem_2"]
    assert result["match_climatologies"]["Deterministic"].members == [None]
    assert len(result["written_paths"]) == _EXPECTED_N_PLOTS


# --- Check 3: N=3, intersection mode -----------------------------------------

def test_three_systems_intersection_mode(tmp_path):
    dirs = {name: tmp_path / name for name in ("A", "B", "C")}
    for d in dirs.values():
        d.mkdir()
    t0 = datetime(2026, 5, 1, 0, 0, 0)
    for name, d in dirs.items():
        _write_case(str(d / "match_init_x.nc"), t0, {(None, 0): ["hit"], (None, 60): ["hit"]})

    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), [{"name": n, "match_dir": str(d)} for n, d in dirs.items()],
                  lead_time={"mode": "intersection"}, output_dir=str(tmp_path / "out"))

    result = run_one_case(str(cfg_path))
    print(f"\n[driver-check3] systems={list(result['match_climatologies'].keys())}")
    assert set(result["match_climatologies"].keys()) == {"A", "B", "C"}
    assert result["config"].lead_time.mode == "intersection"


# --- Check 4: mismatched forecast lengths (different lead-time ranges) -----

def test_mismatched_forecast_lengths_union_mode(tmp_path):
    """System A has a short forecast (0-60min), System B a longer one
    (0-180min) -- union mode must let each contribute its own full range
    independently, not truncate B down to A's shorter range."""
    da, db = tmp_path / "A", tmp_path / "B"
    da.mkdir(); db.mkdir()
    t0 = datetime(2026, 5, 1, 0, 0, 0)
    _write_case(str(da / "match_init_x.nc"), t0, {(None, 0): ["hit"], (None, 60): ["hit"]})
    _write_case(str(db / "match_init_x.nc"), t0, {(None, 0): ["hit"], (None, 60): ["hit"],
                                                     (None, 120): ["hit"], (None, 180): ["hit"]})
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), [{"name": "A", "match_dir": str(da)}, {"name": "B", "match_dir": str(db)}],
                  lead_time={"mode": "union"}, output_dir=str(tmp_path / "out"))

    result = run_one_case(str(cfg_path))
    a_leads = result["match_climatologies"]["A"].lead_hours
    b_leads = result["match_climatologies"]["B"].lead_hours
    print(f"\n[driver-check4] A.lead_hours={a_leads}, B.lead_hours={b_leads}")
    assert len(a_leads) == 2
    assert len(b_leads) == 4  # B's own longer range is NOT truncated to A's


# --- Check 5: mismatched case sets, both require_shared_cases settings ------

def test_mismatched_case_sets_require_shared_cases_true_raises(tmp_path):
    da, db = tmp_path / "A", tmp_path / "B"
    da.mkdir(); db.mkdir()
    t0, t1 = datetime(2026, 5, 1), datetime(2026, 5, 2)
    _write_case(str(da / f"match_init_{t0:%Y%m%d_%H%M%S}.nc"), t0, {(None, 0): ["hit"]})
    _write_case(str(da / f"match_init_{t1:%Y%m%d_%H%M%S}.nc"), t1, {(None, 0): ["hit"]})  # A-only case
    _write_case(str(db / f"match_init_{t0:%Y%m%d_%H%M%S}.nc"), t0, {(None, 0): ["hit"]})
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), [{"name": "A", "match_dir": str(da)}, {"name": "B", "match_dir": str(db)}],
                  output_dir=str(tmp_path / "out"), require_shared_cases=True)

    with pytest.raises(ValueError, match="do not all cover the same set of forecast cases"):
        run_one_case(str(cfg_path))


def test_mismatched_case_sets_require_shared_cases_false_proceeds(tmp_path):
    da, db = tmp_path / "A", tmp_path / "B"
    da.mkdir(); db.mkdir()
    t0, t1 = datetime(2026, 5, 1), datetime(2026, 5, 2)
    _write_case(str(da / f"match_init_{t0:%Y%m%d_%H%M%S}.nc"), t0, {(None, 0): ["hit"]})
    _write_case(str(da / f"match_init_{t1:%Y%m%d_%H%M%S}.nc"), t1, {(None, 0): ["hit"]})
    _write_case(str(db / f"match_init_{t0:%Y%m%d_%H%M%S}.nc"), t0, {(None, 0): ["hit"]})
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), [{"name": "A", "match_dir": str(da)}, {"name": "B", "match_dir": str(db)}],
                  output_dir=str(tmp_path / "out"), require_shared_cases=False)

    result = run_one_case(str(cfg_path))  # must not raise
    print(f"\n[driver-check5] shared_cases={result['case_report'].shared_cases}, "
          f"is_fully_aligned={result['case_report'].is_fully_aligned}")
    assert not result["case_report"].is_fully_aligned
    assert result["case_report"].shared_cases == [t0]
    assert len(result["written_paths"]) == _EXPECTED_N_PLOTS  # still produces full output


# --- Check 6: metrics subset respected --------------------------------------

def test_metrics_subset_controls_which_contingency_plots_are_written(tmp_path):
    d = tmp_path / "solo"
    d.mkdir()
    t0 = datetime(2026, 5, 1, 0, 0, 0)
    _write_case(str(d / "match_init_x.nc"), t0, {(None, 0): ["hit", "miss"]})
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), [{"name": "Solo", "match_dir": str(d)}],
                  metrics=["pod", "csi"], output_dir=str(tmp_path / "out"))

    result = run_one_case(str(cfg_path))
    basenames = {os.path.basename(p) for p in result["written_paths"]}
    print(f"\n[driver-check6] basenames={sorted(basenames)}")
    assert "pod_comparison.png" in basenames
    assert "csi_comparison.png" in basenames
    assert "far_comparison.png" not in basenames
    assert "fbias_comparison.png" not in basenames
    assert len(result["written_paths"]) == 2 + 5 + 2 + 2  # 2 metrics instead of 4


# --- Check 7: real-data smoke tests, both shipped example configs -----------

@_SKIP_NO_SFE_POSTER
def test_real_sfe_3system_smoke(tmp_path):
    result = run_one_case(os.path.join(CONFIGS_DIR, "config_sfe_3system.yaml"))
    print(f"\n[driver-check7] SFE: written={len(result['written_paths'])}, "
          f"shared_cases={len(result['case_report'].shared_cases)}")
    assert len(result["written_paths"]) == _EXPECTED_N_PLOTS
    assert all(os.path.isfile(p) for p in result["written_paths"])
    assert set(result["match_climatologies"].keys()) == {"WoFS", "WoFSCast", "NowcastNet"}
    assert result["match_climatologies"]["NowcastNet"].members == [None]
    assert len(result["case_report"].shared_cases) == 144


@_SKIP_NO_NCAR_MPAS_ANALYSIS
def test_real_mpas_thresholds_smoke(tmp_path):
    result = run_one_case(os.path.join(CONFIGS_DIR, "config_mpas_thresholds.yaml"))
    print(f"\n[driver-check8] MPAS: written={len(result['written_paths'])}, "
          f"shared_cases={len(result['case_report'].shared_cases)}")
    assert len(result["written_paths"]) == _EXPECTED_N_PLOTS
    assert all(os.path.isfile(p) for p in result["written_paths"])
    assert set(result["match_climatologies"].keys()) == {"100km", "200km", "300km", "400km"}
    assert result["case_report"].is_fully_aligned
    assert len(result["case_report"].shared_cases) == 118
    for clim in result["match_climatologies"].values():
        assert len(clim.lead_hours) == 133


# --- Check 9: golden-value regression, real SFE 3-system category totals --

# Real contingency-category totals for the 3-system SFE comparison
# (WoFS/WoFSCast/NowcastNet vs. MRMS), lead_time_mode="union",
# lead_minutes=0..180 by 10min -- the exact settings the (now-retired)
# plot_sfe_multi_system_verification.py driver used to use, confirmed via a
# direct side-by-side run against that driver's own SYSTEMS/LEAD_TIME_MODE/
# LEAD_MINUTES constants (an exact match across all 5 categories for all 3
# systems) before that driver was removed and this fixed as a golden value.
_SFE_DRIVER_GOLDEN_TOTALS = {
    "WoFS": {"hit": 274125, "miss": 226574, "false_alarm": 367093, "truth_extra": 60559, "forecast_extra": 120825},
    "WoFSCast": {"hit": 211792, "miss": 204592, "false_alarm": 257208, "truth_extra": 45352, "forecast_extra": 94817},
    "NowcastNet": {"hit": 11186, "miss": 14289, "false_alarm": 17645, "truth_extra": 2680, "forecast_extra": 5990},
}


@_SKIP_NO_SFE_POSTER
def test_real_sfe_3system_matches_golden_totals():
    """Regression proof (not just 'it runs'): build_multi_system_match_climatology
    against the 3 real sfe_poster directories must reproduce the golden
    category totals above exactly."""
    from py_obj_plotting.climatology import build_multi_system_match_climatology
    from py_obj_plotting.systems import SystemSpec

    fresh_systems = [
        SystemSpec("WoFS", os.path.join(SFE_MATCHES_DIR, "wofs_vs_mrms")),
        SystemSpec("WoFSCast", os.path.join(SFE_MATCHES_DIR, "wofscast_vs_mrms")),
        SystemSpec("NowcastNet", os.path.join(SFE_MATCHES_DIR, "nowcastnet_vs_mrms")),
    ]
    clims, _ = build_multi_system_match_climatology(fresh_systems, lead_time_mode="union", lead_minutes=list(range(0, 190, 10)))

    for name, expected in _SFE_DRIVER_GOLDEN_TOTALS.items():
        actual = {
            "hit": sum(clims[name].n_hits.values()), "miss": sum(clims[name].n_misses.values()),
            "false_alarm": sum(clims[name].n_false_alarms.values()),
            "truth_extra": sum(clims[name].n_truth_extra.values()), "forecast_extra": sum(clims[name].n_forecast_extra.values()),
        }
        print(f"\n[driver-check9] {name}: actual={actual}")
        assert actual == expected, f"{name}: real category totals no longer match the golden reference"


# --- Check 10: real MPAS 0-36h lead-time-SUBSET config ----------------------

@_SKIP_NO_NCAR_MPAS_ANALYSIS
def test_real_mpas_0_36h_subset_config_restricts_lead_range():
    """config_mpas_0_36h.yaml explicitly overrides lead_time.lead_minutes to
    a 37-value 0-36h subset of the real available 0-132h/133-value range --
    confirms the driver actually restricts to just that subset rather than
    silently using everything available (a real, deliberate test of the
    'explicit lead_minutes override' path, distinct from every other sample
    config here, which all auto-discover the full range)."""
    result = run_one_case(os.path.join(CONFIGS_DIR, "config_mpas_0_36h.yaml"))
    print(f"\n[driver-check10] written={len(result['written_paths'])}")
    assert len(result["written_paths"]) == _EXPECTED_N_PLOTS
    for name, clim in result["match_climatologies"].items():
        print(f"  {name}: {len(clim.lead_hours)} lead time(s), max={max(clim.lead_hours)}h")
        assert len(clim.lead_hours) == 37
        assert min(clim.lead_hours) == 0.0
        assert max(clim.lead_hours) == 36.0
    assert result["case_report"].is_fully_aligned
    assert len(result["case_report"].shared_cases) == 118


# --- Check 11: real MRMS truth trace in the object comparison plots ---------

@_SKIP_NO_NCAR_MPAS_ANALYSIS
def test_real_object_comparison_plots_include_mrms_truth_trace(tmp_path):
    """Direct proof (real data, not synthetic) that object_count_comparison.png/
    object_area_comparison.png include a black MRMS reference line with
    real, non-trivial values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from py_obj_plotting.climatology import build_multi_system_object_climatology
    from py_obj_plotting.plots.object_stats_comparison import (
        plot_object_area_comparison_timeseries,
        plot_object_count_comparison_timeseries,
    )
    from py_obj_plotting.systems import SystemSpec, assign_colors

    systems = assign_colors([
        SystemSpec("100km", os.path.join(NCAR_MPAS_DIR, "matched_objects_100km")),
        SystemSpec("400km", os.path.join(NCAR_MPAS_DIR, "matched_objects_400km")),
    ])
    clims, _ = build_multi_system_object_climatology(systems, lead_time_mode="intersection", lead_minutes=[0, 60, 120])
    series = [(s.name, s.color, clims[s.name]) for s in systems]

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    try:
        for plot_fn in (plot_object_count_comparison_timeseries, plot_object_area_comparison_timeseries):
            plot_fn(series, str(tmp_path / f"{plot_fn.__name__}.png"))
            ax = plt.gcf().axes[0]
            black_lines = [l for l in ax.get_lines() if l.get_color() == "black"]
            print(f"\n[driver-check11] {plot_fn.__name__}: black_line_y={black_lines[0].get_ydata() if black_lines else None}")
            assert len(black_lines) == 1
            assert black_lines[0].get_label() == "MRMS"
            assert all(v > 0 for v in black_lines[0].get_ydata())
            plt.close(plt.gcf())
    finally:
        plt.close = orig_close
