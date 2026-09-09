"""Validation of drivers.plot_matching_timeseries's heatmaps:/
storm_mode_timeseries: wiring -- confirms real output files are written
for a synthetic two-system comparison with real area/max_intensity/age/
is_linear data, the two-system diff heatmap is produced when both
heatmaps: and two_system_comparison: are present, and a system with no
tracked age data cleanly skips just the age heatmap (not the whole
feature).

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_heatmap_and_storm_mode_driver.py -v -s
"""

import os
from datetime import datetime, timedelta

import yaml

from python_obj.obj_core import MatchRecord, MatchResult, write_match_file
from py_obj_plotting.drivers.plot_matching_timeseries import run_one_case


def _record(category: str, rid: int, truth_is_linear=None, forecast_is_linear=None, age_seconds=None) -> MatchRecord:
    kwargs = dict(category=category, truth_id=rid, forecast_id=rid, ti_score=0.9 if category == "hit" else 0.0)
    if category in ("hit", "miss", "truth_extra"):
        kwargs.update(truth_area_km2=80.0 + rid, truth_max_intensity=45.0 + rid, truth_is_linear=truth_is_linear, truth_age_seconds=age_seconds)
    if category in ("hit", "false_alarm", "forecast_extra"):
        kwargs["forecast_is_linear"] = forecast_is_linear
    return MatchRecord(**kwargs)


def _write_case(path, init_time, member_records):
    """member_records: {(member_id, lead_min): [MatchRecord, ...]}"""
    results = []
    for (member_id, lead_min), records in member_records.items():
        results.append(MatchResult(records=records, valid_time=init_time + timedelta(minutes=lead_min), member_id=member_id))
    write_match_file(
        path, results, n_truth_source_files=1, n_forecast_source_files=1,
        max_boundary_disp_km=40.0, max_centroid_disp_km=40.0, ti_threshold=0.2, init_time=init_time,
    )


def _two_system_dirs_with_heatmap_data(tmp_path, n_cases=3, track_age=True):
    da, db = tmp_path / "A", tmp_path / "B"
    da.mkdir(); db.mkdir()
    for i in range(n_cases):
        t = datetime(2026, 5, 1 + i)
        age_a = 300.0 if track_age else None
        _write_case(str(da / f"match_init_{t:%Y%m%d_%H%M%S}.nc"), t, {
            (None, 0): [
                _record("hit", 1, truth_is_linear=0, age_seconds=age_a),
                _record("miss", 2, truth_is_linear=0, age_seconds=age_a),
                _record("hit", 3, truth_is_linear=2, age_seconds=age_a),
            ],
            (None, 60): [_record("hit", 4, truth_is_linear=0, age_seconds=age_a)],
        })
        # B is deliberately never tracked (age_seconds=None always) -- used
        # to confirm a system missing age data cleanly skips just that
        # property's heatmap.
        _write_case(str(db / f"match_init_{t:%Y%m%d_%H%M%S}.nc"), t, {
            (None, 0): [
                _record("hit", 1, truth_is_linear=0), _record("hit", 2, truth_is_linear=0),
                _record("miss", 3, truth_is_linear=1),
            ],
            (None, 60): [_record("hit", 4, truth_is_linear=0), _record("miss", 5, truth_is_linear=0)],
        })
    return str(da), str(db)


def _write_config(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def test_heatmaps_produce_per_system_and_diff_files(tmp_path):
    da, db = _two_system_dirs_with_heatmap_data(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {
        "systems": [{"name": "A", "match_dir": da}, {"name": "B", "match_dir": db}],
        "heatmaps": {"area_bins": [80, 85, 90, 95], "max_intensity_bins": [45, 48, 51, 54]},
        "two_system_comparison": {
            "case_scatter_periods": [{"name": "x", "lead_minutes_range": [0, 60]}],
            "histogram_periods": [{"name": "x", "lead_minutes_range": [0, 60]}],
        },
        "output_dir": str(tmp_path / "out"),
    })
    result = run_one_case(str(cfg_path))
    basenames = {os.path.basename(p) for p in result["written_paths"]}
    print(f"\n[heatmap-driver-check1] basenames={sorted(b for b in basenames if 'heatmap' in b)}")
    assert "heatmap_area_A.png" in basenames
    assert "heatmap_area_B.png" in basenames
    assert "heatmap_max_intensity_A.png" in basenames
    assert "heatmap_max_intensity_B.png" in basenames
    assert "heatmap_area_diff.png" in basenames
    assert "heatmap_max_intensity_diff.png" in basenames
    # age was NOT requested (no age_bins in config) -- no age heatmap files at all
    assert not any("age" in b for b in basenames)


def test_heatmap_age_skips_cleanly_for_untracked_system(tmp_path):
    """System B is never tracked (age_seconds always None) -- its own age
    heatmap must be skipped (not raise, not produce a blank/all-NaN plot),
    while system A's (tracked) age heatmap IS produced."""
    da, db = _two_system_dirs_with_heatmap_data(tmp_path, track_age=True)
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {
        "systems": [{"name": "A", "match_dir": da}, {"name": "B", "match_dir": db}],
        "heatmaps": {"age_bins": [0, 300, 600, 900]},
        "two_system_comparison": {
            "case_scatter_periods": [{"name": "x", "lead_minutes_range": [0, 60]}],
            "histogram_periods": [{"name": "x", "lead_minutes_range": [0, 60]}],
        },
        "output_dir": str(tmp_path / "out"),
    })
    result = run_one_case(str(cfg_path))
    basenames = {os.path.basename(p) for p in result["written_paths"]}
    print(f"\n[heatmap-driver-check2] basenames={sorted(b for b in basenames if 'heatmap' in b)}")
    assert "heatmap_age_A.png" in basenames  # A was tracked -- real data
    assert "heatmap_age_B.png" not in basenames  # B was never tracked -- skipped
    assert "heatmap_age_diff.png" not in basenames  # diff needs BOTH sides -- also skipped


def test_storm_mode_timeseries_produces_per_mode_files(tmp_path):
    da, db = _two_system_dirs_with_heatmap_data(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    _write_config(str(cfg_path), {
        "systems": [{"name": "A", "match_dir": da}, {"name": "B", "match_dir": db}],
        "storm_mode_timeseries": {"modes": ["cellular", "linear"]},
        "metrics": ["pod", "csi"],
        "output_dir": str(out_dir),
    })
    result = run_one_case(str(cfg_path))
    basenames = {os.path.basename(p) for p in result["written_paths"]}
    print(f"\n[storm-mode-driver-check1] written={len(result['written_paths'])}, storm_mode_basenames={sorted(b for b in basenames if 'cellular' in b or 'linear' in b)}")
    # every storm-mode file lands flat in out_dir, mode identifiable from
    # the bare filename alone
    assert os.path.isfile(os.path.join(str(out_dir), "pod_comparison_cellular.png"))
    assert os.path.isfile(os.path.join(str(out_dir), "csi_comparison_cellular.png"))
    assert os.path.isfile(os.path.join(str(out_dir), "pod_comparison_linear.png"))
    # requested metrics only include pod/csi, but the FULL diagnostic-property
    # family is always included regardless of cfg.metrics
    assert os.path.isfile(os.path.join(str(out_dir), "area_ratio_comparison_cellular.png"))
    assert os.path.isfile(os.path.join(str(out_dir), "ti_score_comparison_cellular.png"))
    assert os.path.isfile(os.path.join(str(out_dir), "centroid_dist_comparison_linear.png"))
    # far/fbias were NOT requested -- correctly absent even for storm modes
    assert not os.path.isfile(os.path.join(str(out_dir), "far_comparison_cellular.png"))
    # "mixed" was not requested -- no such files at all
    assert not os.path.isfile(os.path.join(str(out_dir), "pod_comparison_mixed.png"))


def test_storm_mode_object_count_area_produces_mrms_reference_line(tmp_path):
    """The storm-mode-filtered time series must ALSO include per-mode object
    count/total area plots, and those plots must include the MRMS
    (truth-side) reference line for objects of that
    SAME storm mode, not just the forecast system(s)' own per-mode counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    da, db = _two_system_dirs_with_heatmap_data(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    _write_config(str(cfg_path), {
        "systems": [{"name": "A", "match_dir": da}, {"name": "B", "match_dir": db}],
        "storm_mode_timeseries": {"modes": ["cellular", "linear"]},
        "metrics": ["pod"],
        "output_dir": str(out_dir),
    })

    orig_close = plt.close
    captured = []  # (title, [line.get_color() for line in ax.get_lines()])

    def _capture_close(fig=None, *a, **k):
        target = fig if fig is not None else plt.gcf()
        ax = target.axes[0]
        captured.append((ax.get_title(), [l.get_color() for l in ax.get_lines()]))
        orig_close(fig)

    plt.close = _capture_close
    result = run_one_case(str(cfg_path))
    plt.close = orig_close

    basenames = {os.path.basename(p) for p in result["written_paths"]}
    print(f"\n[storm-mode-obj-check1] object basenames={sorted(b for b in basenames if 'object_' in b)}")
    assert os.path.isfile(os.path.join(str(out_dir), "object_count_comparison_cellular.png"))
    assert os.path.isfile(os.path.join(str(out_dir), "object_area_comparison_cellular.png"))
    assert os.path.isfile(os.path.join(str(out_dir), "object_count_comparison_linear.png"))
    assert os.path.isfile(os.path.join(str(out_dir), "object_area_comparison_linear.png"))
    # "mixed" was not requested -- no such files at all
    assert not os.path.isfile(os.path.join(str(out_dir), "object_count_comparison_mixed.png"))

    count_titles = [(t, colors) for t, colors in captured if t.startswith("Total object count") and "Cellular" in t]
    assert count_titles, f"expected an object-count title mentioning Cellular, got: {[t for t, _ in captured]}"
    _, colors = count_titles[0]
    print(f"[storm-mode-obj-check1] cellular object-count line colors={colors}")
    assert "black" in colors, "MRMS truth-side reference line (black) must be present on the storm-mode object-count plot"

    area_titles = [(t, colors) for t, colors in captured if t.startswith("Total summed object area") and "Linear" in t]
    assert area_titles, f"expected an object-area title mentioning Linear, got: {[t for t, _ in captured]}"
    _, colors = area_titles[0]
    assert "black" in colors, "MRMS truth-side reference line (black) must be present on the storm-mode object-area plot"


def test_storm_mode_title_indicates_mode(tmp_path):
    """The rendered figure's own title (not just filename/directory) must
    name the storm mode -- verified directly on the axes object, per this
    project's "no direct image analysis" convention (never by reading a
    PNG's rendered pixels)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    da, db = _two_system_dirs_with_heatmap_data(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    _write_config(str(cfg_path), {
        "systems": [{"name": "A", "match_dir": da}, {"name": "B", "match_dir": db}],
        "storm_mode_timeseries": {"modes": ["cellular"]},
        "metrics": ["pod"],
        "output_dir": str(out_dir),
    })

    orig_close = plt.close
    captured_titles = []

    def _capture_close(fig=None, *a, **k):
        target = fig if fig is not None else plt.gcf()
        captured_titles.append(target.axes[0].get_title())
        orig_close(fig)

    plt.close = _capture_close
    run_one_case(str(cfg_path))
    plt.close = orig_close

    print(f"\n[storm-mode-driver-check2] n_titles={len(captured_titles)}, sample={captured_titles[-6:]}")
    pod_titles = [t for t in captured_titles if t.startswith("POD") and "Cellular" in t]
    assert pod_titles, f"expected a 'POD ... (Cellular storm mode)' title, got: {captured_titles}"
    diag_titles = [t for t in captured_titles if "area ratio" in t and "Cellular" in t]
    assert diag_titles, f"expected an area-ratio title mentioning Cellular, got: {captured_titles}"
