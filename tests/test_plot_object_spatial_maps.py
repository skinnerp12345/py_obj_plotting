"""Real, non-skip-marked-where-possible end-to-end validation of
drivers.plot_object_spatial_maps.run_one_case -- category mode with a
member-list + lead-minutes-range selection, and paintball mode with all 5
real members. Uses the same real ncar_mpas_analysis/ dataset exercised
throughout this feature's synthetic+real test files; skip-marked when that
external, non-bundled dataset is absent (matching test_driver_new_features.py's
convention).

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_plot_object_spatial_maps.py -v -s
"""

import os

import pytest
import yaml

from py_obj_plotting.drivers.plot_object_spatial_maps import run_one_case

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NCAR_MPAS_DIR = os.path.join(REPO_ROOT, "ncar_mpas_analysis")
_SKIP_NO_NCAR_MPAS_ANALYSIS = pytest.mark.skipif(
    not os.path.isdir(NCAR_MPAS_DIR), reason="requires the external ncar_mpas_analysis/ dataset (not bundled)",
)


def _write_config(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


@_SKIP_NO_NCAR_MPAS_ANALYSIS
def test_real_category_mode_member_list_and_lead_range(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    _write_config(str(cfg_path), {
        "forecast_object_dir": os.path.join(NCAR_MPAS_DIR, "mpas_objects"),
        "mrms_object_dir": os.path.join(NCAR_MPAS_DIR, "mrms_objects"),
        "match_dir": os.path.join(NCAR_MPAS_DIR, "matched_objects_300km"),
        "init_time": "2023-04-24T00:00:00",
        "color_by": "category",
        "members": {"mode": "list", "members": ["mem_1", "mem_2"]},
        "times": {"mode": "range", "lead_minutes_range": [1680, 1740]},  # 28h, 29h -- both confirmed to have real data
        "output_dir": str(out_dir),
    })

    written = run_one_case(str(cfg_path))
    print(f"\n[obj-spatial-driver-check1] written={len(written)}: {[os.path.basename(p) for p in written]}")
    assert len(written) == 4  # 2 members x 2 lead times
    for p in written:
        assert os.path.isfile(p)
        assert os.path.getsize(p) > 0
    basenames = {os.path.basename(p) for p in written}
    assert "category_mem_1_20230425_050000.png" in basenames
    assert "category_mem_2_20230425_040000.png" in basenames


@_SKIP_NO_NCAR_MPAS_ANALYSIS
def test_real_paintball_all_members_single_time(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    _write_config(str(cfg_path), {
        "forecast_object_dir": os.path.join(NCAR_MPAS_DIR, "mpas_objects"),
        "init_time": "2023-04-24T00:00:00",
        "members": {"mode": "all"},
        "times": {"mode": "single", "lead_minutes": 1740},  # 29h
        "paintball": True,
        "output_dir": str(out_dir),
    })

    written = run_one_case(str(cfg_path))
    print(f"\n[obj-spatial-driver-check2] written={written}")
    assert len(written) == 1
    assert os.path.isfile(written[0])
    assert os.path.getsize(written[0]) > 0
    assert os.path.basename(written[0]) == "paintball_single_20230425_050000.png"


@_SKIP_NO_NCAR_MPAS_ANALYSIS
def test_real_paintball_storm_mode(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    _write_config(str(cfg_path), {
        "forecast_object_dir": os.path.join(NCAR_MPAS_DIR, "mpas_objects"),
        "init_time": "2023-04-24T00:00:00",
        "members": {"mode": "list", "members": ["mem_1", "mem_2"]},
        "times": {"mode": "single", "lead_minutes": 1740},  # 29h
        "color_by": "storm_mode",
        "paintball": True,
        "output_dir": str(out_dir),
    })

    written = run_one_case(str(cfg_path))
    print(f"\n[obj-spatial-driver-check3] written={written}")
    assert len(written) == 1
    assert os.path.isfile(written[0])
    assert os.path.getsize(written[0]) > 0
    assert os.path.basename(written[0]) == "paintball_storm_mode_20230425_050000.png"


@_SKIP_NO_NCAR_MPAS_ANALYSIS
def test_real_paintball_category_with_ensemble_mean_stats(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    _write_config(str(cfg_path), {
        "forecast_object_dir": os.path.join(NCAR_MPAS_DIR, "mpas_objects"),
        "mrms_object_dir": os.path.join(NCAR_MPAS_DIR, "mrms_objects"),
        "match_dir": os.path.join(NCAR_MPAS_DIR, "matched_objects_100km"),
        "init_time": "2023-04-24T00:00:00",
        "members": {"mode": "list", "members": ["mem_1", "mem_2"]},
        "times": {"mode": "single", "lead_minutes": 1740},  # 29h
        "color_by": "category",
        "paintball": True,
        "output_dir": str(out_dir),
    })

    written = run_one_case(str(cfg_path))
    print(f"\n[obj-spatial-driver-check4] written={written}")
    assert len(written) == 1
    assert os.path.isfile(written[0])
    assert os.path.getsize(written[0]) > 0
    assert os.path.basename(written[0]) == "paintball_category_20230425_050000.png"
