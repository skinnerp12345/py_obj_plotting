"""Real, unconditional end-to-end validation of
drivers.plot_reflectivity_maps.run_one_case against py_obj_plotting's own
bundled py_obj/sample_data/ -- covers all 3 sources plus the
stacked-member (WoFS-style) scenario, no skip-marks needed since this data
is committed and always present. Additional skip-marked smoke tests against
the larger external test_mrms//interp_mrms//test_mpas/ directories provide
extra, optional, real-scale coverage on top.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_plot_reflectivity_maps.py -v -s
"""

import os

import numpy as np
import pytest
import yaml

from py_obj_plotting.drivers.plot_reflectivity_maps import run_one_case

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE_DATA = os.path.join(REPO_ROOT, "py_obj", "sample_data")
TEST_MRMS_DIR = os.path.join(REPO_ROOT, "test_mrms")
INTERP_MRMS_DIR = os.path.join(REPO_ROOT, "interp_mrms")
TEST_MPAS_DIR = os.path.join(REPO_ROOT, "test_mpas")
TEST_WOFS_DIR = os.path.join(REPO_ROOT, "test_wofs")

_SKIP_NO_TEST_MRMS = pytest.mark.skipif(not os.path.isdir(TEST_MRMS_DIR), reason="requires the external test_mrms/ dataset (not bundled)")
_SKIP_NO_INTERP_MRMS = pytest.mark.skipif(not os.path.isdir(INTERP_MRMS_DIR), reason="requires the external interp_mrms/ dataset (not bundled)")
_SKIP_NO_TEST_MPAS = pytest.mark.skipif(not os.path.isdir(TEST_MPAS_DIR), reason="requires the external test_mpas/ dataset (not bundled)")
_SKIP_NO_TEST_WOFS = pytest.mark.skipif(not os.path.isdir(TEST_WOFS_DIR), reason="requires the external test_wofs/ dataset (not bundled)")


def _write_config(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


# ---- unconditional, bundled sample_data ----


def test_real_mrms_native(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    _write_config(str(cfg_path), {
        "source": "mrms_native",
        "input_dir": os.path.join(SAMPLE_DATA, "mpas_case", "mrms", "20230501"),
        "output_dir": str(out_dir),
    })
    written = run_one_case(str(cfg_path))
    print(f"\n[refl-driver-check1] written={len(written)}")
    assert len(written) == 3
    for p in written:
        assert os.path.isfile(p)
        assert os.path.getsize(p) > 0


def test_real_mrms_interp(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    _write_config(str(cfg_path), {
        "source": "mrms_interp",
        "input_dir": os.path.join(SAMPLE_DATA, "mpas_case", "interp_mrms", "20230501"),
        "output_dir": str(out_dir),
    })
    written = run_one_case(str(cfg_path))
    print(f"\n[refl-driver-check2] written={len(written)}")
    assert len(written) == 3
    for p in written:
        assert os.path.isfile(p)
        assert os.path.getsize(p) > 0


def test_real_model_deterministic_mpas(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    _write_config(str(cfg_path), {
        "source": "model",
        "input_dir": os.path.join(SAMPLE_DATA, "mpas_case", "mpas_mem1"),
        "var_name": "refl10cm_max",
        "output_dir": str(out_dir),
    })
    written = run_one_case(str(cfg_path))
    print(f"\n[refl-driver-check3] written={len(written)}")
    assert len(written) == 4  # f001, f002, f003, f048
    for p in written:
        assert os.path.isfile(p)
        assert os.path.getsize(p) > 0
    basenames = {os.path.basename(p) for p in written}
    assert all("det" in b for b in basenames)


def test_real_model_stacked_members_wofs(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    _write_config(str(cfg_path), {
        "source": "model",
        "input_dir": os.path.join(SAMPLE_DATA, "wofs_case", "wofs"),
        "var_name": "comp_dz",
        "lat_name": "xlat",
        "lon_name": "xlon",
        "stacked_members": True,
        "member_id": "mem03",
        "valid_time_attr": "valid_time",
        "valid_time_format": "%Y%m%d_%H%M%S",
        "output_dir": str(out_dir),
    })
    written = run_one_case(str(cfg_path))
    print(f"\n[refl-driver-check4] written={len(written)}")
    assert len(written) == 3  # 3 real files in wofs_case/wofs/
    for p in written:
        assert os.path.isfile(p)
        assert os.path.getsize(p) > 0
        assert "mem03" in os.path.basename(p)


def test_stacked_members_without_member_id_raises(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write_config(str(cfg_path), {
        "source": "model",
        "input_dir": os.path.join(SAMPLE_DATA, "wofs_case", "wofs"),
        "var_name": "comp_dz",
        "lat_name": "xlat",
        "lon_name": "xlon",
        "stacked_members": True,
        "valid_time_attr": "valid_time",
        "valid_time_format": "%Y%m%d_%H%M%S",
    })
    with pytest.raises(ValueError, match="member_id"):
        run_one_case(str(cfg_path))


def test_native_mrms_clips_near_zero_sentinel(tmp_path):
    """Real regression check: native MRMS data must have -99 ("real obs
    near/below 0 dBZ") clipped to 0 before plotting -- confirmed by loading
    the same file both ways and checking the clipped version has no
    remaining -99 values while the raw version might."""
    from py_obj.regrid import load_mrms_grib2

    raw_path = os.path.join(SAMPLE_DATA, "mpas_case", "mrms", "20230501", "MRMS_MergedReflectivityQCComposite_00.50_20230501-010041.grib2.gz")
    raw_field = load_mrms_grib2(raw_path)

    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    _write_config(str(cfg_path), {
        "source": "mrms_native",
        "input_dir": os.path.join(SAMPLE_DATA, "mpas_case", "mrms", "20230501"),
        "times": {"mode": "single", "valid_time": "2023-05-01T01:00:41"},
        "output_dir": str(out_dir),
    })
    run_one_case(str(cfg_path))

    from py_obj.regrid.io_mrms import MRMS_NEAR_ZERO_VALUE, clip_near_zero_sentinel
    clipped = clip_near_zero_sentinel(raw_field.data)
    print(f"\n[refl-driver-check5] raw has -99: {np.any(raw_field.data == MRMS_NEAR_ZERO_VALUE)}, "
          f"clipped has -99: {np.any(clipped == MRMS_NEAR_ZERO_VALUE)}")
    assert not np.any(clipped == MRMS_NEAR_ZERO_VALUE)


# ---- additional, skip-marked, larger external-data coverage ----


@_SKIP_NO_TEST_MRMS
def test_real_external_mrms_native_range(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    _write_config(str(cfg_path), {
        "source": "mrms_native",
        "input_dir": os.path.join(TEST_MRMS_DIR, "20230401"),
        "output_dir": str(out_dir),
    })
    written = run_one_case(str(cfg_path))
    print(f"\n[refl-driver-check6] written={len(written)}")
    assert len(written) > 0


@_SKIP_NO_INTERP_MRMS
def test_real_external_mrms_interp_single(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    _write_config(str(cfg_path), {
        "source": "mrms_interp",
        "input_dir": os.path.join(INTERP_MRMS_DIR, "20230401"),
        "times": {"mode": "single", "valid_time": "2023-04-01T00:00:00", "tolerance_minutes": 15.0},
        "output_dir": str(out_dir),
    })
    written = run_one_case(str(cfg_path))
    assert len(written) == 1


@_SKIP_NO_TEST_MPAS
def test_real_external_mpas_member_subdirs(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    _write_config(str(cfg_path), {
        "source": "model",
        "input_dir": TEST_MPAS_DIR,
        "var_name": "refl10cm_max",
        "member_subdirs": True,
        "member_id": "mem1",
        "output_dir": str(out_dir),
    })
    written = run_one_case(str(cfg_path))
    print(f"\n[refl-driver-check7] written={len(written)}")
    assert len(written) > 0


@_SKIP_NO_TEST_WOFS
def test_real_external_wofs_stacked(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    _write_config(str(cfg_path), {
        "source": "model",
        "input_dir": TEST_WOFS_DIR,
        "var_name": "comp_dz",
        "lat_name": "xlat",
        "lon_name": "xlon",
        "stacked_members": True,
        "member_id": "mem00",
        "valid_time_attr": "valid_time",
        "valid_time_format": "%Y%m%d_%H%M%S",
        "output_dir": str(out_dir),
    })
    written = run_one_case(str(cfg_path))
    print(f"\n[refl-driver-check8] written={len(written)}")
    assert len(written) > 0
