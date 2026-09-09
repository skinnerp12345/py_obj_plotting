"""Builds sample_data/mrms_case/ for the MRMS reflectivity/objects tutorial
notebook, from real local data (this developer's own test_mrms/ and
ncar_mpas_analysis/ directories -- neither bundled with this repo).

Case chosen by evidence: 2023-05-13 00:00:15 UTC is the highest-storm-
coverage time across a scan of every May 2023 23Z/00Z file in test_mrms/
(0.97% of valid pixels >=35 dBZ, the highest of 92 scanned times), and its
real object file has all 3 storm modes present (55 cellular / 4 mixed /
2 linear of 61 objects).

Three outputs:
- mrms_native/20230513/<file>.grib2.gz: the raw native file, copied as-is
  (~2 MB, already small -- no regridding needed to plot this panel).
- mrms_interp/interp_mrms_20230513_000015.nc: that file interpolated onto
  the real MPAS full-CONUS target grid (reusing whatever weight cache is
  available; a cold build is a real, one-time ~10 minute cost, not a bug),
  then cropped by row/col to the bounding box of conus_mask_east's own kept
  region -- keeps the bundled file small without needing a lat/lon-space
  crop (which would be unsafe on a projected, non-axis-aligned grid).
- mrms_objects/obj_obs_20230513_000015.nc: the real object file for this
  time, cropped with the IDENTICAL row/col window (so both panels share
  one consistent domain), with its object list filtered to just the
  objects whose id survives the crop.

Idempotent -- safe to re-run; overwrites its own output. Not run
automatically by anything else; this is a provenance/regeneration record,
not part of the library's own runtime.

Run with:
  /opt/anaconda3/envs/pysteps_env/bin/python py_obj_plotting/sample_data/build_mrms_case.py
"""

import os
import shutil
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_THIS_DIR)
REPO_ROOT = os.path.dirname(_PKG_DIR)
sys.path.insert(0, REPO_ROOT)

from python_obj.obj_core import (  # noqa: E402
    IdentificationResult,
    conus_mask_east,
    read_object_file,
    write_object_file,
)
from python_obj.regrid import (  # noqa: E402
    MRMS_MISSING_VALUE,
    GridSpec,
    build_conservative_regridder,
    clip_near_zero_sentinel,
    load_mrms_grib2,
    regrid_field,
    write_interpolated_mrms_netcdf,
)

NATIVE_MRMS_SRC = os.path.join(
    REPO_ROOT, "test_mrms", "20230513", "MRMS_MergedReflectivityQCComposite_00.50_20230513-000015.grib2.gz",
)
TARGET_GRID_OBJECT_FILE = os.path.join(REPO_ROOT, "ncar_mpas_analysis", "mpas_objects", "obj_init_20230512_000000.nc")
MRMS_OBJECT_FILE = os.path.join(REPO_ROOT, "ncar_mpas_analysis", "mrms_objects", "obj_obs_20230513_000015.nc")
WEIGHT_CACHE_DIR = os.path.join(REPO_ROOT, "python_obj", "tests", "_weight_cache")

OUT_DIR = os.path.join(_THIS_DIR, "mrms_case")
NATIVE_OUT_DIR = os.path.join(OUT_DIR, "mrms_native", "20230513")
INTERP_OUT_DIR = os.path.join(OUT_DIR, "mrms_interp")
OBJECTS_OUT_DIR = os.path.join(OUT_DIR, "mrms_objects")


def _crop_bbox_from_mask(keep_mask: np.ndarray, margin: int = 5) -> tuple:
    rows = np.any(keep_mask, axis=1)
    cols = np.any(keep_mask, axis=0)
    row_idx = np.where(rows)[0]
    col_idx = np.where(cols)[0]
    r0 = max(0, row_idx.min() - margin)
    r1 = min(keep_mask.shape[0], row_idx.max() + 1 + margin)
    c0 = max(0, col_idx.min() - margin)
    c1 = min(keep_mask.shape[1], col_idx.max() + 1 + margin)
    return slice(r0, r1), slice(c0, c1)


def main():
    os.makedirs(NATIVE_OUT_DIR, exist_ok=True)
    os.makedirs(INTERP_OUT_DIR, exist_ok=True)
    os.makedirs(OBJECTS_OUT_DIR, exist_ok=True)
    os.makedirs(WEIGHT_CACHE_DIR, exist_ok=True)

    # 1. Native GRIB2, copied as-is -- no regridding needed to plot this panel.
    native_dst = os.path.join(NATIVE_OUT_DIR, os.path.basename(NATIVE_MRMS_SRC))
    shutil.copy2(NATIVE_MRMS_SRC, native_dst)
    print(f"copied native file -> {native_dst}")

    # 2. Interpolate onto the real MPAS full-CONUS target grid.
    target_contents = read_object_file(TARGET_GRID_OBJECT_FILE)
    target_grid = GridSpec(lat2d=target_contents.lat2d, lon2d=target_contents.lon2d)

    field = load_mrms_grib2(NATIVE_MRMS_SRC)
    clipped_data = clip_near_zero_sentinel(field.data)
    source_grid = GridSpec(lat2d=field.lat2d, lon2d=field.lon2d)

    print("building/reusing conservative regridder (may take a while on a cold weight cache)...")
    regridder = build_conservative_regridder(source_grid, target_grid, WEIGHT_CACHE_DIR)
    interp_data, coverage = regrid_field(regridder, clipped_data, fill_value=MRMS_MISSING_VALUE, missing_value=-999.0)
    print(f"interpolation coverage: {coverage.frac_outside_source_bbox:.4%} outside source bbox")

    # 3. Crop both the interpolated field and the real object file to the
    # same row/col bounding box of conus_mask_east's own kept region.
    keep_mask = ~conus_mask_east(target_contents.lat2d, target_contents.lon2d)
    row_slice, col_slice = _crop_bbox_from_mask(keep_mask)
    print(f"crop window: rows {row_slice}, cols {col_slice} (full grid shape {keep_mask.shape})")

    interp_lat2d = target_contents.lat2d[row_slice, col_slice]
    interp_lon2d = target_contents.lon2d[row_slice, col_slice]
    interp_data_cropped = interp_data[row_slice, col_slice]

    interp_out_path = os.path.join(INTERP_OUT_DIR, "interp_mrms_20230513_000015.nc")
    write_interpolated_mrms_netcdf(
        interp_out_path, interp_lat2d, interp_lon2d, interp_data_cropped,
        valid_time=field.valid_time, source_file=os.path.basename(NATIVE_MRMS_SRC),
    )
    print(f"wrote cropped interpolated field -> {interp_out_path}")

    # 4. Crop the real object file identically, filtering its object list to
    # just the ids whose footprint survives the crop.
    obj_contents = read_object_file(MRMS_OBJECT_FILE)
    assert obj_contents.lat2d.shape == keep_mask.shape, "MRMS object file grid must match the target grid used above"
    cropped_labels = obj_contents.labels[row_slice, col_slice]
    surviving_ids = set(np.unique(cropped_labels)) - {0}
    cropped_objects = [o for o in obj_contents.objects if o.id in surviving_ids]
    print(f"objects: {len(obj_contents.objects)} total -> {len(cropped_objects)} survive the crop")

    result = IdentificationResult(
        labels=cropped_labels, objects=cropped_objects,
        valid_time=obj_contents.valid_times[0], member_id=None, init_time=None,
    )
    objects_out_path = os.path.join(OBJECTS_OUT_DIR, "obj_obs_20230513_000015.nc")
    write_object_file(
        objects_out_path, init_time=None,
        lat2d=obj_contents.lat2d[row_slice, col_slice], lon2d=obj_contents.lon2d[row_slice, col_slice],
        results=[result], n_source_files=1,
        thresh_1=obj_contents.thresh_1, thresh_2=obj_contents.thresh_2,
        area_thresh_km2=obj_contents.area_thresh_km2,
        tracked=obj_contents.tracked, storm_mode_classification=obj_contents.storm_mode_classification,
    )
    print(f"wrote cropped object file -> {objects_out_path}")


if __name__ == "__main__":
    main()
