# Sample data

Small, **real** (not synthetic) data bundled with this package so the two
tutorial notebooks (`py_obj_plotting/notebooks/`) and a subset of example
configs can run with zero external data -- just clone and go.

## Provenance

- **`mrms_case/`** (~30 MB) -- a single, objectively-chosen real MRMS
  observation, 2023-05-13 00:00:15 UTC: the highest-storm-coverage time
  found by scanning every May 2023 23Z file and every May/June 2023 00Z
  file in a local `test_mrms/` archive for fraction-of-pixels >=35/40 dBZ
  (0.97% of valid pixels >=35 dBZ, the highest of 92 scanned times). Three
  files:
  - `mrms_native/20230513/*.grib2.gz` -- the raw native file, copied as-is
    (~2 MB).
  - `mrms_interp/interp_mrms_20230513_000015.nc` -- that file interpolated
    onto a real MPAS full-CONUS target grid (reusing `py_obj`'s
    existing conservative-regridding pipeline), then cropped by row/col to
    the bounding box of `conus_mask_east`'s own kept (eastern 2/3 CONUS)
    region (~26 MB -- the eastern 2/3 of CONUS is inherently a large area;
    this is not further compressed beyond the crop itself).
  - `mrms_objects/obj_obs_20230513_000015.nc` -- the real, already-
    identified MRMS object file for this same time, cropped with the
    identical row/col window (~2.9 MB). All 61 real objects (55 cellular,
    4 mixed, 2 linear -- every storm mode genuinely present) survive the
    crop.
- **`wofs_wofscast_case/`** (~880 KB) -- 4 real match files: 2 real
  forecast cases (2026-05-08 00Z and 01Z), each for a real 18-member WoFS
  ensemble and a real 18-member WoFSCast ensemble, both matched against
  MRMS truth. Every file was confirmed to contain all 5 real match
  categories (hit/miss/false_alarm/truth_extra/forecast_extra) before
  bundling -- copied as-is, no trimming needed (already small).

## Regenerating

`build_mrms_case.py` is the exact script that produced `mrms_case/` from
this developer's own local `test_mrms/`/`ncar_mpas_analysis/` directories
(neither bundled here). It is not run automatically by anything else --
it's a provenance record, kept in case the sample ever needs to be
regenerated or extended (e.g. a different case, a different crop). Requires
those larger source directories to exist locally; not runnable from a
fresh clone that only has this trimmed output. `wofs_wofscast_case/` has no
such script -- its 4 files were copied directly from a local `sfe_poster/`
archive with no transformation.

## What's NOT bundled here

Every other example config under `configs/` points at larger, non-bundled
external datasets (`ncar_mpas_analysis/`, `sfe_poster/`) -- "bring your own
larger dataset" reference examples, not meant to run out of a fresh clone.
See the top-level README for details.
