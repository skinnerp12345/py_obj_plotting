# py_obj_plotting notebooks

Two runnable tutorials, each against small, real data bundled in
`py_obj_plotting/sample_data/` -- no external dataset needed (see
`sample_data/README.md` for provenance):

- **`mrms_reflectivity_and_objects.ipynb`** -- native-resolution MRMS
  reflectivity, the same field interpolated onto a real MPAS grid and
  restricted to the eastern 2/3 of CONUS, and MRMS storm objects colored
  by storm mode (cellular/mixed/linear), all for a single, objectively
  chosen high-storm-coverage case (2023-05-13 00:00:15 UTC). Uses
  `configs/config_sample_mrms_reflectivity_native.yaml`/
  `configs/config_sample_mrms_reflectivity_interp.yaml`.
- **`wofs_wofscast_verification_suite.ipynb`** -- the full matching-output
  verification suite (POD/FAR/FBIAS/CSI time series, matched-pair
  diagnostic distributions, performance diagrams, object-property POD
  heatmaps) against a small, real 2-case slice of WoFS and WoFSCast
  forecasts matched to MRMS truth. Uses
  `configs/config_sample_wofs_wofscast_suite.yaml`.

## Prerequisites

- The `pysteps_env` conda environment (`/opt/anaconda3/envs/pysteps_env/bin/python`),
  which has `jupyter`/`nbformat` plus every dependency this package or
  `python_obj` needs.
- No internet access needed -- both notebooks run entirely against bundled
  sample data.

## Running them

```
/opt/anaconda3/envs/pysteps_env/bin/jupyter notebook mrms_reflectivity_and_objects.ipynb
/opt/anaconda3/envs/pysteps_env/bin/jupyter notebook wofs_wofscast_verification_suite.ipynb
```

or, to execute either non-interactively end to end:

```
/opt/anaconda3/envs/pysteps_env/bin/jupyter-nbconvert --to notebook --execute --inplace mrms_reflectivity_and_objects.ipynb
```

(Use the `jupyter-nbconvert` binary directly, not `python -m jupyter
nbconvert` -- on a machine with more than one Jupyter installation, `-m
jupyter` can resolve to a different environment's nbconvert than the one
running the command.)

All outputs are written under `py_obj_plotting/configs/output/`, per the
paths configured in each notebook's sample config. To adapt either
notebook to different data, copy it and change the `CONFIG_PATH` variable
in its Setup cell to point at a different config file -- see
`../CONFIG_REFERENCE.md` for the full config-field reference and
`../README.md` for other example configs.
