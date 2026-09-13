"""Standalone script: MRMS object footprint + centroid maps, "WoFS style"
(state boundaries only, no county/road layer), one PNG per available hourly
timestep for a single day.

Objects shaded by is_linear category (0=cellular=orange, 1=possibly
linear=purple, 2=linear=blue); each object's centroid marked with a small
black circle. Reads py_obj's own object-file schema unchanged via
py_obj.obj_core.read_object_file.

Plain arguments, not a YAML config -- this is one specific analysis; see
py_obj_plotting/README.md.

Run with:
  /opt/anaconda3/envs/pysteps_env/bin/python py_obj_plotting/drivers/plot_mrms_objects.py <YYYYMMDD> [mrms_objects_dir] [out_dir]

Defaults to the real bundled ncar_mpas_analysis/mrms_objects/ directory if
mrms_objects_dir is omitted.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

from py_obj_plotting.plots import plot_mrms_objects_for_day


def main():
    if len(sys.argv) < 2:
        raise ValueError("Usage: plot_mrms_objects.py <YYYYMMDD> [mrms_objects_dir] [out_dir]")
    date_str = sys.argv[1]
    mrms_objects_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_REPO_ROOT, "ncar_mpas_analysis", "mrms_objects")
    out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(_THIS_DIR, "..", "output", "mrms_objects")

    out_paths = plot_mrms_objects_for_day(mrms_objects_dir, date_str, out_dir)
    print(f"\nDone: wrote {len(out_paths)} images for {date_str}.")


if __name__ == "__main__":
    main()
