"""Config-driven driver: standalone composite-reflectivity spatial maps for
a single time or a range of times, from any of 3 sources (forecast model
output, native MRMS observations, already-interpolated MRMS observations)
-- see config.ReflectivityMapConfig, plots/reflectivity_map.py. Independent
of the object-identification pipeline entirely.

Run with:
  /opt/anaconda3/envs/pysteps_env/bin/python py_obj_plotting/drivers/plot_reflectivity_maps.py <config.yaml>
"""

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

from python_obj.obj_core import build_model_manifest
from python_obj.regrid import clip_near_zero_sentinel, load_mrms_grib2, load_mrms_netcdf
from py_obj_plotting.config import load_reflectivity_map_config
from py_obj_plotting.plots.reflectivity_map import ReflectivityFrame, plot_reflectivity_map
from py_obj_plotting.plots.spatial_map import crop_extent_from_mask
from py_obj_plotting.reflectivity_selection import discover_mrms_files_with_time, resolve_absolute_time_selection

_MRMS_DEFAULT_PATTERNS = {"mrms_native": "**/*.grib2*", "mrms_interp": "**/*.nc"}
_MRMS_DEFAULT_LABELS = {"mrms_native": "MRMS (native)", "mrms_interp": "MRMS (interpolated)"}


def _run_model_source(cfg) -> list:
    manifest, loader = build_model_manifest(
        cfg.input_dir, cfg.file_pattern, cfg.member_subdirs, cfg.stacked_members,
        cfg.var_name, cfg.lat_name, cfg.lon_name,
        init_attr=cfg.init_attr, lead_attr=cfg.lead_attr, lead_units=cfg.lead_units, init_format=cfg.init_format,
        valid_time_attr=cfg.valid_time_attr, valid_time_format=cfg.valid_time_format, valid_time_var=cfg.valid_time_var,
        member_subdir_pattern=cfg.member_subdir_pattern,
    )
    if cfg.member_id is not None:
        entries = [e for e in manifest if e.member_id == cfg.member_id]
        if not entries:
            available = sorted({e.member_id for e in manifest})
            raise ValueError(f"_run_model_source: member_id={cfg.member_id!r} not found in manifest (available: {available})")
    else:
        entries = manifest

    entry_by_time = {}
    for e in entries:
        entry_by_time.setdefault(e.valid_time, e)
    available_times = sorted(entry_by_time)
    selected_times = resolve_absolute_time_selection(cfg.times, available_times)
    print(f"  discovered {len(manifest)} manifest entries, {len(available_times)} available time(s) for member {cfg.member_id!r}, "
          f"{len(selected_times)} selected")

    written = []
    extent = cfg.extent
    member_suffix = cfg.member_id if cfg.member_id is not None else "det"
    default_label = f"Model ({cfg.member_id})" if cfg.member_id is not None else "Model"
    for vt in selected_times:
        entry = entry_by_time[vt]
        field = loader(entry.filepath, entry.extra_dim_index)
        if extent is None:
            extent = crop_extent_from_mask(field.lat2d, field.lon2d, np.ones_like(field.data, dtype=bool))
        frame = ReflectivityFrame(field.lat2d, field.lon2d, field.data, field.valid_time, field.missing_value,
                                   label=cfg.label or default_label)
        out_path = os.path.join(cfg.output_dir, f"refl_model_{member_suffix}_{field.valid_time:%Y%m%d_%H%M%S}.png")
        plot_reflectivity_map(frame, out_path, extent=extent, min_dbz=cfg.min_dbz)
        written.append(out_path)
        print(f"  wrote {out_path}")
    return written


def _run_mrms_source(cfg) -> list:
    pattern = cfg.mrms_pattern or _MRMS_DEFAULT_PATTERNS[cfg.source]
    discovered = discover_mrms_files_with_time(cfg.input_dir, pattern)
    if not discovered:
        raise FileNotFoundError(f"_run_mrms_source: no files matching '{pattern}' discovered under '{cfg.input_dir}'")
    by_time = {d.valid_time: d for d in discovered}
    available_times = sorted(by_time)
    selected_times = resolve_absolute_time_selection(cfg.times, available_times)
    print(f"  discovered {len(discovered)} file(s), {len(selected_times)} selected")

    written = []
    extent = cfg.extent
    default_label = _MRMS_DEFAULT_LABELS[cfg.source]
    for vt in selected_times:
        discovered_file = by_time[vt]
        if cfg.source == "mrms_native":
            field = load_mrms_grib2(discovered_file.filepath)
            field.data = clip_near_zero_sentinel(field.data)
        else:
            field = load_mrms_netcdf(discovered_file.filepath, varname=cfg.mrms_varname, lat_name=cfg.mrms_lat_name, lon_name=cfg.mrms_lon_name)

        if extent is None:
            extent = crop_extent_from_mask(field.lat2d, field.lon2d, np.ones_like(field.data, dtype=bool))
        frame = ReflectivityFrame(field.lat2d, field.lon2d, field.data, field.valid_time, field.missing_value,
                                   label=cfg.label or default_label)
        out_path = os.path.join(cfg.output_dir, f"refl_{cfg.source}_{field.valid_time:%Y%m%d_%H%M%S}.png")
        plot_reflectivity_map(frame, out_path, extent=extent, min_dbz=cfg.min_dbz)
        written.append(out_path)
        print(f"  wrote {out_path}")
    return written


def run_one_case(config_path: str, output_dir_override: str | None = None) -> list:
    """`output_dir_override`, if given, replaces the config's own
    output_dir -- lets a caller redirect output without editing the YAML."""
    cfg = load_reflectivity_map_config(config_path)
    if output_dir_override is not None:
        cfg.output_dir = output_dir_override
    print(f"Reflectivity map: source={cfg.source!r}, input_dir={cfg.input_dir}, times.mode={cfg.times.mode!r}")
    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.source == "model":
        return _run_model_source(cfg)
    return _run_mrms_source(cfg)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--output-dir")]
    output_dir_override = None
    for a in sys.argv[1:]:
        if a.startswith("--output-dir="):
            output_dir_override = a.split("=", 1)[1]
    if len(args) != 1:
        print("Usage: plot_reflectivity_maps.py <config.yaml> [--output-dir=<path>]")
        sys.exit(1)
    written = run_one_case(args[0], output_dir_override=output_dir_override)
    print(f"\nDone: wrote {len(written)} image(s).")


if __name__ == "__main__":
    main()
