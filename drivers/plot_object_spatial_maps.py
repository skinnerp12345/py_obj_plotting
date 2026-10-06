"""Config-driven driver: spatial object maps for a single forecast
initialization (see plots/object_spatial_map.py, config.ObjectSpatialMapConfig).
The user picks the init time + member/ensemble subset + a single/range/full
lead-time selection; the driver resolves which real (member, valid_time)
slices exist, reads them, and renders one image per selection (or one
paintball image per selected valid time, combining every selected member).

Run with:
  /opt/anaconda3/envs/pysteps_env/bin/python py_obj_plotting/drivers/plot_object_spatial_maps.py <config.yaml>
"""

import os
import sys

import netCDF4
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

from py_obj.obj_core import conus_mask_east, iter_object_slices, iter_object_slices_lazy, read_match_file, read_object_file
from py_obj_plotting.config import load_object_spatial_map_config
from py_obj_plotting.object_selection import (
    discover_object_file_members_and_times,
    find_forecast_file_for_init,
    find_nearest_mrms_object_file,
    resolve_member_selection,
    resolve_time_selection,
)
from py_obj_plotting.plots.object_spatial_map import ObjectSlice, plot_object_spatial_map, plot_paintball_map
from py_obj_plotting.plots.spatial_map import crop_extent_from_mask


def _read_lat_lon(path: str) -> tuple:
    with netCDF4.Dataset(path, "r") as ds:
        return np.asarray(ds.variables["lat"][:]), np.asarray(ds.variables["lon"][:])


def _match_records_for_slice(match_contents, member_id, valid_time) -> list:
    mask = np.ones(len(match_contents.records), dtype=bool)
    if match_contents.member_ids is not None and match_contents.member_index is not None:
        if member_id not in match_contents.member_ids:
            raise ValueError(f"_match_records_for_slice: member {member_id!r} not found in match file's own member_ids {match_contents.member_ids}")
        mi = match_contents.member_ids.index(member_id)
        mask &= (match_contents.member_index == mi)
    if match_contents.valid_times is not None and match_contents.time_index is not None:
        if valid_time not in match_contents.valid_times:
            raise ValueError(f"_match_records_for_slice: valid_time {valid_time} not found in match file's own valid_times")
        ti = match_contents.valid_times.index(valid_time)
        mask &= (match_contents.time_index == ti)
    return [r for r, m in zip(match_contents.records, mask) if m]


def _truth_slice_for_valid_time(mrms_object_dir: str, valid_time) -> ObjectSlice:
    mrms_path = find_nearest_mrms_object_file(mrms_object_dir, valid_time)
    contents = read_object_file(mrms_path)
    _, truth_valid_time, labels2d, objects = next(iter(iter_object_slices(contents)))
    return ObjectSlice(contents.lat2d, contents.lon2d, labels2d, objects, member_id=None, valid_time=truth_valid_time)


def run_one_case(config_path: str, output_dir_override: str | None = None) -> list:
    """`output_dir_override`, if given, replaces the config's own
    output_dir -- lets a caller redirect output without editing the YAML."""
    cfg = load_object_spatial_map_config(config_path)
    if output_dir_override is not None:
        cfg.output_dir = output_dir_override
    print(f"Object spatial map: forecast_object_dir={cfg.forecast_object_dir}, init_time={cfg.init_time}, "
          f"color_by={cfg.color_by!r}, paintball={cfg.paintball}")

    forecast_path = find_forecast_file_for_init(cfg.forecast_object_dir, cfg.init_time)
    print(f"  forecast file: {forecast_path}")
    member_ids, valid_times, _ = discover_object_file_members_and_times(forecast_path)

    selected_members = resolve_member_selection(cfg.members, member_ids)
    selected_times = resolve_time_selection(cfg.times, valid_times, cfg.init_time)
    print(f"  selected members: {selected_members}")
    print(f"  selected times: {len(selected_times)} of {len(valid_times)} available")

    lat2d, lon2d = _read_lat_lon(forecast_path)
    keep_mask = ~conus_mask_east(lat2d, lon2d)
    extent = crop_extent_from_mask(lat2d, lon2d, keep_mask)

    os.makedirs(cfg.output_dir, exist_ok=True)
    written = []

    selected_members_set = set(selected_members)
    selected_times_set = set(selected_times)
    needs_truth = cfg.mrms_object_dir is not None and (cfg.show_mrms_overlay or cfg.show_truth_hatch or cfg.color_by == "category")

    alpha_kwargs = {}
    if cfg.alpha is not None:
        alpha_kwargs["alpha"] = cfg.alpha
    if cfg.truth_alpha is not None:
        alpha_kwargs["truth_alpha"] = cfg.truth_alpha

    match_contents = None
    if cfg.match_dir is not None:
        match_path = find_forecast_file_for_init(cfg.match_dir, cfg.init_time, pattern=cfg.match_pattern)
        match_contents = read_match_file(match_path)
        print(f"  match file: {match_path} ({len(match_contents.records)} records)")

    if cfg.paintball:
        slices_by_time = {t: {} for t in selected_times}
        for member_id, valid_time, labels2d, objects in iter_object_slices_lazy(forecast_path):
            if member_id in selected_members_set and valid_time in selected_times_set:
                slices_by_time[valid_time][member_id] = ObjectSlice(lat2d, lon2d, labels2d, objects, member_id, valid_time)

        truth_slice_cache: dict = {}

        for valid_time, member_slices_by_id in slices_by_time.items():
            if not member_slices_by_id:
                print(f"  WARNING: no member data at valid_time={valid_time}, skipping paintball image")
                continue
            member_slices = [member_slices_by_id[m] for m in selected_members if m in member_slices_by_id]

            match_records_by_member = None
            if cfg.color_by == "category":
                match_records_by_member = {m: _match_records_for_slice(match_contents, m, valid_time) for m in member_slices_by_id}

            truth_slice = None
            if needs_truth:
                if valid_time not in truth_slice_cache:
                    truth_slice_cache[valid_time] = _truth_slice_for_valid_time(cfg.mrms_object_dir, valid_time)
                truth_slice = truth_slice_cache[valid_time]

            out_path = os.path.join(cfg.output_dir, f"paintball_{cfg.color_by}_{valid_time:%Y%m%d_%H%M%S}.png")
            plot_paintball_map(member_slices, out_path, extent=extent, color_by=cfg.color_by,
                                match_records_by_member=match_records_by_member, truth=truth_slice, **alpha_kwargs)
            written.append(out_path)
            print(f"  wrote {out_path} ({len(member_slices)} members)")
        return written

    # Cache truth slices by valid_time -- a real, non-paintball run with an
    # ensemble selects many members at the SAME set of valid times, and the
    # MRMS truth data doesn't depend on which member is being rendered.
    # Without this, an N-member run would needlessly re-read the same MRMS
    # file N times per valid time.
    truth_slice_cache: dict = {}

    for member_id, valid_time, labels2d, objects in iter_object_slices_lazy(forecast_path):
        if member_id not in selected_members_set or valid_time not in selected_times_set:
            continue
        forecast_slice = ObjectSlice(lat2d, lon2d, labels2d, objects, member_id, valid_time)

        truth_slice = None
        if needs_truth:
            if valid_time not in truth_slice_cache:
                truth_slice_cache[valid_time] = _truth_slice_for_valid_time(cfg.mrms_object_dir, valid_time)
            truth_slice = truth_slice_cache[valid_time]
        match_records = _match_records_for_slice(match_contents, member_id, valid_time) if cfg.color_by == "category" else None

        member_label = member_id if member_id is not None else "det"
        out_path = os.path.join(cfg.output_dir, f"{cfg.color_by}_{member_label}_{valid_time:%Y%m%d_%H%M%S}.png")
        plot_object_spatial_map(
            forecast_slice, out_path, extent=extent, color_by=cfg.color_by,
            truth=truth_slice, show_forecast_hatch=cfg.show_forecast_hatch, show_truth_hatch=cfg.show_truth_hatch,
            match_records=match_records, **alpha_kwargs,
        )
        written.append(out_path)
        print(f"  wrote {out_path}")

    return written


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--output-dir")]
    output_dir_override = None
    for a in sys.argv[1:]:
        if a.startswith("--output-dir="):
            output_dir_override = a.split("=", 1)[1]
    if len(args) != 1:
        print("Usage: plot_object_spatial_maps.py <config.yaml> [--output-dir=<path>]")
        sys.exit(1)
    written = run_one_case(args[0], output_dir_override=output_dir_override)
    print(f"\nDone: wrote {len(written)} image(s).")


if __name__ == "__main__":
    main()
