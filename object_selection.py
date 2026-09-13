"""Member/time selection logic for a single forecast initialization's
object spatial maps (see `plots/object_spatial_map.py`), plus small I/O
helpers for discovering the right files. Pure resolution logic
(`resolve_member_selection`/`resolve_time_selection`) is kept separate from
file I/O so it's independently unit-testable without any real data.
"""

import glob
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import netCDF4

from py_obj.obj_core import read_init_time_only
from py_obj.obj_core.object_io import _read_member_ids, _read_valid_times

_VALID_MEMBER_MODES = ("single", "list", "all")
_VALID_TIME_MODES = ("single", "range", "full")


@dataclass
class MemberSelectionConfig:
    mode: str = "all"  # "single" | "list" | "all"
    members: list | None = None  # required iff mode in ("single", "list")


@dataclass
class TimeSelectionConfig:
    mode: str = "full"  # "single" | "range" | "full"
    lead_minutes: float | None = None  # required iff mode == "single"
    lead_minutes_range: tuple | None = None  # required iff mode == "range"


def resolve_member_selection(cfg: MemberSelectionConfig, available_members: list | None) -> list:
    """`available_members`: None for a deterministic (no member dimension)
    file -- resolves to `[None]` regardless of `cfg.mode` (matches
    `systems.py`'s own "member_id=None as one implicit member" convention),
    since there is nothing to select among."""
    if available_members is None:
        return [None]
    if cfg.mode == "all":
        return list(available_members)
    if cfg.mode == "single":
        if not cfg.members or len(cfg.members) != 1:
            raise ValueError("resolve_member_selection: mode='single' requires exactly one entry in members")
        member = cfg.members[0]
        if member not in available_members:
            raise ValueError(f"resolve_member_selection: member {member!r} not found in available members {available_members}")
        return [member]
    if cfg.mode == "list":
        if not cfg.members:
            raise ValueError("resolve_member_selection: mode='list' requires a non-empty members list")
        missing = [m for m in cfg.members if m not in available_members]
        if missing:
            raise ValueError(f"resolve_member_selection: member(s) {missing} not found in available members {available_members}")
        return list(cfg.members)
    raise ValueError(f"resolve_member_selection: mode={cfg.mode!r} is not one of {_VALID_MEMBER_MODES}")


def resolve_time_selection(cfg: TimeSelectionConfig, available_times: list, init_time: datetime) -> list:
    """Lead time expressed in MINUTES since `init_time` -- reuses the
    project-wide minutes-since-init convention (`systems.load_system()`'s
    own `lead_min_by_ti` calculation)."""
    if cfg.mode == "full":
        return list(available_times)
    if cfg.mode == "single":
        if cfg.lead_minutes is None:
            raise ValueError("resolve_time_selection: mode='single' requires lead_minutes")
        target = init_time + timedelta(minutes=cfg.lead_minutes)
        matches = [t for t in available_times if t == target]
        if not matches:
            available_lead = [round((t - init_time).total_seconds() / 60.0, 2) for t in available_times]
            raise ValueError(
                f"resolve_time_selection: no available time at lead_minutes={cfg.lead_minutes} -- "
                f"available lead minutes: {available_lead}"
            )
        return matches
    if cfg.mode == "range":
        if cfg.lead_minutes_range is None:
            raise ValueError("resolve_time_selection: mode='range' requires lead_minutes_range")
        lo, hi = cfg.lead_minutes_range
        result = [t for t in available_times if lo <= (t - init_time).total_seconds() / 60.0 <= hi]
        if not result:
            raise ValueError(f"resolve_time_selection: no available times within lead_minutes_range={cfg.lead_minutes_range}")
        return result
    raise ValueError(f"resolve_time_selection: mode={cfg.mode!r} is not one of {_VALID_TIME_MODES}")


def discover_object_file_members_and_times(path: str) -> tuple:
    """member_ids (or None), valid_times, init_time (or None) -- metadata
    only, never touches the (possibly multi-GB) `labels` variable. Reuses
    `object_io.py`'s own private member/valid-time readers directly (same
    cross-module private-helper-reuse convention `systems.py` already
    establishes for `aggregate_matches_by_lead_hour._discover`) plus the
    existing public `read_init_time_only`."""
    with netCDF4.Dataset(path, "r") as ds:
        member_ids = _read_member_ids(ds)
        valid_times = _read_valid_times(ds)
    init_time = read_init_time_only(path)
    return member_ids, valid_times, init_time


_TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})")


def find_forecast_file_for_init(forecast_object_dir: str, init_time: datetime, pattern: str = "*.nc") -> str:
    """Globs `forecast_object_dir`, uses `read_init_time_only` per
    candidate to find the one file matching `init_time`. Falls back to
    parsing the init time from the filename (`_TIMESTAMP_RE`, the same
    `<YYYYMMDD>_<HHMMSS>` convention `obj_init_*.nc`/`match_init_*.nc` both
    use) when the attribute is absent -- a real, confirmed gap: the
    `ncar_mpas_analysis/matched_objects_*km/` match files predate the
    stored `init_time` global attribute entirely (same backward-compat
    situation already handled elsewhere in this project, e.g.
    `aggregate_matches_by_lead_hour.py`'s own `_parse_init_time_from_filename`)."""
    candidates = sorted(glob.glob(os.path.join(forecast_object_dir, pattern)))
    if not candidates:
        raise FileNotFoundError(f"find_forecast_file_for_init: no files matching '{pattern}' under '{forecast_object_dir}'")
    for path in candidates:
        file_init_time = read_init_time_only(path)
        if file_init_time is None:
            m = _TIMESTAMP_RE.search(os.path.basename(path))
            if m is not None:
                file_init_time = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        if file_init_time == init_time:
            return path
    raise FileNotFoundError(f"find_forecast_file_for_init: no file under '{forecast_object_dir}' has init_time={init_time}")


def find_nearest_mrms_object_file(
    mrms_object_dir: str, valid_time: datetime, tolerance_minutes: float = 20.0, pattern: str = "obj_obs_*.nc",
) -> str:
    """Nearest-timestamp MRMS object file lookup, promoted into one shared
    location -- the identical regex/tolerance-window logic was previously
    duplicated verbatim in `python_test_plot/plot_mrms_vs_mem1_objects.py`
    and `python_test_plot/plot_match_categories.py`."""
    candidates = glob.glob(os.path.join(mrms_object_dir, pattern))
    best_file, best_diff = None, None
    for f in candidates:
        m = _TIMESTAMP_RE.search(os.path.basename(f))
        if m is None:
            continue
        file_time = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        diff = abs((file_time - valid_time).total_seconds())
        if best_diff is None or diff < best_diff:
            best_file, best_diff = f, diff
    if best_file is None or best_diff > tolerance_minutes * 60:
        raise FileNotFoundError(f"find_nearest_mrms_object_file: no MRMS object file within {tolerance_minutes} min of {valid_time}")
    return best_file
