"""Absolute-datetime time selection + MRMS file discovery for the standalone
composite-reflectivity spatial map (see `plots/reflectivity_map.py`,
`drivers/plot_reflectivity_maps.py`). Mirrors `object_selection.py`'s own
SHAPE (pure resolver dataclass + small I/O helpers in one file), but is
genuinely a different selection semantic, not an extension of it: MRMS
observations have no init/lead-time concept at all, so selection here is
expressed as ABSOLUTE datetimes (a single time, an inclusive [start, end]
range, or every discovered time), never lead-minutes-since-init.
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime

from py_obj.regrid import discover_mrms_files

_VALID_ABS_TIME_MODES = ("single", "range", "full")


@dataclass
class AbsoluteTimeSelectionConfig:
    mode: str = "full"  # "single" | "range" | "full"
    valid_time: datetime | None = None  # required iff mode == "single"
    valid_time_range: tuple | None = None  # (start, end), required iff mode == "range"
    tolerance_minutes: float = 10.0  # nearest-match window, mode == "single" only


def resolve_absolute_time_selection(cfg: AbsoluteTimeSelectionConfig, available_times: list) -> list:
    """"full": every available time, sorted. "range": times within
    [start, end] INCLUSIVE (a range's own bounds are its own tolerance --
    no fuzzing needed). "single": the NEAREST available time to
    `cfg.valid_time` within `tolerance_minutes` -- real MRMS timestamps are
    irregularly jittered (e.g. `010041`, `020038`), so exact equality would
    be unusable; same nearest-with-tolerance idea as
    `object_selection.find_nearest_mrms_object_file`, just against a plain
    datetime list instead of files."""
    if cfg.mode == "full":
        return sorted(available_times)
    if cfg.mode == "range":
        if cfg.valid_time_range is None:
            raise ValueError("resolve_absolute_time_selection: mode='range' requires valid_time_range")
        lo, hi = cfg.valid_time_range
        result = sorted(t for t in available_times if lo <= t <= hi)
        if not result:
            raise ValueError(f"resolve_absolute_time_selection: no available times within valid_time_range={cfg.valid_time_range}")
        return result
    if cfg.mode == "single":
        if cfg.valid_time is None:
            raise ValueError("resolve_absolute_time_selection: mode='single' requires valid_time")
        if not available_times:
            raise ValueError("resolve_absolute_time_selection: mode='single' requires at least one available time")
        best = min(available_times, key=lambda t: abs((t - cfg.valid_time).total_seconds()))
        diff_minutes = abs((best - cfg.valid_time).total_seconds()) / 60.0
        if diff_minutes > cfg.tolerance_minutes:
            raise ValueError(
                f"resolve_absolute_time_selection: no available time within {cfg.tolerance_minutes} min "
                f"of {cfg.valid_time} (nearest was {best}, {diff_minutes:.1f} min away)"
            )
        return [best]
    raise ValueError(f"resolve_absolute_time_selection: mode={cfg.mode!r} is not one of {_VALID_ABS_TIME_MODES}")


@dataclass
class DiscoveredMrmsFile:
    filepath: str
    valid_time: datetime  # parsed from the filename ONLY -- for selection purposes; the
                           # actual load re-derives the authoritative valid_time from the file itself


_TIMESTAMP_RE = re.compile(r"(\d{8})[-_](\d{6})")  # handles both native MRMS's dash and interpolated MRMS's underscore


def discover_mrms_files_with_time(input_dir: str, pattern: str) -> list:
    """Wraps `py_obj.regrid.discover_mrms_files` (already handles the
    `YYYYMMDD/`-per-day-subdirectory layout for both native and interpolated
    MRMS) and attaches a filename-parsed valid_time to each result, WITHOUT
    opening/decompressing any file. A filename that doesn't match
    `_TIMESTAMP_RE` is silently skipped (not raised on) -- `discover_mrms_files`'s
    own glob can pick up unrelated non-data files in the same directory,
    matching this project's established "positive inclusion, skip what
    doesn't parse" convention rather than failing the whole discovery pass."""
    files = discover_mrms_files(input_dir, pattern)
    result = []
    for f in files:
        m = _TIMESTAMP_RE.search(os.path.basename(f))
        if m is None:
            continue
        valid_time = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        result.append(DiscoveredMrmsFile(filepath=f, valid_time=valid_time))
    return result
