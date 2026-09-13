"""Core (non-plotting) aggregation logic: pools per-(member, lead_hour)
contingency-table counts (hits, misses, false_alarms, truth_extra,
forecast_extra) from many match_init_<init_time>.nc files (file_grouping=
"init_snapshot" match files, one per forecast case -- every member x every
lead time already matched against MRMS truth).

This needs no MRMS/object files and no time-tolerance matching at all --
matching already happened upstream (py_obj.obj_core.run_matching_series),
so a match file's own category column is already the ground truth for
hit/miss/false_alarm/truth_extra/forecast_extra. This module only pools
those counts across cases.

Reads only the metadata a match file needs for pooling (category counts and
per-record float diagnostic fields), never a full read_object_file()-style
read of unrelated arrays.
"""

import glob
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import netCDF4
import numpy as np

# Fallback source of a match file's init_time when the file itself predates
# the init_time global attribute (real situation: the 118 real
# ncar_mpas_analysis/matched_objects_*km/ files were generated before this
# attribute existed and will not be regenerated) -- parse it from the
# established match_init_<YYYYMMDD>_<HHMMSS>.nc filename convention instead.
# Mirrors py_obj/drivers/run_matching_per_case.py's own
# _parse_filename_timestamp() pattern exactly.
_FILENAME_TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})")


def _parse_init_time_from_filename(path: str) -> datetime | None:
    m = _FILENAME_TIMESTAMP_RE.search(os.path.basename(path))
    if m is None:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def _seconds_to_dt(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)


@dataclass
class MatchLeadHourClimatology:
    lead_hours: list  # sorted unique lead hours found across all match files
    members: list  # sorted unique member ids found across all match files
    n_hits: dict = field(default_factory=dict)  # (member, lead_hour) -> int
    n_misses: dict = field(default_factory=dict)
    n_false_alarms: dict = field(default_factory=dict)
    n_truth_extra: dict = field(default_factory=dict)
    n_forecast_extra: dict = field(default_factory=dict)
    n_cases_per_lead_hour: dict = field(default_factory=dict)  # lead_hour -> int
    n_files_used_stored_init_time: int = 0  # diagnostic: how many files had a real init_time attribute
    n_files_used_filename_init_time: int = 0  # diagnostic: how many fell back to filename parsing


@dataclass
class _MatchMetadata:
    member_ids: list | None
    valid_times: list
    init_time: datetime | None
    init_time_from_filename: bool
    category: np.ndarray
    member_index: np.ndarray | None
    time_index: np.ndarray | None


def _read_match_metadata(path: str) -> _MatchMetadata:
    """Metadata-only match-file read -- category/member_index/time_index/
    member_id/valid_time only, never the float diagnostic fields. init_time
    prefers the file's own global attribute (added after this project's
    first 118 real match files were generated); falls back to parsing it
    from the filename when absent -- see module docstring."""
    with netCDF4.Dataset(path, "r") as ds:
        if "member_id" in ds.variables:
            member_ids = list(ds.variables["member_id"][:])
        elif hasattr(ds, "member_id"):
            member_ids = [ds.member_id]
        else:
            member_ids = None

        if "valid_time" in ds.variables:
            valid_times = [_seconds_to_dt(t) for t in ds.variables["valid_time"][:]]
        else:
            valid_times = [_seconds_to_dt(ds.valid_time)]

        init_time_from_filename = False
        if hasattr(ds, "init_time"):
            init_time = datetime.fromisoformat(ds.init_time)
        else:
            init_time = _parse_init_time_from_filename(path)
            init_time_from_filename = init_time is not None

        category = np.asarray(ds.variables["category"][:], dtype=str)
        member_index = np.asarray(ds.variables["match_member_index"][:]) if "match_member_index" in ds.variables else None
        time_index = np.asarray(ds.variables["match_time_index"][:]) if "match_time_index" in ds.variables else None

    return _MatchMetadata(member_ids, valid_times, init_time, init_time_from_filename, category, member_index, time_index)


def _discover(dir_path: str, pattern: str) -> list:
    files = sorted(glob.glob(os.path.join(dir_path, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found under '{dir_path}'")
    return files


def build_match_lead_hour_climatology(
    match_dir: str,
    match_pattern: str = "match_init_*.nc",
) -> MatchLeadHourClimatology:
    """match_dir: one match_init_*.nc per forecast case (file_grouping=
    "init_snapshot" -- every member x every lead time for that case, already
    matched against truth)."""
    match_files = _discover(match_dir, match_pattern)

    n_hits: dict = {}
    n_misses: dict = {}
    n_false_alarms: dict = {}
    n_truth_extra: dict = {}
    n_forecast_extra: dict = {}
    n_cases_per_lead_hour: dict = {}
    n_files_used_stored_init_time = 0
    n_files_used_filename_init_time = 0
    members_seen: set = set()
    lead_hours_seen: set = set()

    _count_dicts = {
        "hit": n_hits, "miss": n_misses, "false_alarm": n_false_alarms,
        "truth_extra": n_truth_extra, "forecast_extra": n_forecast_extra,
    }

    for match_file in match_files:
        meta = _read_match_metadata(match_file)
        if meta.init_time is None:
            raise ValueError(f"'{match_file}' has no init_time (neither a stored attribute nor a parseable filename) -- expected an init_snapshot match file")
        if meta.init_time_from_filename:
            n_files_used_filename_init_time += 1
        else:
            n_files_used_stored_init_time += 1
        if meta.member_ids is None or meta.member_index is None or meta.time_index is None:
            raise ValueError(
                f"'{match_file}' is missing member_id/match_member_index/match_time_index -- "
                "expected a full ensemble/lead-time (init_snapshot) match file"
            )

        lead_hour_by_time_idx = {}
        for ti, vt in enumerate(meta.valid_times):
            lead_hour = (vt - meta.init_time).total_seconds() / 3600.0
            lead_hour_by_time_idx[ti] = lead_hour
            lead_hours_seen.add(lead_hour)

        # one case contributes once per lead_hour, not once per (member, lead_hour)
        for lead_hour in lead_hour_by_time_idx.values():
            n_cases_per_lead_hour[lead_hour] = n_cases_per_lead_hour.get(lead_hour, 0) + 1

        for mi, member_id in enumerate(meta.member_ids):
            members_seen.add(member_id)
            member_mask = meta.member_index == mi
            for ti, lead_hour in lead_hour_by_time_idx.items():
                mask = member_mask & (meta.time_index == ti)
                key = (member_id, lead_hour)
                categories_here = meta.category[mask]
                for cat, target in _count_dicts.items():
                    n = int(np.sum(categories_here == cat))
                    target[key] = target.get(key, 0) + n

    return MatchLeadHourClimatology(
        lead_hours=sorted(lead_hours_seen),
        members=sorted(members_seen),
        n_hits=n_hits,
        n_misses=n_misses,
        n_false_alarms=n_false_alarms,
        n_truth_extra=n_truth_extra,
        n_forecast_extra=n_forecast_extra,
        n_cases_per_lead_hour=n_cases_per_lead_hour,
        n_files_used_stored_init_time=n_files_used_stored_init_time,
        n_files_used_filename_init_time=n_files_used_filename_init_time,
    )


# Storm-mode-filtered pooling: the SAME contingency-count pooling as
# build_match_lead_hour_climatology() above, restricted to records of one
# storm mode (0=cellular, 1=mixed, 2=linear -- see
# py_obj/obj_core/matching.py's is_linear convention), so the
# resulting MatchLeadHourClimatology flows completely unchanged through
# the existing compute_verification_stats() -> comparison-plot pipeline.
# truth_is_linear/forecast_is_linear are never read by the plain
# _read_match_metadata() above (deliberately -- that reader stays minimal
# for the common pooled-POD/FAR/CSI path); a separate reader here reads
# them ADDITIONALLY, not by bloating the shared one.

_TRUTH_KEYED_CATEGORIES = ("hit", "miss", "truth_extra")  # rows with a real truth object
_FORECAST_KEYED_CATEGORIES = ("false_alarm", "forecast_extra")  # rows with NO truth object at all


@dataclass
class _StormModeMatchMetadata:
    member_ids: list | None
    valid_times: list
    init_time: datetime | None
    init_time_from_filename: bool
    category: np.ndarray
    truth_is_linear: np.ndarray  # int; -1 sentinel where absent/not applicable
    forecast_is_linear: np.ndarray
    member_index: np.ndarray | None
    time_index: np.ndarray | None


def _read_match_metadata_with_storm_mode(path: str) -> _StormModeMatchMetadata:
    with netCDF4.Dataset(path, "r") as ds:
        if "member_id" in ds.variables:
            member_ids = list(ds.variables["member_id"][:])
        elif hasattr(ds, "member_id"):
            member_ids = [ds.member_id]
        else:
            member_ids = None

        if "valid_time" in ds.variables:
            valid_times = [_seconds_to_dt(t) for t in ds.variables["valid_time"][:]]
        else:
            valid_times = [_seconds_to_dt(ds.valid_time)]

        init_time_from_filename = False
        if hasattr(ds, "init_time"):
            init_time = datetime.fromisoformat(ds.init_time)
        else:
            init_time = _parse_init_time_from_filename(path)
            init_time_from_filename = init_time is not None

        category = np.asarray(ds.variables["category"][:], dtype=str)
        member_index = np.asarray(ds.variables["match_member_index"][:]) if "match_member_index" in ds.variables else None
        time_index = np.asarray(ds.variables["match_time_index"][:]) if "match_time_index" in ds.variables else None

        def _i(name):
            # Backward compatibility: same hasattr-gated fallback every
            # other optional field reader in this project already uses --
            # a file predating storm_mode_classification simply has no
            # is_linear data, so every record is excluded from every mode.
            if name not in ds.variables:
                return np.full(len(category), -1, dtype=int)
            return np.asarray(ds.variables[name][:], dtype=int)

        truth_is_linear = _i("truth_is_linear")
        forecast_is_linear = _i("forecast_is_linear")

    return _StormModeMatchMetadata(
        member_ids, valid_times, init_time, init_time_from_filename,
        category, truth_is_linear, forecast_is_linear, member_index, time_index,
    )


def _effective_storm_mode(meta: _StormModeMatchMetadata) -> np.ndarray:
    """One storm-mode value per record: truth_is_linear for hit/miss/
    truth_extra rows (a real truth object exists); forecast_is_linear for
    false_alarm/forecast_extra rows (NO truth object at all on those rows,
    so truth_is_linear is always the -1 sentinel there -- a real,
    necessary asymmetry, not an oversight)."""
    truth_keyed = np.isin(meta.category, _TRUTH_KEYED_CATEGORIES)
    return np.where(truth_keyed, meta.truth_is_linear, meta.forecast_is_linear)


def build_match_lead_hour_climatology_by_storm_mode(
    match_dir: str, storm_mode: int, match_pattern: str = "match_init_*.nc",
) -> MatchLeadHourClimatology:
    """Same pooling restricted to records whose EFFECTIVE storm mode (see
    _effective_storm_mode()) equals `storm_mode` (0=cellular, 1=mixed,
    2=linear). Unlike build_match_lead_hour_climatology() (which strictly
    requires a real ensemble/init_snapshot shape), this function ALSO
    accepts a deterministic (no member dimension) match file -- pooled
    under a single member=None key, matching systems.py's own
    load_system() tolerance -- since this feeds
    drivers.plot_matching_timeseries, which must support both shapes.
    Requires storm_mode_classification to have been enabled at
    identification time -- a match file without it has
    truth_is_linear/forecast_is_linear entirely absent (read as the -1
    sentinel here), so every record is excluded from every storm mode,
    producing an all-zero-count climatology rather than raising (matches
    this project's established convention of degrading gracefully on an
    absent optional field) -- callers should treat an unexpectedly
    all-zero result as a signal to check whether storm_mode_classification
    was actually enabled for this data, not assume it means "no objects
    of this type occurred."
    """
    match_files = _discover(match_dir, match_pattern)

    n_hits: dict = {}
    n_misses: dict = {}
    n_false_alarms: dict = {}
    n_truth_extra: dict = {}
    n_forecast_extra: dict = {}
    n_cases_per_lead_hour: dict = {}
    n_files_used_stored_init_time = 0
    n_files_used_filename_init_time = 0
    members_seen: set = set()
    lead_hours_seen: set = set()

    _count_dicts = {
        "hit": n_hits, "miss": n_misses, "false_alarm": n_false_alarms,
        "truth_extra": n_truth_extra, "forecast_extra": n_forecast_extra,
    }

    for match_file in match_files:
        meta = _read_match_metadata_with_storm_mode(match_file)
        if meta.init_time is None:
            raise ValueError(f"'{match_file}' has no init_time (neither a stored attribute nor a parseable filename) -- expected an init_snapshot match file")
        if meta.init_time_from_filename:
            n_files_used_filename_init_time += 1
        else:
            n_files_used_stored_init_time += 1

        # Unlike build_match_lead_hour_climatology (which strictly requires
        # a real ensemble/init_snapshot shape), this function ALSO tolerates
        # a deterministic (no member dimension) match file -- matching
        # systems.py's own load_system() convention -- since the driver
        # feeding this needs to support both shapes, same as the rest of
        # plot_matching_timeseries.py's pooled comparison path already does.
        has_members = meta.member_ids is not None and meta.member_index is not None

        mode_mask = _effective_storm_mode(meta) == storm_mode

        lead_hour_by_time_idx = {}
        for ti, vt in enumerate(meta.valid_times):
            lead_hour = (vt - meta.init_time).total_seconds() / 3600.0
            lead_hour_by_time_idx[ti] = lead_hour
            lead_hours_seen.add(lead_hour)

        for lead_hour in lead_hour_by_time_idx.values():
            n_cases_per_lead_hour[lead_hour] = n_cases_per_lead_hour.get(lead_hour, 0) + 1

        def _time_mask(ti):
            if meta.time_index is not None:
                return meta.time_index == ti
            return np.ones(len(meta.category), dtype=bool)

        if has_members:
            for mi, member_id in enumerate(meta.member_ids):
                members_seen.add(member_id)
                member_mask = meta.member_index == mi
                for ti, lead_hour in lead_hour_by_time_idx.items():
                    mask = member_mask & _time_mask(ti) & mode_mask
                    key = (member_id, lead_hour)
                    categories_here = meta.category[mask]
                    for cat, target in _count_dicts.items():
                        n = int(np.sum(categories_here == cat))
                        target[key] = target.get(key, 0) + n
        else:
            members_seen.add(None)
            for ti, lead_hour in lead_hour_by_time_idx.items():
                mask = _time_mask(ti) & mode_mask
                key = (None, lead_hour)
                categories_here = meta.category[mask]
                for cat, target in _count_dicts.items():
                    n = int(np.sum(categories_here == cat))
                    target[key] = target.get(key, 0) + n

    return MatchLeadHourClimatology(
        lead_hours=sorted(lead_hours_seen),
        members=sorted(members_seen),
        n_hits=n_hits,
        n_misses=n_misses,
        n_false_alarms=n_false_alarms,
        n_truth_extra=n_truth_extra,
        n_forecast_extra=n_forecast_extra,
        n_cases_per_lead_hour=n_cases_per_lead_hour,
        n_files_used_stored_init_time=n_files_used_stored_init_time,
        n_files_used_filename_init_time=n_files_used_filename_init_time,
    )
