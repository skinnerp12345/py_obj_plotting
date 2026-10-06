"""Core (non-plotting) aggregation logic: pre-convective-initiation (CI)
forecast-skill statistics from match_init_<init_time>.nc files (file_grouping=
"init_snapshot" match files, one per forecast case).

A "CI candidate" object on either side of a match record is one whose own
tracked age is less than the elapsed lead time at that valid time (i.e.
age_seconds < lead_seconds, equivalently "age relative to init" < 0 -- the
same transform aggregate_object_property_pod.py already uses for its own
age-based heatmap) -- the object did not yet exist when the forecast/
tracked series began, so its appearance IS a genesis event within the
window being verified. This applies independently and symmetrically to the
truth (MRMS) side and the forecast side: a forecast object's own age is
bounded by construction by how long its OWN tracked series has run (a
model's tracking starts at that model's own lead=0), so
forecast_age_seconds < lead_seconds identifies a forecast object that
formed partway through the run, not one carried over from the initial
condition (which would instead satisfy forecast_age_seconds == lead_seconds
exactly, having been tracked continuously since lead=0).

A "hit" row only counts as a genuine CI hit when BOTH sides are CI
candidates AND their age difference is within `age_diff_threshold_minutes`
-- requiring both sides (not just the truth side, as the literal "observed
object age < lead time" framing alone would suggest) rules out the case
where a young truth object happens to spatially overlap an OLD, carried-
over forecast object: that pairing is a real spatial hit, but not a correct
CI prediction, since the model never actually initiated anything new there.
A "hit" row that fails this joint test is reclassified: a CI miss on the
truth-CI population (the real, new truth storm was not correctly
predicted) and, independently, a CI false alarm on the forecast-CI
population (the model's own newly-initiated object has no valid,
correctly-timed truth counterpart) -- mirroring how truth_extra/
forecast_extra already represent asymmetric near-miss outcomes elsewhere in
this schema, rather than inventing one shared reclassified label.

Returns the SAME MatchLeadHourClimatology shape build_match_lead_hour_climatology
already produces (keyed by (member, lead_hour)) -- not a bespoke result type
-- so it flows completely unchanged through compute_verification_stats() and
every existing POD/FAR/FBIAS/CSI comparison plot; no new plotting code is
needed for the CI skill time series.

truth_age_seconds/forecast_age_seconds are NaN wherever that side has no
object on a given row (miss rows have no forecast object; false_alarm rows
have no truth object) or wherever that side's series wasn't tracked at
identification time at all -- `NaN < x` and `abs(NaN - x)` are always False
in numpy, so every CI-candidate mask and the threshold comparison already
self-exclude both "not applicable" and "untracked" rows with no special-case
handling needed, the same property aggregate_object_property_pod.py already
relies on for np.digitize.
"""

from dataclasses import dataclass
from datetime import datetime

import netCDF4
import numpy as np

from .aggregate_matches_by_lead_hour import (
    MatchLeadHourClimatology,
    _discover,
    _parse_init_time_from_filename,
    _seconds_to_dt,
)


@dataclass
class _CIMatchMetadata:
    member_ids: list | None
    valid_times: list
    init_time: datetime | None
    init_time_from_filename: bool
    category: np.ndarray
    truth_age_seconds: np.ndarray
    forecast_age_seconds: np.ndarray
    member_index: np.ndarray | None
    time_index: np.ndarray | None


def _read_ci_metadata(path: str) -> _CIMatchMetadata:
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

        def _f(name):
            # Backward compatibility: same hasattr-gated fallback every
            # other optional float field reader in this project already
            # uses -- a file predating truth_age_seconds/forecast_age_seconds
            # (or whose relevant series simply wasn't tracked) has no real
            # age data, and every CI mask below already treats NaN as "not
            # a CI candidate" with no extra handling required.
            if name not in ds.variables:
                return np.full(len(category), np.nan)
            return np.asarray(ds.variables[name][:], dtype=np.float64)

        return _CIMatchMetadata(
            member_ids=member_ids, valid_times=valid_times, init_time=init_time,
            init_time_from_filename=init_time_from_filename, category=category,
            truth_age_seconds=_f("truth_age_seconds"), forecast_age_seconds=_f("forecast_age_seconds"),
            member_index=member_index, time_index=time_index,
        )


def _lead_seconds_per_record(meta: _CIMatchMetadata) -> tuple[np.ndarray, dict]:
    lead_hour_by_time_idx = {ti: (vt - meta.init_time).total_seconds() / 3600.0 for ti, vt in enumerate(meta.valid_times)}
    time_index = meta.time_index if meta.time_index is not None else np.zeros(len(meta.category), dtype=int)
    lead_hour_per_record = np.array([lead_hour_by_time_idx[ti] for ti in time_index])
    return lead_hour_per_record, lead_hour_by_time_idx


@dataclass
class TrackingRestartReport:
    """Likely truth (MRMS) tracking-run restarts found inside forecast windows
    -- see count_tracking_restarts. Report only: nothing is corrected."""
    n_files: int
    n_flagged_slices: int  # (forecast case, valid time) slices flagged
    flagged_valid_times: list  # sorted distinct datetimes
    n_truth_objects_reset: int  # truth objects on flagged slices (one member's rows), all age 0


def count_tracking_restarts(
    match_dir: str, match_pattern: str = "match_init_*.nc", min_objects: int = 3,
) -> TrackingRestartReport:
    """Flags (case, valid time) slices after init where EVERY truth object
    (>= min_objects of them) has age 0 -- the signature of the truth series'
    tracking restarting (e.g. MRMS tracking restarts after a data gap, giving
    every existing storm age 0). Such objects look like brand-new storms, so
    they are counted as CI by build_ci_skill_climatology/
    build_ci_age_difference_distribution and as age-0 by the age heatmaps;
    this function only counts them so callers can report the exposure.

    Truth objects are identical across members, so one member's rows are used
    (the first member present in each file; all rows if no member dimension)."""
    match_files = _discover(match_dir, match_pattern)
    flagged_times = set()
    n_slices = n_objects = 0
    truth_categories = ("hit", "miss", "truth_extra")
    for match_file in match_files:
        meta = _read_ci_metadata(match_file)
        if meta.init_time is None:
            continue
        time_index = meta.time_index if meta.time_index is not None else np.zeros(len(meta.category), dtype=int)
        one_member = (meta.member_index == np.min(meta.member_index)) if meta.member_index is not None and len(meta.member_index) else np.ones(len(meta.category), dtype=bool)
        truth_rows = one_member & np.isin(meta.category, truth_categories)
        for ti, vt in enumerate(meta.valid_times):
            if vt <= meta.init_time:
                continue
            ages = meta.truth_age_seconds[truth_rows & (time_index == ti)]
            ages = ages[np.isfinite(ages)]
            if len(ages) >= min_objects and np.all(ages == 0):
                n_slices += 1
                n_objects += len(ages)
                flagged_times.add(vt)
    return TrackingRestartReport(
        n_files=len(match_files), n_flagged_slices=n_slices,
        flagged_valid_times=sorted(flagged_times), n_truth_objects_reset=n_objects,
    )


def build_ci_skill_climatology(
    match_dir: str,
    age_diff_threshold_minutes: float,
    match_pattern: str = "match_init_*.nc",
    include_near_miss_candidates: bool = False,
) -> MatchLeadHourClimatology:
    """match_dir: one match_init_*.nc per forecast case (file_grouping=
    "init_snapshot"), same convention every other module in this file family
    requires.

    include_near_miss_candidates (default False, matching
    aggregate_object_property_pod.py's own hit/miss-only scope precedent):
    when True, truth_extra/forecast_extra rows whose (near-miss) side is
    itself a CI candidate are folded into n_truth_extra/n_forecast_extra
    (never into hits/misses/false_alarms, exactly mirroring how the
    standard, non-CI schema already treats these two categories) --
    compute_fbias's extended terms then pick these up automatically with no
    further code needed; when False those rows are excluded entirely, and
    n_truth_extra/n_forecast_extra stay at 0, degrading FBIAS to the plain
    3-category formula.
    """
    match_files = _discover(match_dir, match_pattern)
    threshold_seconds = age_diff_threshold_minutes * 60.0

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

    for match_file in match_files:
        meta = _read_ci_metadata(match_file)
        if meta.init_time is None:
            raise ValueError(f"'{match_file}' has no init_time (neither a stored attribute nor a parseable filename) -- expected an init_snapshot match file")
        if meta.init_time_from_filename:
            n_files_used_filename_init_time += 1
        else:
            n_files_used_stored_init_time += 1

        # Tolerates a deterministic (no member dimension) match file too,
        # same as build_match_lead_hour_climatology_by_storm_mode -- this
        # feeds the same driver, which must support both shapes.
        has_members = meta.member_ids is not None and meta.member_index is not None

        lead_hour_per_record, lead_hour_by_time_idx = _lead_seconds_per_record(meta)
        for lead_hour in lead_hour_by_time_idx.values():
            lead_hours_seen.add(lead_hour)
            n_cases_per_lead_hour[lead_hour] = n_cases_per_lead_hour.get(lead_hour, 0) + 1
        # Whole seconds: lead_hour * 3600 carries ~1e-12 s float error (e.g. leads of 125/250/265 min
        # overshoot), which let an object present since init (age == lead exactly) pass
        # 'age < lead' and be miscounted as a CI candidate.
        lead_seconds_per_record = np.round(lead_hour_per_record * 3600.0)

        truth_ci = meta.truth_age_seconds < lead_seconds_per_record
        forecast_ci = meta.forecast_age_seconds < lead_seconds_per_record
        age_diff = meta.truth_age_seconds - meta.forecast_age_seconds  # observed - forecast (only |age_diff| used here)
        is_ci_hit = (meta.category == "hit") & truth_ci & forecast_ci & (np.abs(age_diff) <= threshold_seconds)

        truth_pop_categories = ("hit", "miss", "truth_extra") if include_near_miss_candidates else ("hit", "miss")
        forecast_pop_categories = ("hit", "false_alarm", "forecast_extra") if include_near_miss_candidates else ("hit", "false_alarm")
        ci_truth_pop = np.isin(meta.category, truth_pop_categories) & truth_ci
        ci_forecast_pop = np.isin(meta.category, forecast_pop_categories) & forecast_ci

        # A "hit" row failing the joint CI test contributes to BOTH the
        # truth-CI-population's miss count and the forecast-CI-population's
        # false-alarm count independently (see module docstring) -- these
        # are boolean masks, not a single-assignment category label, since
        # one row can legitimately feed two different populations at once.
        is_ci_miss = (ci_truth_pop & (meta.category == "miss")) | (ci_truth_pop & (meta.category == "hit") & ~is_ci_hit)
        is_ci_false_alarm = (ci_forecast_pop & (meta.category == "false_alarm")) | (ci_forecast_pop & (meta.category == "hit") & ~is_ci_hit)
        is_ci_truth_extra = ci_truth_pop & (meta.category == "truth_extra")
        is_ci_forecast_extra = ci_forecast_pop & (meta.category == "forecast_extra")

        def _accumulate(mask: np.ndarray, key: tuple) -> None:
            n_hits[key] = n_hits.get(key, 0) + int(np.sum(mask & is_ci_hit))
            n_misses[key] = n_misses.get(key, 0) + int(np.sum(mask & is_ci_miss))
            n_false_alarms[key] = n_false_alarms.get(key, 0) + int(np.sum(mask & is_ci_false_alarm))
            n_truth_extra[key] = n_truth_extra.get(key, 0) + int(np.sum(mask & is_ci_truth_extra))
            n_forecast_extra[key] = n_forecast_extra.get(key, 0) + int(np.sum(mask & is_ci_forecast_extra))

        def _time_mask(ti):
            if meta.time_index is not None:
                return meta.time_index == ti
            return np.ones(len(meta.category), dtype=bool)

        if has_members:
            for mi, member_id in enumerate(meta.member_ids):
                members_seen.add(member_id)
                member_mask = meta.member_index == mi
                for ti, lead_hour in lead_hour_by_time_idx.items():
                    _accumulate(member_mask & _time_mask(ti), (member_id, lead_hour))
        else:
            members_seen.add(None)
            for ti, lead_hour in lead_hour_by_time_idx.items():
                _accumulate(_time_mask(ti), (None, lead_hour))

    return MatchLeadHourClimatology(
        lead_hours=sorted(lead_hours_seen),
        members=sorted(members_seen),
        n_hits=n_hits, n_misses=n_misses, n_false_alarms=n_false_alarms,
        n_truth_extra=n_truth_extra, n_forecast_extra=n_forecast_extra,
        n_cases_per_lead_hour=n_cases_per_lead_hour,
        n_files_used_stored_init_time=n_files_used_stored_init_time,
        n_files_used_filename_init_time=n_files_used_filename_init_time,
    )


@dataclass
class CIAgeDifferenceDistribution:
    age_diff_seconds: np.ndarray  # truth_age_seconds - forecast_age_seconds (observed - forecast), one entry per matched CI pair;
                                  # age is time since first detection, so positive = forecast object younger = model LATE
    lead_hours: np.ndarray  # parallel array
    n_files_used_stored_init_time: int = 0
    n_files_used_filename_init_time: int = 0


def build_ci_age_difference_distribution(
    match_dir: str,
    match_pattern: str = "match_init_*.nc",
) -> CIAgeDifferenceDistribution:
    """age_diff_seconds for every 'hit' row where BOTH the truth and
    forecast objects are themselves CI candidates (age_seconds < lead_seconds
    on both sides) -- i.e. every case where truth and forecast agree a NEW
    storm formed here, at ANY timing offset (not restricted to pairs passing
    age_diff_threshold_minutes, unlike build_ci_skill_climatology) -- the
    population the early/late timing-bias histogram is built from.

    Sign: observed - forecast age. Ages count up from 0 at an object's first
    detection, so a positive value (forecast object younger) means the model
    initiated the storm LATER than observed; negative means EARLIER."""
    match_files = _discover(match_dir, match_pattern)
    all_diffs = []
    all_lead_hours = []
    n_files_used_stored_init_time = 0
    n_files_used_filename_init_time = 0

    for match_file in match_files:
        meta = _read_ci_metadata(match_file)
        if meta.init_time is None:
            raise ValueError(f"'{match_file}' has no init_time (neither a stored attribute nor a parseable filename) -- expected an init_snapshot match file")
        if meta.init_time_from_filename:
            n_files_used_filename_init_time += 1
        else:
            n_files_used_stored_init_time += 1

        lead_hour_per_record, _ = _lead_seconds_per_record(meta)
        # Whole seconds: lead_hour * 3600 carries ~1e-12 s float error (e.g. leads of 125/250/265 min
        # overshoot), which let an object present since init (age == lead exactly) pass
        # 'age < lead' and be miscounted as a CI candidate.
        lead_seconds_per_record = np.round(lead_hour_per_record * 3600.0)

        truth_ci = meta.truth_age_seconds < lead_seconds_per_record
        forecast_ci = meta.forecast_age_seconds < lead_seconds_per_record
        both_ci_hit = (meta.category == "hit") & truth_ci & forecast_ci

        all_diffs.append((meta.truth_age_seconds - meta.forecast_age_seconds)[both_ci_hit])  # observed - forecast
        all_lead_hours.append(lead_hour_per_record[both_ci_hit])

    return CIAgeDifferenceDistribution(
        age_diff_seconds=np.concatenate(all_diffs) if all_diffs else np.array([]),
        lead_hours=np.concatenate(all_lead_hours) if all_lead_hours else np.array([]),
        n_files_used_stored_init_time=n_files_used_stored_init_time,
        n_files_used_filename_init_time=n_files_used_filename_init_time,
    )
