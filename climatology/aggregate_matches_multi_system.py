"""Core (non-plotting) aggregation logic: pools per-(member, lead_minutes)
contingency-table counts AND per-lead_minutes diagnostic-property
distributions across several INDEPENDENTLY forecast-vs-MRMS-matched systems
(e.g. WoFS, WoFSCast, NowcastNet), restricted to (forecast cycle, lead time)
combinations where every system has a matched result -- a fair, apples-to-
apples comparison, mirroring py_obj's own "shared valid times" precedent
(climatology.aggregate_shared_valid_times) but applied to match/contingency
counts, keyed by (cycle init_time, lead_minutes) rather than a flat valid_time
set.

Built on top of ..systems.SystemSpec/load_system and
..multi_system.load_systems/compute_included_keys (the N-system comparison
layer): a caller passes a list of SystemSpec (the SAME mechanism whether
"systems" means different forecast systems or different matching-threshold
variants of one system), plus an explicit lead_time_mode ("intersection",
strict joint-availability, or "union", where each system's own presence at
a lead time suffices independently -- see multi_system.py's own module
docstring for how to reconstruct a mixed per-lead-minute policy via
composition, if ever needed).

Reuses aggregate_diagnostic_distributions_by_lead_hour.py's metadata-only
float-field reader unchanged (_read_diagnostic_metadata) for the two
functions that need it -- this module only adds the re-keying from
lead_minutes into the EXISTING MatchLeadHourClimatology/
DiagnosticDistributionClimatology shapes (lead_minutes/60.0), so
compute_verification_stats()/the existing comparison-plotting functions
need zero changes to consume the result.
"""

from dataclasses import dataclass, field
from datetime import datetime

import netCDF4
import numpy as np

from ..multi_system import MultiSystemAvailabilityReport, compute_included_keys, load_systems
from ..systems import SystemSpec
from .aggregate_diagnostic_distributions_by_lead_hour import (
    DiagnosticDistributionClimatology,
    _haversine_km,
    _read_diagnostic_metadata,
)
from .aggregate_matches_by_lead_hour import (
    MatchLeadHourClimatology,
    _discover,
    _parse_init_time_from_filename,
    _seconds_to_dt,
)

# A truth/forecast object's canonical categorization -- the ONE row where its
# own area_km2/mean_intensity should be counted. Necessary because a
# truth_extra/forecast_extra row ALSO carries its own best near-miss
# candidate's full field set on the OTHER side (see
# py_obj/obj_core/matching.py's _record_from_objects: truth/forecast
# fields are populated whenever that side's object is not None, regardless of
# category) -- so a forecast object that lost a contested match and is
# truly categorized "forecast_extra" on its own row can ALSO appear as the
# (non-canonical) forecast-side candidate on a DIFFERENT truth object's
# truth_extra row. Masking by category membership rather than by
# np.isfinite(...) is what avoids double-counting that same object twice.
_TRUTH_CANONICAL_CATEGORIES = ("hit", "miss", "truth_extra")
_FORECAST_CANONICAL_CATEGORIES = ("hit", "false_alarm", "forecast_extra")


def build_multi_system_match_climatology(
    systems: list[SystemSpec], lead_time_mode: str = "intersection", lead_minutes: list | None = None,
) -> tuple[dict, MultiSystemAvailabilityReport]:
    """systems: list[SystemSpec]. Returns ({system_name: MatchLeadHourClimatology}, availability_report).

    Each returned MatchLeadHourClimatology's lead_hours is populated with
    ONLY the lead_minutes/60.0 values that system actually has included data
    for -- e.g. WoFSCast's own lead_hours list correctly starts at 20 min
    (1/3 hour), never 0 or 10 min, rather than including those lead times
    with spurious zero counts. This is what makes those points render as "no
    line" rather than a misleading flat 0.0 when plotted -- no NaN-filling
    needed, the point is simply absent from that system's own series.
    """
    loaded = load_systems(systems)
    included, report = compute_included_keys(loaded, lead_time_mode=lead_time_mode, lead_minutes=lead_minutes)

    _CATEGORIES = ("hit", "miss", "false_alarm", "truth_extra", "forecast_extra")
    climatologies = {}

    for name, ls in loaded.items():
        n_hits, n_misses, n_false_alarms, n_truth_extra, n_forecast_extra = {}, {}, {}, {}, {}
        n_cases_per_lead_hour: dict = {}
        members_seen: set = set()
        _count_dicts = dict(zip(_CATEGORIES, (n_hits, n_misses, n_false_alarms, n_truth_extra, n_forecast_extra)))

        for key in included[name]:
            lead_hour = key[1] / 60.0
            n_cases_per_lead_hour[lead_hour] = n_cases_per_lead_hour.get(lead_hour, 0) + 1
            for member_id, cat_arrays in ls.records_by_key.get(key, {}).items():
                members_seen.add(member_id)
                mkey = (member_id, lead_hour)
                cats = np.concatenate(cat_arrays) if cat_arrays else np.array([], dtype=str)
                for cat, target in _count_dicts.items():
                    target[mkey] = target.get(mkey, 0) + int(np.sum(cats == cat))

        climatologies[name] = MatchLeadHourClimatology(
            lead_hours=sorted(n_cases_per_lead_hour.keys()),
            members=sorted(members_seen),
            n_hits=n_hits, n_misses=n_misses, n_false_alarms=n_false_alarms,
            n_truth_extra=n_truth_extra, n_forecast_extra=n_forecast_extra,
            n_cases_per_lead_hour=n_cases_per_lead_hour,
        )

    return climatologies, report


def build_multi_system_diagnostic_climatology(
    systems: list[SystemSpec], lead_time_mode: str = "intersection", lead_minutes: list | None = None,
) -> tuple[dict, MultiSystemAvailabilityReport]:
    """Same shared-availability filtering as build_multi_system_match_climatology,
    applied to hit-pair diagnostic properties instead of contingency counts.
    A second read pass is needed (the lightweight metadata read used for
    availability doesn't carry the float diagnostic fields), filtered down
    to only the (cycle, lead_min) keys the availability pass already decided
    to include -- avoids reading floats for files/times that will be
    excluded anyway.
    """
    loaded = load_systems(systems)
    included, report = compute_included_keys(loaded, lead_time_mode=lead_time_mode, lead_minutes=lead_minutes)

    climatologies = {}
    for spec in systems:
        name = spec.name
        files = _discover(spec.match_dir, spec.match_pattern)
        area_ratio: dict = {}
        max_intensity_diff: dict = {}
        mean_intensity_diff: dict = {}
        centroid_dist_km: dict = {}
        ti_score: dict = {}
        n_hits_per_lead_hour: dict = {}

        for f in files:
            meta = _read_diagnostic_metadata(f)
            if meta.init_time is None:
                raise ValueError(f"'{f}' has no init_time -- expected an init_snapshot match file")

            lead_min_by_ti = {
                ti: round((vt - meta.init_time).total_seconds() / 60.0) for ti, vt in enumerate(meta.valid_times)
            }
            for ti, lm in lead_min_by_ti.items():
                key = (meta.init_time, lm)
                if key not in included[name]:
                    continue
                lead_hour = lm / 60.0
                # match_time_index is only written when a file spans >1
                # distinct valid_time (write_match_file's own use_time_dim
                # rule) -- a file with exactly one lead time has no such
                # variable, so every record in it belongs to that file's
                # single (implicit) time index 0 (same convention as
                # systems.py's load_system()).
                time_mask = (meta.time_index == ti) if meta.time_index is not None else np.ones(len(meta.category), dtype=bool)
                mask = time_mask & (meta.category == "hit")
                n = int(np.sum(mask))
                if n == 0:
                    continue

                area_ratio.setdefault(lead_hour, []).extend(
                    (meta.forecast_area_km2[mask] / meta.truth_area_km2[mask]).tolist()
                )
                max_intensity_diff.setdefault(lead_hour, []).extend(
                    (meta.forecast_max_intensity[mask] - meta.truth_max_intensity[mask]).tolist()
                )
                mean_intensity_diff.setdefault(lead_hour, []).extend(
                    (meta.forecast_mean_intensity[mask] - meta.truth_mean_intensity[mask]).tolist()
                )
                centroid_dist_km.setdefault(lead_hour, []).extend(
                    _haversine_km(
                        meta.truth_centroid_lat[mask], meta.truth_centroid_lon[mask],
                        meta.forecast_centroid_lat[mask], meta.forecast_centroid_lon[mask],
                    ).tolist()
                )
                ti_score.setdefault(lead_hour, []).extend(meta.ti_score[mask].tolist())
                n_hits_per_lead_hour[lead_hour] = n_hits_per_lead_hour.get(lead_hour, 0) + n

        climatologies[name] = DiagnosticDistributionClimatology(
            lead_hours=sorted(n_hits_per_lead_hour.keys()),
            area_ratio=area_ratio, max_intensity_diff=max_intensity_diff,
            mean_intensity_diff=mean_intensity_diff, centroid_dist_km=centroid_dist_km,
            ti_score=ti_score, n_hits_per_lead_hour=n_hits_per_lead_hour,
        )

    return climatologies, report


@dataclass
class ObjectStatsMultiSystemClimatology:
    """Per-system object-level statistics -- every truth/forecast object a
    match file carries (see _TRUTH_CANONICAL_CATEGORIES/_FORECAST_CANONICAL_CATEGORIES
    above for how each object is counted exactly once), NOT restricted to
    hit pairs like DiagnosticDistributionClimatology. Same shared-
    availability lead-time filtering as the other multi-system climatologies.
    """
    lead_hours: list
    members: list  # [None] for a deterministic system with no member dimension
    forecast_count: dict = field(default_factory=dict)  # (member, lead_hour) -> int
    forecast_area_km2: dict = field(default_factory=dict)  # (member, lead_hour) -> float (summed)
    forecast_area_values: list = field(default_factory=list)  # flat, pooled across every included (member, lead_hour) -- for box-whisker
    forecast_mean_intensity_values: list = field(default_factory=list)
    # Truth-side fields are SHARED/identical across every system's own
    # climatology object (see build_multi_system_object_climatology) -- MRMS
    # truth objects don't depend on which forecast system's match file they
    # were read from, so rather than pool a slightly different truth subset
    # per system (which would differ only at relaxed lead minutes, where
    # systems' own inclusion sets diverge), one shared truth series -- pooled
    # from the UNION of every system's own included keys -- is computed once
    # and attached to all of them, so any single system's own climatology
    # object can serve as the MRMS reference for plotting.
    truth_lead_hours: list = field(default_factory=list)
    truth_count: dict = field(default_factory=dict)  # lead_hour -> int
    truth_area_km2: dict = field(default_factory=dict)  # lead_hour -> float (summed)
    truth_area_values: list = field(default_factory=list)
    truth_mean_intensity_values: list = field(default_factory=list)


@dataclass
class _ObjectStatsMetadata:
    member_ids: list | None
    valid_times: list
    init_time: object
    category: np.ndarray
    member_index: np.ndarray | None
    time_index: np.ndarray | None
    truth_area_km2: np.ndarray
    forecast_area_km2: np.ndarray
    truth_mean_intensity: np.ndarray
    forecast_mean_intensity: np.ndarray


def _read_object_stats_metadata(path: str) -> _ObjectStatsMetadata:
    """Combines _read_match_metadata's member/category fields and
    _read_diagnostic_metadata's area/intensity fields in ONE read pass --
    neither existing reader alone carries everything this needs (member
    identity AND the float object-property fields together), and a third
    full read of every file (on top of the presence pass) is already the
    established two-pass pattern here (see build_multi_system_diagnostic_climatology).

    init_time prefers the file's own global attribute, falling back to
    parsing it from the match_init_<YYYYMMDD>_<HHMMSS>.nc filename when
    absent -- the same fallback _read_match_metadata already has. Real bug
    found running this against ncar_mpas_analysis/ for the first time: this
    reader had NO fallback at all (unlike its sibling), raising outright on
    every one of those real files, which predate the stored init_time
    attribute -- this code path had only ever been exercised against
    sfe_poster/ data, where the attribute is always present."""
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

        if hasattr(ds, "init_time"):
            init_time = datetime.fromisoformat(ds.init_time)
        else:
            init_time = _parse_init_time_from_filename(path)
        category = np.asarray(ds.variables["category"][:], dtype=str)
        member_index = np.asarray(ds.variables["match_member_index"][:]) if "match_member_index" in ds.variables else None
        time_index = np.asarray(ds.variables["match_time_index"][:]) if "match_time_index" in ds.variables else None

        def _f(name):
            return np.asarray(ds.variables[name][:], dtype=np.float64)

        return _ObjectStatsMetadata(
            member_ids=member_ids, valid_times=valid_times, init_time=init_time,
            category=category, member_index=member_index, time_index=time_index,
            truth_area_km2=_f("truth_area_km2"), forecast_area_km2=_f("forecast_area_km2"),
            truth_mean_intensity=_f("truth_mean_intensity"), forecast_mean_intensity=_f("forecast_mean_intensity"),
        )


def build_multi_system_object_climatology(
    systems: list[SystemSpec], lead_time_mode: str = "intersection", lead_minutes: list | None = None,
) -> tuple[dict, MultiSystemAvailabilityReport]:
    """systems: list[SystemSpec]. Per-system total forecast object count/
    summed area by lead hour, plus flat pooled forecast area/mean-intensity
    value lists (for box-whisker plots aggregated across every included lead
    hour and member) -- plus one SHARED truth (MRMS) series attached
    identically to every system's own climatology object (see the dataclass
    docstring for why: truth objects are system-independent, so they are
    pooled once from the union of every system's own included keys, not
    recomputed separately per system).

    Truth-side stats are pooled once per (cycle, lead_min) key -- not once
    per member -- since the same MRMS truth field is independently re-matched
    by every member but is not itself a per-member quantity; exactly one
    member's own canonical category assignment (member index 0) is used to
    select the full set of truth objects at that key, since every truth
    object present is assigned to exactly one of hit/miss/truth_extra by any
    given member's own match run.
    """
    loaded = load_systems(systems)
    included, report = compute_included_keys(loaded, lead_time_mode=lead_time_mode, lead_minutes=lead_minutes)

    # ---- forecast side: genuinely per-system ----
    climatologies = {}
    for spec in systems:
        name = spec.name
        files = _discover(spec.match_dir, spec.match_pattern)
        forecast_count: dict = {}
        forecast_area_km2: dict = {}
        forecast_area_values: list = []
        forecast_mean_intensity_values: list = []
        members_seen: set = set()
        lead_hours_seen: set = set()

        for f in files:
            meta = _read_object_stats_metadata(f)
            if meta.init_time is None:
                raise ValueError(f"'{f}' has no init_time -- expected an init_snapshot match file")

            lead_min_by_ti = {
                ti: round((vt - meta.init_time).total_seconds() / 60.0) for ti, vt in enumerate(meta.valid_times)
            }

            has_members = meta.member_ids is not None and meta.member_index is not None
            member_ids = meta.member_ids if has_members else [None]
            forecast_canonical = np.isin(meta.category, _FORECAST_CANONICAL_CATEGORIES)

            for ti, lm in lead_min_by_ti.items():
                key = (meta.init_time, lm)
                if key not in included[name]:
                    continue
                lead_hour = lm / 60.0
                lead_hours_seen.add(lead_hour)

                # See build_multi_system_diagnostic_climatology's comment:
                # match_time_index is absent for a single-lead-time file.
                time_mask = (meta.time_index == ti) if meta.time_index is not None else np.ones(len(meta.category), dtype=bool)
                for mi, member_id in enumerate(member_ids):
                    members_seen.add(member_id)
                    fmask = forecast_canonical & time_mask
                    if has_members:
                        fmask = fmask & (meta.member_index == mi)
                    mkey = (member_id, lead_hour)
                    n_fc = int(np.sum(fmask))
                    forecast_count[mkey] = forecast_count.get(mkey, 0) + n_fc
                    forecast_area_km2[mkey] = forecast_area_km2.get(mkey, 0.0) + float(meta.forecast_area_km2[fmask].sum())
                    forecast_area_values.extend(meta.forecast_area_km2[fmask].tolist())
                    forecast_mean_intensity_values.extend(meta.forecast_mean_intensity[fmask].tolist())

        climatologies[name] = ObjectStatsMultiSystemClimatology(
            lead_hours=sorted(lead_hours_seen),
            members=sorted(members_seen),
            forecast_count=forecast_count, forecast_area_km2=forecast_area_km2,
            forecast_area_values=forecast_area_values,
            forecast_mean_intensity_values=forecast_mean_intensity_values,
        )

    # ---- truth side: ONE shared series, pooled from the union of every
    # system's own included keys, each key read from whichever system's file
    # happens to contain it (first one encountered in systems' own list
    # order) -- correct because MRMS truth objects are identical regardless
    # of which forecast system's match file they were read from. ----
    combined_keys: set = set()
    for name in included:
        combined_keys |= included[name]

    truth_count: dict = {}
    truth_area_km2: dict = {}
    truth_area_values: list = []
    truth_mean_intensity_values: list = []
    truth_lead_hours_seen: set = set()
    keys_done: set = set()

    for spec in systems:
        if combined_keys == keys_done:
            break  # every needed key already found in an earlier system
        files = _discover(spec.match_dir, spec.match_pattern)
        for f in files:
            if combined_keys == keys_done:
                break
            meta = _read_object_stats_metadata(f)
            lead_min_by_ti = {
                ti: round((vt - meta.init_time).total_seconds() / 60.0) for ti, vt in enumerate(meta.valid_times)
            }
            has_members = meta.member_ids is not None and meta.member_index is not None
            truth_canonical = np.isin(meta.category, _TRUTH_CANONICAL_CATEGORIES)

            for ti, lm in lead_min_by_ti.items():
                key = (meta.init_time, lm)
                if key not in combined_keys or key in keys_done:
                    continue
                keys_done.add(key)
                lead_hour = lm / 60.0
                truth_lead_hours_seen.add(lead_hour)

                # See build_multi_system_diagnostic_climatology's comment:
                # match_time_index is absent for a single-lead-time file.
                time_mask = (meta.time_index == ti) if meta.time_index is not None else np.ones(len(meta.category), dtype=bool)
                truth_mask = truth_canonical & time_mask
                if has_members:
                    truth_mask = truth_mask & (meta.member_index == 0)
                n_truth = int(np.sum(truth_mask))
                truth_count[lead_hour] = truth_count.get(lead_hour, 0) + n_truth
                truth_area_km2[lead_hour] = truth_area_km2.get(lead_hour, 0.0) + float(meta.truth_area_km2[truth_mask].sum())
                truth_area_values.extend(meta.truth_area_km2[truth_mask].tolist())
                truth_mean_intensity_values.extend(meta.truth_mean_intensity[truth_mask].tolist())

    truth_lead_hours = sorted(truth_lead_hours_seen)
    for name in climatologies:
        climatologies[name].truth_lead_hours = truth_lead_hours
        climatologies[name].truth_count = truth_count
        climatologies[name].truth_area_km2 = truth_area_km2
        climatologies[name].truth_area_values = truth_area_values
        climatologies[name].truth_mean_intensity_values = truth_mean_intensity_values

    return climatologies, report


@dataclass
class _ObjectStatsMetadataWithStormMode(_ObjectStatsMetadata):
    truth_is_linear: np.ndarray  # int; -1 sentinel where absent/not applicable
    forecast_is_linear: np.ndarray


def _read_object_stats_metadata_with_storm_mode(path: str) -> _ObjectStatsMetadataWithStormMode:
    """Same fields as _read_object_stats_metadata() PLUS truth_is_linear/
    forecast_is_linear -- read ADDITIONALLY, mirroring
    aggregate_matches_by_lead_hour.py's _read_match_metadata_with_storm_mode's
    own -1-sentinel backward-compat pattern for a match file predating
    storm_mode_classification."""
    base = _read_object_stats_metadata(path)
    with netCDF4.Dataset(path, "r") as ds:
        def _i(name):
            if name not in ds.variables:
                return np.full(len(base.category), -1, dtype=int)
            return np.asarray(ds.variables[name][:], dtype=int)

        truth_is_linear = _i("truth_is_linear")
        forecast_is_linear = _i("forecast_is_linear")

    return _ObjectStatsMetadataWithStormMode(
        member_ids=base.member_ids, valid_times=base.valid_times, init_time=base.init_time,
        category=base.category, member_index=base.member_index, time_index=base.time_index,
        truth_area_km2=base.truth_area_km2, forecast_area_km2=base.forecast_area_km2,
        truth_mean_intensity=base.truth_mean_intensity, forecast_mean_intensity=base.forecast_mean_intensity,
        truth_is_linear=truth_is_linear, forecast_is_linear=forecast_is_linear,
    )


def build_multi_system_object_climatology_by_storm_mode(
    systems: list[SystemSpec], storm_mode: int, lead_time_mode: str = "intersection", lead_minutes: list | None = None,
) -> tuple[dict, MultiSystemAvailabilityReport]:
    """Same shape/pooling as build_multi_system_object_climatology(), restricted
    to objects of one storm mode (0=cellular, 1=mixed, 2=linear -- see
    py_obj/obj_core/matching.py's is_linear convention). Unlike the
    truth/forecast category-membership asymmetry
    build_match_lead_hour_climatology_by_storm_mode() has to handle (a
    false_alarm row has NO truth object at all, so a merged "effective"
    storm mode is needed there), no such merge is needed here: forecast-side
    counting is already restricted to _FORECAST_CANONICAL_CATEGORIES rows
    (every one of which carries a real forecast object, hence a real
    forecast_is_linear), and truth-side counting is already restricted to
    _TRUTH_CANONICAL_CATEGORIES rows (every one of which carries a real
    truth object, hence a real truth_is_linear) -- so each side's own
    is_linear field is simply ANDed onto that side's existing canonical-
    category mask, independently.

    Requires storm_mode_classification to have been enabled at
    identification time -- a match file without it has truth_is_linear/
    forecast_is_linear entirely absent (read as the -1 sentinel here), so
    every record is excluded from every storm mode, producing an
    all-zero-count climatology rather than raising (same established
    degrade-gracefully convention as the match/diagnostic storm-mode
    siblings).
    """
    loaded = load_systems(systems)
    included, report = compute_included_keys(loaded, lead_time_mode=lead_time_mode, lead_minutes=lead_minutes)

    # ---- forecast side: genuinely per-system ----
    climatologies = {}
    for spec in systems:
        name = spec.name
        files = _discover(spec.match_dir, spec.match_pattern)
        forecast_count: dict = {}
        forecast_area_km2: dict = {}
        forecast_area_values: list = []
        forecast_mean_intensity_values: list = []
        members_seen: set = set()
        lead_hours_seen: set = set()

        for f in files:
            meta = _read_object_stats_metadata_with_storm_mode(f)
            if meta.init_time is None:
                raise ValueError(f"'{f}' has no init_time -- expected an init_snapshot match file")

            lead_min_by_ti = {
                ti: round((vt - meta.init_time).total_seconds() / 60.0) for ti, vt in enumerate(meta.valid_times)
            }

            has_members = meta.member_ids is not None and meta.member_index is not None
            member_ids = meta.member_ids if has_members else [None]
            forecast_canonical = np.isin(meta.category, _FORECAST_CANONICAL_CATEGORIES) & (meta.forecast_is_linear == storm_mode)

            for ti, lm in lead_min_by_ti.items():
                key = (meta.init_time, lm)
                if key not in included[name]:
                    continue
                lead_hour = lm / 60.0
                lead_hours_seen.add(lead_hour)

                time_mask = (meta.time_index == ti) if meta.time_index is not None else np.ones(len(meta.category), dtype=bool)
                for mi, member_id in enumerate(member_ids):
                    members_seen.add(member_id)
                    fmask = forecast_canonical & time_mask
                    if has_members:
                        fmask = fmask & (meta.member_index == mi)
                    mkey = (member_id, lead_hour)
                    n_fc = int(np.sum(fmask))
                    forecast_count[mkey] = forecast_count.get(mkey, 0) + n_fc
                    forecast_area_km2[mkey] = forecast_area_km2.get(mkey, 0.0) + float(meta.forecast_area_km2[fmask].sum())
                    forecast_area_values.extend(meta.forecast_area_km2[fmask].tolist())
                    forecast_mean_intensity_values.extend(meta.forecast_mean_intensity[fmask].tolist())

        climatologies[name] = ObjectStatsMultiSystemClimatology(
            lead_hours=sorted(lead_hours_seen),
            members=sorted(members_seen),
            forecast_count=forecast_count, forecast_area_km2=forecast_area_km2,
            forecast_area_values=forecast_area_values,
            forecast_mean_intensity_values=forecast_mean_intensity_values,
        )

    # ---- truth side: ONE shared series, pooled from the union of every
    # system's own included keys -- same rationale as
    # build_multi_system_object_climatology(). ----
    combined_keys: set = set()
    for name in included:
        combined_keys |= included[name]

    truth_count: dict = {}
    truth_area_km2: dict = {}
    truth_area_values: list = []
    truth_mean_intensity_values: list = []
    truth_lead_hours_seen: set = set()
    keys_done: set = set()

    for spec in systems:
        if combined_keys == keys_done:
            break
        files = _discover(spec.match_dir, spec.match_pattern)
        for f in files:
            if combined_keys == keys_done:
                break
            meta = _read_object_stats_metadata_with_storm_mode(f)
            lead_min_by_ti = {
                ti: round((vt - meta.init_time).total_seconds() / 60.0) for ti, vt in enumerate(meta.valid_times)
            }
            has_members = meta.member_ids is not None and meta.member_index is not None
            truth_canonical = np.isin(meta.category, _TRUTH_CANONICAL_CATEGORIES) & (meta.truth_is_linear == storm_mode)

            for ti, lm in lead_min_by_ti.items():
                key = (meta.init_time, lm)
                if key not in combined_keys or key in keys_done:
                    continue
                keys_done.add(key)
                lead_hour = lm / 60.0
                truth_lead_hours_seen.add(lead_hour)

                time_mask = (meta.time_index == ti) if meta.time_index is not None else np.ones(len(meta.category), dtype=bool)
                truth_mask = truth_canonical & time_mask
                if has_members:
                    truth_mask = truth_mask & (meta.member_index == 0)
                n_truth = int(np.sum(truth_mask))
                truth_count[lead_hour] = truth_count.get(lead_hour, 0) + n_truth
                truth_area_km2[lead_hour] = truth_area_km2.get(lead_hour, 0.0) + float(meta.truth_area_km2[truth_mask].sum())
                truth_area_values.extend(meta.truth_area_km2[truth_mask].tolist())
                truth_mean_intensity_values.extend(meta.truth_mean_intensity[truth_mask].tolist())

    truth_lead_hours = sorted(truth_lead_hours_seen)
    for name in climatologies:
        climatologies[name].truth_lead_hours = truth_lead_hours
        climatologies[name].truth_count = truth_count
        climatologies[name].truth_area_km2 = truth_area_km2
        climatologies[name].truth_area_values = truth_area_values
        climatologies[name].truth_mean_intensity_values = truth_mean_intensity_values

    return climatologies, report
