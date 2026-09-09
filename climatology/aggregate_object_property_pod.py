"""Core (non-plotting) aggregation logic: POD binned by a truth (MRMS)
object's own property (area, max intensity, or tracked age) vs. forecast
lead hour, from match_init_<init_time>.nc files.

Ported from python_base/wofs_verif_cbook.py's heat_map_fig()/calc_pod_bins()
-- POD is CONDITIONED ON TRUTH OBJECTS (binned by area_km2/max_intensity/
age of the actual observed storm), restricted to hit/miss category rows
only (a miss still has a real truth-side object with a real property
value, just no matching forecast object; false_alarm/truth_extra/
forecast_extra rows are excluded, matching the legacy's own scope for this
diagnostic). Pooled across every member and case sharing a lead hour (a
heat map shows one POD grid, not one per ensemble member).

`age` uses "age relative to init" (age_seconds - lead_hours*3600), the
exact same transform the legacy used -- this normalizes for forecast
progression so the SAME physical storm's maturity is comparable across
lead times, not just its raw age at whatever valid time it happened to be
observed. `truth_age_seconds` is NaN for any match file whose truth series
wasn't tracked at identification time (see python_obj/obj_core/matching.py's
age_seconds field) -- np.digitize places NaN values outside every real
bin (confirmed: digitize(nan, edges) == len(edges), i.e. "above the last
edge"), so they're automatically excluded by the same in-range mask used
for real out-of-range values, without any special-case NaN handling
needed here.
"""

from dataclasses import dataclass, field
from datetime import datetime

import netCDF4
import numpy as np

from .aggregate_matches_by_lead_hour import _discover, _parse_init_time_from_filename, _seconds_to_dt

PROPERTIES = ("area", "max_intensity", "age")


@dataclass
class ObjectPropertyPodClimatology:
    lead_hours: list  # sorted unique lead hours found across all match files
    members: list  # sorted unique member ids found across all match files
    bin_edges: dict = field(default_factory=dict)  # property -> np.ndarray (N+1 edges -> N bins), only for requested properties
    n_hits: dict = field(default_factory=dict)  # property -> {(bin_index, lead_hour): int}
    n_misses: dict = field(default_factory=dict)  # property -> {(bin_index, lead_hour): int}
    n_files_used_stored_init_time: int = 0
    n_files_used_filename_init_time: int = 0


@dataclass
class _HeatmapMetadata:
    member_ids: list | None
    valid_times: list
    init_time: datetime | None
    init_time_from_filename: bool
    category: np.ndarray
    truth_area_km2: np.ndarray
    truth_max_intensity: np.ndarray
    truth_age_seconds: np.ndarray
    member_index: np.ndarray | None
    time_index: np.ndarray | None


def _read_heatmap_metadata(path: str) -> _HeatmapMetadata:
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
            # Backward compatibility: truth_age_seconds was added after some
            # real on-disk match files were already written -- same
            # hasattr-gated fallback the rest of this project's optional
            # field readers already use.
            if name not in ds.variables:
                return np.full(len(category), np.nan)
            return np.asarray(ds.variables[name][:], dtype=np.float64)

        return _HeatmapMetadata(
            member_ids=member_ids, valid_times=valid_times, init_time=init_time,
            init_time_from_filename=init_time_from_filename, category=category,
            truth_area_km2=_f("truth_area_km2"), truth_max_intensity=_f("truth_max_intensity"),
            truth_age_seconds=_f("truth_age_seconds"),
            member_index=member_index, time_index=time_index,
        )


_PROPERTY_VALUES = {
    "area": lambda meta: meta.truth_area_km2,
    "max_intensity": lambda meta: meta.truth_max_intensity,
    "age": lambda meta: meta.truth_age_seconds,
}


def build_object_property_pod_climatology(
    match_dir: str,
    bin_edges: dict,  # property -> np.ndarray of bin edges; any non-empty subset of PROPERTIES
    match_pattern: str = "match_init_*.nc",
) -> ObjectPropertyPodClimatology:
    """match_dir: one match_init_*.nc per forecast case (file_grouping=
    "init_snapshot" -- every member x every lead time for that case,
    already matched against truth), same convention
    aggregate_matches_by_lead_hour.build_match_lead_hour_climatology()
    requires."""
    if not bin_edges:
        raise ValueError("build_object_property_pod_climatology: bin_edges must specify at least one property")
    unknown = set(bin_edges) - set(PROPERTIES)
    if unknown:
        raise ValueError(f"build_object_property_pod_climatology: unknown propert(ies) {sorted(unknown)}, expected subset of {PROPERTIES}")

    match_files = _discover(match_dir, match_pattern)

    n_hits = {p: {} for p in bin_edges}
    n_misses = {p: {} for p in bin_edges}
    n_files_used_stored_init_time = 0
    n_files_used_filename_init_time = 0
    members_seen: set = set()
    lead_hours_seen: set = set()

    for match_file in match_files:
        meta = _read_heatmap_metadata(match_file)
        if meta.init_time is None:
            raise ValueError(f"'{match_file}' has no init_time (neither a stored attribute nor a parseable filename) -- expected an init_snapshot match file")
        if meta.init_time_from_filename:
            n_files_used_filename_init_time += 1
        else:
            n_files_used_stored_init_time += 1
        # This function pools across every member regardless (never keys
        # its output by member), so -- unlike aggregate_matches_by_lead_hour's
        # sibling family -- a real member dimension is not actually
        # required; a deterministic (no member dimension) match file is
        # tolerated too, matching systems.py's own load_system() convention
        # and drivers.plot_matching_timeseries's support for both shapes.
        members_seen.update(meta.member_ids if meta.member_ids is not None else [None])

        lead_hour_by_time_idx = {ti: (vt - meta.init_time).total_seconds() / 3600.0 for ti, vt in enumerate(meta.valid_times)}
        lead_hours_seen.update(lead_hour_by_time_idx.values())
        time_index = meta.time_index if meta.time_index is not None else np.zeros(len(meta.category), dtype=int)
        lead_hour_per_record = np.array([lead_hour_by_time_idx[ti] for ti in time_index])

        hit_or_miss_mask = (meta.category == "hit") | (meta.category == "miss")
        hit_mask = meta.category == "hit"
        miss_mask = meta.category == "miss"

        for prop, edges in bin_edges.items():
            n_bins = len(edges) - 1
            values = _PROPERTY_VALUES[prop](meta)
            if prop == "age":
                values = values - lead_hour_per_record * 3600.0
            bin_idx = np.digitize(values, edges) - 1
            in_range = hit_or_miss_mask & (bin_idx >= 0) & (bin_idx < n_bins)
            if not np.any(in_range):
                continue

            for lh in np.unique(lead_hour_per_record[in_range]):
                lh_mask = in_range & (lead_hour_per_record == lh)
                hit_counts = np.bincount(bin_idx[lh_mask & hit_mask], minlength=n_bins)
                miss_counts = np.bincount(bin_idx[lh_mask & miss_mask], minlength=n_bins)
                for b in range(n_bins):
                    if hit_counts[b]:
                        key = (b, float(lh))
                        n_hits[prop][key] = n_hits[prop].get(key, 0) + int(hit_counts[b])
                    if miss_counts[b]:
                        key = (b, float(lh))
                        n_misses[prop][key] = n_misses[prop].get(key, 0) + int(miss_counts[b])

    return ObjectPropertyPodClimatology(
        lead_hours=sorted(lead_hours_seen), members=sorted(members_seen),
        bin_edges={p: np.asarray(e) for p, e in bin_edges.items()},
        n_hits=n_hits, n_misses=n_misses,
        n_files_used_stored_init_time=n_files_used_stored_init_time,
        n_files_used_filename_init_time=n_files_used_filename_init_time,
    )
