"""Core (non-plotting) aggregation logic: pools raw per-record diagnostic
property differences (area ratio, max/mean intensity difference, centroid
distance, Total Interest) for matched ("hit" category only) object pairs,
by forecast lead hour, from match_init_<init_time>.nc files.

Adapted from python_base/wofs_verif_cbook.py's time_series_diagnostic()/
time_series_displacement() (which pool hit-pair values by lead time across
all ensemble members) -- but this module keeps the full pooled list of raw
values per lead hour (not just a mean/bootstrap-CI), since the downstream
plot needs the 10th/90th percentile and median of the actual distribution.

Only "hit" category records are used (confirmed as the legacy convention:
truth_extra/forecast_extra -- near-misses that lost the greedy assignment --
are excluded, matching python_base's own scope for these diagnostics).

centroid_dist_km is computed here via haversine on the already-stored
truth_centroid_lat/lon + forecast_centroid_lat/lon fields, NOT read from a
stored centroid_dist_km variable -- confirmed real match files (all four
ncar_mpas_analysis/matched_objects_*km/ sets) predate that field (added to
py_obj after these files were generated) and are not being regenerated
(478 cases x ~110s would be many hours); computing it from data already
present in every existing file avoids that entirely, for old and new files
alike. This is a standard great-circle distance, not the exact LCC-
projected planar distance py_obj uses internally for matching (which
depends on grid-specific projection parameters not reconstructable from a
match file alone) -- close enough for a diagnostic distribution plot at
storm-object separation scales (up to a few hundred km), but not bit-
identical to the value used inside the matching decision itself.

area_ratio = forecast_area_km2 / truth_area_km2, plotted on a LINEAR scale
(not python_base's log2 transform) -- log2 was that codebase's fix for a
MEAN statistic being skewed by outliers; median/percentiles (used here) are
already outlier-robust, so the transform's justification doesn't apply.
"""

from dataclasses import dataclass, field
from datetime import datetime

import netCDF4
import numpy as np

from .aggregate_matches_by_lead_hour import _discover, _parse_init_time_from_filename, _seconds_to_dt

_EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1r, lon1r, lat2r, lon2r = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return _EARTH_RADIUS_KM * c


@dataclass
class DiagnosticDistributionClimatology:
    lead_hours: list  # sorted unique lead hours found across all match files
    area_ratio: dict = field(default_factory=dict)  # lead_hour -> list[float], hit pairs only
    max_intensity_diff: dict = field(default_factory=dict)
    mean_intensity_diff: dict = field(default_factory=dict)
    centroid_dist_km: dict = field(default_factory=dict)
    ti_score: dict = field(default_factory=dict)
    n_hits_per_lead_hour: dict = field(default_factory=dict)
    n_files_used_stored_init_time: int = 0
    n_files_used_filename_init_time: int = 0


@dataclass
class _DiagnosticMetadata:
    time_index: np.ndarray
    valid_times: list
    init_time: object
    init_time_from_filename: bool
    category: np.ndarray
    truth_area_km2: np.ndarray
    forecast_area_km2: np.ndarray
    truth_max_intensity: np.ndarray
    forecast_max_intensity: np.ndarray
    truth_mean_intensity: np.ndarray
    forecast_mean_intensity: np.ndarray
    truth_centroid_lat: np.ndarray
    truth_centroid_lon: np.ndarray
    forecast_centroid_lat: np.ndarray
    forecast_centroid_lon: np.ndarray
    ti_score: np.ndarray


def _read_diagnostic_metadata(path: str) -> _DiagnosticMetadata:
    with netCDF4.Dataset(path, "r") as ds:
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

        time_index = np.asarray(ds.variables["match_time_index"][:]) if "match_time_index" in ds.variables else None
        category = np.asarray(ds.variables["category"][:], dtype=str)

        def _f(name):
            return np.asarray(ds.variables[name][:], dtype=np.float64)

        return _DiagnosticMetadata(
            time_index=time_index, valid_times=valid_times, init_time=init_time,
            init_time_from_filename=init_time_from_filename, category=category,
            truth_area_km2=_f("truth_area_km2"), forecast_area_km2=_f("forecast_area_km2"),
            truth_max_intensity=_f("truth_max_intensity"), forecast_max_intensity=_f("forecast_max_intensity"),
            truth_mean_intensity=_f("truth_mean_intensity"), forecast_mean_intensity=_f("forecast_mean_intensity"),
            truth_centroid_lat=_f("truth_centroid_lat"), truth_centroid_lon=_f("truth_centroid_lon"),
            forecast_centroid_lat=_f("forecast_centroid_lat"), forecast_centroid_lon=_f("forecast_centroid_lon"),
            ti_score=_f("ti_score"),
        )


def build_diagnostic_distribution_climatology(
    match_dir: str,
    match_pattern: str = "match_init_*.nc",
) -> DiagnosticDistributionClimatology:
    match_files = _discover(match_dir, match_pattern)

    area_ratio: dict = {}
    max_intensity_diff: dict = {}
    mean_intensity_diff: dict = {}
    centroid_dist_km: dict = {}
    ti_score: dict = {}
    n_hits_per_lead_hour: dict = {}
    n_files_used_stored_init_time = 0
    n_files_used_filename_init_time = 0
    lead_hours_seen: set = set()

    for match_file in match_files:
        meta = _read_diagnostic_metadata(match_file)
        if meta.init_time is None:
            raise ValueError(f"'{match_file}' has no init_time (neither a stored attribute nor a parseable filename)")
        if meta.time_index is None:
            raise ValueError(f"'{match_file}' is missing match_time_index -- expected an init_snapshot match file")
        if meta.init_time_from_filename:
            n_files_used_filename_init_time += 1
        else:
            n_files_used_stored_init_time += 1

        hit_mask = meta.category == "hit"

        lead_hour_by_time_idx = {ti: (vt - meta.init_time).total_seconds() / 3600.0 for ti, vt in enumerate(meta.valid_times)}
        for lh in lead_hour_by_time_idx.values():
            lead_hours_seen.add(lh)

        centroid_dist = _haversine_km(meta.truth_centroid_lat, meta.truth_centroid_lon,
                                       meta.forecast_centroid_lat, meta.forecast_centroid_lon)

        for ti, lead_hour in lead_hour_by_time_idx.items():
            mask = hit_mask & (meta.time_index == ti)
            n = int(np.sum(mask))
            if n == 0:
                continue
            n_hits_per_lead_hour[lead_hour] = n_hits_per_lead_hour.get(lead_hour, 0) + n
            area_ratio.setdefault(lead_hour, []).extend((meta.forecast_area_km2[mask] / meta.truth_area_km2[mask]).tolist())
            max_intensity_diff.setdefault(lead_hour, []).extend((meta.forecast_max_intensity[mask] - meta.truth_max_intensity[mask]).tolist())
            mean_intensity_diff.setdefault(lead_hour, []).extend((meta.forecast_mean_intensity[mask] - meta.truth_mean_intensity[mask]).tolist())
            centroid_dist_km.setdefault(lead_hour, []).extend(centroid_dist[mask].tolist())
            ti_score.setdefault(lead_hour, []).extend(meta.ti_score[mask].tolist())

    return DiagnosticDistributionClimatology(
        lead_hours=sorted(lead_hours_seen),
        area_ratio=area_ratio,
        max_intensity_diff=max_intensity_diff,
        mean_intensity_diff=mean_intensity_diff,
        centroid_dist_km=centroid_dist_km,
        ti_score=ti_score,
        n_hits_per_lead_hour=n_hits_per_lead_hour,
        n_files_used_stored_init_time=n_files_used_stored_init_time,
        n_files_used_filename_init_time=n_files_used_filename_init_time,
    )


# Storm-mode-filtered variant -- the SAME pooling as
# build_diagnostic_distribution_climatology() above, restricted to "hit"
# rows of one storm mode. Simpler than the contingency-table storm-mode
# split (aggregate_matches_by_lead_hour.build_match_lead_hour_climatology_by_storm_mode):
# a "hit" row ALWAYS has a real truth object (never a false_alarm-style
# row with no truth side at all), so the storm mode is unambiguously
# truth_is_linear -- no truth/forecast-keyed asymmetry needed here.

@dataclass
class _DiagnosticMetadataWithStormMode(_DiagnosticMetadata):
    truth_is_linear: np.ndarray = None  # int; -1 sentinel where absent


def _read_diagnostic_metadata_with_storm_mode(path: str) -> _DiagnosticMetadataWithStormMode:
    base = _read_diagnostic_metadata(path)
    with netCDF4.Dataset(path, "r") as ds:
        if "truth_is_linear" in ds.variables:
            truth_is_linear = np.asarray(ds.variables["truth_is_linear"][:], dtype=int)
        else:
            # Backward compatibility: a file predating storm_mode_classification
            # has no truth_is_linear at all -- every record excluded from
            # every mode, same convention as the contingency-table sibling.
            truth_is_linear = np.full(len(base.category), -1, dtype=int)
    return _DiagnosticMetadataWithStormMode(**{**vars(base), "truth_is_linear": truth_is_linear})


def build_diagnostic_distribution_climatology_by_storm_mode(
    match_dir: str, storm_mode: int, match_pattern: str = "match_init_*.nc",
) -> DiagnosticDistributionClimatology:
    """Same pooling as build_diagnostic_distribution_climatology(),
    restricted to "hit" rows whose truth_is_linear equals `storm_mode`
    (0=cellular, 1=mixed, 2=linear). A match file predating
    storm_mode_classification degrades to an all-empty climatology for
    every mode (truth_is_linear read as the -1 sentinel), not an error --
    same convention as this module's contingency-table sibling."""
    match_files = _discover(match_dir, match_pattern)

    area_ratio: dict = {}
    max_intensity_diff: dict = {}
    mean_intensity_diff: dict = {}
    centroid_dist_km: dict = {}
    ti_score: dict = {}
    n_hits_per_lead_hour: dict = {}
    n_files_used_stored_init_time = 0
    n_files_used_filename_init_time = 0
    lead_hours_seen: set = set()

    for match_file in match_files:
        meta = _read_diagnostic_metadata_with_storm_mode(match_file)
        if meta.init_time is None:
            raise ValueError(f"'{match_file}' has no init_time (neither a stored attribute nor a parseable filename)")
        if meta.time_index is None:
            raise ValueError(f"'{match_file}' is missing match_time_index -- expected an init_snapshot match file")
        if meta.init_time_from_filename:
            n_files_used_filename_init_time += 1
        else:
            n_files_used_stored_init_time += 1

        hit_mask = (meta.category == "hit") & (meta.truth_is_linear == storm_mode)

        lead_hour_by_time_idx = {ti: (vt - meta.init_time).total_seconds() / 3600.0 for ti, vt in enumerate(meta.valid_times)}
        for lh in lead_hour_by_time_idx.values():
            lead_hours_seen.add(lh)

        centroid_dist = _haversine_km(meta.truth_centroid_lat, meta.truth_centroid_lon,
                                       meta.forecast_centroid_lat, meta.forecast_centroid_lon)

        for ti, lead_hour in lead_hour_by_time_idx.items():
            mask = hit_mask & (meta.time_index == ti)
            n = int(np.sum(mask))
            if n == 0:
                continue
            n_hits_per_lead_hour[lead_hour] = n_hits_per_lead_hour.get(lead_hour, 0) + n
            area_ratio.setdefault(lead_hour, []).extend((meta.forecast_area_km2[mask] / meta.truth_area_km2[mask]).tolist())
            max_intensity_diff.setdefault(lead_hour, []).extend((meta.forecast_max_intensity[mask] - meta.truth_max_intensity[mask]).tolist())
            mean_intensity_diff.setdefault(lead_hour, []).extend((meta.forecast_mean_intensity[mask] - meta.truth_mean_intensity[mask]).tolist())
            centroid_dist_km.setdefault(lead_hour, []).extend(centroid_dist[mask].tolist())
            ti_score.setdefault(lead_hour, []).extend(meta.ti_score[mask].tolist())

    return DiagnosticDistributionClimatology(
        lead_hours=sorted(lead_hours_seen),
        area_ratio=area_ratio,
        max_intensity_diff=max_intensity_diff,
        mean_intensity_diff=mean_intensity_diff,
        centroid_dist_km=centroid_dist_km,
        ti_score=ti_score,
        n_hits_per_lead_hour=n_hits_per_lead_hour,
        n_files_used_stored_init_time=n_files_used_stored_init_time,
        n_files_used_filename_init_time=n_files_used_filename_init_time,
    )
