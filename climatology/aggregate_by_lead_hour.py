"""Core (non-plotting) aggregation logic: builds a per-forecast-lead-hour
composite reflectivity climatology from many MPAS ensemble forecast cases
and matching MRMS observations.

For each forecast lead hour, pools every available MPAS case's histogram
into one running distribution per (member, lead_hour), and separately pools
the corresponding MRMS observation -- matched to each case's own real valid
time (init_time + lead_hour), not just hour-of-day -- into one running
distribution per lead_hour. This lets a downstream caller compute, e.g.,
"what MPAS dBZ value corresponds to the same percentile as MRMS's 40 dBZ,
pooled across every spring-2023-2025 case, specifically at lead hour 24"
for every lead hour and every member.

Reuses python_obj's own histogram I/O and time-matching helpers unchanged
(python_obj.histogram.read_histogram_file, python_obj.time_utils.
nearest_within_tolerance) -- this package builds on python_obj, it never
modifies it.
"""

import glob
import os
from dataclasses import dataclass

import numpy as np

from python_obj.histogram import read_histogram_file

from .case_time_matching import resolve_case_matches


@dataclass
class LeadHourClimatology:
    bins: np.ndarray  # bin edges, shared across MPAS and MRMS (verified equal)
    lead_hours: list  # sorted unique lead hours found across all MPAS cases
    members: list  # sorted unique member ids found across all MPAS cases
    mpas_hist: dict  # (member, lead_hour) -> summed histogram (np.ndarray)
    mrms_hist: dict  # lead_hour -> summed, valid-time-matched histogram (np.ndarray)
    n_cases_per_lead_hour: dict  # lead_hour -> number of MPAS case files contributing
    n_mrms_matched_per_lead_hour: dict  # lead_hour -> number of cases whose MRMS match was found
    unmatched: list  # list of (mpas_file, lead_hour, valid_time) with no MRMS match within tolerance

    @property
    def n_unmatched(self) -> int:
        return len(self.unmatched)


def _discover(dir_path: str, pattern: str) -> list:
    files = sorted(glob.glob(os.path.join(dir_path, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found under '{dir_path}'")
    return files


def _load_all_mrms_slices(mrms_files: list):
    """Flattens every MRMS file's own slices into one time-sorted list --
    needed because a case's large lead hour can push its valid_time into a
    different MRMS day-file than its own init date (confirmed a real
    occurrence in this project's WoFS/MRMS work), so matching must search
    across every MRMS file, not just the one matching the case's own date."""
    bins = None
    hist_by_time = {}
    for f in mrms_files:
        contents = read_histogram_file(f)
        if bins is None:
            bins = contents.bins
        elif not np.array_equal(bins, contents.bins):
            raise ValueError(f"'{f}' has different bin edges than earlier MRMS files -- cannot aggregate")
        for s in contents.slices:
            hist_by_time[s.valid_time] = s.hist
    valid_times = sorted(hist_by_time.keys())
    return bins, valid_times, hist_by_time


def build_lead_hour_climatology(
    mpas_dir: str,
    mrms_dir: str,
    mpas_pattern: str = "hist_model_*.nc",
    mrms_pattern: str = "hist_mrms_*.nc",
    valid_time_tolerance_minutes: float = 10.0,
) -> LeadHourClimatology:
    """mpas_dir/mrms_dir: directories of already-built histogram files (one
    hist_model_*.nc per MPAS forecast case, one hist_mrms_*.nc per MRMS day)
    -- produced by python_obj.drivers.build_histogram_model/build_histogram_mrms,
    read here unchanged via python_obj.histogram.read_histogram_file.

    valid_time_tolerance_minutes: max allowed mismatch between an MPAS
    slice's own valid_time and the nearest available MRMS slice's valid_time
    for them to be considered a match; a case/lead_hour with no match within
    tolerance is skipped and reported in `unmatched`, never silently dropped.
    """
    mpas_files = _discover(mpas_dir, mpas_pattern)
    mrms_files = _discover(mrms_dir, mrms_pattern)

    mrms_bins, mrms_valid_times, mrms_hist_by_time = _load_all_mrms_slices(mrms_files)

    bins = None
    mpas_hist: dict = {}
    mrms_hist: dict = {}
    n_cases_per_lead_hour: dict = {}
    n_mrms_matched_per_lead_hour: dict = {}
    unmatched = []
    members_seen = set()
    lead_hours_seen = set()

    for mpas_file in mpas_files:
        contents = read_histogram_file(mpas_file)
        if bins is None:
            bins = contents.bins
            if not np.array_equal(bins, mrms_bins):
                raise ValueError("MPAS bins and MRMS bins differ -- cannot aggregate onto a shared bin axis")
        elif not np.array_equal(bins, contents.bins):
            raise ValueError(f"'{mpas_file}' has different bin edges than earlier MPAS files -- cannot aggregate")

        for s in contents.slices:
            if s.member_id is None or s.lead_hours is None:
                raise ValueError(
                    f"'{mpas_file}' has a slice with missing member_id/lead_hours -- "
                    "expected a full ensemble/lead-time series"
                )
            members_seen.add(s.member_id)
            lead_hours_seen.add(s.lead_hours)

            key = (s.member_id, s.lead_hours)
            if key not in mpas_hist:
                mpas_hist[key] = np.zeros_like(s.hist)
            mpas_hist[key] += s.hist

        # De-duplicate the MRMS match per (case, lead_hour): every member of
        # one case shares the exact same valid_time at a given lead_hour
        # (confirmed on real data), so the MRMS lookup/accumulation happens
        # only once per lead_hour per case -- counting it once per member
        # instead would inflate MRMS's climatological weight by the member
        # count for no reason (the observation doesn't depend on the model
        # member being compared against it). Shared with the object-based
        # climatology via case_time_matching.resolve_case_matches().
        case_valid_times_by_lead_hour = {s.lead_hours: s.valid_time for s in contents.slices}
        matches = resolve_case_matches(case_valid_times_by_lead_hour, mrms_valid_times, valid_time_tolerance_minutes)

        for lead_hour, matched_vt in matches.items():
            n_cases_per_lead_hour[lead_hour] = n_cases_per_lead_hour.get(lead_hour, 0) + 1
            if matched_vt is None:
                unmatched.append((mpas_file, lead_hour, case_valid_times_by_lead_hour[lead_hour]))
            else:
                if lead_hour not in mrms_hist:
                    mrms_hist[lead_hour] = np.zeros_like(mrms_hist_by_time[matched_vt])
                mrms_hist[lead_hour] += mrms_hist_by_time[matched_vt]
                n_mrms_matched_per_lead_hour[lead_hour] = n_mrms_matched_per_lead_hour.get(lead_hour, 0) + 1

    return LeadHourClimatology(
        bins=bins,
        lead_hours=sorted(lead_hours_seen),
        members=sorted(members_seen),
        mpas_hist=mpas_hist,
        mrms_hist=mrms_hist,
        n_cases_per_lead_hour=n_cases_per_lead_hour,
        n_mrms_matched_per_lead_hour=n_mrms_matched_per_lead_hour,
        unmatched=unmatched,
    )
