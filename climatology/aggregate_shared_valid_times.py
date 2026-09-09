"""Core (non-plotting) aggregation logic: builds one reflectivity
climatology per source (MRMS + N model systems), restricted to valid times
where every source has data -- a fair, apples-to-apples comparison across
sources with different, non-identical output cadences (confirmed real:
MRMS/WoFS ~5-min, WoFSCast/NowcastNet ~10-min).

For ensemble sources (member_id populated per slice), builds one
climatology per member first, then the source's own climatology is the
per-bin MEAN of its members' individual PDFs -- not a sum of member counts
then one normalization, so every member is weighted equally regardless of
how much data it happened to contribute (explicit, confirmed choice).

Reuses python_obj's own histogram I/O/aggregation math unchanged
(python_obj.histogram.read_histogram_file/histogram_to_pdf/histogram_to_cdf)
-- this package builds on python_obj, it never modifies it. Sums directly
from already-loaded in-memory slices rather than python_obj.histogram.
sum_histograms() -- an ensemble source needs one pass per member, and
sum_histograms() re-reads every file from disk per call, which would mean
re-reading each file once per member (confirmed real cost: a real 18-member
WoFS source would re-read ~200 files 18 times each).
"""

import glob
import os
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from python_obj.histogram import histogram_to_cdf, histogram_to_pdf, read_histogram_file


@dataclass
class SharedValidTimesClimatology:
    bins: np.ndarray  # bin edges, shared across every source (verified equal)
    shared_valid_times: list  # sorted list[datetime], present in every source
    system_names: list  # e.g. ["MRMS", "WoFS", "WoFSCast", "NowcastNet"] -- MRMS first by convention
    ensemble_systems: set  # subset of system_names built as an ensemble mean of member PDFs
    pdf: dict  # system_name -> np.ndarray (ensemble-mean PDF for ensemble sources, plain PDF otherwise)
    cdf: dict  # system_name -> np.ndarray
    member_pdf: dict  # system_name -> {member_id: np.ndarray}, only present for ensemble_systems
    contributing_files_by_date: dict = field(default_factory=dict)
    # system_name -> {date: set(file_path)} -- every file that contributed at
    # least one shared-valid-time slice, needed for the "forecast
    # initializations" summary table (one file == one forecast cycle for
    # every real source in this project's own established convention).


def _discover(dir_path: str, pattern: str) -> list:
    files = sorted(glob.glob(os.path.join(dir_path, pattern), recursive=True))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found under '{dir_path}'")
    return files


def _read_all(files: list) -> list:
    return [(f, read_histogram_file(f)) for f in files]


def _valid_time_set(contents_list: list) -> set:
    s = set()
    for _, c in contents_list:
        s.update(sl.valid_time for sl in c.slices)
    return s


def build_shared_valid_times_climatology(
    model_dirs: dict,
    ensemble_systems: set,
    mrms_dir: str,
    mrms_name: str = "MRMS",
    pattern: str = "**/*.nc",
) -> SharedValidTimesClimatology:
    """model_dirs: {system_name: directory} for every non-MRMS source, each a
    directory of already-built hist_model_*.nc files (produced by
    python_obj.drivers.build_histogram_model). mrms_dir: directory of
    hist_mrms_*.nc files (produced by python_obj.drivers.build_histogram_mrms).

    ensemble_systems: subset of model_dirs' keys whose slices carry a real
    member_id (e.g. an 18-member WoFS-style ensemble) -- these get a
    per-member climatology built first, then averaged; every other source
    (including MRMS, which is never an ensemble) gets one direct climatology.

    Raises if no valid_time is shared by every source, or if any source's
    bin edges differ from MRMS's -- summing/comparing histograms built with
    different bins would be meaningless, not silently reconciled.
    """
    mrms_files = _discover(mrms_dir, pattern)
    mrms_contents = _read_all(mrms_files)
    bins = mrms_contents[0][1].bins

    system_contents = {mrms_name: mrms_contents}
    valid_time_sets = {mrms_name: _valid_time_set(mrms_contents)}

    for name, d in model_dirs.items():
        files = _discover(d, pattern)
        contents = _read_all(files)
        for f, c in contents:
            if not np.array_equal(c.bins, bins):
                raise ValueError(
                    f"'{f}' ({name}) has different bin edges than {mrms_name} -- "
                    "cannot aggregate onto a shared bin axis"
                )
        system_contents[name] = contents
        valid_time_sets[name] = _valid_time_set(contents)

    shared = set.intersection(*valid_time_sets.values())
    if not shared:
        raise ValueError(
            "No valid_time is shared by every source -- cannot build a comparison climatology. "
            f"Per-source distinct valid_time counts: {({k: len(v) for k, v in valid_time_sets.items()})}"
        )
    shared_sorted = sorted(shared)

    system_names = [mrms_name] + list(model_dirs.keys())
    pdf: dict = {}
    cdf: dict = {}
    member_pdf: dict = {}
    contributing_files_by_date: dict = {}

    for name in system_names:
        contents = system_contents[name]

        contributing: dict = defaultdict(set)
        for f, c in contents:
            for sl in c.slices:
                if sl.valid_time in shared:
                    contributing[sl.valid_time.date()].add(f)
        contributing_files_by_date[name] = dict(contributing)

        if name in ensemble_systems:
            member_ids = sorted({sl.member_id for _, c in contents for sl in c.slices})
            per_member_pdf = {}
            for member in member_ids:
                total = np.zeros(len(bins) - 1, dtype=np.int64)
                for f, c in contents:
                    for sl in c.slices:
                        if sl.valid_time in shared and sl.member_id == member:
                            total += sl.hist
                if total.sum() == 0:
                    raise ValueError(f"'{name}' member '{member}' has zero counts at the shared valid times")
                per_member_pdf[member] = histogram_to_pdf(bins, total)
            member_pdf[name] = per_member_pdf
            pdf[name] = np.mean([per_member_pdf[m] for m in member_ids], axis=0)
            cdf[name] = np.cumsum(pdf[name])
        else:
            total = np.zeros(len(bins) - 1, dtype=np.int64)
            for f, c in contents:
                for sl in c.slices:
                    if sl.valid_time in shared:
                        total += sl.hist
            pdf[name] = histogram_to_pdf(bins, total)
            cdf[name] = histogram_to_cdf(bins, total)

    return SharedValidTimesClimatology(
        bins=bins,
        shared_valid_times=shared_sorted,
        system_names=system_names,
        ensemble_systems=set(ensemble_systems),
        pdf=pdf,
        cdf=cdf,
        member_pdf=member_pdf,
        contributing_files_by_date=contributing_files_by_date,
    )


def build_shared_times_summary_table(clim: SharedValidTimesClimatology) -> list:
    """One row per date with at least one shared valid time, plus a final
    TOTAL row. Columns: date, n_shared_valid_times, and one
    '{system}_n_initializations' column per source -- the count of distinct
    contributing files (== distinct forecast cycles) on that date. MRMS gets
    a count too (its own per-day file, not a "forecast" in the same sense,
    but tracked the same way for a uniform table shape).

    The TOTAL row's per-system count is the number of DISTINCT files across
    every date (a set union), not a sum of the per-date counts -- a real
    forecast cycle can span two calendar dates (confirmed real: a WoFSCast
    cycle's valid times can run from one evening into the next morning), so
    summing per-date counts would double-count that one file."""
    dates = sorted({vt.date() for vt in clim.shared_valid_times})
    rows = []
    total_shared = 0
    for d in dates:
        n_shared = sum(1 for vt in clim.shared_valid_times if vt.date() == d)
        row = {"date": d.isoformat(), "n_shared_valid_times": n_shared}
        for name in clim.system_names:
            n_init = len(clim.contributing_files_by_date.get(name, {}).get(d, set()))
            row[f"{name}_n_initializations"] = n_init
        total_shared += n_shared
        rows.append(row)

    total_row = {"date": "TOTAL", "n_shared_valid_times": total_shared}
    for name in clim.system_names:
        all_files = set()
        for files_for_date in clim.contributing_files_by_date.get(name, {}).values():
            all_files |= files_for_date
        total_row[f"{name}_n_initializations"] = len(all_files)
    rows.append(total_row)
    return rows
