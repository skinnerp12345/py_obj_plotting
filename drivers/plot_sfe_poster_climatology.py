"""Standalone script: composite reflectivity climatology comparison across
WoFS, WoFSCast, and NowcastNet against MRMS, restricted to samples where all
three model systems share the SAME forecast initialization AND the SAME
valid time (not just any valid time from any cycle) -- an apples-to-apples
"same forecast opportunity" comparison -- and MRMS has observational data at
that valid time.

Each model system's true init_time is recovered per slice as
`valid_time - lead_hours` (already-embedded, real per-slice data -- no
filename parsing needed) EXCEPT WoFSCast, whose `lead_hours` is still
uniformly 0.0 for every slice (a separate, known, not-yet-fixed limitation --
see CLAUDE.md's `valid_time_var` entry). For WoFSCast, the true init is
recovered from the file's own anchor (its minimum valid_time, i.e. the whole
cycle's own first output) minus 20 minutes -- a real, confirmed offset (user-
reported and independently verified against real data: WoFS-vlab's real init
17:00 vs. WoFSCast's own anchor 17:20 for the matching real cycle) between
how WoFSCast labels its own initialization and the convention WoFS/NowcastNet
already share with each other (no correction needed between those two,
confirmed: NowcastNet's valid_time - lead_hours correctly recovers its true
init even though its first output lags init by 10 minutes).

This is a one-off analysis script, not a permanent py_obj_plotting
capability -- the matching logic below is deliberately inline here rather
than a new climatology/ module, per explicit request. It still reuses the
already-existing, genuinely reusable plotting layer
(plots.plot_pdf_cdf_per_system/plot_combined_pdf/plot_combined_cdf/
compute_percentile_table, climatology.build_shared_times_summary_table)
unchanged, by constructing the same SharedValidTimesClimatology shape their
existing code already expects.

Run with:
  /opt/anaconda3/envs/pysteps_env/bin/python py_obj_plotting/drivers/plot_sfe_poster_climatology.py [sfe_poster_dir] [out_dir]

Defaults to the real ../sfe_poster/ directory (external, not bundled) and
writes plots to <sfe_poster_dir>/histogram_images/, tables to
<sfe_poster_dir>/ directly, if omitted.
"""

import csv
import glob
import os
import sys
from collections import defaultdict
from datetime import timedelta

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

import numpy as np

from python_obj.histogram import histogram_to_cdf, histogram_to_pdf, read_histogram_file
from py_obj_plotting.climatology import SharedValidTimesClimatology, build_shared_times_summary_table
from py_obj_plotting.plots import compute_percentile_table, plot_combined_cdf, plot_combined_pdf, plot_pdf_cdf_per_system

MRMS_NAME = "MRMS"
MODEL_DIR_NAMES = {
    "WoFS": "wofs_vlab_histograms",
    "WoFSCast": "wofscast_histograms",
    "NowcastNet": "nowcastnet_histograms",
}
ENSEMBLE_SYSTEMS = {"WoFS", "WoFSCast"}
MRMS_DIR_NAME = "mrms_histograms"
THRESHOLDS = (40.0, 45.0)
WOFSCAST_INIT_TIME_CORRECTION_MINUTES = 20  # WoFSCast's own file anchor is this much LATER
                                             # than the true, WoFS/NowcastNet-comparable init time
                                             # (real, user-confirmed and independently re-verified offset)


def _discover(dir_path: str, pattern: str = "**/*.nc") -> list:
    files = sorted(glob.glob(os.path.join(dir_path, pattern), recursive=True))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found under '{dir_path}'")
    return files


def _true_init_times_for_file(path: str, contents, needs_wofscast_correction: bool) -> list:
    """Returns a list of true_init_time, index-aligned with contents.slices.

    needs_wofscast_correction=True: one value for the whole file (uniform
    across the cycle) -- the file's own anchor (min valid_time, i.e. the
    cycle's first output) minus WOFSCAST_INIT_TIME_CORRECTION_MINUTES, since
    WoFSCast's lead_hours is still uniformly 0.0 (broken, unusable here).

    needs_wofscast_correction=False: one value per slice, computed directly
    from that slice's own real lead_hours (valid_time - lead_hours) -- no
    filename/anchor guessing needed, this already recovers the true init
    time exactly (confirmed real: NowcastNet's first output lags its own
    init by 10 minutes, and this formula still recovers the correct init)."""
    if needs_wofscast_correction:
        anchor = min(sl.valid_time for sl in contents.slices)
        true_init = anchor - timedelta(minutes=WOFSCAST_INIT_TIME_CORRECTION_MINUTES)
        return [true_init] * len(contents.slices)

    result = []
    for sl in contents.slices:
        if sl.lead_hours is None:
            raise ValueError(f"'{path}' has a slice with no lead_hours -- cannot recover its true init_time")
        result.append(sl.valid_time - timedelta(hours=sl.lead_hours))
    return result


def _build_matched_init_valid_climatology(
    model_dirs: dict, ensemble_systems: set, mrms_dir: str, mrms_name: str = MRMS_NAME,
) -> SharedValidTimesClimatology:
    mrms_files = _discover(mrms_dir)
    mrms_contents = [(f, read_histogram_file(f)) for f in mrms_files]
    bins = mrms_contents[0][1].bins
    mrms_valid_times = {sl.valid_time for _, c in mrms_contents for sl in c.slices}

    # --- per-system: {(true_init_time, valid_time): {member_id: hist}} ----
    system_by_key: dict = {}
    system_contents: dict = {}
    for name, d in model_dirs.items():
        files = _discover(d)
        contents = [(f, read_histogram_file(f)) for f in files]
        for f, c in contents:
            if not np.array_equal(c.bins, bins):
                raise ValueError(f"'{f}' ({name}) has different bin edges than {mrms_name} -- cannot aggregate")
        system_contents[name] = contents

        by_key: dict = defaultdict(dict)
        for f, c in contents:
            true_inits = _true_init_times_for_file(f, c, needs_wofscast_correction=(name == "WoFSCast"))
            for sl, true_init in zip(c.slices, true_inits):
                by_key[(true_init, sl.valid_time)][sl.member_id] = sl.hist
        system_by_key[name] = by_key

    # --- (init, valid) pairs present in every model system, AND whose
    # valid_time MRMS also has (MRMS has no init_time concept) -----------
    model_names = list(model_dirs.keys())
    common_keys = set(system_by_key[model_names[0]].keys())
    for name in model_names[1:]:
        common_keys &= set(system_by_key[name].keys())
    matched_keys = {k for k in common_keys if k[1] in mrms_valid_times}
    if not matched_keys:
        raise ValueError(
            "No (init_time, valid_time) pair is shared by all model systems with a matching MRMS "
            "valid_time -- cannot build a matched comparison."
        )
    shared_valid_times = sorted({k[1] for k in matched_keys})
    print(f"Matched (init_time, valid_time) pairs across all 3 model systems + MRMS: {len(matched_keys)} "
          f"({len(shared_valid_times)} distinct valid times)")

    # --- MRMS aggregation over the matched valid times ---------------------
    mrms_total = np.zeros(len(bins) - 1, dtype=np.int64)
    mrms_contributing: dict = defaultdict(set)
    for f, c in mrms_contents:
        for sl in c.slices:
            if sl.valid_time in shared_valid_times:
                mrms_total += sl.hist
                mrms_contributing[sl.valid_time.date()].add(f)

    system_names = [mrms_name] + model_names
    pdf = {mrms_name: histogram_to_pdf(bins, mrms_total)}
    cdf = {mrms_name: histogram_to_cdf(bins, mrms_total)}
    member_pdf: dict = {}
    contributing_files_by_date = {mrms_name: dict(mrms_contributing)}

    for name in model_names:
        by_key = system_by_key[name]
        contents = system_contents[name]

        contributing: dict = defaultdict(set)
        for f, c in contents:
            true_inits = _true_init_times_for_file(f, c, needs_wofscast_correction=(name == "WoFSCast"))
            for sl, true_init in zip(c.slices, true_inits):
                if (true_init, sl.valid_time) in matched_keys:
                    contributing[sl.valid_time.date()].add(f)
        contributing_files_by_date[name] = dict(contributing)

        if name in ensemble_systems:
            member_ids = sorted({m for k in matched_keys for m in by_key[k].keys()})
            per_member_pdf = {}
            for member in member_ids:
                total = np.zeros(len(bins) - 1, dtype=np.int64)
                for k in matched_keys:
                    h = by_key[k].get(member)
                    if h is not None:
                        total += h
                if total.sum() == 0:
                    raise ValueError(f"'{name}' member '{member}' has zero counts at the matched (init, valid) pairs")
                per_member_pdf[member] = histogram_to_pdf(bins, total)
            member_pdf[name] = per_member_pdf
            pdf[name] = np.mean([per_member_pdf[m] for m in member_ids], axis=0)
            cdf[name] = np.cumsum(pdf[name])
        else:
            total = np.zeros(len(bins) - 1, dtype=np.int64)
            for k in matched_keys:
                for h in by_key[k].values():
                    total += h
            pdf[name] = histogram_to_pdf(bins, total)
            cdf[name] = histogram_to_cdf(bins, total)

    return SharedValidTimesClimatology(
        bins=bins,
        shared_valid_times=shared_valid_times,
        system_names=system_names,
        ensemble_systems=set(ensemble_systems),
        pdf=pdf,
        cdf=cdf,
        member_pdf=member_pdf,
        contributing_files_by_date=contributing_files_by_date,
    )


def _write_csv(rows: list, out_path: str) -> None:
    if not rows:
        raise ValueError(f"_write_csv: no rows to write for '{out_path}'")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_table(rows: list, title: str) -> None:
    print(f"\n{title}")
    if not rows:
        return
    headers = list(rows[0].keys())
    widths = [max(len(h), max(len(f"{r[h]:.2f}" if isinstance(r[h], float) else str(r[h])) for r in rows)) for h in headers]
    print("  " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    for r in rows:
        cells = [f"{r[h]:.2f}" if isinstance(r[h], float) else str(r[h]) for h in headers]
        print("  " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)))


def run_one_case(sfe_poster_dir: str, out_dir: str) -> dict:
    mrms_dir = os.path.join(sfe_poster_dir, MRMS_DIR_NAME)
    model_dirs = {name: os.path.join(sfe_poster_dir, dirname) for name, dirname in MODEL_DIR_NAMES.items()}

    print(f"Building matched-init-and-valid-time climatology: MRMS='{mrms_dir}', models={model_dirs}")
    clim = _build_matched_init_valid_climatology(
        model_dirs=model_dirs, ensemble_systems=ENSEMBLE_SYSTEMS, mrms_dir=mrms_dir, mrms_name=MRMS_NAME,
    )
    print(f"Found {len(clim.shared_valid_times)} matched valid times across all 4 sources "
          f"({clim.shared_valid_times[0]} to {clim.shared_valid_times[-1]})")

    os.makedirs(out_dir, exist_ok=True)
    per_system_paths = plot_pdf_cdf_per_system(clim, out_dir)
    combined_pdf_path = os.path.join(out_dir, "pdf_combined.png")
    combined_cdf_path = os.path.join(out_dir, "cdf_combined.png")
    combined_cdf_zoom_path = os.path.join(out_dir, "cdf_combined_30_70dbz.png")
    plot_combined_pdf(clim, combined_pdf_path)
    plot_combined_cdf(clim, combined_cdf_path, thresholds=THRESHOLDS)
    plot_combined_cdf(clim, combined_cdf_zoom_path, thresholds=THRESHOLDS, xlim=(30.0, 70.0), ylim=(0.9, 1.0))
    for p in per_system_paths + [combined_pdf_path, combined_cdf_path, combined_cdf_zoom_path]:
        print(f"wrote {p}")

    percentile_rows = compute_percentile_table(clim, thresholds=THRESHOLDS, mrms_name=MRMS_NAME)
    summary_rows = build_shared_times_summary_table(clim)

    _print_table(percentile_rows, "Matched-percentile threshold table (MRMS vs. each system):")
    _print_table(summary_rows, "Shared-times summary table (per date):")

    percentile_csv_path = os.path.join(sfe_poster_dir, "percentile_table.csv")
    summary_csv_path = os.path.join(sfe_poster_dir, "shared_times_summary.csv")
    _write_csv(percentile_rows, percentile_csv_path)
    _write_csv(summary_rows, summary_csv_path)
    print(f"\nwrote {percentile_csv_path}")
    print(f"wrote {summary_csv_path}")

    return {
        "climatology": clim,
        "percentile_rows": percentile_rows,
        "summary_rows": summary_rows,
        "per_system_plot_paths": per_system_paths,
        "combined_pdf_path": combined_pdf_path,
        "combined_cdf_path": combined_cdf_path,
        "combined_cdf_zoom_path": combined_cdf_zoom_path,
        "percentile_csv_path": percentile_csv_path,
        "summary_csv_path": summary_csv_path,
    }


def main():
    sfe_poster_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_REPO_ROOT, "sfe_poster")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(sfe_poster_dir, "histogram_images")

    run_one_case(sfe_poster_dir, out_dir)


if __name__ == "__main__":
    main()
