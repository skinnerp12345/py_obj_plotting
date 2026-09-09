"""Standalone script: composite reflectivity climatology comparison between
a multi-member MPAS ensemble and matching MRMS observations, aggregated by
forecast lead hour across every available spring 2023/2024/2025 case.

For each forecast lead hour (0-132h), pools every MPAS case's histogram at
that lead hour (per member) and the corresponding MRMS observation (matched
to each case's own real valid time, not just hour-of-day) into one running
climatological distribution. Then computes, per member and per lead hour,
the MPAS dBZ value at the same percentile as MRMS's own 40/45 dBZ threshold
in the pooled MRMS distribution -- and separately, the percentile MRMS's own
40/45 dBZ threshold occurs at in that pooled distribution.

Plain arguments, not a YAML config -- this is one specific analysis, not
(yet) a generalized reusable pipeline stage; see py_obj_plotting/README.md.

Run with:
  /opt/anaconda3/envs/pysteps_env/bin/python py_obj_plotting/drivers/plot_mpas_vs_mrms_climatology.py [mpas_dir] [mrms_dir] [out_dir]

Defaults to the real bundled ncar_mpas_analysis/ directories if omitted.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

from py_obj_plotting.climatology import build_lead_hour_climatology
from py_obj_plotting.plots import (
    average_over_lead_hour_range,
    compute_ensemble_mean_series,
    compute_matched_percentiles,
    plot_matched_value_timeseries,
    plot_source_percentile_timeseries,
    range_over_lead_hour_range,
)

THRESHOLDS = (40.0, 45.0)
VALID_TIME_TOLERANCE_MINUTES = 10.0
SPIN_UP_HOURS = 12.0  # excluded from the full-period ensemble mean summary
DAILY_WINDOW_START_HOUR = 18.0  # first recurring 6h window: 18-24h, then +24h each day
DAILY_WINDOW_WIDTH_HOURS = 6.0
DAILY_WINDOW_STEP_HOURS = 24.0


def run_one_case(mpas_dir: str, mrms_dir: str, out_dir: str) -> dict:
    print(f"Building lead-hour climatology from '{mpas_dir}' (MPAS) and '{mrms_dir}' (MRMS)")
    clim = build_lead_hour_climatology(
        mpas_dir, mrms_dir, valid_time_tolerance_minutes=VALID_TIME_TOLERANCE_MINUTES,
    )
    print(f"Found {len(clim.lead_hours)} lead hours ({min(clim.lead_hours):.0f}-{max(clim.lead_hours):.0f}h), "
          f"{len(clim.members)} members: {clim.members}")

    n_cases = sorted(set(clim.n_cases_per_lead_hour.values()))
    if len(n_cases) == 1:
        print(f"Contributing MPAS cases per lead hour: {n_cases[0]} (uniform across all lead hours)")
    else:
        print(f"Contributing MPAS cases per lead hour varies: min={min(n_cases)}, max={max(n_cases)}")

    n_matched = sorted(set(clim.n_mrms_matched_per_lead_hour.get(lh, 0) for lh in clim.lead_hours))
    if len(n_matched) == 1:
        print(f"MRMS-matched cases per lead hour: {n_matched[0]} (uniform across all lead hours)")
    else:
        print(f"MRMS-matched cases per lead hour varies: min={min(n_matched)}, max={max(n_matched)}")
    print(f"Total unmatched (case, lead_hour) pairs (no MRMS within {VALID_TIME_TOLERANCE_MINUTES:.0f} min): {clim.n_unmatched}")
    if clim.unmatched:
        for mpas_file, lead_hour, valid_time in clim.unmatched[:10]:
            print(f"  unmatched: {os.path.basename(mpas_file)} lead={lead_hour:.0f}h valid={valid_time}")
        if len(clim.unmatched) > 10:
            print(f"  ... and {len(clim.unmatched) - 10} more")

    matched_value, source_percentile = compute_matched_percentiles(clim, thresholds=THRESHOLDS)
    ensemble_mean = compute_ensemble_mean_series(clim, matched_value)

    max_lead_hour = max(clim.lead_hours)

    print(f"\nEnsemble mean matched dBZ value, {SPIN_UP_HOURS:.0f}-{max_lead_hour:.0f}h "
          f"(full forecast minus {SPIN_UP_HOURS:.0f}h spin-up):")
    for t in THRESHOLDS:
        mean_val = average_over_lead_hour_range(ensemble_mean[t], clim.lead_hours, SPIN_UP_HOURS, max_lead_hour)
        print(f"  {t:.0f} dBZ threshold -> {mean_val:.2f} dBZ")

    print(f"\nMRMS source percentile (member-independent), {SPIN_UP_HOURS:.0f}-{max_lead_hour:.0f}h "
          f"(full forecast minus {SPIN_UP_HOURS:.0f}h spin-up):")
    for t in THRESHOLDS:
        mean_pct = average_over_lead_hour_range(source_percentile[t], clim.lead_hours, SPIN_UP_HOURS, max_lead_hour)
        pct_min, pct_max = range_over_lead_hour_range(source_percentile[t], clim.lead_hours, SPIN_UP_HOURS, max_lead_hour)
        print(f"  {t:.0f} dBZ threshold -> mean {mean_pct:.3f}%, range [{pct_min:.3f}%, {pct_max:.3f}%]")

    print(f"\nEnsemble mean matched dBZ value, recurring {DAILY_WINDOW_WIDTH_HOURS:.0f}h daily windows "
          f"starting at {DAILY_WINDOW_START_HOUR:.0f}h (every {DAILY_WINDOW_STEP_HOURS:.0f}h):")
    window_start = DAILY_WINDOW_START_HOUR
    while window_start + DAILY_WINDOW_WIDTH_HOURS <= max_lead_hour:
        window_end = window_start + DAILY_WINDOW_WIDTH_HOURS
        print(f"  {window_start:.0f}-{window_end:.0f}h:")
        for t in THRESHOLDS:
            mean_val = average_over_lead_hour_range(ensemble_mean[t], clim.lead_hours, window_start, window_end)
            print(f"    {t:.0f} dBZ threshold -> {mean_val:.2f} dBZ")
        window_start += DAILY_WINDOW_STEP_HOURS

    os.makedirs(out_dir, exist_ok=True)
    matched_value_path = os.path.join(out_dir, "matched_value_by_lead_hour.png")
    source_percentile_path = os.path.join(out_dir, "mrms_source_percentile_by_lead_hour.png")
    plot_matched_value_timeseries(clim, matched_value, matched_value_path, thresholds=THRESHOLDS)
    plot_source_percentile_timeseries(clim, source_percentile, source_percentile_path, thresholds=THRESHOLDS)
    print(f"wrote {matched_value_path}")
    print(f"wrote {source_percentile_path}")

    return {
        "climatology": clim,
        "matched_value": matched_value,
        "source_percentile": source_percentile,
        "ensemble_mean": ensemble_mean,
        "matched_value_path": matched_value_path,
        "source_percentile_path": source_percentile_path,
    }


def main():
    mpas_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_REPO_ROOT, "ncar_mpas_analysis", "mpas_histograms")
    mrms_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_REPO_ROOT, "ncar_mpas_analysis", "mrms_histogram")
    out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(_THIS_DIR, "..", "output", "mpas_vs_mrms_climatology")

    run_one_case(mpas_dir, mrms_dir, out_dir)


if __name__ == "__main__":
    main()
