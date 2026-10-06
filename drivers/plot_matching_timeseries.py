"""Generic, config-driven driver: N-system (or N-matching-threshold)
comparison plots built from object-matching output -- POD/FAR/FBIAS/CSI
(whichever the config's `metrics:` list names), matched-pair diagnostic-
property distributions (area ratio, intensity differences, centroid
distance, Total Interest), object-level stats (count/area by lead time,
area/mean-intensity boxplots), performance diagrams (case-scatter per
lead-time-range period, lead-time trajectory), two-system CSI-difference
diagnostics (colorbar performance diagram + POD/FAR/FBIAS/CSI difference
histograms), and single-case-vs-baseline highlighting.

One driver + one config file per comparison scenario -- the `systems:`
mechanism is identical whether "systems" means different forecast systems
(config_sfe_3system.yaml) or different matching-threshold variants of one
system (config_mpas_thresholds.yaml).

Before pooling, always runs the shared-forecast-dataset check
(py_obj_plotting.multi_system.check_shared_forecast_cases) -- under the
config's require_shared_cases=True (the default), raises a clear, named
error listing exactly which forecast cases are mismatched across the
systems being compared, rather than silently pooling a biased subset.
require_shared_cases=False proceeds using only the cases shared by every
system, still printing the same report.

Tick spacing/display unit (minutes vs. hours) is derived automatically from
each plot's own actual lead-time range
(plots.lead_hour_timeseries.adaptive_lead_hour_xticks) -- no per-scenario
xtick_fn choice needed.

Two entry points: run_one_case(config_path) for a FLAT, single-analysis
config file; run_suite(config_path) for a SUITE config file (a top-level
`analyses:` list) -- both call the exact same per-analysis logic
(_run_from_config), so every field/feature behaves identically regardless
of which file shape it's reached through.

Run with:
  /opt/anaconda3/envs/pysteps_env/bin/python py_obj_plotting/drivers/plot_matching_timeseries.py <config.yaml> [--output-dir=<path>]
"""

import os
import re
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

from py_obj_plotting.climatology import (
    build_ci_age_difference_distribution,
    build_ci_skill_climatology,
    build_diagnostic_distribution_climatology_by_storm_mode,
    build_match_lead_hour_climatology_by_storm_mode,
    build_multi_system_diagnostic_climatology,
    build_multi_system_match_climatology,
    build_multi_system_object_climatology,
    build_multi_system_object_climatology_by_storm_mode,
    build_object_property_pod_climatology,
    build_per_case_match_climatologies,
    compute_verification_stats,
    count_tracking_restarts,
    split_baseline_and_highlight,
)
from py_obj_plotting.config import STORM_MODE_TO_INT, load_config, load_suite_config
from py_obj_plotting.multi_system import check_shared_forecast_cases, load_systems
from py_obj_plotting.plots import (
    plot_area_ratio_comparison_timeseries,
    plot_centroid_dist_comparison_timeseries,
    plot_ci_age_difference_histogram,
    plot_csi_comparison_timeseries,
    plot_far_comparison_timeseries,
    plot_fbias_comparison_timeseries,
    plot_max_intensity_diff_comparison_timeseries,
    plot_mean_intensity_diff_comparison_timeseries,
    plot_object_area_boxplot,
    plot_object_area_comparison_timeseries,
    plot_object_count_comparison_timeseries,
    plot_object_mean_intensity_boxplot,
    plot_pod_comparison_timeseries,
    plot_ti_score_comparison_timeseries,
)
from py_obj_plotting.plots.lead_hour_timeseries import adaptive_lead_hour_xticks
from py_obj_plotting.plots.object_property_heatmap import (
    plot_object_property_pod_diff_heatmap,
    plot_object_property_pod_heatmap,
)
from py_obj_plotting.plots.performance_diagram import (
    plot_case_scatter_performance_diagram,
    plot_lead_time_trajectory_performance_diagram,
    plot_two_system_case_diff_performance_diagram,
)
from py_obj_plotting.plots.single_case_highlight import (
    plot_single_case_highlight_case_scatter,
    plot_single_case_highlight_timeseries,
)
from py_obj_plotting.plots.two_system_diff_histograms import plot_two_system_diff_histograms
from py_obj_plotting.systems import assign_colors

_HEATMAP_Y_LABELS = {
    "area": "Truth Object Area (km$^2$)",
    "max_intensity": "Truth Object Max Intensity (dBZ)",
    "age": "Truth Object Age Relative to Init (s)",
}

_METRIC_PLOTS = {
    "pod": (plot_pod_comparison_timeseries, "pod_comparison.png"),
    "far": (plot_far_comparison_timeseries, "far_comparison.png"),
    "fbias": (plot_fbias_comparison_timeseries, "fbias_comparison.png"),
    "csi": (plot_csi_comparison_timeseries, "csi_comparison.png"),
}
_METRIC_TITLES = {
    "pod": "POD vs. lead hour",
    "far": "FAR vs. lead hour",
    "fbias": "Frequency Bias vs. lead hour",
    "csi": "CSI vs. lead hour",
}
_DIAGNOSTIC_PLOTS = [
    (plot_area_ratio_comparison_timeseries, "area_ratio_comparison.png", "Matched-pair area ratio vs. lead hour"),
    (plot_max_intensity_diff_comparison_timeseries, "max_intensity_diff_comparison.png", "Matched-pair max intensity difference vs. lead hour"),
    (plot_mean_intensity_diff_comparison_timeseries, "mean_intensity_diff_comparison.png", "Matched-pair mean intensity difference vs. lead hour"),
    (plot_centroid_dist_comparison_timeseries, "centroid_dist_comparison.png", "Matched-pair centroid distance vs. lead hour"),
    (plot_ti_score_comparison_timeseries, "ti_score_comparison.png", "Matched-pair Total Interest vs. lead hour"),
]
_OBJECT_TIMESERIES_PLOTS = [
    (plot_object_count_comparison_timeseries, "object_count_comparison.png"),
    (plot_object_area_comparison_timeseries, "object_area_comparison.png"),
]
_OBJECT_TIMESERIES_TITLES = {
    "object_count_comparison.png": "Total object count vs. lead hour",
    "object_area_comparison.png": "Total summed object area vs. lead hour",
}
_OBJECT_BOXPLOTS = [
    (plot_object_area_boxplot, "object_area_boxplot.png"),
    (plot_object_mean_intensity_boxplot, "object_mean_intensity_boxplot.png"),
]
_TIMESERIES_YLABELS = {"pod": "Probability of Detection", "far": "False Alarm Ratio",
                        "fbias": "Frequency Bias", "csi": "Critical Success Index"}
_TIMESERIES_YLIMS = {"pod": (0.0, 1.0), "far": (0.0, 1.0), "fbias": None, "csi": (0.0, 1.0)}


def _slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")


def _with_suffix(fname: str, suffix: str) -> str:
    """'pod_comparison.png' + 'cellular' -> 'pod_comparison_cellular.png' --
    keeps storm-mode output files uniquely named in the same flat
    output_dir as everything else."""
    base, ext = os.path.splitext(fname)
    return f"{base}_{suffix}{ext}"


def _case_scatter_case_grouping(cfg) -> str:
    if cfg.performance_diagrams is not None and cfg.performance_diagrams.case_scatter is not None:
        return cfg.performance_diagrams.case_scatter.case_grouping
    return "init_time"


def _case_scatter_day_start_hour(cfg) -> int:
    if cfg.performance_diagrams is not None and cfg.performance_diagrams.case_scatter is not None:
        return cfg.performance_diagrams.case_scatter.day_start_hour
    return 0


def _resolve_two_system_periods(cfg, explicit_periods, context: str) -> list:
    if explicit_periods is not None:
        return explicit_periods
    if cfg.performance_diagrams is not None and cfg.performance_diagrams.case_scatter is not None:
        return cfg.performance_diagrams.case_scatter.periods
    raise ValueError(
        f"{context}: two_system_comparison has no periods of its own, and performance_diagrams.case_scatter "
        "is absent -- set one or the other"
    )


def _report_tracking_restarts(systems, cache: dict) -> None:
    """Prints, per system, how many likely truth-tracking restarts fall inside
    its forecast windows (see climatology.count_tracking_restarts). Report
    only -- these objects are counted as new storms by the CI and age
    analyses; nothing is corrected. Cached so it runs at most once per system."""
    for s in systems:
        if s.name not in cache:
            cache[s.name] = count_tracking_restarts(s.match_dir, match_pattern=s.match_pattern)
        r = cache[s.name]
        if r.n_flagged_slices == 0:
            print(f"  {s.name}: no truth-tracking restarts detected inside forecast windows")
            continue
        times = ", ".join(t.strftime("%Y-%m-%d %H:%MZ") for t in r.flagged_valid_times)
        print(f"  {s.name}: WARNING {r.n_flagged_slices} (case, valid time) slice(s) look like truth-tracking "
              f"restarts (every truth object age 0), {r.n_truth_objects_reset} truth object(s) reset -- "
              f"counted as new storms by the CI/age analyses (not corrected). Valid times: {times}")


def _run_single_case_highlight(cfg) -> dict:
    sch = cfg.single_case_highlight
    system_spec = next(s for s in cfg.systems if s.name == sch.system)
    print(f"Single-case-highlight mode: system={sch.system!r}, case={sch.case!r}")

    case_grouping = _case_scatter_case_grouping(cfg)
    per_case = build_per_case_match_climatologies(
        [system_spec], lead_time_mode=cfg.lead_time.mode, lead_minutes=cfg.lead_time.lead_minutes,
        case_grouping=case_grouping, day_start_hour=_case_scatter_day_start_hour(cfg),
    )[sch.system]
    print(f"  {len(per_case)} case(s) available for {sch.system!r} (case_grouping={case_grouping!r})")

    baseline_clims, highlighted_clim = split_baseline_and_highlight(per_case, sch.case, sch.baseline_cases)
    print(f"  {len(baseline_clims)} baseline case(s), highlighting {sch.case!r}")

    baseline_stats = [
        compute_verification_stats(c, bootstrap_samples=cfg.bootstrap.samples, ci_interval=cfg.bootstrap.ci_interval)
        for c in baseline_clims
    ]
    highlighted_stats = compute_verification_stats(
        highlighted_clim, bootstrap_samples=cfg.bootstrap.samples, ci_interval=cfg.bootstrap.ci_interval,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    written = []

    for metric in cfg.metrics:
        out_path = os.path.join(cfg.output_dir, f"{metric}_highlight.png")
        plot_single_case_highlight_timeseries(
            baseline_stats, highlighted_stats, metric, out_path,
            ylabel=_TIMESERIES_YLABELS[metric], title=f"{metric.upper()}: {sch.case} vs. other cases",
            highlight_color=sch.highlight_color, highlight_label=sch.case,
            ylim=_TIMESERIES_YLIMS[metric], xtick_fn=adaptive_lead_hour_xticks,
        )
        written.append(out_path)

    if cfg.performance_diagrams is not None:
        pd = cfg.performance_diagrams
        if pd.case_scatter is not None:
            for period in pd.case_scatter.periods:
                out_path = os.path.join(cfg.output_dir, f"case_scatter_highlight_{_slugify(period.name)}.png")
                plot_single_case_highlight_case_scatter(
                    baseline_clims, highlighted_clim, period.lead_minutes_range, out_path,
                    highlight_color=sch.highlight_color, highlight_label=sch.case,
                    marker_size_basis=pd.case_scatter.marker_size_basis, title=f"Case scatter: {period.name}",
                )
                written.append(out_path)
        # Note: pd.lead_time_trajectory (the N-system trajectory diagram)
        # has no single-case-highlight equivalent -- that variant was
        # removed as unnecessary; ignored here even if configured.

    print()
    for p in written:
        print(f"wrote {p}")

    return {
        "config": cfg,
        "per_case_climatologies": {sch.system: per_case},
        "baseline_climatologies": baseline_clims,
        "highlighted_climatology": highlighted_clim,
        "baseline_stats": baseline_stats,
        "highlighted_stats": highlighted_stats,
        "written_paths": written,
    }


def _run_comparison(cfg) -> dict:
    systems = assign_colors(cfg.systems)
    system_names = [s.name for s in systems]
    colors = {s.name: s.color for s in systems}

    print(f"Systems: {system_names}")
    loaded = load_systems(systems)

    case_report = check_shared_forecast_cases(loaded, require_shared_cases=cfg.require_shared_cases)
    n_all_cases = len(set().union(*(set(ls.init_times) for ls in loaded.values())))
    print(f"Shared forecast cases: {len(case_report.shared_cases)}/{n_all_cases}")
    for name, cases in case_report.system_only_cases.items():
        if cases:
            print(f"  {name}-only cases (no other system has them): {len(cases)}")
    if case_report.partial_cases:
        print(f"  {len(case_report.partial_cases)} case(s) present in only some systems")

    print(f"\nBuilding multi-system match climatology (lead_time_mode={cfg.lead_time.mode!r})...")
    match_clims, avail_report = build_multi_system_match_climatology(
        systems, lead_time_mode=cfg.lead_time.mode, lead_minutes=cfg.lead_time.lead_minutes,
    )
    for name, clim in match_clims.items():
        print(f"  {name}: {len(clim.lead_hours)} lead time(s), members={clim.members}")
    if avail_report.excluded_partial_lead_minutes:
        print(f"  {len(avail_report.excluded_partial_lead_minutes)} (cycle, lead) combination(s) excluded "
              f"(partial availability under intersection mode)")

    print("\nComputing verification stats "
          f"({'/'.join(m.upper() for m in cfg.metrics)}, {cfg.bootstrap.ci_interval:.0f}% bootstrap CI)...")
    verification_stats = {
        name: compute_verification_stats(clim, bootstrap_samples=cfg.bootstrap.samples, ci_interval=cfg.bootstrap.ci_interval)
        for name, clim in match_clims.items()
    }

    print("Building multi-system diagnostic-distribution climatology...")
    diag_clims, _ = build_multi_system_diagnostic_climatology(
        systems, lead_time_mode=cfg.lead_time.mode, lead_minutes=cfg.lead_time.lead_minutes,
    )
    print("Building multi-system object-stats climatology...")
    object_clims, _ = build_multi_system_object_climatology(
        systems, lead_time_mode=cfg.lead_time.mode, lead_minutes=cfg.lead_time.lead_minutes,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    verification_series = [(n, colors[n], verification_stats[n]) for n in system_names]
    diagnostic_series = [(n, colors[n], diag_clims[n]) for n in system_names]
    object_series = [(n, colors[n], object_clims[n]) for n in system_names]

    written = []
    for metric in cfg.metrics:
        plot_fn, fname = _METRIC_PLOTS[metric]
        out_path = os.path.join(cfg.output_dir, fname)
        plot_fn(verification_series, out_path, xtick_fn=adaptive_lead_hour_xticks)
        written.append(out_path)

    for plot_fn, fname, _title in _DIAGNOSTIC_PLOTS:
        out_path = os.path.join(cfg.output_dir, fname)
        plot_fn(diagnostic_series, out_path, xtick_fn=adaptive_lead_hour_xticks)
        written.append(out_path)

    for plot_fn, fname in _OBJECT_TIMESERIES_PLOTS:
        out_path = os.path.join(cfg.output_dir, fname)
        plot_fn(object_series, out_path, xtick_fn=adaptive_lead_hour_xticks)
        written.append(out_path)

    for plot_fn, fname in _OBJECT_BOXPLOTS:
        out_path = os.path.join(cfg.output_dir, fname)
        plot_fn(object_series, out_path)
        written.append(out_path)

    if cfg.storm_mode_timeseries is not None:
        smt = cfg.storm_mode_timeseries
        print(f"\nBuilding storm-mode-filtered match climatologies (modes={smt.modes})...")
        for mode_name in smt.modes:
            mode_int = STORM_MODE_TO_INT[mode_name]
            mode_label = mode_name.capitalize()

            mode_clims = {
                s.name: build_match_lead_hour_climatology_by_storm_mode(s.match_dir, storm_mode=mode_int, match_pattern=s.match_pattern)
                for s in systems
            }
            for name, clim in mode_clims.items():
                n_total = sum(clim.n_hits.values()) + sum(clim.n_misses.values()) + sum(clim.n_false_alarms.values())
                print(f"  {name} ({mode_name}): {n_total} total hit/miss/false_alarm record(s)")
                if n_total == 0:
                    print(f"    WARNING: zero records for {name!r}/{mode_name!r} -- check storm_mode_classification was enabled at identification time")

            mode_stats = {
                name: compute_verification_stats(clim, bootstrap_samples=cfg.bootstrap.samples, ci_interval=cfg.bootstrap.ci_interval)
                for name, clim in mode_clims.items()
            }
            mode_series = [(n, colors[n], mode_stats[n]) for n in system_names]
            for metric in cfg.metrics:
                plot_fn, fname = _METRIC_PLOTS[metric]
                out_path = os.path.join(cfg.output_dir, _with_suffix(fname, mode_name))
                plot_fn(mode_series, out_path, xtick_fn=adaptive_lead_hour_xticks,
                        title=f"{_METRIC_TITLES[metric]} ({mode_label} storm mode)")
                written.append(out_path)

            # Full diagnostic-property family too (area ratio, intensity
            # diffs, centroid distance, TI score), hit-pair-only.
            mode_diag_clims = {
                s.name: build_diagnostic_distribution_climatology_by_storm_mode(s.match_dir, storm_mode=mode_int, match_pattern=s.match_pattern)
                for s in systems
            }
            for name, clim in mode_diag_clims.items():
                n_total = sum(clim.n_hits_per_lead_hour.values())
                print(f"  {name} ({mode_name}) diagnostic hit-pairs: {n_total}")
            mode_diag_series = [(n, colors[n], mode_diag_clims[n]) for n in system_names]
            for plot_fn, fname, title in _DIAGNOSTIC_PLOTS:
                out_path = os.path.join(cfg.output_dir, _with_suffix(fname, mode_name))
                plot_fn(mode_diag_series, out_path, xtick_fn=adaptive_lead_hour_xticks,
                        title=f"{title} ({mode_label} storm mode)")
                written.append(out_path)

            # Object count/total area by storm mode, including the MRMS
            # truth-side reference line for objects of the same storm mode
            # (not just the forecast side), via
            # build_multi_system_object_climatology_by_storm_mode's
            # shared-truth-series mechanism.
            mode_object_clims, _ = build_multi_system_object_climatology_by_storm_mode(
                systems, storm_mode=mode_int, lead_time_mode=cfg.lead_time.mode, lead_minutes=cfg.lead_time.lead_minutes,
            )
            for name, clim in mode_object_clims.items():
                n_fc = sum(clim.forecast_count.values())
                n_truth = sum(clim.truth_count.values())
                print(f"  {name} ({mode_name}) objects: {n_fc} forecast, {n_truth} truth (MRMS)")
                if n_fc == 0 and n_truth == 0:
                    print(f"    WARNING: zero objects for {name!r}/{mode_name!r} -- check storm_mode_classification was enabled at identification time")
            mode_object_series = [(n, colors[n], mode_object_clims[n]) for n in system_names]
            for plot_fn, fname in _OBJECT_TIMESERIES_PLOTS:
                out_path = os.path.join(cfg.output_dir, _with_suffix(fname, mode_name))
                plot_fn(mode_object_series, out_path, xtick_fn=adaptive_lead_hour_xticks,
                        title=f"{_OBJECT_TIMESERIES_TITLES[fname]} ({mode_label} storm mode)")
                written.append(out_path)

    restart_reports: dict = {}
    if cfg.convection_initiation is not None:
        ci_cfg = cfg.convection_initiation
        print("\nChecking for truth-tracking restarts inside forecast windows (affects CI)...")
        _report_tracking_restarts(systems, restart_reports)
        print(f"\nBuilding convection-initiation (CI) skill climatologies "
              f"(age_diff_threshold_minutes={ci_cfg.age_diff_threshold_minutes})...")
        ci_clims = {}
        for s in systems:
            clim = build_ci_skill_climatology(
                s.match_dir, age_diff_threshold_minutes=ci_cfg.age_diff_threshold_minutes,
                match_pattern=s.match_pattern, include_near_miss_candidates=ci_cfg.include_near_miss_candidates,
            )
            ci_clims[s.name] = clim
            n_total = sum(clim.n_hits.values()) + sum(clim.n_misses.values()) + sum(clim.n_false_alarms.values())
            print(f"  {s.name}: {n_total} CI-candidate hit/miss/false_alarm record(s)")
            if n_total == 0:
                print(f"    WARNING: zero CI records for {s.name!r} -- check the FORECAST series "
                      "was tracked (model.track: true) at identification time, not just the truth series")

        ci_stats = {
            name: compute_verification_stats(clim, bootstrap_samples=cfg.bootstrap.samples, ci_interval=cfg.bootstrap.ci_interval)
            for name, clim in ci_clims.items()
        }
        ci_series = [(n, colors[n], ci_stats[n]) for n in system_names]
        for metric in cfg.metrics:
            plot_fn, fname = _METRIC_PLOTS[metric]
            out_path = os.path.join(cfg.output_dir, _with_suffix(fname, "ci"))
            plot_fn(ci_series, out_path, xtick_fn=adaptive_lead_hour_xticks,
                    title=f"{_METRIC_TITLES[metric]} (convection initiation)")
            written.append(out_path)

        print("Building CI age-difference distributions...")
        for s in systems:
            dist = build_ci_age_difference_distribution(s.match_dir, match_pattern=s.match_pattern)
            out_path = os.path.join(cfg.output_dir, f"ci_age_diff_histogram_{_slugify(s.name)}.png")
            n = plot_ci_age_difference_histogram(
                dist, out_path, age_diff_threshold_minutes=ci_cfg.age_diff_threshold_minutes,
                title=f"{s.name}: CI timing-error distribution", color=colors[s.name],
            )
            print(f"  {s.name}: {n} matched CI pair(s) -- wrote {out_path}")
            written.append(out_path)

    per_case_climatologies = None
    if cfg.performance_diagrams is not None or cfg.two_system_comparison is not None:
        case_grouping = _case_scatter_case_grouping(cfg)
        print(f"\nBuilding per-case match climatologies (case_grouping={case_grouping!r})...")
        per_case_climatologies = build_per_case_match_climatologies(
            systems, lead_time_mode=cfg.lead_time.mode, lead_minutes=cfg.lead_time.lead_minutes,
            case_grouping=case_grouping, day_start_hour=_case_scatter_day_start_hour(cfg),
        )
        for name, cases in per_case_climatologies.items():
            print(f"  {name}: {len(cases)} case(s)")

    if cfg.performance_diagrams is not None:
        pd = cfg.performance_diagrams
        if pd.case_scatter is not None:
            for period in pd.case_scatter.periods:
                cs_series = [(n, colors[n], per_case_climatologies[n]) for n in system_names]
                out_path = os.path.join(cfg.output_dir, f"case_scatter_{_slugify(period.name)}.png")
                case_indices = plot_case_scatter_performance_diagram(
                    cs_series, period.lead_minutes_range, out_path,
                    marker_size_basis=pd.case_scatter.marker_size_basis, title=f"Case scatter: {period.name}",
                )
                index_to_label = {i: label for label, i in case_indices.items()}
                print(f"  case scatter ({period.name}) index -> case_label: "
                      f"{ {i: index_to_label[i] for i in sorted(index_to_label)} }")
                written.append(out_path)
        if pd.lead_time_trajectory is not None:
            traj_series = [(n, colors[n], verification_stats[n]) for n in system_names]
            out_path = os.path.join(cfg.output_dir, "lead_time_trajectory.png")
            plot_lead_time_trajectory_performance_diagram(
                traj_series, out_path, label_interval_minutes=pd.lead_time_trajectory.label_interval_minutes,
                title="Lead-time trajectory",
            )
            written.append(out_path)

    if cfg.two_system_comparison is not None:
        tsc = cfg.two_system_comparison
        name_a, name_b = system_names[0], system_names[1]
        system_a = (name_a, colors[name_a], per_case_climatologies[name_a])
        system_b = (name_b, colors[name_b], per_case_climatologies[name_b])

        cs_periods = _resolve_two_system_periods(cfg, tsc.case_scatter_periods, "two_system_comparison.case_scatter_periods")
        for period in cs_periods:
            out_path = os.path.join(cfg.output_dir, f"two_system_diff_{_slugify(period.name)}.png")
            report = plot_two_system_case_diff_performance_diagram(
                system_a, system_b, period.lead_minutes_range, out_path,
                colormap=tsc.colormap, diff_levels=tsc.diff_levels, title=f"CSI difference: {period.name}",
            )
            print(f"  two-system diff ({period.name}): {len(report.matched_cases)} matched case(s)")
            written.append(out_path)

        hist_periods = _resolve_two_system_periods(cfg, tsc.histogram_periods, "two_system_comparison.histogram_periods")
        for period in hist_periods:
            filename_prefix = f"diff_histogram_{_slugify(period.name)}_"
            report = plot_two_system_diff_histograms(
                system_a, system_b, period.lead_minutes_range, cfg.output_dir,
                title_prefix=period.name, filename_prefix=filename_prefix,
            )
            print(f"  diff histograms ({period.name}): {len(report.matched_cases)} matched case(s)")
            written.extend(
                os.path.join(cfg.output_dir, f) for f in sorted(os.listdir(cfg.output_dir))
                if f.startswith(filename_prefix)
            )

    if cfg.heatmaps is not None:
        hm = cfg.heatmaps
        bin_edges = {p: np.asarray(v, dtype=float) for p, v in hm.bin_edges().items()}
        print(f"\nBuilding object-property POD climatologies (properties={list(bin_edges)})...")
        if "age" in bin_edges:
            print("  truth-tracking restarts (affect the age heatmaps):")
            _report_tracking_restarts(systems, restart_reports)
        heatmap_clims = {}
        for s in systems:
            clim = build_object_property_pod_climatology(s.match_dir, bin_edges=bin_edges, match_pattern=s.match_pattern)
            heatmap_clims[s.name] = clim
            for prop in bin_edges:
                n_total = sum(clim.n_hits[prop].values()) + sum(clim.n_misses[prop].values())
                note = " (tracking likely wasn't enabled for the truth series)" if prop == "age" and n_total == 0 else ""
                print(f"  {s.name} [{prop}]: {n_total} hit/miss record(s) with real data{note}")

        for name in system_names:
            clim = heatmap_clims[name]
            for prop in bin_edges:
                if sum(clim.n_hits[prop].values()) + sum(clim.n_misses[prop].values()) == 0:
                    print(f"  skipping heatmap_{prop}_{_slugify(name)}.png -- no real '{prop}' data for {name!r}")
                    continue
                out_path = os.path.join(cfg.output_dir, f"heatmap_{prop}_{_slugify(name)}.png")
                plot_object_property_pod_heatmap(clim, prop, _HEATMAP_Y_LABELS[prop], out_path, title=f"{name}: POD by {prop}")
                written.append(out_path)

        if cfg.two_system_comparison is not None and len(systems) == 2:
            name_a, name_b = system_names[0], system_names[1]
            clim_a, clim_b = heatmap_clims[name_a], heatmap_clims[name_b]
            for prop in bin_edges:
                total_a = sum(clim_a.n_hits[prop].values()) + sum(clim_a.n_misses[prop].values())
                total_b = sum(clim_b.n_hits[prop].values()) + sum(clim_b.n_misses[prop].values())
                if total_a == 0 or total_b == 0:
                    print(f"  skipping heatmap_{prop}_diff.png -- insufficient '{prop}' data ({name_a}={total_a}, {name_b}={total_b})")
                    continue
                out_path = os.path.join(cfg.output_dir, f"heatmap_{prop}_diff.png")
                plot_object_property_pod_diff_heatmap(
                    clim_a, clim_b, prop, _HEATMAP_Y_LABELS[prop], out_path,
                    ci_samples=cfg.bootstrap.samples, ci_interval=cfg.bootstrap.ci_interval,
                    difference_label=f"{name_a} - {name_b}", title=f"POD difference by {prop}: {name_a} - {name_b}",
                )
                written.append(out_path)

    print()
    for p in written:
        print(f"wrote {p}")

    return {
        "config": cfg,
        "case_report": case_report,
        "availability_report": avail_report,
        "match_climatologies": match_clims,
        "verification_stats": verification_stats,
        "diagnostic_climatologies": diag_clims,
        "object_climatologies": object_clims,
        "per_case_climatologies": per_case_climatologies,
        "written_paths": written,
    }


def _run_from_config(cfg) -> dict:
    if cfg.single_case_highlight is not None:
        return _run_single_case_highlight(cfg)
    return _run_comparison(cfg)


def run_one_case(config_path: str, output_dir_override: str | None = None) -> dict:
    """Runs a FLAT, single-analysis config file (no top-level `analyses:`
    list). `output_dir_override`, if given, replaces the config's own
    output_dir -- lets a caller redirect output without editing the YAML."""
    cfg = load_config(config_path)
    if output_dir_override is not None:
        cfg.output_dir = output_dir_override
    return _run_from_config(cfg)


def run_suite(config_path: str, output_dir_override: str | None = None) -> dict:
    """Runs a SUITE config file (a top-level `analyses:` list, sharing one
    `systems:` registry) -- one call to _run_from_config() per analysis,
    the exact same per-analysis logic run_one_case() uses for a flat file.
    `output_dir_override`, if given, replaces every analysis' own
    output_dir with `<output_dir_override>/<analysis_name>` so the
    analyses stay distinguishable. Returns {analysis_name: result_dict}."""
    suite = load_suite_config(config_path)
    results = {}
    for name, cfg in suite.items():
        print(f"\n=== Analysis: {name} ===")
        if output_dir_override is not None:
            cfg.output_dir = os.path.join(output_dir_override, name)
        results[name] = _run_from_config(cfg)
    return results


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--output-dir")]
    output_dir_override = None
    for a in sys.argv[1:]:
        if a.startswith("--output-dir="):
            output_dir_override = a.split("=", 1)[1]
    if len(args) != 1:
        print("Usage: plot_matching_timeseries.py <config.yaml> [--output-dir=<path>]")
        sys.exit(1)
    run_one_case(args[0], output_dir_override=output_dir_override)


if __name__ == "__main__":
    main()
