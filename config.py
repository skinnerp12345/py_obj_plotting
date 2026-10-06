"""Config layer for the matching-output time-series redesign, mirroring
py_obj.config's own conventions: dataclasses + load_config(path), a
namelist-style YAML file, path-shaped fields resolved relative to the
CONFIG FILE'S OWN directory (not the caller's working directory), clear
named-error validation for a present-but-incomplete section.

Unlike py_obj.config's Config (several independently optional
top-level sections spanning several unrelated driver purposes --
interpolation, object ID, matching, histograms), this config has exactly
one job per analysis -- describing one matching-output time-series
comparison -- so its schema is flat at the top level rather than wrapped
in an optional section.

`systems:` is the one core list -- both "systems being compared" (e.g.
WoFS vs. WoFSCast vs. NowcastNet) and "matching thresholds being compared"
(e.g. ncar_mpas_analysis's 100/200/300/400 km directories) use the
identical mechanism (see systems.SystemSpec); the two real example
configs shipped alongside this module
(config_sfe_3system.yaml, config_mpas_thresholds.yaml) demonstrate both
cases through the same schema.

Two top-level file SHAPES are supported (load_config() detects which):
- FLAT (no top-level `analyses:` key): today's original shape, one file =
  one analysis. `load_config()` returns a single MatchingTimeseriesConfig.
- SUITE (`analyses:` present): the top-level `systems:` becomes a shared
  REGISTRY, and each `analyses:` entry describes one independent analysis,
  selecting a subset of the registry by NAME (or every registered system,
  if it omits its own `systems:`). Use `load_suite_config()` for this
  shape -- it returns {analysis_name: MatchingTimeseriesConfig}, reusing
  the exact same per-analysis parsing as the flat shape (`_parse_analysis`),
  so both shapes support every field identically. This exists because more
  analyses are expected to be added to this suite over time, and a user
  should be able to define the whole suite for one dataset from a single
  file rather than maintaining many near-duplicate flat files.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime

import yaml

from .climatology.contingency_stats import DEFAULT_BOOTSTRAP_SAMPLES, DEFAULT_CI_INTERVAL
from .object_selection import _VALID_MEMBER_MODES, _VALID_TIME_MODES, MemberSelectionConfig, TimeSelectionConfig
from .reflectivity_selection import _VALID_ABS_TIME_MODES, AbsoluteTimeSelectionConfig
from .systems import SystemSpec

_VALID_LEAD_TIME_MODES = ("intersection", "union")
_VALID_METRICS = ("pod", "far", "fbias", "csi")
_VALID_CASE_GROUPINGS = ("init_time", "day")
_VALID_MARKER_SIZE_BASES = ("truth", "forecast")
_VALID_STORM_MODES = ("cellular", "mixed", "linear")
STORM_MODE_TO_INT = {"cellular": 0, "mixed": 1, "linear": 2}  # matches py_obj's is_linear convention
_VALID_SPATIAL_COLOR_BY = ("single", "storm_mode", "age", "category")
_VALID_REFLECTIVITY_SOURCES = ("model", "mrms_native", "mrms_interp")


@dataclass
class LeadTimeConfig:
    mode: str = "intersection"  # or "union" -- see multi_system.compute_included_keys
    # Explicit override; when None (the default), the candidate lead-time
    # axis is auto-discovered as the union of every system's own
    # lead_minutes (see systems.load_system) -- no hand-enumeration needed.
    lead_minutes: list | None = None


@dataclass
class BootstrapConfig:
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES
    ci_interval: float = DEFAULT_CI_INTERVAL


@dataclass
class CaseScatterPeriod:
    name: str
    lead_minutes_range: tuple  # (lo, hi) in MINUTES, inclusive


@dataclass
class CaseScatterConfig:
    case_grouping: str = "init_time"  # or "day" (WoFS-style: bucket same-forecast-day init_times together)
    day_start_hour: int = 0  # 'day' grouping only: UTC hour a forecast day starts (0 = calendar date;
                             # e.g. 12 keeps a WoFS day's 17Z-03Z inits in one case)
    marker_size_basis: str = "truth"  # or "forecast" -- see aggregate_matches_per_case.case_object_count
    periods: list = field(default_factory=list)  # list[CaseScatterPeriod], one case-scatter diagram per entry


@dataclass
class LeadTimeTrajectoryConfig:
    label_interval_minutes: float | None = None  # e.g. 30 for WoFS, 720 for MPAS; None = no interval labels


@dataclass
class PerformanceDiagramsConfig:
    case_scatter: CaseScatterConfig | None = None
    lead_time_trajectory: LeadTimeTrajectoryConfig | None = None


@dataclass
class TwoSystemComparisonConfig:
    """Only valid when an analysis's own `systems:` resolves to EXACTLY 2
    entries -- validated once systems are resolved, see _parse_analysis()."""
    colormap: str = "RdBu_r"
    diff_levels: list | None = None  # None = default np.arange(-0.1, 0.11, 0.01), see plots.performance_diagram
    # None = reuse performance_diagrams.case_scatter.periods for that same analysis;
    # give an explicit list[CaseScatterPeriod] to use a different set just for these two features.
    case_scatter_periods: list | None = None
    histogram_periods: list | None = None


@dataclass
class SingleCaseHighlightConfig:
    """Mutually exclusive with comparing multiple systems within one
    analysis -- 'system' must name exactly one of that analysis's own
    resolved systems."""
    system: str
    case: str
    baseline_cases: list | None = None  # None = every other available case
    highlight_color: tuple | None = None  # None = a sensible bold default (see plots.single_case_highlight)


@dataclass
class HeatmapsConfig:
    """POD-by-object-property-vs-lead-time heatmaps (see
    plots.object_property_heatmap). Each of the 3 properties is
    independently optional -- omitting one skips just that property's
    heatmap. Bin edges are in the property's own natural unit: area_bins
    in km^2, max_intensity_bins in dBZ, age_bins in SECONDS relative to
    init (age_seconds - lead_minutes*60 -- see
    aggregate_object_property_pod's module docstring). age_bins only
    produces real data if the truth series was tracked at identification
    time; a match file without tracking has no real age data at all (see
    py_obj/obj_core/matching.py's age_seconds field), which the
    driver detects and reports rather than silently plotting a blank
    heatmap. When exactly 2 systems are being compared AND
    two_system_comparison: is also present in this analysis, one
    difference heatmap (with bootstrap-significance shading) is produced
    per requested property too, alongside each system's own heatmap."""
    area_bins: list | None = None
    max_intensity_bins: list | None = None
    age_bins: list | None = None

    def bin_edges(self) -> dict:
        out = {}
        if self.area_bins is not None:
            out["area"] = self.area_bins
        if self.max_intensity_bins is not None:
            out["max_intensity"] = self.max_intensity_bins
        if self.age_bins is not None:
            out["age"] = self.age_bins
        return out


@dataclass
class StormModeTimeseriesConfig:
    """The existing POD/FAR/FBIAS/CSI comparison time series (see
    MatchingTimeseriesConfig.metrics), recomputed separately for each
    requested storm mode (see py_obj's is_linear classification:
    0=cellular, 1=mixed, 2=linear) -- one additional bundle of time-series
    plots per mode, written flat into output_dir with the mode name suffixed
    onto each filename. Requires
    storm_mode_classification to have been enabled at identification
    time; a match file without it degrades to an all-zero-count
    climatology for every mode (see
    aggregate_matches_by_lead_hour.build_match_lead_hour_climatology_by_storm_mode's
    own docstring) rather than raising -- the driver reports this rather
    than silently producing empty plots."""
    modes: list = field(default_factory=lambda: list(_VALID_STORM_MODES))


@dataclass
class ConvectionInitiationConfig:
    """Pre-convection-initiation (CI) forecast-skill statistics (see
    climatology.aggregate_ci_skill). A "CI candidate" object on either side
    of a match record is one whose own tracked age is less than the elapsed
    lead time at that valid time (age_seconds < lead_seconds -- "age
    relative to init" < 0, the same transform HeatmapsConfig's age_bins
    already uses); a "hit" row only counts as a genuine CI hit when BOTH
    the truth and forecast objects are themselves CI candidates AND their
    age difference is within age_diff_threshold_minutes -- see the module
    docstring for why both sides (not just the truth side) are required.

    Requires the FORECAST series to have been tracked at identification
    time (model.track: true in py_obj's config), not just the truth series
    -- a match file whose forecast side was never tracked has no real
    forecast_age_seconds data at all (see py_obj/obj_core/matching.py's
    age_seconds field), which the driver detects and reports rather than
    silently producing an all-zero/blank result.

    include_near_miss_candidates (default False, matching HeatmapsConfig's/
    aggregate_object_property_pod.py's own hit/miss-only scope precedent):
    whether truth_extra/forecast_extra rows are folded into the CI
    contingency table's extended (FBIAS-only) terms, or excluded entirely."""
    age_diff_threshold_minutes: float
    include_near_miss_candidates: bool = False


@dataclass
class MatchingTimeseriesConfig:
    systems: list  # list[SystemSpec]
    lead_time: LeadTimeConfig = field(default_factory=LeadTimeConfig)
    metrics: list = field(default_factory=lambda: ["pod", "far", "fbias", "csi"])
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    output_dir: str = "output"
    # Shared-forecast-dataset check: True raises a clear, named error
    # listing exactly which forecast cases are mismatched across the
    # systems being compared; False reports the same mismatch (never
    # silent) and proceeds using only the cases shared by every system.
    require_shared_cases: bool = True
    performance_diagrams: PerformanceDiagramsConfig | None = None
    two_system_comparison: TwoSystemComparisonConfig | None = None
    single_case_highlight: SingleCaseHighlightConfig | None = None
    heatmaps: HeatmapsConfig | None = None
    storm_mode_timeseries: StormModeTimeseriesConfig | None = None
    convection_initiation: ConvectionInitiationConfig | None = None


def _parse_lead_minutes_range(raw, context: str) -> tuple:
    """lo == hi is allowed and means "exactly this one lead minute" (e.g.
    a single-instant period like "24h") -- not just a genuine range."""
    if not (isinstance(raw, (list, tuple)) and len(raw) == 2):
        raise ValueError(f"{context}: lead_minutes_range must be a [lo, hi] pair, got {raw!r}")
    lo, hi = raw
    if lo > hi:
        raise ValueError(f"{context}: lead_minutes_range [lo, hi] requires lo <= hi, got {raw!r}")
    return (lo, hi)


def _parse_case_scatter_periods(raw_periods, context: str) -> list:
    periods = []
    for i, p in enumerate(raw_periods):
        missing = [k for k in ("name", "lead_minutes_range") if k not in p]
        if missing:
            raise ValueError(f"{context}: periods[{i}] is missing required field(s): {', '.join(missing)}")
        periods.append(CaseScatterPeriod(
            name=p["name"],
            lead_minutes_range=_parse_lead_minutes_range(p["lead_minutes_range"], f"{context}: periods[{i}]"),
        ))
    return periods


def _parse_performance_diagrams(raw: dict, context: str) -> PerformanceDiagramsConfig:
    case_scatter = None
    if "case_scatter" in raw:
        cs = raw["case_scatter"] or {}
        case_grouping = cs.get("case_grouping", "init_time")
        if case_grouping not in _VALID_CASE_GROUPINGS:
            raise ValueError(f"{context}: case_scatter.case_grouping={case_grouping!r} is not one of {_VALID_CASE_GROUPINGS}")
        marker_size_basis = cs.get("marker_size_basis", "truth")
        if marker_size_basis not in _VALID_MARKER_SIZE_BASES:
            raise ValueError(f"{context}: case_scatter.marker_size_basis={marker_size_basis!r} is not one of {_VALID_MARKER_SIZE_BASES}")
        day_start_hour = cs.get("day_start_hour", 0)
        if isinstance(day_start_hour, bool) or not isinstance(day_start_hour, int) or not 0 <= day_start_hour <= 23:
            raise ValueError(f"{context}: case_scatter.day_start_hour must be an integer 0-23, got {day_start_hour!r}")
        if day_start_hour != 0 and case_grouping != "day":
            raise ValueError(f"{context}: case_scatter.day_start_hour only applies with case_grouping: day")
        raw_periods = cs.get("periods") or []
        if not raw_periods:
            raise ValueError(f"{context}: case_scatter is present but has no 'periods:' -- at least one is required")
        case_scatter = CaseScatterConfig(
            case_grouping=case_grouping, day_start_hour=day_start_hour, marker_size_basis=marker_size_basis,
            periods=_parse_case_scatter_periods(raw_periods, f"{context}: case_scatter"),
        )

    lead_time_trajectory = None
    if "lead_time_trajectory" in raw:
        ltt = raw["lead_time_trajectory"] or {}
        lead_time_trajectory = LeadTimeTrajectoryConfig(label_interval_minutes=ltt.get("label_interval_minutes"))

    return PerformanceDiagramsConfig(case_scatter=case_scatter, lead_time_trajectory=lead_time_trajectory)


def _parse_two_system_comparison(raw: dict, n_systems: int, context: str) -> TwoSystemComparisonConfig:
    if n_systems != 2:
        raise ValueError(
            f"{context}: two_system_comparison requires exactly 2 systems, but this analysis resolves to {n_systems}"
        )
    diff_levels = raw.get("diff_levels")
    case_scatter_periods = raw.get("case_scatter_periods")
    histogram_periods = raw.get("histogram_periods")
    return TwoSystemComparisonConfig(
        colormap=raw.get("colormap", "RdBu_r"),
        diff_levels=list(diff_levels) if diff_levels is not None else None,
        case_scatter_periods=_parse_case_scatter_periods(case_scatter_periods, f"{context}: two_system_comparison.case_scatter_periods") if case_scatter_periods is not None else None,
        histogram_periods=_parse_case_scatter_periods(histogram_periods, f"{context}: two_system_comparison.histogram_periods") if histogram_periods is not None else None,
    )


def _parse_single_case_highlight(raw: dict, system_names: list, context: str) -> SingleCaseHighlightConfig:
    missing = [k for k in ("system", "case") if k not in raw]
    if missing:
        raise ValueError(f"{context}: single_case_highlight is missing required field(s): {', '.join(missing)}")
    if raw["system"] not in system_names:
        raise ValueError(
            f"{context}: single_case_highlight.system={raw['system']!r} is not among this analysis's own "
            f"systems: {system_names}"
        )
    baseline_cases = raw.get("baseline_cases")
    highlight_color = raw.get("highlight_color")
    return SingleCaseHighlightConfig(
        system=raw["system"], case=raw["case"],
        baseline_cases=list(baseline_cases) if baseline_cases is not None else None,
        highlight_color=tuple(highlight_color) if highlight_color is not None else None,
    )


def _parse_heatmaps(raw: dict, context: str) -> HeatmapsConfig:
    area_bins = raw.get("area_bins")
    max_intensity_bins = raw.get("max_intensity_bins")
    age_bins = raw.get("age_bins")
    if area_bins is None and max_intensity_bins is None and age_bins is None:
        raise ValueError(f"{context}: heatmaps is present but has no area_bins/max_intensity_bins/age_bins -- at least one is required")
    return HeatmapsConfig(
        area_bins=list(area_bins) if area_bins is not None else None,
        max_intensity_bins=list(max_intensity_bins) if max_intensity_bins is not None else None,
        age_bins=list(age_bins) if age_bins is not None else None,
    )


def _parse_storm_mode_timeseries(raw: dict, context: str) -> StormModeTimeseriesConfig:
    modes = list(raw.get("modes", _VALID_STORM_MODES))
    invalid = [m for m in modes if m not in _VALID_STORM_MODES]
    if invalid:
        raise ValueError(f"{context}: storm_mode_timeseries.modes contains invalid entries {invalid}, must be from {_VALID_STORM_MODES}")
    if not modes:
        raise ValueError(f"{context}: storm_mode_timeseries.modes must be non-empty")
    return StormModeTimeseriesConfig(modes=modes)


def _parse_convection_initiation(raw: dict, context: str) -> ConvectionInitiationConfig:
    if "age_diff_threshold_minutes" not in raw:
        raise ValueError(f"{context}: convection_initiation is missing required field: age_diff_threshold_minutes")
    return ConvectionInitiationConfig(
        age_diff_threshold_minutes=raw["age_diff_threshold_minutes"],
        include_near_miss_candidates=raw.get("include_near_miss_candidates", False),
    )


def _parse_systems_list(raw_systems: list, base_dir: str, context: str) -> list:
    systems = []
    seen_names = set()
    for i, s in enumerate(raw_systems):
        missing = [k for k in ("name", "match_dir") if k not in s]
        if missing:
            raise ValueError(f"{context}: systems[{i}] is missing required field(s): {', '.join(missing)}")
        if s["name"] in seen_names:
            raise ValueError(f"{context}: duplicate system name '{s['name']}' in 'systems:' list")
        seen_names.add(s["name"])

        match_dir = s["match_dir"]
        if not os.path.isabs(match_dir):
            match_dir = os.path.normpath(os.path.join(base_dir, match_dir))

        systems.append(SystemSpec(
            name=s["name"],
            match_dir=match_dir,
            color=tuple(s["color"]) if "color" in s else None,
            match_pattern=s.get("match_pattern", "match_init_*.nc"),
        ))
    return systems


def _parse_analysis(raw: dict, systems: list, base_dir: str, context: str) -> MatchingTimeseriesConfig:
    """Parses every field EXCEPT `systems:`/`name:` (already resolved by
    the caller -- load_config() from the flat file's own `systems:` list,
    load_suite_config() from one analysis entry's resolved registry
    subset) -- the one shared implementation both file shapes use, so
    every field behaves identically regardless of which shape reached it.
    """
    lt_section = raw.get("lead_time") or {}
    mode = lt_section.get("mode", "intersection")
    if mode not in _VALID_LEAD_TIME_MODES:
        raise ValueError(f"{context}: lead_time.mode={mode!r} is not one of {_VALID_LEAD_TIME_MODES}")
    lead_minutes_raw = lt_section.get("lead_minutes")
    lead_time = LeadTimeConfig(mode=mode, lead_minutes=list(lead_minutes_raw) if lead_minutes_raw is not None else None)

    metrics = list(raw.get("metrics", ["pod", "far", "fbias", "csi"]))
    invalid_metrics = [m for m in metrics if m not in _VALID_METRICS]
    if invalid_metrics:
        raise ValueError(f"{context}: metrics contains invalid entries {invalid_metrics}, must be from {_VALID_METRICS}")

    bs_section = raw.get("bootstrap") or {}
    bootstrap = BootstrapConfig(
        samples=bs_section.get("samples", DEFAULT_BOOTSTRAP_SAMPLES),
        ci_interval=bs_section.get("ci_interval", DEFAULT_CI_INTERVAL),
    )

    output_dir = raw.get("output_dir", "output")
    if not os.path.isabs(output_dir):
        output_dir = os.path.normpath(os.path.join(base_dir, output_dir))

    performance_diagrams = _parse_performance_diagrams(raw["performance_diagrams"], f"{context}: performance_diagrams") if "performance_diagrams" in raw else None
    two_system_comparison = _parse_two_system_comparison(raw["two_system_comparison"], len(systems), f"{context}: two_system_comparison") if "two_system_comparison" in raw else None
    single_case_highlight = _parse_single_case_highlight(raw["single_case_highlight"], [s.name for s in systems], f"{context}: single_case_highlight") if "single_case_highlight" in raw else None
    heatmaps = _parse_heatmaps(raw["heatmaps"], f"{context}: heatmaps") if "heatmaps" in raw else None
    storm_mode_timeseries = _parse_storm_mode_timeseries(raw["storm_mode_timeseries"] or {}, f"{context}: storm_mode_timeseries") if "storm_mode_timeseries" in raw else None
    convection_initiation = _parse_convection_initiation(raw["convection_initiation"], f"{context}: convection_initiation") if "convection_initiation" in raw else None

    return MatchingTimeseriesConfig(
        systems=systems,
        lead_time=lead_time,
        metrics=metrics,
        bootstrap=bootstrap,
        output_dir=output_dir,
        require_shared_cases=raw.get("require_shared_cases", True),
        performance_diagrams=performance_diagrams,
        two_system_comparison=two_system_comparison,
        single_case_highlight=single_case_highlight,
        heatmaps=heatmaps,
        storm_mode_timeseries=storm_mode_timeseries,
        convection_initiation=convection_initiation,
    )


def load_config(path: str) -> MatchingTimeseriesConfig:
    """Parses the FLAT, single-analysis file shape (no top-level
    `analyses:` key) -- unchanged behavior from before this feature was
    added. Use load_suite_config() for a file with an `analyses:` list."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"'{path}' did not parse into a mapping at the top level")
    if "analyses" in raw:
        raise ValueError(f"'{path}' has an 'analyses:' list -- use load_suite_config() for a multi-analysis suite file")

    base_dir = os.path.dirname(os.path.abspath(path))
    raw_systems = raw.get("systems")
    if not raw_systems:
        raise ValueError(f"'{path}' has no 'systems:' list (or it is empty) -- at least one system is required")

    systems = _parse_systems_list(raw_systems, base_dir, f"'{path}'")
    return _parse_analysis(raw, systems, base_dir, f"'{path}'")


def load_suite_config(path: str) -> dict:
    """Parses the SUITE file shape (`analyses:` present): the top-level
    `systems:` is a shared registry; each `analyses:` entry resolves its
    own `systems:` (a list of NAMES) against that registry, or uses every
    registered system if it omits `systems:` entirely. Returns
    {analysis_name: MatchingTimeseriesConfig}, one entry per analysis,
    each parsed via the exact same _parse_analysis() the flat shape uses.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"'{path}' did not parse into a mapping at the top level")
    if "analyses" not in raw:
        raise ValueError(f"'{path}' has no 'analyses:' list -- use load_config() for a flat, single-analysis file")

    base_dir = os.path.dirname(os.path.abspath(path))
    registry_raw = raw.get("systems") or []
    registry_list = _parse_systems_list(registry_raw, base_dir, f"'{path}': systems registry") if registry_raw else []
    registry = {s.name: s for s in registry_list}

    analyses_raw = raw.get("analyses")
    if not analyses_raw:
        raise ValueError(f"'{path}' has an 'analyses:' key but it is empty -- at least one analysis is required")

    result = {}
    seen_names = set()
    for i, entry in enumerate(analyses_raw):
        name = entry.get("name", f"analysis_{i}")
        context = f"'{path}': analyses[{i}] ('{name}')"
        if name in seen_names:
            raise ValueError(f"'{path}': duplicate analysis name '{name}' in 'analyses:' list")
        seen_names.add(name)

        if "systems" in entry:
            unknown = [n for n in entry["systems"] if n not in registry]
            if unknown:
                raise ValueError(f"{context}: references unknown system name(s) {unknown} -- not in the top-level 'systems:' registry")
            entry_systems = [registry[n] for n in entry["systems"]]
        else:
            if not registry:
                raise ValueError(f"{context}: has no 'systems:' and the top-level registry is empty")
            entry_systems = list(registry.values())

        result[name] = _parse_analysis(entry, entry_systems, base_dir, context)

    return result


@dataclass
class ObjectSpatialMapConfig:
    """Config for a single forecast initialization's object spatial maps
    (see plots/object_spatial_map.py, drivers/plot_object_spatial_maps.py).
    A STANDALONE top-level config shape (own load_object_spatial_map_config,
    own YAML file), not nested inside MatchingTimeseriesConfig -- this
    feature's data dependencies (object-file grids/labels, optionally a
    match file) are structurally different from the systems:-registry-
    driven matching-timeseries family (every existing section there
    consumes SystemSpec.match_dir, and a match file alone has no grid/
    labels/lat-lon at all), so nesting it there would force every user of
    this feature to also declare an irrelevant placeholder systems: entry.

    mrms_object_dir/match_dir are each independently optional -- required
    together only when color_by="category" (needs both to build categorical
    fills + the instantaneous stats box); mrms_object_dir alone is required
    whenever show_mrms_overlay or show_truth_hatch is requested outside
    category mode."""
    forecast_object_dir: str
    init_time: datetime
    mrms_object_dir: str | None = None
    match_dir: str | None = None
    match_pattern: str = "match_init_*.nc"
    members: MemberSelectionConfig = field(default_factory=MemberSelectionConfig)
    times: TimeSelectionConfig = field(default_factory=TimeSelectionConfig)
    color_by: str = "single"
    show_mrms_overlay: bool = False
    show_forecast_hatch: bool = False
    show_truth_hatch: bool = False
    paintball: bool = False
    output_dir: str = "output"


def _parse_member_selection(raw: dict, context: str) -> MemberSelectionConfig:
    mode = raw.get("mode", "all")
    if mode not in _VALID_MEMBER_MODES:
        raise ValueError(f"{context}: mode={mode!r} is not one of {_VALID_MEMBER_MODES}")
    members = list(raw["members"]) if "members" in raw else None
    if mode in ("single", "list") and not members:
        raise ValueError(f"{context}: mode={mode!r} requires a non-empty 'members' list")
    return MemberSelectionConfig(mode=mode, members=members)


def _parse_time_selection(raw: dict, context: str) -> TimeSelectionConfig:
    mode = raw.get("mode", "full")
    if mode not in _VALID_TIME_MODES:
        raise ValueError(f"{context}: mode={mode!r} is not one of {_VALID_TIME_MODES}")
    lead_minutes = raw.get("lead_minutes")
    lead_minutes_range = _parse_lead_minutes_range(raw["lead_minutes_range"], context) if "lead_minutes_range" in raw else None
    if mode == "single" and lead_minutes is None:
        raise ValueError(f"{context}: mode='single' requires 'lead_minutes'")
    if mode == "range" and lead_minutes_range is None:
        raise ValueError(f"{context}: mode='range' requires 'lead_minutes_range'")
    return TimeSelectionConfig(mode=mode, lead_minutes=lead_minutes, lead_minutes_range=lead_minutes_range)


def _parse_object_spatial_map(raw: dict, base_dir: str, context: str) -> ObjectSpatialMapConfig:
    missing = [k for k in ("forecast_object_dir", "init_time") if k not in raw]
    if missing:
        raise ValueError(f"{context}: missing required field(s): {', '.join(missing)}")

    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))

    forecast_object_dir = _resolve(raw["forecast_object_dir"])
    mrms_object_dir = _resolve(raw["mrms_object_dir"]) if raw.get("mrms_object_dir") else None
    match_dir = _resolve(raw["match_dir"]) if raw.get("match_dir") else None
    init_time = datetime.fromisoformat(str(raw["init_time"]))

    color_by = raw.get("color_by", "single")
    if color_by not in _VALID_SPATIAL_COLOR_BY:
        raise ValueError(f"{context}: color_by={color_by!r} is not one of {_VALID_SPATIAL_COLOR_BY}")
    paintball = raw.get("paintball", False)

    show_mrms_overlay = raw.get("show_mrms_overlay", False)
    show_truth_hatch = raw.get("show_truth_hatch", False)
    if color_by == "category":
        # Paintball category mode shades each member's own forecast
        # footprint by its own match-category assignment -- it never
        # renders a truth layer at all (see plots.object_spatial_map.
        # plot_paintball_map), so mrms_object_dir is genuinely unneeded
        # there, unlike single-panel category mode (which DOES render a
        # truth-side categorical layer and requires it).
        if match_dir is None or (not paintball and mrms_object_dir is None):
            requirement = "match_dir" if paintball else "both mrms_object_dir and match_dir"
            raise ValueError(f"{context}: color_by='category' requires {requirement}")
    if color_by != "category" and (show_mrms_overlay or show_truth_hatch) and mrms_object_dir is None:
        raise ValueError(f"{context}: show_mrms_overlay/show_truth_hatch require mrms_object_dir")

    members = _parse_member_selection(raw.get("members") or {}, f"{context}: members")
    times = _parse_time_selection(raw.get("times") or {}, f"{context}: times")

    output_dir = raw.get("output_dir", "output")
    if not os.path.isabs(output_dir):
        output_dir = os.path.normpath(os.path.join(base_dir, output_dir))

    return ObjectSpatialMapConfig(
        forecast_object_dir=forecast_object_dir,
        init_time=init_time,
        mrms_object_dir=mrms_object_dir,
        match_dir=match_dir,
        match_pattern=raw.get("match_pattern", "match_init_*.nc"),
        members=members,
        times=times,
        color_by=color_by,
        show_mrms_overlay=show_mrms_overlay,
        show_forecast_hatch=raw.get("show_forecast_hatch", False),
        show_truth_hatch=show_truth_hatch,
        paintball=paintball,
        output_dir=output_dir,
    )


def load_object_spatial_map_config(path: str) -> ObjectSpatialMapConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"'{path}' did not parse into a mapping at the top level")
    base_dir = os.path.dirname(os.path.abspath(path))
    return _parse_object_spatial_map(raw, base_dir, f"'{path}'")


@dataclass
class ReflectivityMapConfig:
    """Standalone config for the composite-reflectivity spatial map (see
    plots/reflectivity_map.py, drivers/plot_reflectivity_maps.py) -- a raw
    continuous-dBZ-field plot, independent of the object-identification
    pipeline entirely. `source` selects which loader/discovery mechanism
    applies:

    - "model": forecast output (MPAS/WoFS/NowcastNet), via
      py_obj.obj_core.manifest.build_model_manifest. `var_name` is
      required. `member_subdirs`/`stacked_members` mirror that function's
      own semantics exactly (MPAS-style member subdirectories vs. WoFS-style
      one-file-with-all-members-stacked); whenever either is set,
      `member_id` is REQUIRED -- this feature renders exactly one member
      per config, never an ensemble aggregation/overlay (see
      plot_paintball_map for that different, larger capability).
    - "mrms_native": raw MRMS GRIB2 observations, via load_mrms_grib2.
    - "mrms_interp": already-interpolated MRMS netCDF observations, via
      load_mrms_netcdf.
    """
    source: str
    input_dir: str
    times: AbsoluteTimeSelectionConfig = field(default_factory=AbsoluteTimeSelectionConfig)
    min_dbz: float = 20.0
    extent: list | None = None
    label: str | None = None
    output_dir: str = "output"

    # source == "model" only. init_attr/lead_attr/init_format default to
    # load_model_netcdf's own MPAS-style arithmetic-mode defaults
    # (NOT None) -- build_model_manifest always passes these through as
    # real arguments regardless of whether its own caller set them, which
    # would otherwise silently override load_model_netcdf's own defaults
    # with None and raise (`hasattr(ds, None)` is invalid) for any config
    # that relies on those defaults implicitly, e.g. a plain MPAS case with
    # no valid_time_attr/valid_time_var override.
    var_name: str | None = None
    lat_name: str = "latitude"
    lon_name: str = "longitude"
    file_pattern: str = "*.nc"
    member_subdirs: bool = False
    stacked_members: bool = False
    member_subdir_pattern: str = "*"
    member_id: str | None = None
    init_attr: str = "initializationTime"
    lead_attr: str = "forecastHour"
    lead_units: str = "hours"
    init_format: str = "%Y%m%d%H"
    valid_time_attr: str | None = None
    valid_time_format: str | None = None
    valid_time_var: str | None = None

    # source in ("mrms_native", "mrms_interp") only
    mrms_pattern: str | None = None
    mrms_varname: str = "refl_consv"
    mrms_lat_name: str = "lat"
    mrms_lon_name: str = "lon"


def _parse_absolute_time_selection(raw: dict, context: str) -> AbsoluteTimeSelectionConfig:
    mode = raw.get("mode", "full")
    if mode not in _VALID_ABS_TIME_MODES:
        raise ValueError(f"{context}: mode={mode!r} is not one of {_VALID_ABS_TIME_MODES}")

    valid_time = datetime.fromisoformat(str(raw["valid_time"])) if "valid_time" in raw else None

    valid_time_range = None
    if "valid_time_range" in raw:
        raw_range = raw["valid_time_range"]
        if not (isinstance(raw_range, (list, tuple)) and len(raw_range) == 2):
            raise ValueError(f"{context}: valid_time_range must be a [start, end] pair, got {raw_range!r}")
        start, end = (datetime.fromisoformat(str(v)) for v in raw_range)
        if start > end:
            raise ValueError(f"{context}: valid_time_range [start, end] requires start <= end, got {raw_range!r}")
        valid_time_range = (start, end)

    if mode == "single" and valid_time is None:
        raise ValueError(f"{context}: mode='single' requires 'valid_time'")
    if mode == "range" and valid_time_range is None:
        raise ValueError(f"{context}: mode='range' requires 'valid_time_range'")

    return AbsoluteTimeSelectionConfig(
        mode=mode, valid_time=valid_time, valid_time_range=valid_time_range,
        tolerance_minutes=raw.get("tolerance_minutes", 10.0),
    )


def _parse_reflectivity_map(raw: dict, base_dir: str, context: str) -> ReflectivityMapConfig:
    missing = [k for k in ("source", "input_dir") if k not in raw]
    if missing:
        raise ValueError(f"{context}: missing required field(s): {', '.join(missing)}")

    source = raw["source"]
    if source not in _VALID_REFLECTIVITY_SOURCES:
        raise ValueError(f"{context}: source={source!r} is not one of {_VALID_REFLECTIVITY_SOURCES}")

    input_dir = raw["input_dir"]
    if not os.path.isabs(input_dir):
        input_dir = os.path.normpath(os.path.join(base_dir, input_dir))

    member_subdirs = raw.get("member_subdirs", False)
    stacked_members = raw.get("stacked_members", False)
    member_id = raw.get("member_id")

    var_name = raw.get("var_name")
    if source == "model":
        if var_name is None:
            raise ValueError(f"{context}: source='model' requires 'var_name'")
        if (member_subdirs or stacked_members) and member_id is None:
            raise ValueError(
                f"{context}: member_subdirs/stacked_members requires an explicit 'member_id' -- "
                "ensemble aggregation is not supported here (see the paintball capability for that)"
            )

    times = _parse_absolute_time_selection(raw.get("times") or {}, f"{context}: times")

    output_dir = raw.get("output_dir", "output")
    if not os.path.isabs(output_dir):
        output_dir = os.path.normpath(os.path.join(base_dir, output_dir))

    return ReflectivityMapConfig(
        source=source,
        input_dir=input_dir,
        times=times,
        min_dbz=raw.get("min_dbz", 20.0),
        extent=list(raw["extent"]) if "extent" in raw else None,
        label=raw.get("label"),
        output_dir=output_dir,
        var_name=var_name,
        lat_name=raw.get("lat_name", "latitude"),
        lon_name=raw.get("lon_name", "longitude"),
        file_pattern=raw.get("file_pattern", "*.nc"),
        member_subdirs=member_subdirs,
        stacked_members=stacked_members,
        member_subdir_pattern=raw.get("member_subdir_pattern", "*"),
        member_id=member_id,
        init_attr=raw.get("init_attr", "initializationTime"),
        lead_attr=raw.get("lead_attr", "forecastHour"),
        lead_units=raw.get("lead_units", "hours"),
        init_format=raw.get("init_format", "%Y%m%d%H"),
        valid_time_attr=raw.get("valid_time_attr"),
        valid_time_format=raw.get("valid_time_format"),
        valid_time_var=raw.get("valid_time_var"),
        mrms_pattern=raw.get("mrms_pattern"),
        mrms_varname=raw.get("mrms_varname", "refl_consv"),
        mrms_lat_name=raw.get("mrms_lat_name", "lat"),
        mrms_lon_name=raw.get("mrms_lon_name", "lon"),
    )


def load_reflectivity_map_config(path: str) -> ReflectivityMapConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"'{path}' did not parse into a mapping at the top level")
    base_dir = os.path.dirname(os.path.abspath(path))
    return _parse_reflectivity_map(raw, base_dir, f"'{path}'")
