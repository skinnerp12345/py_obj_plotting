# Config reference

Exhaustive listing of every field across the three YAML-config-driven
drivers. This is the dry, complete reference; see `README.md` for a
narrative walkthrough of what each capability does and why.

All path-shaped fields (`match_dir`, `input_dir`, `output_dir`,
`forecast_object_dir`, `mrms_object_dir`) resolve relative to the config
file's own directory when not absolute.

## `drivers/plot_matching_timeseries.py` -- `MatchingTimeseriesConfig`

Loaded via `config.load_config(path)` for a flat, single-analysis file, or
`config.load_suite_config(path)` for a suite file (top-level `analyses:`
list, sharing one `systems:` registry -- every field below applies
identically inside each `analyses:` entry).

| Field | Type | Default | Notes |
|---|---|---|---|
| `systems` | list of system entries | required | See "System entry" below. At least one required. |
| `lead_time.mode` | `"intersection"` \| `"union"` | `"intersection"` | Which `(cycle, lead_minutes)` combinations are pooled across systems. |
| `lead_time.lead_minutes` | list of float \| `null` | `null` (auto) | Explicit lead-minute axis; `null` auto-discovers the union of every system's own lead minutes. |
| `metrics` | list of `"pod"`\|`"far"`\|`"fbias"`\|`"csi"` | all four | Which contingency-table time series to render. |
| `bootstrap.samples` | int | `1000` | Bootstrap resample count for CI widths. |
| `bootstrap.ci_interval` | float | `95` | CI interval, percent. |
| `output_dir` | str | `"output"` | Every plot is written flat here with a unique filename. |
| `require_shared_cases` | bool | `true` | `true` raises a named error listing mismatched forecast cases across systems; `false` reports the same mismatch and proceeds on the shared subset. |
| `performance_diagrams` | section \| `null` | `null` | See below. |
| `two_system_comparison` | section \| `null` | `null` | Only valid when `systems` resolves to exactly 2 entries. See below. |
| `single_case_highlight` | section \| `null` | `null` | Mutually exclusive with normal N-system comparison for that analysis. See below. |
| `heatmaps` | section \| `null` | `null` | See below. |
| `storm_mode_timeseries` | section \| `null` | `null` | See below. |

### System entry (`systems:` list items)

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | str | required | Must be unique within the list. |
| `match_dir` | str | required | Directory of `match_init_*.nc` files (`python_obj.obj_core.run_matching_series`, `file_grouping="init_snapshot"`). |
| `color` | `[r, g, b]` \| `null` | `null` (auto-assigned) | Auto-assigns from a default palette in list order if omitted. |
| `match_pattern` | str (glob) | `"match_init_*.nc"` | Restricts which files in `match_dir` are read -- e.g. to bundle only a small subset. |

### `performance_diagrams:`

| Field | Type | Default | Notes |
|---|---|---|---|
| `case_scatter.case_grouping` | `"init_time"` \| `"day"` | `"init_time"` | `"day"` buckets same-calendar-day initializations into one case (WoFS-style). |
| `case_scatter.marker_size_basis` | `"truth"` \| `"forecast"` | `"truth"` | Which side's object count sizes each marker. |
| `case_scatter.periods` | list of `{name, lead_minutes_range: [lo, hi]}` | `[]` | One case-scatter diagram per period. |
| `lead_time_trajectory.label_interval_minutes` | float \| `null` | `null` (no labels) | Interval between labeled markers along the SR-POD trajectory. |

### `two_system_comparison:`

| Field | Type | Default | Notes |
|---|---|---|---|
| `colormap` | str (matplotlib colormap name) | `"RdBu_r"` | Used for the CSI-difference performance diagram. |
| `diff_levels` | list of float \| `null` | `null` (`np.arange(-0.1, 0.11, 0.01)`) | Color-level boundaries for the diff colormap. |
| `case_scatter_periods` | list of period \| `null` | `null` (reuses `performance_diagrams.case_scatter.periods`) | Periods for the two-system diff performance diagram. |
| `histogram_periods` | list of period \| `null` | `null` (reuses `performance_diagrams.case_scatter.periods`) | Periods for the POD/FAR/FBIAS/CSI difference histograms. |

### `single_case_highlight:`

| Field | Type | Default | Notes |
|---|---|---|---|
| `system` | str | required | Must name exactly one of this analysis's own resolved systems. |
| `case` | str | required | The case label to highlight. |
| `baseline_cases` | list of str \| `null` | `null` (every other available case) | Restrict the baseline population. |
| `highlight_color` | `[r, g, b]` \| `null` | `null` (a bold default) | Color for the highlighted case's own line/marker. |

### `heatmaps:`

| Field | Type | Default | Notes |
|---|---|---|---|
| `area_bins` | list of float \| `null` | `null` (skipped) | Bin edges, truth object area in km². |
| `max_intensity_bins` | list of float \| `null` | `null` (skipped) | Bin edges, truth object max reflectivity in dBZ. |
| `age_bins` | list of float \| `null` | `null` (skipped) | Bin edges, truth object age relative to init, in seconds. Requires the truth series to have been tracked at identification time. |

Each property is independently optional. When exactly 2 systems are
compared and `two_system_comparison:` is also present, a difference
heatmap is produced per requested property too.

### `storm_mode_timeseries:`

| Field | Type | Default | Notes |
|---|---|---|---|
| `modes` | list of `"cellular"`\|`"mixed"`\|`"linear"` | all three | Recomputes the full POD/FAR/FBIAS/CSI + diagnostic + object-count/area bundle separately per mode. Requires `storm_mode_classification` to have been enabled at identification time. |

## `drivers/plot_object_spatial_maps.py` -- `ObjectSpatialMapConfig`

Loaded via `config.load_object_spatial_map_config(path)`. Flat, standalone
schema (not nested inside the matching-timeseries config).

| Field | Type | Default | Notes |
|---|---|---|---|
| `forecast_object_dir` | str | required | Directory of forecast object files for this one initialization. |
| `init_time` | ISO datetime string | required | Selects the one forecast file matching this init time. |
| `mrms_object_dir` | str \| `null` | `null` | Required for `show_mrms_overlay`/`show_truth_hatch`, and (together with `match_dir`) for `color_by: category`. |
| `match_dir` | str \| `null` | `null` | Directory of `match_init_*.nc` files. Required whenever `color_by: category`. |
| `match_pattern` | str (glob) | `"match_init_*.nc"` | Restricts which files in `match_dir` are read. |
| `members.mode` | `"all"` \| `"single"` \| `"list"` | `"all"` | Which ensemble member(s) to render. |
| `members.members` | list of str | required iff `mode` is `single`/`list` | The member name(s) to select. |
| `times.mode` | `"full"` \| `"single"` \| `"range"` | `"full"` | Which lead time(s) to render. |
| `times.lead_minutes` | float | required iff `mode: single` | Exact lead-minutes-since-init to select. |
| `times.lead_minutes_range` | `[lo, hi]` | required iff `mode: range` | Inclusive lead-minutes-since-init range. |
| `color_by` | `"single"`\|`"storm_mode"`\|`"age"`\|`"category"` | `"single"` | Primary fill style -- see README for what each renders. |
| `show_mrms_overlay` | bool | `false` | Draws the MRMS/truth footprint alongside the forecast layer. |
| `show_forecast_hatch` | bool | `false` | Independent storm-mode hatch overlay on the forecast layer. |
| `show_truth_hatch` | bool | `false` | Independent storm-mode hatch overlay on the truth layer. |
| `paintball` | bool | `false` | Renders an ensemble "paintball" plot (every selected member's footprint on one shared domain) instead of one map per member. |
| `output_dir` | str | `"output"` | Every plot is written flat here with a unique filename. |

## `drivers/plot_reflectivity_maps.py` -- `ReflectivityMapConfig`

Loaded via `config.load_reflectivity_map_config(path)`. Flat, standalone
schema.

| Field | Type | Default | Notes |
|---|---|---|---|
| `source` | `"model"`\|`"mrms_native"`\|`"mrms_interp"` | required | Which loader/discovery mechanism applies. |
| `input_dir` | str | required | Directory to discover files in. |
| `times.mode` | `"full"`\|`"single"`\|`"range"` | `"full"` | Which valid time(s) to render. |
| `times.valid_time` | ISO datetime string | required iff `mode: single` | Nearest available time within `tolerance_minutes`. |
| `times.valid_time_range` | `[start, end]` ISO datetimes | required iff `mode: range` | Inclusive range. |
| `times.tolerance_minutes` | float | `10.0` | Nearest-match window, `mode: single` only. |
| `min_dbz` | float | `20.0` | Values below this are not shaded. |
| `extent` | `[lon_min, lon_max, lat_min, lat_max]` \| `null` | `null` (auto, from data) | Fixed map extent override. |
| `label` | str \| `null` | `null` | Optional label text added to the plot. |
| `output_dir` | str | `"output"` | Every plot is written flat here with a unique filename. |
| `var_name` | str \| `null` | `null` | **`source: model` only, required.** Variable to plot. |
| `lat_name` / `lon_name` | str | `"latitude"` / `"longitude"` | `source: model` only. |
| `file_pattern` | str (glob) | `"*.nc"` | `source: model` only. |
| `member_subdirs` | bool | `false` | `source: model` only, MPAS-style one-subdirectory-per-member. Requires `member_id` when set. |
| `stacked_members` | bool | `false` | `source: model` only, WoFS-style one-file-with-all-members-stacked. Requires `member_id` when set. |
| `member_subdir_pattern` | str (glob) | `"*"` | `source: model` only, restricts which subdirectories count as members when `member_subdirs: true`. |
| `member_id` | str \| `null` | `null` | `source: model` only. Required whenever `member_subdirs`/`stacked_members` is set -- this feature renders exactly one member, never an ensemble aggregation. |
| `init_attr` | str | `"initializationTime"` | `source: model` only, arithmetic (init+lead) time mode. |
| `lead_attr` | str | `"forecastHour"` | `source: model` only, arithmetic time mode. |
| `lead_units` | str | `"hours"` | `source: model` only, unit of `lead_attr`'s raw value. |
| `init_format` | str | `"%Y%m%d%H"` | `source: model` only, arithmetic time mode. |
| `valid_time_attr` | str \| `null` | `null` | `source: model` only, string valid-time mode (alternative to arithmetic mode). |
| `valid_time_format` | str \| `null` | `null` | `source: model` only, paired with `valid_time_attr`. |
| `valid_time_var` | str \| `null` | `null` | `source: model` only, CF-time-coordinate override -- takes precedence over `valid_time_attr` when both could apply. |
| `mrms_pattern` | str (glob) \| `null` | `null` | `source` in (`mrms_native`, `mrms_interp`) only. |
| `mrms_varname` | str | `"refl_consv"` | `source` in (`mrms_native`, `mrms_interp`) only. |
| `mrms_lat_name` / `mrms_lon_name` | str | `"lat"` / `"lon"` | `source` in (`mrms_native`, `mrms_interp`) only. |

## Command-line output override

`plot_matching_timeseries.py`, `plot_object_spatial_maps.py`, and
`plot_reflectivity_maps.py` all accept an optional trailing
`--output-dir=<path>` argument that overrides the config's own
`output_dir` without editing the YAML file. `<path>` may be absolute or a
path relative to the current working directory, and is not restricted to
living under `configs/output/`.
