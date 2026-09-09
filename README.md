# py_obj_plotting

Verification plotting and climatology suite for
[`python_obj`](https://github.com/skinnerp12345/py_obj) match/object output:
POD/FAR/FBIAS/CSI and matched-pair diagnostic time series, performance
diagrams, object-property POD heatmaps, storm-mode filtering, spatial maps
of identified objects, and composite-reflectivity spatial maps. A
standalone, complementary package -- imports `python_obj` as a dependency
and never modifies it.

## Installation

```bash
git clone https://github.com/skinnerp12345/py_obj.git python_obj
cd python_obj
conda env create -f environment.yml
conda activate python_obj_env
pip install -e . --no-deps
cd ..

git clone <this repo's URL> py_obj_plotting
cd py_obj_plotting
pip install -e . --no-deps
```

`py_obj_plotting` needs `python_obj` importable in the same environment --
its `environment.yml` mirrors `python_obj`'s own dependency set (every code
path here that touches raw MRMS/model files or masking goes through
`python_obj`, which pulls those dependencies in transitively even for
callers that only exercise a subset). The `pip install -e .` step registers
this checkout as the importable package `py_obj_plotting` independent of
the checkout folder's own name, exactly like `python_obj`'s own install
step.

## What's here

```
py_obj_plotting/
  systems.py              SystemSpec + load_system() -- the core per-system loader
  multi_system.py         N-system comparison layer: lead_time_mode (intersection/
                           union), check_shared_forecast_cases()
  object_selection.py     member/time selection for object spatial maps
  reflectivity_selection.py  absolute-datetime selection for reflectivity maps
  config.py               YAML config layer for all 3 driver families below
  configs/                real, runnable example configs (bring-your-own-data
                           and small bundled-sample-data configs alike)
  sample_data/            small, bundled, committed sample data for the
                           notebooks and a subset of example configs
  notebooks/               runnable Jupyter tutorials against sample_data/
  climatology/             aggregation logic (no plotting)
    aggregate_by_lead_hour.py
    aggregate_matches_multi_system.py    pooled-across-all-cases aggregation
    aggregate_matches_per_case.py        PER-CASE aggregation (case-scatter/diff/highlight)
    aggregate_matches_by_lead_hour.py    pooled contingency counts + storm-mode-filtered variant
    aggregate_object_property_pod.py     POD binned by truth object area/max_intensity/age
  plots/                  matched-percentile computation + matplotlib plotting
    lead_hour_timeseries.py
    performance_diagram.py               SR-vs-POD diagrams (case-scatter, two-system
                                          CSI-diff, lead-time trajectory)
    two_system_diff_histograms.py        POD/FAR/FBIAS/CSI difference histograms
    single_case_highlight.py             single-case-vs-baseline rendering variants
    object_property_heatmap.py           POD-by-object-property heatmaps (+ two-system diff)
    object_spatial_map.py                spatial maps of identified objects (+ paintball)
    reflectivity_map.py                  composite-reflectivity spatial maps
    mrms_object_map.py                   standalone MRMS-only object-footprint maps
  drivers/                thin CLI scripts, one per capability family
    plot_matching_timeseries.py
    plot_object_spatial_maps.py
    plot_reflectivity_maps.py
    plot_mrms_objects.py
    plot_mpas_vs_mrms_climatology.py
    plot_sfe_poster_climatology.py
  tests/                  pytest suite, synthetic fixtures + real-data smoke tests
                          (skip-marked when the external dataset isn't present)
  cb_colors.py             self-contained color constants
```

See `CONFIG_REFERENCE.md` for every config field across all three
config-driven drivers.

## Object-matching output time series (POD/FAR/FBIAS/CSI + matched-pair diagnostics)

`drivers/plot_matching_timeseries.py <config.yaml>` compares N forecast
systems (or N matching-threshold variants of one system -- both reduce to
the identical mechanism) against MRMS truth, purely from a config file --
no hardcoded per-scenario driver script. Built on:

- `systems.SystemSpec`/`load_system()`: one system = a name + a directory
  of `match_init_*.nc` files (`python_obj.obj_core.run_matching_series`
  output). Lead time is always normalized to **minutes** internally
  (never hours, to avoid float-precision drift at fine cadences);
  `member_id` is treated as a fully opaque string throughout (no format
  assumed -- real data has both `mem00`-style and `mem_1`-style
  conventions side by side).
- `multi_system.compute_included_keys(loaded, lead_time_mode=...)`: derives
  which `(forecast case, lead minute)` combinations are actually pooled.
  `"intersection"` (default) requires every system to have it;
  `"union"` lets each system contribute its own presence independently
  (needed for real systems with different startup-availability gaps, e.g.
  WoFSCast has no matched objects before lead=20min).
- `multi_system.check_shared_forecast_cases()`: a separate, case-level
  check (distinct from lead-time coverage) -- under the config's
  `require_shared_cases: true` (default), raises a clear, named error if
  the systems being compared don't cover the same set of forecast
  cases/init-times, rather than silently pooling a biased subset.
  `require_shared_cases: false` proceeds on the shared subset, still
  reporting the mismatch.
- `plots.lead_hour_timeseries.adaptive_lead_hour_xticks`: derives tick
  spacing and display unit (minutes vs. hours) automatically from each
  plot's own actual lead-time range -- no per-scenario x-tick choice.
- A system with no member dimension at all (`members == [None]`, e.g. real
  NowcastNet match files) renders as one bold line with its own bootstrap
  CI band, not member-alpha spaghetti drawn over a fake "1 member."

Two real example configs (points at external, non-bundled data --
`sfe_poster/`/`ncar_mpas_analysis/`, same "bring your own larger dataset"
convention as `python_obj`): `configs/config_sfe_3system.yaml` (WoFS vs.
WoFSCast vs. NowcastNet, mixed ensemble/deterministic, 0-6h at 5-10min
cadence) and `configs/config_mpas_thresholds.yaml` (4 matching-threshold
variants of one MPAS ensemble, 0-132h hourly) -- plus 3 pairwise SFE
configs and a 2-threshold MPAS config demonstrating the identical
mechanism at N=2. Run with:

```bash
/opt/anaconda3/envs/pysteps_env/bin/python py_obj_plotting/drivers/plot_matching_timeseries.py py_obj_plotting/configs/config_sfe_3system.yaml
```

Every plot writes flat into `output_dir` with a unique filename (no
per-storm-mode or per-period subdirectories). Pass `--output-dir=<path>` on
the command line to redirect output without editing the config file --
`<path>` can be anywhere, absolute or relative, not just under
`configs/output/`.

### Performance diagrams (Success Ratio vs. POD), per-case markers, two-system diagnostics

`plots.performance_diagram.init_performance_diagram()` is a faithful port
of the legacy `python_base/wofs_verif_cbook.py`'s `perf_diagram_init()`/
`perf_diagram_cbar_init()` styling (CSI contour lines, frequency-bias
radial lines, axis geometry) -- numerically verified against the legacy's
own exact contour levels/colors, not eyeballed. Three diagram/plot
families are built on top of `climatology.aggregate_matches_per_case`
(a NEW, non-flattening aggregation path -- `build_per_case_match_climatologies()`
groups by forecast case instead of pooling across every case, returning
the same `MatchLeadHourClimatology` shape scoped to one case, so
`compute_verification_stats()` needs no changes to consume it):

- **Case-scatter performance diagrams** (`plot_case_scatter_performance_diagram`,
  config `performance_diagrams.case_scatter`): one marker per forecast
  case, aggregated over a user-configured lead-time-range "period" (e.g.
  `60-90min` for WoFS, `32-37h` for MPAS -- any number of periods,
  each producing its own diagram). `case_grouping: init_time` treats
  each real initialization as its own case (MPAS-style); `case_grouping:
  day` buckets every same-calendar-day initialization into one case
  (WoFS-style -- confirmed real: 190 real initializations collapse to 24
  real days). Marker size reflects `case_object_count()` (truth- or
  forecast-side, `marker_size_basis:`); ensembles are represented by
  their **ensemble mean only** (one marker per case, not one per member).
  Each marker is labeled with a compact **1-based chronological integer
  index** (earliest case = 1, shared across every system in the diagram
  so the same real-world case always gets the same number), not the full
  date-string case identifier -- the function returns the
  `dict[case_label, index]` mapping so callers can report which real case
  a given number refers to.
- **Two-system CSI-difference performance diagrams**
  (`plot_two_system_case_diff_performance_diagram`, config
  `two_system_comparison`, only valid for exactly 2 systems): restricted
  to **matched cases** (the intersection of both systems' own case sets,
  always reported, never silently assumed) -- one shared
  `CSI(system_a) - CSI(system_b)` value colors BOTH systems' markers for
  that case (`RdBu_r` + `BoundaryNorm`, warm = system_a higher), each
  labeled with the same chronological integer index as above in **plain
  thin black text** (no outline). The two systems are distinguished by
  each marker's own **edge linestyle** instead -- system_a solid,
  system_b dashed -- since colored/outlined text was found to be masked
  against similarly-colored diff-colormap marker fills.
- **Lead-time-trajectory performance diagrams**
  (`plot_lead_time_trajectory_performance_diagram`, config
  `performance_diagrams.lead_time_trajectory`): reuses the *same* pooled
  `VerificationStats` the POD/FAR/etc. time series already compute (zero
  new aggregation) -- one (SR, POD) trajectory per system across every
  available lead time, lead-time indicated by both a per-segment alpha
  fade (early -> late) and interval-labeled markers (`label_interval_minutes`,
  e.g. 30 for WoFS, 720 for MPAS). N-system comparison only -- no
  single-case-highlight equivalent (see below).
- **Two-system metric-difference histograms**
  (`plots.two_system_diff_histograms.plot_two_system_diff_histograms`,
  same `two_system_comparison` config): 4 histograms (POD/FAR/FBIAS/CSI),
  one value per matched case (ensemble-mean-based), showing the
  distribution of `system_a - system_b` across every matched
  initialization. Every metric uses a fixed bin width (0.025) and a
  dynamically-computed x-axis range: symmetric around 0, sized to the
  largest-magnitude difference actually present in the data, floored at
  `[-0.4, 0.4]` for POD/FAR and `[-0.2, 0.2]` for CSI (no floor for FBIAS,
  which has no natural bound). Bars are colored by the sign of their own
  bin center for all 4 metrics -- positive (system_a scoring higher) in
  orange, negative (system_b scoring higher) in blue.

### Single-case-vs-baseline mode

`single_case_highlight:` (config) isolates ONE case of ONE system against
a baseline of its other available cases -- **not** a system comparison.
Applies to 2 of the families above (`plots.single_case_highlight`; no
lead-time-trajectory variant -- removed as unnecessary):

- **Time series**: the baseline population is summarized as a shaded
  **percentile-band distribution** per lead hour (10th-90th and 25th-75th
  percentiles across whichever baseline cases have data at that lead
  hour, shaded gray, plus a black median line) rather than one line per
  baseline case (illegible once there are more than a handful of cases);
  the highlighted case's own line is plotted on top, full alpha, in a
  bold `highlight_color`.
- **Case-scatter performance diagram**: baseline and highlighted markers
  share **one linear size scale** based on `case_object_count()` (the
  same scale `plot_case_scatter_performance_diagram` uses) -- cases with
  the same object count render at the same size regardless of which one
  is highlighted. Baseline markers render in a darker gray
  (low alpha, no legend); the highlighted marker uses `highlight_color`
  at full alpha with a black edge, labeled **above** the marker (not
  centered within it) in a larger, bold font.

Mutually exclusive with `systems:` comparison mode within one analysis --
when `single_case_highlight:` is present, the driver renders ONLY the
highlight variants for that one named system, never both.

### Object-property POD heatmaps

`heatmaps:` (config) produces POD-vs-lead-time heatmaps binned by a truth
(MRMS) object's own property -- direct ports of the legacy
`python_base/wofs_verif_cbook.py`'s `heat_map_fig()`/`heat_map_diff_fig()`
visual style (`Reds`/`RdBu_r`, `BoundaryNorm`-stepped `pcolor`, numeric
cell labels every 4th lead-time column, zero-count cells left blank).
Three independently optional properties, each gated by its own bin-edges
field (omitting one skips just that property's heatmap):

- `area_bins`: truth object area, in **km²**.
- `max_intensity_bins`: truth object max reflectivity, in **dBZ**.
- `age_bins`: truth object age **relative to init** (`age_seconds -
  lead_minutes*60`, the same transform the legacy used, in **seconds**)
  -- only produces real data if the truth series was **tracked** at
  identification time (`observations.track: true` in `python_obj`'s own
  config). A match file whose truth series wasn't tracked has no real age
  data at all; the driver detects this per system and skips just that
  heatmap with an explicit report (`skipping heatmap_age_<system>.png --
  no real 'age' data for '<system>'`), rather than silently plotting a
  blank one -- confirmed on real data (both bundled example configs'
  MPAS/WoFS/WoFSCast match archives predate this schema field and
  correctly report 0 real age records).

POD is conditioned on truth objects (hit/miss rows only, binned by that
row's own truth-side property value -- a miss still has a real truth
object with a real area/intensity/age, just no matching forecast object),
pooled across every member and case sharing a `(property bin, lead hour)`
cell. When exactly 2 systems are compared **and** `two_system_comparison:`
is also present, one **difference heatmap** per requested property is
produced too (`POD_A - POD_B`, `RdBu_r`, a dynamic symmetric-around-zero
color range with a floor -- the same convention
`plots.two_system_diff_histograms` uses, not the legacy's hardcoded
±0.4), with bootstrap-significance alpha shading (full alpha where a
cell's `|diff|` exceeds its own bootstrap CI width via the new
`contingency_stats.bootstrap_pod_diff_ci_width()`, 0.5 alpha otherwise).

Object age is a genuinely new capability end to end, not just a plotting
addition: `StormObject.age_seconds` existed at identification time but
was never persisted to match files before this feature -- schema extended
(`python_obj/obj_core/matching.py`'s `_MATCH_RECORD_SIDE_FIELDS`,
`match_io.py`'s `_MATCH_FLOAT_FIELDS`) and validated on a real regenerated
tracked case (437/585 real, non-null `truth_age_seconds` values, exactly
matching every non-`false_alarm` row).

### Storm-mode-filtered time series

`storm_mode_timeseries:` (config, `modes: [cellular, mixed, linear]` by
default) recomputes the same POD/FAR/FBIAS/CSI comparison time series
(`metrics:`) separately for each requested storm mode (`is_linear`:
0=cellular, 1=mixed, 2=linear -- see `python_obj`'s linear-classification
work), written under `output_dir/storm_mode_<mode>/`. Filters at the raw
per-record level **before** pooling: hit/miss/truth_extra rows are keyed
by `truth_is_linear`; false_alarm/forecast_extra rows (no truth object at
all) are keyed by `forecast_is_linear` instead -- a real, necessary
asymmetry. The resulting per-mode `MatchLeadHourClimatology` flows
completely unchanged through the existing `compute_verification_stats` ->
comparison-plot pipeline (zero plotting-layer changes needed). Validated
that cellular+mixed+linear counts sum back exactly to the unfiltered
pooled totals on synthetic data, and confirmed on real MPAS/WoFS/WoFSCast
data (hundreds of thousands of real per-mode records).

Both of the above tolerate **either** a real ensemble (init_snapshot,
multiple named members) **or** a deterministic (no member dimension)
match file -- matching `systems.py`'s own `load_system()` convention,
since `drivers.plot_matching_timeseries` must support both shapes.

### Suite-style parent config (multiple analyses from one file)

A single config file can define an entire *suite* of independent
analyses for one dataset: a shared `systems:` **registry** at the top
level, plus an `analyses:` list where each entry selects a subset of the
registry by name (or every registered system, if it omits its own
`systems:`) and gets its own independent `lead_time:`/`metrics:`/
`performance_diagrams:`/`two_system_comparison:`/`single_case_highlight:`/
`output_dir:`. `config.load_suite_config(path)` parses this shape (a file
with a top-level `analyses:` key); `config.load_config(path)` still
parses today's flat, single-analysis shape unchanged (a file with no
`analyses:` key) -- both reuse the identical per-analysis parsing, so
every field behaves the same regardless of which shape reaches it.
`drivers.plot_matching_timeseries.run_suite(path)` runs every analysis in
one call, alongside the existing `run_one_case(path)` for a flat file.

## Object spatial maps

`drivers/plot_object_spatial_maps.py <config.yaml> [--output-dir=<path>]`
renders one identified-storm-object map per selected (member, valid time)
of a single forecast initialization -- `plots.object_spatial_map`, config
`ObjectSpatialMapConfig`. `color_by` selects one mutually-exclusive primary
fill style for the forecast layer:

- `single`: one flat color per footprint pixel.
- `storm_mode`: categorical, keyed by `StormObject.is_linear`
  (0=cellular, 1=mixed, 2=linear).
- `age`: continuous colormap + colorbar, keyed by `StormObject.age_seconds`
  (in minutes) -- an object with no tracked age maps to NaN rather than
  raising, so an entirely-untracked file degrades to an all-NaN layer.
- `category`: hit/miss/false_alarm/extra, colored to match the matching
  pipeline's own category convention (blue/red/orange/gray). Requires both
  `mrms_object_dir` and `match_dir`; also renders an instantaneous
  POD/FAR/FBIAS/CSI annotation box computed directly from that slice's own
  match records.

An MRMS/truth overlay (`show_mrms_overlay`) and independent storm-mode
hatch toggles (`show_forecast_hatch`/`show_truth_hatch`) are available
regardless of `color_by`. `paintball: true` switches to an ensemble
"paintball" plot instead -- every selected member's own footprint at one
shared valid time, overlaid on one shared domain, with its own `color_by`
support for `single` (uniform low-alpha), `storm_mode`, and `category`
(each member's own hit/miss/false_alarm/extra assignment, with
ensemble-mean POD/FAR/FBIAS/CSI annotated).

`members:`/`times:` (`MemberSelectionConfig`/`TimeSelectionConfig`) each
independently select `all`/`single`/a `list` of members, and `full`/
`single`/a `range` of lead-minutes-since-init -- the same selection
mechanism regardless of which one is used, resolved against whatever the
forecast file itself actually contains.

```bash
/opt/anaconda3/envs/pysteps_env/bin/python py_obj_plotting/drivers/plot_object_spatial_maps.py py_obj_plotting/configs/config_object_spatial_map_wofs.yaml
```

## Composite-reflectivity spatial maps

`drivers/plot_reflectivity_maps.py <config.yaml> [--output-dir=<path>]`
renders a raw, continuous dBZ field geographically -- independent of the
object-identification pipeline entirely (`plots.reflectivity_map`, config
`ReflectivityMapConfig`). `source` selects the loader:

- `model`: forecast output (MPAS/WoFS/NowcastNet-style), via
  `python_obj`'s model manifest builder. `var_name` is required;
  `member_subdirs`/`stacked_members` mirror that function's own semantics
  (MPAS-style member subdirectories vs. WoFS-style one-file-with-all-
  members-stacked) -- whenever either is set, `member_id` is required,
  since this feature renders exactly one member per config, never an
  ensemble aggregation (see the paintball capability above for that).
- `mrms_native`: raw MRMS GRIB2 observations.
- `mrms_interp`: already-interpolated MRMS netCDF observations.

`times:` (`AbsoluteTimeSelectionConfig`) selects `full` (every discovered
time), a `single` absolute `valid_time` (nearest match within
`tolerance_minutes`, since real MRMS timestamps are irregularly jittered),
or a `[start, end]` `valid_time_range`.

```bash
/opt/anaconda3/envs/pysteps_env/bin/python py_obj_plotting/drivers/plot_reflectivity_maps.py py_obj_plotting/configs/config_reflectivity_map_mpas.yaml
```

## Standalone MRMS object-footprint maps

`drivers/plot_mrms_objects.py <YYYYMMDD> [mrms_objects_dir] [out_dir]`: one
PNG per available hourly MRMS object file for a single day, "WoFS style"
(state boundaries only) -- objects shaded by `is_linear` category, each
centroid marked with a small circle. Plain arguments, not a YAML config,
since this is one specific analysis rather than a general capability.

```bash
/opt/anaconda3/envs/pysteps_env/bin/python py_obj_plotting/drivers/plot_mrms_objects.py 20230513
```

## MPAS-vs-MRMS matched-percentile climatology by forecast lead hour

`climatology.build_lead_hour_climatology(mpas_dir, mrms_dir, ...)` reads
every `hist_model_*.nc` (one per MPAS ensemble forecast case, produced by
`python_obj.drivers.build_histogram_model`) and every `hist_mrms_*.nc` (one
per MRMS day, produced by `python_obj.drivers.build_histogram_mrms`) under
the given directories, and pools them into:

- one composite reflectivity distribution per `(member, lead_hour)` for
  MPAS -- summed across every available case
- one distribution per `lead_hour` for MRMS -- matched to each case's own
  real valid time (`init_time + lead_hour`, not just hour-of-day), so a
  large lead hour that pushes past midnight is matched against the correct
  MRMS file even when it's a different calendar day than the case's own
  init date. Counted **once per case per lead hour**, not once per member,
  since the observation doesn't depend on which model member it's being
  compared against.

A `(case, lead_hour)` pair with no MRMS observation within the configured
tolerance (default +/-10 minutes) is skipped and reported in
`LeadHourClimatology.unmatched`, never silently dropped.

`plots.compute_matched_percentiles(clim, thresholds=(40.0, 45.0))` then
calls `python_obj.histogram.match_percentile_threshold` (unmodified) for
every `(lead_hour, member, threshold)` combination, giving:

- `matched_value[threshold][member]` -- the MPAS dBZ value at the same
  percentile as MRMS's own threshold value, one per lead hour
- `source_percentile[threshold]` -- the percentile MRMS's own threshold
  value occurs at in the pooled MRMS distribution, one per lead hour
  (member-independent)

`plots.plot_matched_value_timeseries()`/`plots.plot_source_percentile_timeseries()`
render these as two figures: matched dBZ value vs. lead hour (all members
sharing one color per threshold -- `cb_colors.blue5` for 40 dBZ,
`cb_colors.orange5` for 45 dBZ, no per-member distinction), and MRMS's own
percentile vs. lead hour (`cb_colors.gray5` for 40 dBZ, black for 45 dBZ).

Run the full comparison against the real bundled `ncar_mpas_analysis/` data:

```bash
/opt/anaconda3/envs/pysteps_env/bin/python py_obj_plotting/drivers/plot_mpas_vs_mrms_climatology.py
```

Writes `matched_value_by_lead_hour.png` and
`mrms_source_percentile_by_lead_hour.png` to
`py_obj_plotting/output/mpas_vs_mrms_climatology/` by default. Pass
`[mpas_dir] [mrms_dir] [out_dir]` to point at different data.

## Multi-system shared-valid-time climatology comparison (WoFS/WoFSCast/NowcastNet vs. MRMS)

`climatology.build_shared_valid_times_climatology(model_dirs, ensemble_systems, mrms_dir)`
reads every source's already-built `hist_model_*.nc`/`hist_mrms_*.nc` files
(one per source directory) and restricts the comparison to valid times where
**every** source has data -- a fair comparison across sources with
different, non-identical output cadences (confirmed real: MRMS/WoFS ~5-min,
WoFSCast/NowcastNet ~10-min). For ensemble sources (`member_id` populated
per slice, e.g. WoFS/WoFSCast's real 18-member ensembles), builds one
climatology per member first, then averages the members' individual PDFs
-- **not** a sum of member counts normalized once -- so every member is
weighted equally regardless of how much data it happened to contribute
(confirmed to matter: with unequal per-member totals, averaging PDFs and
summing-then-normalizing give genuinely different results).

`plots.compute_percentile_table()` finds the percentile MRMS's own 40/45
dBZ threshold occurs at in MRMS's shared-valid-time climatology, then the
dBZ value at that same percentile in every other source's climatology
(reuses `python_obj.histogram.value_at_percentile` unchanged).
`plots.plot_pdf_cdf_per_system()`/`plots.plot_combined_pdf()`/
`plots.plot_combined_cdf()` render one PDF+CDF figure per source plus one
combined overlay figure each for PDF and CDF (never a dual-axis PDF+CDF
combination -- different measures/scales get separate panels).
`climatology.build_shared_times_summary_table()` reports, per date, the
number of shared valid times and the number of distinct forecast cycles
(files) per source that contributed at least one -- the TOTAL row correctly
deduplicates a forecast cycle whose valid times span midnight (confirmed
real for WoFSCast), rather than double-counting it.

Run against the real bundled `../sfe_poster/` directories:

```bash
/opt/anaconda3/envs/pysteps_env/bin/python py_obj_plotting/drivers/plot_sfe_poster_climatology.py
```

Writes `pdf_cdf_<system>.png`, `pdf_combined.png`, `cdf_combined.png` to
`sfe_poster/histogram_images/`, and `percentile_table.csv`/
`shared_times_summary.csv` to `sfe_poster/` directly. Pass
`[sfe_poster_dir] [out_dir]` to point at different data. Real run: 1158
shared valid times across 24 dates; e.g. MRMS's 40 dBZ (99.1st percentile)
matches WoFS 46.2 dBZ, WoFSCast 43.8 dBZ, NowcastNet 47.8 dBZ at the same
percentile in each system's own climatology.

## Notebooks

`notebooks/` has runnable Jupyter tutorials against the small, bundled
`sample_data/` -- see `notebooks/README.md` for what each one covers and
how to run it.

## Tests

```bash
/opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/ -v
```

Most tests use small, synthetic, hand-computed fixtures, plus a
smaller set of real-data checks against this package's own bundled
`sample_data/` and `python_obj`'s bundled `sample_data/` (unconditional,
always run) alongside additional real-scale checks against larger external
datasets (`ncar_mpas_analysis/`, `sfe_poster/`, etc.) that are not bundled
and skip cleanly when absent, consistent with `python_obj`'s own "bring
your own larger dataset" precedent for scenarios too big to ship with the
repo.
