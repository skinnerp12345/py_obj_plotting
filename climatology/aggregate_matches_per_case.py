"""Per-case (per forecast initialization, or per calendar day for
high-cadence systems like WoFS) match-file aggregation -- deliberately
NOT flattened across cases, unlike aggregate_matches_multi_system.py's
pooled path. Feeds the case-scatter performance diagrams, the two-system
per-case CSI-difference diagnostics, the two-system metric-difference
histograms, and the single-case-vs-baseline highlighting mode -- all of
which need one MatchLeadHourClimatology-shaped object PER CASE rather than
one pooled across every case.

Reuses systems.load_systems()/multi_system.compute_included_keys() exactly
as aggregate_matches_multi_system.py does for the pooled path -- the only
difference is the final grouping step: cases are kept separate here
instead of being flattened into (member, lead_hour) totals summed across
every case.

Returns the EXISTING MatchLeadHourClimatology dataclass, scoped to one
case, so climatology.compute_verification_stats() (unmodified) already
knows how to turn any one of these into per-case POD/FAR/FBIAS/CSI -- no
new per-lead-hour stats math needed anywhere in this module, only new
data-shaping. case_metrics()/case_object_count() below are a further,
period-aggregated convenience layer on top of that (for scatter-style
diagrams that need one point per case, not a full lead-hour curve).
"""

from datetime import timedelta
from typing import Literal

import numpy as np

from ..multi_system import compute_included_keys, load_systems
from ..systems import SystemSpec
from .aggregate_matches_by_lead_hour import MatchLeadHourClimatology
from .contingency_stats import compute_csi, compute_far, compute_fbias, compute_pod

_CATEGORIES = ("hit", "miss", "false_alarm", "truth_extra", "forecast_extra")
_VALID_CASE_GROUPINGS = ("init_time", "day")


def _forecast_day(init_time, day_start_hour: int):
    """The date a 'day'-grouped case belongs to: init_time's own UTC date,
    shifted back by day_start_hour -- e.g. day_start_hour=12 puts a 02Z
    May 7 init into the "2026-05-06" case, so one WoFS day (roughly 17Z to
    03Z) stays one case instead of splitting at 00 UTC."""
    return (init_time - timedelta(hours=day_start_hour)).date()


def _case_key(init_time, case_grouping: str, day_start_hour: int = 0):
    """The grouping key used to bucket cases together -- 'day' buckets
    every init_time sharing a forecast day (see _forecast_day) into one case
    (WoFS-style, many initializations/day); 'init_time' keeps every real
    init_time as its own case (MPAS-style, one case per daily run)."""
    if case_grouping == "day":
        return _forecast_day(init_time, day_start_hour)
    if case_grouping == "init_time":
        return init_time
    raise ValueError(f"case_grouping must be one of {_VALID_CASE_GROUPINGS}, got {case_grouping!r}")


def _case_label(init_time, case_grouping: str, day_start_hour: int = 0) -> str:
    if case_grouping == "day":
        return _forecast_day(init_time, day_start_hour).strftime("%Y-%m-%d")
    if case_grouping == "init_time":
        return init_time.strftime("%Y-%m-%d %H%MZ")
    raise ValueError(f"case_grouping must be one of {_VALID_CASE_GROUPINGS}, got {case_grouping!r}")


def build_per_case_match_climatologies(
    systems: list[SystemSpec],
    lead_time_mode: Literal["intersection", "union"] = "intersection",
    lead_minutes: list | None = None,
    case_grouping: Literal["init_time", "day"] = "init_time",
    day_start_hour: int = 0,
) -> dict[str, dict[str, MatchLeadHourClimatology]]:
    """Returns {system_name: {case_label: MatchLeadHourClimatology}} -- one
    climatology PER CASE, not pooled across cases like
    build_multi_system_match_climatology(). Same lead_time_mode/lead_minutes
    semantics as that function (see multi_system.compute_included_keys).

    day_start_hour (0-23, 'day' grouping only; default 0 = UTC calendar
    date): the UTC hour a forecast day starts at, so e.g. 12 groups WoFS's
    17Z-03Z initializations into one case labeled by the evening's date."""
    if case_grouping not in _VALID_CASE_GROUPINGS:
        raise ValueError(f"case_grouping must be one of {_VALID_CASE_GROUPINGS}, got {case_grouping!r}")
    if isinstance(day_start_hour, bool) or not isinstance(day_start_hour, int) or not 0 <= day_start_hour <= 23:
        raise ValueError(f"day_start_hour must be an integer 0-23, got {day_start_hour!r}")

    loaded = load_systems(systems)
    included, _ = compute_included_keys(loaded, lead_time_mode=lead_time_mode, lead_minutes=lead_minutes)

    result: dict[str, dict[str, MatchLeadHourClimatology]] = {}
    for spec in systems:
        name = spec.name
        ls = loaded[name]

        keys_by_case: dict = {}
        for key in included[name]:
            init_time, _lm = key
            keys_by_case.setdefault(_case_key(init_time, case_grouping, day_start_hour), []).append(key)

        cases: dict[str, MatchLeadHourClimatology] = {}
        for case_key, case_keys in keys_by_case.items():
            n_hits, n_misses, n_false_alarms, n_truth_extra, n_forecast_extra = {}, {}, {}, {}, {}
            n_cases_per_lead_hour: dict = {}
            members_seen: set = set()
            _count_dicts = dict(zip(_CATEGORIES, (n_hits, n_misses, n_false_alarms, n_truth_extra, n_forecast_extra)))

            for key in case_keys:
                _init_time, lm = key
                lead_hour = lm / 60.0
                n_cases_per_lead_hour[lead_hour] = n_cases_per_lead_hour.get(lead_hour, 0) + 1
                for member_id, cat_arrays in ls.records_by_key.get(key, {}).items():
                    members_seen.add(member_id)
                    mkey = (member_id, lead_hour)
                    cats = np.concatenate(cat_arrays) if cat_arrays else np.array([], dtype=str)
                    for cat, target in _count_dicts.items():
                        target[mkey] = target.get(mkey, 0) + int(np.sum(cats == cat))

            # every key bucketed under one case_key shares the same real
            # init_time (init_time grouping) or the same forecast day
            # (day grouping), so any one of them yields the correct label.
            label = _case_label(case_keys[0][0], case_grouping, day_start_hour)
            if label in cases:
                raise ValueError(
                    f"System '{name}': case label '{label}' is not unique under case_grouping="
                    f"{case_grouping!r} -- two distinct cases formatted to the same label string"
                )

            cases[label] = MatchLeadHourClimatology(
                lead_hours=sorted(n_cases_per_lead_hour.keys()),
                members=sorted(members_seen),
                n_hits=n_hits, n_misses=n_misses, n_false_alarms=n_false_alarms,
                n_truth_extra=n_truth_extra, n_forecast_extra=n_forecast_extra,
                n_cases_per_lead_hour=n_cases_per_lead_hour,
            )
        result[name] = cases

    return result


def _target_lead_hours(clim: MatchLeadHourClimatology, lead_minutes_range: tuple | None) -> list:
    if lead_minutes_range is None:
        return list(clim.lead_hours)
    lo, hi = lead_minutes_range[0] / 60.0, lead_minutes_range[1] / 60.0
    return [lh for lh in clim.lead_hours if lo <= lh <= hi]


def case_object_count(
    clim: MatchLeadHourClimatology, lead_minutes_range: tuple | None = None, basis: str = "truth",
) -> int:
    """Total object count for one case's own MatchLeadHourClimatology,
    within lead_minutes_range (an inclusive (lo, hi) pair in MINUTES; None
    = the case's own full range) -- the marker-size input for the
    case-scatter performance diagrams.

    basis='truth' (default): the real MRMS/observed sample size behind
    POD (hits+misses+truth_extra) -- matches the legacy `hits+misses`
    sizing convention, extended to include truth_extra since that's also
    a real truth object. Uses ONE representative member (not summed
    across members), since truth objects are member-independent -- the
    same MRMS field is independently re-matched by every member -- so
    summing across members would inflate the count by ensemble size
    (mirrors aggregate_matches_multi_system.build_multi_system_object_climatology's
    own established "one representative member" canonical-truth convention).

    basis='forecast': hits+false_alarms+forecast_extra, averaged across
    members (ensemble mean, per this project's confirmed convention) --
    forecast objects ARE genuinely distinct per member, so a raw sum
    across members would scale with ensemble size rather than reflecting
    a typical member's own object count.
    """
    if basis not in ("truth", "forecast"):
        raise ValueError(f"basis must be 'truth' or 'forecast', got {basis!r}")
    if not clim.members:
        return 0

    target_lead_hours = _target_lead_hours(clim, lead_minutes_range)

    if basis == "truth":
        member = sorted(clim.members, key=lambda m: (m is None, m))[0]
        return sum(
            clim.n_hits.get((member, lh), 0) + clim.n_misses.get((member, lh), 0) + clim.n_truth_extra.get((member, lh), 0)
            for lh in target_lead_hours
        )

    total = sum(
        clim.n_hits.get((m, lh), 0) + clim.n_false_alarms.get((m, lh), 0) + clim.n_forecast_extra.get((m, lh), 0)
        for m in clim.members for lh in target_lead_hours
    )
    return round(total / len(clim.members))


def case_metrics(clim: MatchLeadHourClimatology, lead_minutes_range: tuple | None = None) -> dict:
    """Ensemble-mean POD/FAR/FBIAS/CSI for one case's own
    MatchLeadHourClimatology, aggregated over lead_minutes_range (an
    inclusive (lo, hi) pair in MINUTES; None = the case's own full range).

    Counts are summed across the target lead hours WITHIN one member
    first (not an average of per-lead-hour ratios, which would distort
    under unequal per-lead-hour sample sizes), giving one ratio per
    member for the whole range; those per-member ratios are then averaged
    across members (ensemble mean) -- the same two-step convention this
    project already uses for its pooled verification-stats ensemble mean
    (average of per-member ratios, not one ratio from pooled counts).
    """
    target_lead_hours = _target_lead_hours(clim, lead_minutes_range)
    if not clim.members or not target_lead_hours:
        return {"pod": 0.0, "far": 0.0, "fbias": 0.0, "csi": 0.0}

    pods, fars, fbiases, csis = [], [], [], []
    for member in clim.members:
        h = sum(clim.n_hits.get((member, lh), 0) for lh in target_lead_hours)
        m_ = sum(clim.n_misses.get((member, lh), 0) for lh in target_lead_hours)
        f = sum(clim.n_false_alarms.get((member, lh), 0) for lh in target_lead_hours)
        te = sum(clim.n_truth_extra.get((member, lh), 0) for lh in target_lead_hours)
        fe = sum(clim.n_forecast_extra.get((member, lh), 0) for lh in target_lead_hours)
        pods.append(compute_pod(h, m_))
        fars.append(compute_far(h, f))
        fbiases.append(compute_fbias(h, m_, f, te, fe))
        csis.append(compute_csi(h, m_, f))

    return {
        "pod": float(np.mean(pods)), "far": float(np.mean(fars)),
        "fbias": float(np.mean(fbiases)), "csi": float(np.mean(csis)),
    }


def split_baseline_and_highlight(
    per_case_climatologies: dict, case: str, baseline_cases: list | None = None,
) -> tuple:
    """per_case_climatologies: one system's own {case_label: MatchLeadHourClimatology}
    dict (from build_per_case_match_climatologies). Returns
    (baseline: list[MatchLeadHourClimatology], highlighted: MatchLeadHourClimatology)
    -- the shared partitioning logic behind the single-case-vs-baseline
    mode's three rendering variants (time series, case-scatter performance
    diagram, lead-time-trajectory performance diagram).

    baseline_cases=None (default): every OTHER case in per_case_climatologies.
    baseline_cases=[...]: an explicit list of case_labels to restrict the
    baseline population to (the highlighted case itself is excluded even
    if present in the list, since it's rendered separately, not as part
    of the baseline population).
    """
    if case not in per_case_climatologies:
        raise ValueError(
            f"case {case!r} not found among {len(per_case_climatologies)} available case(s): "
            f"{sorted(per_case_climatologies.keys())}"
        )
    highlighted = per_case_climatologies[case]

    if baseline_cases is None:
        baseline_labels = [c for c in per_case_climatologies if c != case]
    else:
        missing = [c for c in baseline_cases if c not in per_case_climatologies]
        if missing:
            raise ValueError(f"baseline_cases contains unknown case(s): {missing}")
        baseline_labels = [c for c in baseline_cases if c != case]

    baseline = [per_case_climatologies[c] for c in baseline_labels]
    return baseline, highlighted
