"""N-system comparison layer: derives which (forecast case, lead_minutes)
combinations are actually available/included across N SystemSpecs, per an
explicit, orthogonal `lead_time_mode` -- "intersection" (default: strict,
every system must have that (cycle, lead_minutes) combination) or "union"
(every system contributes whatever it independently has). Either way, any
partial-availability gap is always reported via MultiSystemAvailabilityReport,
never silently pooled or dropped.

Generalizes an explicit two-value policy layered on top of systems.py's
auto-discovered LoadedSystem.lead_minutes. A caller who needs a mixed
per-lead-minute-set behavior (e.g. "strict
except these 3 early lead minutes") can reconstruct it by calling
compute_included_keys() twice (once "union" restricted to the relaxed lead
minutes, once "intersection" restricted to the rest) and unioning the two
`included` dicts -- see py_obj_plotting/tests/test_multi_system.py's
regression test for a concrete worked example against real data.
"""

from dataclasses import dataclass
from typing import Literal

from .systems import LoadedSystem, SystemSpec, load_system


@dataclass
class MultiSystemAvailabilityReport:
    lead_minutes: list
    lead_time_mode: str
    shared_cycles: list  # init_times present (at all) in every system's own match files
    system_only_cycles: dict  # system_name -> cycles found in that system but no other
    excluded_partial_lead_minutes: dict  # (cycle_init_time, lead_minutes) -> [systems missing it] -- only
                                          # populated for "intersection"-mode lead minutes, where a
                                          # partial-availability (cycle, lead) combination is a real,
                                          # reportable exclusion rather than something silently pooled


@dataclass
class CaseAlignmentReport:
    """Design #7: whether the systems being compared actually cover the
    same set of forecast cases (init_times), a genuinely different failure
    mode than lead-time coverage differing (MultiSystemAvailabilityReport,
    above) -- case-level mismatch can exist even when lead-time handling is
    otherwise correct."""
    shared_cases: list  # init_times present in EVERY system being compared
    system_only_cases: dict  # system_name -> cases found ONLY in that system (no other system has them at all)
    partial_cases: dict  # case -> sorted list of system names that DO have it, for any case present
                          # in SOME but not ALL systems -- a strict superset of system_only_cases (also
                          # catches e.g. a case shared by 2 of 3 systems but missing from the 3rd, which
                          # would not show up as "only" in any single system's own list)
    is_fully_aligned: bool  # True iff every case found in ANY system is found in EVERY system


def check_shared_forecast_cases(
    loaded: dict[str, LoadedSystem], require_shared_cases: bool = True,
) -> CaseAlignmentReport:
    """Reports (never silently drops) whether the systems being compared
    cover the same forecast cases. require_shared_cases=True (the config
    default) raises a clear, named error listing exactly which cases are
    mismatched when they don't fully agree; require_shared_cases=False
    returns the same report without raising, so a caller can proceed using
    only shared_cases."""
    system_names = list(loaded.keys())
    per_system_cases = {name: set(ls.init_times) for name, ls in loaded.items()}
    all_cases = set().union(*per_system_cases.values()) if per_system_cases else set()
    shared = set.intersection(*per_system_cases.values()) if per_system_cases else set()

    system_only = {}
    for name in system_names:
        others = (
            set().union(*(v for n, v in per_system_cases.items() if n != name))
            if len(system_names) > 1 else set()
        )
        system_only[name] = sorted(per_system_cases[name] - others)

    partial = {}
    for case in sorted(all_cases - shared):
        partial[case] = sorted(name for name in system_names if case in per_system_cases[name])

    report = CaseAlignmentReport(
        shared_cases=sorted(shared), system_only_cases=system_only,
        partial_cases=partial, is_fully_aligned=(shared == all_cases),
    )

    if require_shared_cases and not report.is_fully_aligned:
        lines = [f"  {name}: {len(cases)} exclusively-own case(s)" for name, cases in system_only.items() if cases]
        raise ValueError(
            f"check_shared_forecast_cases: the {len(system_names)} systems being compared do not all "
            f"cover the same set of forecast cases (require_shared_cases=True). "
            f"{len(shared)}/{len(all_cases)} total case(s) are shared by every system; "
            f"{len(report.partial_cases)} case(s) are present in only some systems.\n"
            + "\n".join(lines)
            + "\nSet require_shared_cases=False to proceed using only the shared subset (never silently pooled)."
        )
    return report


def load_systems(systems: list[SystemSpec]) -> dict[str, LoadedSystem]:
    """One load_system() call per SystemSpec. Duplicate names are almost
    certainly a caller mistake (silently overwriting one system's own
    availability with another's), so this raises rather than picking one."""
    out: dict[str, LoadedSystem] = {}
    for spec in systems:
        if spec.name in out:
            raise ValueError(f"duplicate SystemSpec name '{spec.name}' -- system names must be unique")
        out[spec.name] = load_system(spec)
    return out


def compute_included_keys(
    loaded: dict[str, LoadedSystem],
    lead_time_mode: Literal["intersection", "union"] = "intersection",
    lead_minutes: list | None = None,
) -> tuple[dict[str, set], MultiSystemAvailabilityReport]:
    """Decides, per system, which (init_time, lead_minutes) keys actually
    get included in a downstream aggregation.

    lead_minutes=None (default) uses the union of every system's own
    auto-discovered lead_minutes (systems.load_system) as the candidate
    axis -- no caller hand-enumeration required. Pass an explicit list to
    restrict to a subset.

    lead_time_mode="intersection" (default): a (cycle, lead_minutes)
    combination is only included if EVERY system has it -- partial
    availability is excluded and reported (excluded_partial_lead_minutes),
    never silently pooled from whichever systems happen to have it.
    lead_time_mode="union": each system's own presence at a given
    lead_minutes suffices independently -- a system missing a lead_minutes
    another system has simply doesn't contribute a point there; there is no
    exclusion event, since there was nothing to reconcile with a system that
    doesn't have that lead_minutes at all.
    """
    if lead_time_mode not in ("intersection", "union"):
        raise ValueError(f"lead_time_mode must be 'intersection' or 'union', got {lead_time_mode!r}")

    system_names = list(loaded.keys())
    if lead_minutes is None:
        candidate_minutes = sorted(set().union(*(set(ls.lead_minutes) for ls in loaded.values()))) if loaded else []
    else:
        candidate_minutes = sorted(lead_minutes)
    wanted_minutes = set(candidate_minutes)

    present_keys = {
        name: {k for k in ls.records_by_key.keys() if k[1] in wanted_minutes}
        for name, ls in loaded.items()
    }

    per_system_cycles = {name: {c for c, _ in keys} for name, keys in present_keys.items()}
    shared_cycles = set.intersection(*per_system_cycles.values()) if per_system_cycles else set()
    system_only_cycles = {}
    for name in system_names:
        others = (
            set().union(*(v for n, v in per_system_cycles.items() if n != name))
            if len(system_names) > 1 else set()
        )
        system_only_cycles[name] = sorted(per_system_cycles[name] - others)

    all_cycles = set().union(*per_system_cycles.values()) if per_system_cycles else set()

    included: dict = {name: set() for name in system_names}
    excluded_partial: dict = {}

    for cycle in all_cycles:
        for lm in candidate_minutes:
            present_in = [name for name in system_names if (cycle, lm) in present_keys[name]]
            if not present_in:
                continue  # no system has this (cycle, lead) at all -- nothing to include or report
            if lead_time_mode == "union":
                for name in present_in:
                    included[name].add((cycle, lm))
            elif len(present_in) == len(system_names):
                for name in system_names:
                    included[name].add((cycle, lm))
            else:
                missing = [n for n in system_names if n not in present_in]
                excluded_partial[(cycle, lm)] = missing

    report = MultiSystemAvailabilityReport(
        lead_minutes=candidate_minutes, lead_time_mode=lead_time_mode,
        shared_cycles=sorted(shared_cycles), system_only_cycles=system_only_cycles,
        excluded_partial_lead_minutes=excluded_partial,
    )
    return included, report
