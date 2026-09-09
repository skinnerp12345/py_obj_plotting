"""SystemSpec: the one core abstraction that unifies "systems being
compared" (e.g. WoFS vs. WoFSCast vs. NowcastNet) and "matching thresholds
being compared" (e.g. ncar_mpas_analysis's 100/200/300/400 km directories)
into a single list mechanism -- both cases already reduce to "N named
directories of match_init_*.nc files, each optionally colored." One list of
SystemSpec replaces the legacy scripts' hardcoded MATCH_DIRS/SYSTEM_COLORS/
THRESHOLDS constants.

load_system() is the paired loader: wraps the already-fixed read_match_file
machinery's metadata reader (aggregate_matches_by_lead_hour._read_match_metadata,
reused not reimplemented), normalizes lead time to minutes always -- never
hours -- (avoids float-precision drift converting a 5-minute cadence into
fractional hours; minutes divide evenly into every real cadence found in
this project's example data: 5, 10, 60 min), and treats member_id as a
fully opaque string throughout (no format assumed anywhere -- confirmed
real, incompatible conventions exist side by side in this project's own
example data: "mem00".."mem17" vs. "mem_1".."mem_5"). Auto-discovers which
lead minutes/forecast cases/members a system actually has data for by
reading its files, rather than requiring a caller to hand-enumerate a
lead_minutes list up front.
"""

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .cb_colors import blue5, gray5, green5, orange5, purple5
from .climatology.aggregate_matches_by_lead_hour import _discover, _read_match_metadata

# Fixed order, never cycled -- a 6th system needs an explicit SystemSpec.color
# rather than silently wrapping back to blue5 (dataviz convention: color
# follows the entity, assigned once in a stable order).
DEFAULT_COLOR_PALETTE = [blue5, orange5, green5, purple5, gray5]


@dataclass
class SystemSpec:
    """One forecast/matching system (or matching-threshold variant) to
    compare. `match_dir` holds file_grouping="init_snapshot"-shaped
    match_init_*.nc files (python_obj.obj_core.run_matching_series output).
    `color` auto-assigns from DEFAULT_COLOR_PALETTE in list order if left
    None -- see assign_colors()."""
    name: str
    match_dir: str
    color: tuple | None = None
    match_pattern: str = "match_init_*.nc"


def assign_colors(systems: list[SystemSpec]) -> list[SystemSpec]:
    """Returns NEW SystemSpec instances with any missing .color filled in
    from DEFAULT_COLOR_PALETTE, in list order. Systems that already specify
    a color are left untouched and don't consume a palette slot."""
    n_needed = sum(1 for s in systems if s.color is None)
    if n_needed > len(DEFAULT_COLOR_PALETTE):
        raise ValueError(
            f"{n_needed} systems need an auto-assigned color but DEFAULT_COLOR_PALETTE "
            f"only has {len(DEFAULT_COLOR_PALETTE)} entries -- pass explicit SystemSpec.color values"
        )
    out = []
    palette_idx = 0
    for s in systems:
        if s.color is not None:
            out.append(s)
            continue
        out.append(dataclasses.replace(s, color=DEFAULT_COLOR_PALETTE[palette_idx]))
        palette_idx += 1
    return out


@dataclass
class LoadedSystem:
    """One SystemSpec's fully-discovered content: which forecast cases
    (init_times), members (opaque strings; None means deterministic -- no
    member dimension in ANY of this system's files), and lead minutes
    (ALWAYS minutes, auto-discovered from the files themselves) it actually
    has data for."""
    spec: SystemSpec
    files: list
    init_times: list  # sorted, unique datetime -- forecast cases found
    member_ids: list | None  # sorted opaque strings; None = deterministic
    lead_minutes: list  # sorted, unique floats (whole-minute values), across every file
    # records_by_key[(init_time, lead_minutes)][member_id] -> list of category
    # ndarrays (member_id is None for a deterministic system's one implicit
    # member, matching this project's existing object-file convention for a
    # non-ensemble source). A list, not a single concatenated array, since in
    # principle more than one file could contribute to the same key --
    # concatenation is left to the aggregation layer.
    records_by_key: dict = field(default_factory=dict)
    n_files_used_stored_init_time: int = 0
    n_files_used_filename_init_time: int = 0


def load_system(spec: SystemSpec) -> LoadedSystem:
    files = _discover(spec.match_dir, spec.match_pattern)

    init_times_seen: set = set()
    member_ids_seen: set = set()
    lead_minutes_seen: set = set()
    records_by_key: dict = {}
    n_stored, n_filename = 0, 0
    saw_any_members = False
    saw_any_deterministic = False

    for f in files:
        meta = _read_match_metadata(f)
        if meta.init_time is None:
            raise ValueError(
                f"'{f}' has no init_time (neither a stored attribute nor a parseable "
                f"match_init_<YYYYMMDD>_<HHMMSS>.nc filename) -- SystemSpec requires every "
                "match file to belong to a resolvable forecast case"
            )
        if meta.init_time_from_filename:
            n_filename += 1
        else:
            n_stored += 1
        init_times_seen.add(meta.init_time)

        lead_min_by_ti = {
            ti: round((vt - meta.init_time).total_seconds() / 60.0) for ti, vt in enumerate(meta.valid_times)
        }
        lead_minutes_seen.update(lead_min_by_ti.values())

        has_members = meta.member_ids is not None and meta.member_index is not None
        if has_members:
            saw_any_members = True
            member_ids_seen.update(meta.member_ids)
        else:
            saw_any_deterministic = True

        # time_index is only written when a file spans >1 distinct valid_time
        # (write_match_file's own use_time_dim rule) -- a file with exactly
        # one lead time has no match_time_index variable at all, so every
        # record in it belongs to that file's single (implicit) time index 0.
        def _time_mask(ti: int) -> np.ndarray:
            if meta.time_index is not None:
                return meta.time_index == ti
            return np.ones(len(meta.category), dtype=bool)

        for ti, lm in lead_min_by_ti.items():
            key = (meta.init_time, lm)
            tmask = _time_mask(ti)
            if has_members:
                for mi, member_id in enumerate(meta.member_ids):
                    mask = (meta.member_index == mi) & tmask
                    records_by_key.setdefault(key, {}).setdefault(member_id, []).append(meta.category[mask])
            else:
                records_by_key.setdefault(key, {}).setdefault(None, []).append(meta.category[tmask])

    if saw_any_members and saw_any_deterministic:
        raise ValueError(
            f"System '{spec.name}' ({spec.match_dir}) has some files with a member dimension "
            "and some without -- a system must be consistently ensemble or deterministic"
        )

    return LoadedSystem(
        spec=spec,
        files=files,
        init_times=sorted(init_times_seen),
        member_ids=sorted(member_ids_seen) if saw_any_members else None,
        lead_minutes=sorted(lead_minutes_seen),
        records_by_key=records_by_key,
        n_files_used_stored_init_time=n_stored,
        n_files_used_filename_init_time=n_filename,
    )
