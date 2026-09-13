"""Shared case/lead-hour <-> MRMS real-valid-time matching logic, used by
histogram-based (aggregate_by_lead_hour.py) lead-hour climatologies.

Handles only the "which MRMS valid_time matches this one forecast case's
own valid_time at a given lead hour, resolved once per case regardless of
how many per-lead-hour sub-items (ensemble members) share it" problem.
Callers supply their own per-case lead_hour->valid_time mapping and their
own MRMS candidate-time list, since discovering those differs by file
format -- that discovery logic is deliberately NOT shared, only the
matching/dedup itself.
"""

from py_obj.time_utils import nearest_within_tolerance


def resolve_case_matches(
    case_valid_times_by_lead_hour: dict,
    mrms_valid_times: list,
    tolerance_minutes: float,
) -> dict:
    """case_valid_times_by_lead_hour: {lead_hour: valid_time} for ONE case,
    already deduped across whatever per-lead-hour sub-items exist (e.g.
    ensemble members) -- every member of one case shares the exact same
    valid_time at a given lead_hour (confirmed real behavior for both the
    histogram and object data this project works with), so the caller only
    needs to supply one valid_time per lead_hour, not one per (member,
    lead_hour).

    Returns {lead_hour: matched_mrms_valid_time_or_None} -- None means no
    MRMS valid_time was found within tolerance_minutes; callers report this
    explicitly (e.g. in an `unmatched` list) rather than silently dropping it.
    """
    return {
        lead_hour: nearest_within_tolerance(valid_time, mrms_valid_times, tolerance_minutes)
        for lead_hour, valid_time in case_valid_times_by_lead_hour.items()
    }
