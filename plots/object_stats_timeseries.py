"""Shared member/ensemble-mean line-drawing helpers, reused by several
comparison-plot modules (object_stats_comparison.py, performance_diagram.py,
single_case_highlight.py, verification_stats_timeseries.py) that each draw
thin, semi-transparent per-member lines plus one thicker, fully-opaque
ensemble-mean line for a forecast series, optionally alongside a single
black MRMS reference line.
"""

import numpy as np

from .. import cb_colors

_MEMBER_LINEWIDTH = 0.8
_MEMBER_ALPHA = 0.35
_ENSEMBLE_MEAN_LINEWIDTH_INCREASE = 0.5
_MRMS_LINEWIDTH = 1.7  # matches the ensemble-mean weight -- MRMS has no member spread of its own


def _member_series(values_by_member_lead: dict, members: list, lead_hours: list) -> dict:
    return {m: [values_by_member_lead.get((m, lh), 0) for lh in lead_hours] for m in members}


def _ensemble_mean(series_by_member: dict, members: list) -> list:
    stacked = np.array([series_by_member[m] for m in members], dtype=float)
    return np.mean(stacked, axis=0).tolist()


def _mrms_series(values_by_lead: dict, lead_hours: list) -> list:
    return [values_by_lead.get(lh, float("nan")) for lh in lead_hours]


def _plot_member_ensemble_lines(
    ax, lead_hours: list, members: list, values_by_member_lead: dict,
    color=cb_colors.orange5, series_label: str = "MPAS",
) -> None:
    """Draws one climatology's thin, semi-transparent member lines + one
    thicker, fully-opaque ensemble-mean line onto an EXISTING ax -- no
    bootstrap CI band, since there is none for a raw summed count/area
    quantity, unlike a contingency-table statistic."""
    per_member = _member_series(values_by_member_lead, members, lead_hours)
    ensemble_mean = _ensemble_mean(per_member, members)
    for i, m in enumerate(members):
        label = f"{series_label} (member)" if i == 0 else None
        ax.plot(lead_hours, per_member[m], color=color, linestyle="-",
                 linewidth=_MEMBER_LINEWIDTH, alpha=_MEMBER_ALPHA, label=label)
    ax.plot(lead_hours, ensemble_mean, color=color, linestyle="-",
             linewidth=_MEMBER_LINEWIDTH + _ENSEMBLE_MEAN_LINEWIDTH_INCREASE, alpha=1.0,
             label=None)
