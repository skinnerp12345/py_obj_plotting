"""Shared low-level drawing helper for verification-statistics (POD, FAR,
FBIAS, CSI) time series over forecast lead hour: member lines + ensemble-
mean line + shaded bootstrap-CI band, built on top of
climatology.compute_verification_stats's pooled per-(member, lead_hour)
contingency-table metrics + bootstrap CI widths.

Factored to take a caller-supplied `ax` and `color` rather than creating its
own figure, so a multi-system comparison plot (verification_stats_comparison.py)
can call it once per system on one shared figure/ax.
"""

import matplotlib

matplotlib.use("Agg")

from .. import cb_colors
from .object_stats_timeseries import _ensemble_mean, _member_series

_MEMBER_LINEWIDTH = 0.8
_MEMBER_ALPHA = 0.35
_ENSEMBLE_MEAN_LINEWIDTH_INCREASE = 0.5
_CI_SHADE_ALPHA = 0.2
_DEFAULT_COLOR = cb_colors.orange5


def _plot_member_ensemble_series(
    ax, lead_hours: list, members: list,
    values_by_member_lead: dict, ci_width_by_member_lead: dict,
    color=_DEFAULT_COLOR, series_label: str = "MPAS",
) -> None:
    """Draws one climatology's member lines + ensemble-mean line + shaded
    bootstrap-CI band onto an EXISTING ax, in a given color.

    values_by_member_lead/ci_width_by_member_lead: dict[(member, lead_hour)]
    -> float, e.g. VerificationStats.pod / VerificationStats.pod_ci_width.
    A CI width is the FULL width of the interval (upper - lower percentile,
    see contingency_stats.bootstrap_ci_widths), so the band is drawn at
    +/- half of it -- the interval the width came from, centered on the line.

    A DETERMINISTIC system (members == [None], the sentinel this project
    uses throughout for "no member dimension at all," e.g. a deterministic
    forecast) renders as one bold line with its own bootstrap-CI band, not
    member-alpha spaghetti drawn over a fake "1 member" -- a real 1-member
    ensemble (a real member name, not the None sentinel) still gets the
    member-line treatment.
    """
    per_member = _member_series(values_by_member_lead, members, lead_hours)
    ensemble_mean = _ensemble_mean(per_member, members)

    per_member_ci = _member_series(ci_width_by_member_lead, members, lead_hours)
    mean_ci_width = _ensemble_mean(per_member_ci, members)
    lower = [m - w / 2.0 for m, w in zip(ensemble_mean, mean_ci_width)]
    upper = [m + w / 2.0 for m, w in zip(ensemble_mean, mean_ci_width)]
    ax.fill_between(lead_hours, lower, upper, color=color, alpha=_CI_SHADE_ALPHA, linewidth=0)

    if members == [None]:
        ax.plot(lead_hours, ensemble_mean, color=color, linestyle="-",
                 linewidth=_MEMBER_LINEWIDTH + _ENSEMBLE_MEAN_LINEWIDTH_INCREASE, alpha=1.0,
                 label=series_label)
        return

    for i, m in enumerate(members):
        label = f"{series_label} (member)" if i == 0 else None
        ax.plot(lead_hours, per_member[m], color=color, linestyle="-",
                 linewidth=_MEMBER_LINEWIDTH, alpha=_MEMBER_ALPHA, label=label)
    ax.plot(lead_hours, ensemble_mean, color=color, linestyle="-",
             linewidth=_MEMBER_LINEWIDTH + _ENSEMBLE_MEAN_LINEWIDTH_INCREASE, alpha=1.0,
             label=None)
