"""Combines MatchLeadHourClimatology's pooled contingency-table counts with
contingency_stats.py's pure formulas/bootstrap into one lead-hour-indexed
VerificationStats -- computed once per (member, lead_hour) bin so all four
metrics' point estimates and CI widths are available without repeating the
(relatively expensive, 1000-sample) bootstrap once per plot.
"""

from dataclasses import dataclass, field

from .aggregate_matches_by_lead_hour import MatchLeadHourClimatology
from .contingency_stats import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_CI_INTERVAL,
    bootstrap_ci_widths,
    compute_csi,
    compute_far,
    compute_fbias,
    compute_pod,
)


@dataclass
class VerificationStats:
    lead_hours: list
    members: list
    pod: dict = field(default_factory=dict)  # (member, lead_hour) -> float
    far: dict = field(default_factory=dict)
    fbias: dict = field(default_factory=dict)
    csi: dict = field(default_factory=dict)
    pod_ci_width: dict = field(default_factory=dict)
    far_ci_width: dict = field(default_factory=dict)
    fbias_ci_width: dict = field(default_factory=dict)
    csi_ci_width: dict = field(default_factory=dict)


def compute_verification_stats(
    clim: MatchLeadHourClimatology,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    ci_interval: float = DEFAULT_CI_INTERVAL,
) -> VerificationStats:
    stats = VerificationStats(lead_hours=clim.lead_hours, members=clim.members)

    for member in clim.members:
        for lh in clim.lead_hours:
            key = (member, lh)
            h = clim.n_hits.get(key, 0)
            m = clim.n_misses.get(key, 0)
            f = clim.n_false_alarms.get(key, 0)
            te = clim.n_truth_extra.get(key, 0)
            fe = clim.n_forecast_extra.get(key, 0)

            stats.pod[key] = compute_pod(h, m)
            stats.far[key] = compute_far(h, f)
            stats.csi[key] = compute_csi(h, m, f)
            stats.fbias[key] = compute_fbias(h, m, f, te, fe)

            pod_w, far_w, fbias_w, csi_w = bootstrap_ci_widths(
                h, m, f, te, fe, samples=bootstrap_samples, interval=ci_interval,
            )
            stats.pod_ci_width[key] = pod_w
            stats.far_ci_width[key] = far_w
            stats.fbias_ci_width[key] = fbias_w
            stats.csi_ci_width[key] = csi_w

    return stats
