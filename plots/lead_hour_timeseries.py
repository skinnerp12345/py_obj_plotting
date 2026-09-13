"""Matched-percentile computation + time-series plotting over forecast lead
hour, built on top of climatology.aggregate_by_lead_hour's pooled
per-(member, lead_hour) MPAS and per-lead_hour MRMS distributions.

Reuses py_obj.histogram.match_percentile_threshold unchanged for the
actual CDF/percentile math -- this module only loops over lead hours/
members/thresholds and calls it, plus does the matplotlib presentation.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from py_obj.histogram import match_percentile_threshold

from .. import cb_colors
from ..climatology import LeadHourClimatology

_NAN = float("nan")

_DEFAULT_XLIM_PAD_MINUTES = 5.0


def _lead_hour_xlim(lead_hours: list, pad_minutes: float = _DEFAULT_XLIM_PAD_MINUTES) -> tuple:
    """(lo, hi) x-axis limits for a lead-hour time series, padded by
    pad_minutes on each side (converted to hours) so the first/last plotted
    points are visibly clear of the axis edge rather than sitting exactly on
    it. Shared by every lead-hour time series plot in this package."""
    pad_hours = pad_minutes / 60.0
    return min(lead_hours) - pad_hours, max(lead_hours) + pad_hours


def _apply_lead_hour_xticks(ax, lead_hours: list) -> None:
    """Tick marks every 6h, text labels only every 12h (the in-between 6h
    ticks are unlabeled) -- shared by both plots so they stay visually
    consistent with each other. Suited to multi-day forecast lead ranges
    (e.g. MPAS's 0-132h); for a short (few-hour), fine-cadence lead range,
    see _apply_lead_hour_xticks_fine() instead."""
    lo, hi = min(lead_hours), max(lead_hours)
    ticks = list(np.arange(lo - (lo % 6), hi + 1, 6))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t)}" if t % 12 == 0 else "" for t in ticks])


def _apply_lead_hour_xticks_fine(
    ax, lead_hours: list, tick_step_hours: float = 0.5, label_step_hours: float = 1.0,
) -> None:
    """Same tick-marks-vs-labels split as _apply_lead_hour_xticks(), but with
    caller-configurable steps -- for a short-range, fine-cadence lead axis
    (e.g. 0-3h at 10-min output) where a fixed 6h/12h step would put at most
    one labeled tick on the whole plot. Labels are placed only where a tick
    falls on a label_step_hours boundary (checked via a small floating-point
    tolerance, since lead_hours values like 1/6 h don't land on exact
    multiples otherwise)."""
    lo, hi = min(lead_hours), max(lead_hours)
    n_ticks = int(round((hi - lo) / tick_step_hours)) + 1
    ticks = [lo + i * tick_step_hours for i in range(n_ticks)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([
        f"{t:.2g}" if abs((t / label_step_hours) - round(t / label_step_hours)) < 1e-6 else ""
        for t in ticks
    ])


_MINUTE_DISPLAY_CANDIDATES_MIN = [5.0, 10.0, 15.0, 20.0, 30.0, 60.0]
_HOUR_DISPLAY_CANDIDATES_MIN = [60.0, 120.0, 180.0, 360.0, 720.0, 1440.0, 2880.0, 5760.0]
_HOUR_DISPLAY_THRESHOLD_MINUTES = 360.0  # >6h -> display in hours; <=6h -> display in minutes


def _choose_tick_step(span: float, candidate_steps: list, target_n_ticks: int = 10) -> float:
    """Picks the candidate step producing closest to target_n_ticks tick
    marks over the given span (ties favor the earlier/coarser candidate)."""
    if span <= 0:
        return candidate_steps[0]
    best, best_diff = candidate_steps[-1], None
    for step in candidate_steps:
        n_ticks = span / step + 1
        diff = abs(n_ticks - target_n_ticks)
        if best_diff is None or diff < best_diff - 1e-9:
            best, best_diff = step, diff
    return best


def adaptive_lead_hour_xticks(ax, lead_hours: list, target_n_ticks: int = 10) -> None:
    """Derives sensible tick-mark/label spacing directly from the actual
    lead_hours range -- replaces the previous hardcoded coarse
    (_apply_lead_hour_xticks: fixed 6h tick/12h label, tuned for MPAS-style
    multi-day ranges) vs. fine (_apply_lead_hour_xticks_fine: caller-
    configurable step, tuned for SFE-style short/fine-cadence ranges)
    binary choice with one function that adapts automatically. Same
    (ax, lead_hours) plug-in signature as those two, so it drops straight
    into any existing xtick_fn= call site.

    Display unit (minutes vs. hours) is chosen automatically from the span:
    <=6h displays in minutes (SFE-style short ranges), >6h displays in
    hours (MPAS-style multi-day ranges) -- matches this project's own
    established examples for each real case. Tick/label step reasoning is
    always done in minutes (this project's canonical lead-time unit)
    internally, even though the underlying x-axis DATA stays in hours
    (unchanged, so this is a true drop-in for existing call sites, which
    all already plot against a *_hours list). Because the display unit can
    change, this function ALSO sets ax.set_xlabel(...) (overriding
    whatever the caller set beforehand) so the axis label text always
    matches the unit actually shown in the tick labels -- the one
    deliberate difference from the two fixed-unit tick functions above,
    which never need to touch the label."""
    lo_h, hi_h = min(lead_hours), max(lead_hours)
    span_minutes = (hi_h - lo_h) * 60.0

    if span_minutes <= _HOUR_DISPLAY_THRESHOLD_MINUTES:
        candidates_min, unit_minutes = _MINUTE_DISPLAY_CANDIDATES_MIN, True
    else:
        candidates_min, unit_minutes = _HOUR_DISPLAY_CANDIDATES_MIN, False

    tick_step_min = _choose_tick_step(span_minutes, candidates_min, target_n_ticks=target_n_ticks)
    label_step_min = tick_step_min * 2.0  # matches the established 2x tick:label ratio (6h/12h, 0.5h/1h)
    tick_step_h = tick_step_min / 60.0

    lo_snap_h = lo_h - (lo_h % tick_step_h)
    n_ticks = int(round((hi_h - lo_snap_h) / tick_step_h)) + 1
    ticks_h = [lo_snap_h + i * tick_step_h for i in range(n_ticks)]

    labels = []
    for t_h in ticks_h:
        t_min = t_h * 60.0
        on_label_boundary = abs((t_min / label_step_min) - round(t_min / label_step_min)) < 1e-6
        if not on_label_boundary:
            labels.append("")
        else:
            labels.append(f"{t_min:.0f}" if unit_minutes else f"{t_h:.0f}")

    ax.set_xticks(ticks_h)
    ax.set_xticklabels(labels)
    ax.set_xlabel(f"Forecast lead time ({'minutes' if unit_minutes else 'hours'})")


def compute_matched_percentiles(clim: LeadHourClimatology, thresholds: tuple = (40.0, 45.0)) -> tuple:
    """For every lead hour in clim.lead_hours, every member, and every
    threshold: the MPAS dBZ value at the same percentile as MRMS's own
    threshold value in the pooled-for-that-lead-hour MRMS distribution.

    Returns (matched_value, source_percentile):
      matched_value[threshold][member] -- list[float], one per clim.lead_hours
        (index-aligned), NaN where no MRMS match (or no MPAS data for that
        member/lead_hour) exists for that lead hour.
      source_percentile[threshold] -- list[float] (0-100 scale), one per
        clim.lead_hours, member-independent (MRMS's own distribution only);
        NaN where MRMS has no pooled distribution for that lead hour.
    """
    matched_value = {t: {m: [] for m in clim.members} for t in thresholds}
    source_percentile = {t: [] for t in thresholds}

    for lead_hour in clim.lead_hours:
        mrms_h = clim.mrms_hist.get(lead_hour)
        for t in thresholds:
            if mrms_h is None:
                source_percentile[t].append(_NAN)
                for m in clim.members:
                    matched_value[t][m].append(_NAN)
                continue

            pct_this_lead_hour = _NAN
            for m in clim.members:
                mpas_h = clim.mpas_hist.get((m, lead_hour))
                if mpas_h is None:
                    matched_value[t][m].append(_NAN)
                    continue
                pct, val = match_percentile_threshold(clim.bins, mrms_h, t, clim.bins, mpas_h)
                matched_value[t][m].append(val)
                pct_this_lead_hour = pct * 100.0  # identical across members -- MRMS-only quantity
            source_percentile[t].append(pct_this_lead_hour)

    return matched_value, source_percentile


def compute_ensemble_mean_series(clim: LeadHourClimatology, matched_value: dict) -> dict:
    """Ensemble mean of matched_value[threshold][member] across members, at
    each lead hour -- returns {threshold: list[float]}, index-aligned with
    clim.lead_hours. NaN-safe (np.nanmean) even though real data has no
    per-lead-hour gaps once pooled across every contributing case."""
    ensemble_mean = {}
    for t, per_member in matched_value.items():
        stacked = np.array([per_member[m] for m in clim.members], dtype=float)
        ensemble_mean[t] = np.nanmean(stacked, axis=0).tolist()
    return ensemble_mean


def average_over_lead_hour_range(series: list, lead_hours: list, lo: float, hi: float) -> float:
    """Mean of `series` (index-aligned with lead_hours) over lead_hours in
    the inclusive range [lo, hi]. NaN-safe."""
    values = [v for lh, v in zip(lead_hours, series) if lo <= lh <= hi]
    if not values:
        raise ValueError(f"average_over_lead_hour_range: no lead hours found in [{lo}, {hi}]")
    return float(np.nanmean(values))


def range_over_lead_hour_range(series: list, lead_hours: list, lo: float, hi: float) -> tuple:
    """(min, max) of `series` (index-aligned with lead_hours) over lead_hours
    in the inclusive range [lo, hi]. NaN-safe."""
    values = [v for lh, v in zip(lead_hours, series) if lo <= lh <= hi]
    if not values:
        raise ValueError(f"range_over_lead_hour_range: no lead hours found in [{lo}, {hi}]")
    return float(np.nanmin(values)), float(np.nanmax(values))


def plot_matched_value_timeseries(
    clim: LeadHourClimatology, matched_value: dict, out_path: str,
    thresholds: tuple = (40.0, 45.0),
    member_alpha: float = 0.35,
    ensemble_mean_linewidth_increase: float = 0.5,
) -> None:
    """One figure, one line per (member, threshold) -- every member sharing
    the same color/linestyle within a threshold (per explicit request: no
    per-member visual distinction yet), so this reads as two clusters of
    thin, heavily semi-transparent lines (blue = 40 dBZ matched value, orange
    = 45 dBZ), plus one fully opaque, thicker ensemble-mean line per
    threshold on top."""
    threshold_colors = {40.0: cb_colors.blue5, 45.0: cb_colors.orange5}
    member_linewidth = 0.8
    ensemble_mean = compute_ensemble_mean_series(clim, matched_value)

    fig, ax = plt.subplots(figsize=(10, 6))
    for t in thresholds:
        color = threshold_colors.get(t, "black")
        for i, m in enumerate(clim.members):
            label = f"{t:.0f} dBZ matched value (member)" if i == 0 else None  # one legend entry per threshold
            ax.plot(clim.lead_hours, matched_value[t][m], color=color, linestyle="-",
                     linewidth=member_linewidth, alpha=member_alpha, label=label)
        ax.plot(clim.lead_hours, ensemble_mean[t], color=color, linestyle="-",
                 linewidth=member_linewidth + ensemble_mean_linewidth_increase, alpha=1.0,
                 label=f"{t:.0f} dBZ matched value (ensemble mean)")

    ax.set_xlabel("Forecast lead hour")
    ax.set_ylabel("Matched MPAS composite reflectivity (dBZ)")
    ax.set_title("MPAS ensemble dBZ value matched to MRMS's 40/45 dBZ percentile, by lead hour")
    ax.set_xlim(*_lead_hour_xlim(clim.lead_hours))
    ax.set_ylim(40, 60)
    _apply_lead_hour_xticks(ax, clim.lead_hours)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_source_percentile_timeseries(
    clim: LeadHourClimatology, source_percentile: dict, out_path: str,
    thresholds: tuple = (40.0, 45.0),
) -> None:
    """One figure: the percentile MRMS's own 40/45 dBZ threshold occurs at in
    the pooled-for-that-lead-hour MRMS distribution -- member-independent."""
    threshold_colors = {40.0: cb_colors.gray5, 45.0: "black"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for t in thresholds:
        color = threshold_colors.get(t, "black")
        ax.plot(clim.lead_hours, source_percentile[t], color=color, linestyle="-", linewidth=1.5,
                label=f"MRMS {t:.0f} dBZ percentile")

    ax.set_xlabel("Forecast lead hour")
    ax.set_ylabel("Percentile (%)")
    ax.set_title("MRMS 40/45 dBZ percentile in the pooled MRMS distribution, by lead hour")
    ax.set_xlim(*_lead_hour_xlim(clim.lead_hours))
    ax.set_ylim(99.5, 100)
    _apply_lead_hour_xticks(ax, clim.lead_hours)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
