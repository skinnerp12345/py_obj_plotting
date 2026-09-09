"""adaptive_lead_hour_xticks() validation: confirms it derives distinct,
sensible tick/label spacing (and display unit) automatically from the
actual lead_hours range, replacing the old hardcoded coarse/fine binary
choice -- checked against the real ranges this project's own example data
actually has (SFE-style short/fine-cadence, MPAS-style 132h multi-day),
with exact values computed directly (not eyeballed) before being written
into these assertions.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_adaptive_lead_ticks.py -v -s
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from py_obj_plotting.plots.lead_hour_timeseries import adaptive_lead_hour_xticks


def _apply(lead_hours):
    fig, ax = plt.subplots()
    adaptive_lead_hour_xticks(ax, lead_hours)
    ticks = list(ax.get_xticks())
    labels = [t.get_text() for t in ax.get_xticklabels()]
    xlabel = ax.get_xlabel()
    plt.close(fig)
    return ticks, labels, xlabel


# --- Check 1: real SFE range (WoFS/WoFSCast, 0-6h at 5-min cadence) --------

def test_sfe_wofs_range_uses_minutes_display_with_sensible_spacing():
    lead_hours = [i / 60.0 * 5 for i in range(73)]  # 0..360 min by 5
    ticks, labels, xlabel = _apply(lead_hours)
    print(f"\n[adaptive-check1] n_ticks={len(ticks)}, labels={labels}, xlabel={xlabel!r}")
    assert xlabel == "Forecast lead time (minutes)"
    assert ticks == [i * 0.5 for i in range(13)]  # 0, 0.5, 1.0, ..., 6.0h -> 30-min tick step
    assert labels == ["0", "", "60", "", "120", "", "180", "", "240", "", "300", "", "360"]


# --- Check 2: real SFE range (NowcastNet, 10-180min at 10-min cadence) ----

def test_sfe_nowcastnet_range_uses_minutes_display():
    lead_hours = [i / 60.0 for i in range(10, 190, 10)]
    ticks, labels, xlabel = _apply(lead_hours)
    print(f"\n[adaptive-check2] n_ticks={len(ticks)}, labels={labels}, xlabel={xlabel!r}")
    assert xlabel == "Forecast lead time (minutes)"
    assert len(ticks) == 10
    assert labels == ["0", "", "40", "", "80", "", "120", "", "160", ""]


# --- Check 3: real MPAS range (0-132h) -- must switch to hours display ----

def test_mpas_132h_range_uses_hours_display_with_coarser_spacing():
    lead_hours = list(range(133))
    ticks, labels, xlabel = _apply(lead_hours)
    print(f"\n[adaptive-check3] n_ticks={len(ticks)}, labels={labels}, xlabel={xlabel!r}")
    assert xlabel == "Forecast lead time (hours)"
    assert ticks == [i * 12.0 for i in range(12)]  # 0, 12, 24, ..., 132h -> 12h tick step
    assert labels == ["0", "", "24", "", "48", "", "72", "", "96", "", "120", ""]


# --- Check 4: distinctness -- the whole point of "adaptive" ----------------

def test_short_and_long_ranges_produce_genuinely_different_spacing():
    """Direct proof this is NOT still a hardcoded binary choice: two very
    differently-scaled ranges must yield different tick steps AND
    different display units, derived purely from each range's own span."""
    short_ticks, _, short_label = _apply([i / 60.0 for i in range(0, 190, 10)])
    long_ticks, _, long_label = _apply(list(range(133)))

    short_step_min = round((short_ticks[1] - short_ticks[0]) * 60.0)
    long_step_min = round((long_ticks[1] - long_ticks[0]) * 60.0)
    print(f"\n[adaptive-check4] short: step={short_step_min}min, unit={short_label}; "
          f"long: step={long_step_min}min, unit={long_label}")
    assert short_label != long_label
    assert short_step_min != long_step_min
    assert short_step_min < long_step_min  # short range must get finer spacing, not coarser


# --- Check 5: edge cases -----------------------------------------------------

def test_single_lead_hour_does_not_crash():
    ticks, labels, xlabel = _apply([2.0])
    print(f"\n[adaptive-check5] single lead hour: ticks={ticks}")
    assert len(ticks) >= 1


def test_target_n_ticks_is_respected_approximately():
    """A caller-requested tick density should visibly change the result
    (not be silently ignored) -- fewer requested ticks over the same span
    must produce a coarser (larger) step."""
    lead_hours = list(range(133))
    fig, ax = plt.subplots()
    adaptive_lead_hour_xticks(ax, lead_hours, target_n_ticks=5)
    few_ticks = list(ax.get_xticks())
    plt.close(fig)

    fig, ax = plt.subplots()
    adaptive_lead_hour_xticks(ax, lead_hours, target_n_ticks=20)
    many_ticks = list(ax.get_xticks())
    plt.close(fig)

    print(f"\n[adaptive-check6] target=5 -> {len(few_ticks)} ticks; target=20 -> {len(many_ticks)} ticks")
    assert len(few_ticks) < len(many_ticks)
