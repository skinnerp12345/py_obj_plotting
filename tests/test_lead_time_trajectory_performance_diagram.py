"""Validation of plots.performance_diagram.plot_lead_time_trajectory_performance_diagram:
synthetic checks for the trajectory line (alpha fading with lead time) and
interval labeling, plus real end-to-end checks against already-known/
already-validated VerificationStats from the real SFE 3-system and MPAS
2-threshold comparisons.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_lead_time_trajectory_performance_diagram.py -v -s
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection

from py_obj_plotting import cb_colors
from py_obj_plotting.climatology import (
    VerificationStats,
    build_multi_system_match_climatology,
    compute_verification_stats,
)
from py_obj_plotting.plots.performance_diagram import plot_lead_time_trajectory_performance_diagram
from py_obj_plotting.systems import SystemSpec, assign_colors

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SFE_MATCHES_DIR = os.path.join(REPO_ROOT, "sfe_poster", "sfe_matches")
NCAR_MPAS_DIR = os.path.join(REPO_ROOT, "ncar_mpas_analysis")
_SKIP_NO_SFE_POSTER = pytest.mark.skipif(
    not os.path.isdir(SFE_MATCHES_DIR), reason="requires the external sfe_poster/sfe_matches/ dataset (not bundled)",
)
_SKIP_NO_NCAR_MPAS_ANALYSIS = pytest.mark.skipif(
    not os.path.isdir(NCAR_MPAS_DIR), reason="requires the external ncar_mpas_analysis/ dataset (not bundled)",
)


def _stats_fixture(lead_hours, pod_by_lh, far_by_lh, members=("mem_1",)) -> VerificationStats:
    pod = {(m, lh): pod_by_lh[lh] for m in members for lh in lead_hours}
    far = {(m, lh): far_by_lh[lh] for m in members for lh in lead_hours}
    zero = {(m, lh): 0.0 for m in members for lh in lead_hours}
    return VerificationStats(
        lead_hours=list(lead_hours), members=list(members),
        pod=pod, pod_ci_width=zero, far=far, far_ci_width=zero,
        csi=zero, csi_ci_width=zero, fbias=zero, fbias_ci_width=zero,
    )


def test_trajectory_line_alpha_increases_with_lead_time(tmp_path):
    lead_hours = [0.0, 1.0, 2.0, 3.0]
    pod_by_lh = {lh: 0.5 for lh in lead_hours}
    far_by_lh = {lh: 0.3 for lh in lead_hours}
    stats = _stats_fixture(lead_hours, pod_by_lh, far_by_lh)
    series = [("A", cb_colors.blue5, stats)]

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_lead_time_trajectory_performance_diagram(series, out_path=str(tmp_path / "out.png"), label_interval_minutes=None)
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    lcs = [c for c in ax.collections if isinstance(c, LineCollection)]
    print(f"\n[trajectory-check1] n_line_collections={len(lcs)}")
    assert len(lcs) == 1
    colors = lcs[0].get_colors()
    alphas = [c[3] for c in colors]
    print(f"[trajectory-check1] alphas={alphas}")
    assert len(alphas) == len(lead_hours) - 1  # n_segments = n_points - 1
    assert alphas == sorted(alphas)  # strictly non-decreasing with lead time
    assert alphas[0] < alphas[-1]
    plt.close(fig)


def test_interval_labels_appear_only_at_configured_interval(tmp_path):
    lead_hours = [0.0, 0.5, 1.0, 1.5, 2.0]  # 0, 30, 60, 90, 120 min
    pod_by_lh = {lh: 0.5 for lh in lead_hours}
    far_by_lh = {lh: 0.3 for lh in lead_hours}
    stats = _stats_fixture(lead_hours, pod_by_lh, far_by_lh)
    series = [("A", cb_colors.blue5, stats)]

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_lead_time_trajectory_performance_diagram(series, out_path=str(tmp_path / "out.png"), label_interval_minutes=60)
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    lead_labels = [t.get_text() for t in ax.texts if t.get_text() in ("0m", "30m", "1h", "90m", "2h")]
    print(f"\n[trajectory-check2] lead_labels={lead_labels}")
    # every 60min: 0, 60, 120 qualify -- NOT 30 or 90 -- but only HALF of
    # the qualifying markers are actually drawn (every other one), so 60min
    # ("1h") is dropped, leaving just the first and third qualifying points.
    assert set(lead_labels) == {"0m", "2h"}
    plt.close(fig)


def test_two_systems_with_different_lead_time_coverage_share_aligned_markers(tmp_path):
    """Real reported issue: comparing two systems with different lead-time
    coverage (e.g. WoFS vs. WoFSCast, where WoFSCast has fewer available
    lead times) previously computed each system's qualifying/halved
    marker set independently from its OWN lead_hours, so the two
    trajectories' markers could land at entirely different, non-
    overlapping lead times. Fixed by using the INTERSECTION of every
    system's lead_hours as the shared candidate set before halving.

    Constructed so the bug is unambiguous: computed independently, system
    A's own qualifying set is {0,60,120}min and system B's is {0,90}min
    -- disjoint past 0min. Using the shared intersection, BOTH systems
    must show markers at exactly {0,90}min (rendered as "0m"/"2h" --
    _format_lead_label's existing, unchanged rounding-to-nearest-hour
    behavior for >=60min renders 90min as "2h", not "90m")."""
    lead_hours_a = [m / 60.0 for m in (0, 15, 30, 45, 60, 75, 90, 105, 120)]
    lead_hours_b = [m / 60.0 for m in (0, 30, 90, 120)]
    stats_a = _stats_fixture(lead_hours_a, {lh: 0.5 for lh in lead_hours_a}, {lh: 0.3 for lh in lead_hours_a})
    stats_b = _stats_fixture(lead_hours_b, {lh: 0.5 for lh in lead_hours_b}, {lh: 0.3 for lh in lead_hours_b})
    series = [("A", cb_colors.blue5, stats_a), ("B", cb_colors.orange5, stats_b)]

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_lead_time_trajectory_performance_diagram(series, out_path=str(tmp_path / "out.png"), label_interval_minutes=30)
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    scatters = [c for c in ax.collections if isinstance(c, matplotlib.collections.PathCollection)
                and c.get_sizes()[0] == pytest.approx(40.0)]  # _TRAJECTORY_LABEL_MARKER_SIZE, excludes contour artifacts
    a_markers = [c for c in scatters if tuple(c.get_facecolor()[0][:3]) == pytest.approx(cb_colors.blue5)]
    b_markers = [c for c in scatters if tuple(c.get_facecolor()[0][:3]) == pytest.approx(cb_colors.orange5)]
    print(f"\n[trajectory-shared] n_a_markers={len(a_markers)}, n_b_markers={len(b_markers)}")
    # both systems get exactly 2 markers (the shared {0min, 90min}), not
    # their own independently-computed (and disjoint-past-0) sets.
    assert len(a_markers) == 2
    assert len(b_markers) == 2

    lead_labels = sorted({t.get_text() for t in ax.texts if t.get_text() in ("0m", "30m", "60m", "90m", "1h", "2h")})
    print(f"[trajectory-shared] lead_labels={lead_labels}")
    assert lead_labels == ["0m", "2h"]  # both systems' labels, deduplicated (90min renders as "2h")
    plt.close(fig)


def test_no_interval_labels_when_none(tmp_path):
    lead_hours = [0.0, 0.5, 1.0]
    stats = _stats_fixture(lead_hours, {lh: 0.5 for lh in lead_hours}, {lh: 0.3 for lh in lead_hours})
    series = [("A", cb_colors.blue5, stats)]

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_lead_time_trajectory_performance_diagram(series, out_path=str(tmp_path / "out.png"), label_interval_minutes=None)
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    lead_labels = [t.get_text() for t in ax.texts if t.get_text() in ("0m", "30m", "60m")]
    print(f"\n[trajectory-check3] lead_labels={lead_labels}")
    assert lead_labels == []
    plt.close(fig)


def test_empty_series_raises(tmp_path):
    with pytest.raises(ValueError, match="series must be non-empty"):
        plot_lead_time_trajectory_performance_diagram([], out_path=str(tmp_path / "out.png"))


# --- Real data ----------------------------------------------------------------

@_SKIP_NO_SFE_POSTER
def test_real_sfe_3system_trajectory(tmp_path):
    systems = assign_colors([
        SystemSpec("WoFS", os.path.join(SFE_MATCHES_DIR, "wofs_vs_mrms")),
        SystemSpec("WoFSCast", os.path.join(SFE_MATCHES_DIR, "wofscast_vs_mrms")),
        SystemSpec("NowcastNet", os.path.join(SFE_MATCHES_DIR, "nowcastnet_vs_mrms")),
    ])
    clims, _ = build_multi_system_match_climatology(systems, lead_time_mode="union")
    vs = {s.name: compute_verification_stats(clims[s.name]) for s in systems}
    series = [(s.name, s.color, vs[s.name]) for s in systems]

    out_path = str(tmp_path / "sfe_trajectory.png")
    plot_lead_time_trajectory_performance_diagram(series, out_path=out_path, label_interval_minutes=30, title="SFE trajectory")
    print(f"\n[trajectory-check4] n_lead_hours: {[(n, len(vs[n].lead_hours)) for n in ('WoFS','WoFSCast','NowcastNet')]}")
    assert os.path.isfile(out_path)
    assert len(vs["WoFS"].lead_hours) == 73
    assert len(vs["WoFSCast"].lead_hours) == 35
    assert len(vs["NowcastNet"].lead_hours) == 18


@_SKIP_NO_NCAR_MPAS_ANALYSIS
def test_real_mpas_trajectory_12h_labels(tmp_path):
    systems = assign_colors([
        SystemSpec("100km", os.path.join(NCAR_MPAS_DIR, "matched_objects_100km")),
        SystemSpec("400km", os.path.join(NCAR_MPAS_DIR, "matched_objects_400km")),
    ])
    clims, _ = build_multi_system_match_climatology(systems, lead_time_mode="intersection")
    vs = {s.name: compute_verification_stats(clims[s.name]) for s in systems}
    series = [(s.name, s.color, vs[s.name]) for s in systems]

    orig_close = plt.close
    plt.close = lambda *a, **k: None
    plot_lead_time_trajectory_performance_diagram(series, out_path=str(tmp_path / "mpas_traj.png"), label_interval_minutes=720)
    plt.close = orig_close

    fig = plt.gcf()
    ax = fig.axes[0]
    hour_labels = {t.get_text() for t in ax.texts if t.get_text().endswith("h") or t.get_text() == "0m"}
    print(f"\n[trajectory-check5] hour_labels={sorted(hour_labels)}")
    # only HALF of the 12h-interval-qualifying points (0,12,24,...,132,
    # 12 total) are actually drawn -- every other one, starting from 0.
    all_qualifying_minutes = list(range(0, 133 * 60, 12 * 60))
    expected_minutes = all_qualifying_minutes[::2]
    expected = {("0m" if m < 60 else f"{m // 60}h") for m in expected_minutes}
    assert hour_labels == expected
    plt.close(fig)
