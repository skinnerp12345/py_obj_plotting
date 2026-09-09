"""Validation of climatology.verification_stats.compute_verification_stats --
confirms it correctly wires MatchLeadHourClimatology's pooled counts through
contingency_stats.py's formulas/bootstrap for every (member, lead_hour) bin,
against hand-computed expected point estimates.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_verification_stats.py -v -s
"""

from py_obj_plotting.climatology import MatchLeadHourClimatology, compute_verification_stats


def test_compute_verification_stats_matches_hand_computed_point_estimates():
    clim = MatchLeadHourClimatology(
        lead_hours=[0.0, 1.0],
        members=["mem_1", "mem_2"],
        n_hits={("mem_1", 0.0): 8, ("mem_1", 1.0): 0, ("mem_2", 0.0): 0, ("mem_2", 1.0): 0},
        n_misses={("mem_1", 0.0): 2, ("mem_1", 1.0): 0, ("mem_2", 0.0): 0, ("mem_2", 1.0): 0},
        n_false_alarms={("mem_1", 0.0): 4, ("mem_1", 1.0): 0, ("mem_2", 0.0): 0, ("mem_2", 1.0): 0},
        n_truth_extra={("mem_1", 0.0): 1, ("mem_1", 1.0): 0, ("mem_2", 0.0): 0, ("mem_2", 1.0): 0},
        n_forecast_extra={("mem_1", 0.0): 3, ("mem_1", 1.0): 0, ("mem_2", 0.0): 0, ("mem_2", 1.0): 0},
    )

    stats = compute_verification_stats(clim, bootstrap_samples=200)

    print(f"\n[verif-stats-check1] pod={stats.pod[('mem_1', 0.0)]}, far={stats.far[('mem_1', 0.0)]}, "
          f"fbias={stats.fbias[('mem_1', 0.0)]}, csi={stats.csi[('mem_1', 0.0)]}")
    assert abs(stats.pod[("mem_1", 0.0)] - 8 / 10) < 1e-9
    assert abs(stats.far[("mem_1", 0.0)] - 4 / 12) < 1e-9
    assert abs(stats.csi[("mem_1", 0.0)] - 8 / 14) < 1e-9
    assert abs(stats.fbias[("mem_1", 0.0)] - 15 / 11) < 1e-9

    # zero-count bin (mem_1 @ lead=1, and all of mem_2) must be exactly 0, not NaN/crash
    assert stats.pod[("mem_1", 1.0)] == 0.0
    assert stats.pod[("mem_2", 0.0)] == 0.0

    # CI widths must exist for every bin and be non-negative
    for key in [("mem_1", 0.0), ("mem_1", 1.0), ("mem_2", 0.0), ("mem_2", 1.0)]:
        assert stats.pod_ci_width[key] >= 0.0
        assert stats.far_ci_width[key] >= 0.0
        assert stats.fbias_ci_width[key] >= 0.0
        assert stats.csi_ci_width[key] >= 0.0

    assert stats.lead_hours == [0.0, 1.0]
    assert stats.members == ["mem_1", "mem_2"]
