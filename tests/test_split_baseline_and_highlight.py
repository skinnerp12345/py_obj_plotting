"""Validation of climatology.aggregate_matches_per_case.split_baseline_and_highlight():
the shared partitioning logic behind the single-case-vs-baseline mode's
three rendering variants.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_split_baseline_and_highlight.py -v -s
"""

import pytest

from py_obj_plotting.climatology import split_baseline_and_highlight


def test_default_baseline_is_every_other_case():
    cases = {"c1": "clim1", "c2": "clim2", "c3": "clim3"}
    baseline, highlighted = split_baseline_and_highlight(cases, "c2")
    print(f"\n[split-check1] baseline={baseline}, highlighted={highlighted}")
    assert set(baseline) == {"clim1", "clim3"}
    assert highlighted == "clim2"


def test_explicit_baseline_list_restricts_population():
    cases = {"c1": "clim1", "c2": "clim2", "c3": "clim3", "c4": "clim4"}
    baseline, highlighted = split_baseline_and_highlight(cases, "c2", baseline_cases=["c1", "c3"])
    print(f"\n[split-check2] baseline={baseline}")
    assert set(baseline) == {"clim1", "clim3"}
    assert highlighted == "clim2"


def test_highlighted_case_excluded_from_explicit_baseline_even_if_present():
    cases = {"c1": "clim1", "c2": "clim2", "c3": "clim3"}
    baseline, highlighted = split_baseline_and_highlight(cases, "c2", baseline_cases=["c1", "c2", "c3"])
    print(f"\n[split-check3] baseline={baseline}")
    assert "clim2" not in baseline
    assert set(baseline) == {"clim1", "clim3"}


def test_missing_highlight_case_raises():
    cases = {"c1": "clim1"}
    with pytest.raises(ValueError, match="not found"):
        split_baseline_and_highlight(cases, "c99")


def test_unknown_baseline_case_raises():
    cases = {"c1": "clim1", "c2": "clim2"}
    with pytest.raises(ValueError, match="unknown case"):
        split_baseline_and_highlight(cases, "c1", baseline_cases=["c2", "c99"])
