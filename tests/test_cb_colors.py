"""Validation of cb_colors.py's reflectivity colormap constant.

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_cb_colors.py -v -s
"""

import numpy as np
import pytest

from py_obj_plotting import cb_colors


def test_nws_dz_cmap_has_eleven_colors():
    assert cb_colors.nws_dz_cmap.N == 11


def test_reflectivity_levels_match_expected_range():
    print(f"\n[cb-colors-check1] levels={cb_colors.REFLECTIVITY_LEVELS}")
    assert np.allclose(cb_colors.REFLECTIVITY_LEVELS, np.arange(20.0, 75.0, 5.0))
    assert len(cb_colors.REFLECTIVITY_LEVELS) == 11


def test_nws_dz_cmap_first_color_is_green():
    color = cb_colors.nws_dz_cmap(0)
    print(f"\n[cb-colors-check2] first color={color}")
    assert color == pytest.approx((0.0, 1.0, 0.0, 1.0))
