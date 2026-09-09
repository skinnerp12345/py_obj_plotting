"""Small, self-contained color constants for py_obj_plotting -- deliberately
NOT importing python_base/wofs_colortables.py, since py_obj_plotting (like
python_obj) is meant to be able to stand alone with no dependency on the
legacy python_base/ directory. Values copied from wofs_colortables.py's own
ColorBrewer-derived palette (only the specific entries this package uses).
"""

import numpy as np
from matplotlib.colors import ListedColormap

blue5 = (107 / 255., 174 / 255., 214 / 255.)
orange5 = (253 / 255., 141 / 255., 60 / 255.)
gray5 = (150 / 255., 150 / 255., 150 / 255.)
gray7 = (82 / 255., 82 / 255., 82 / 255.)
purple5 = (158 / 255., 154 / 255., 200 / 255.)
green5 = (116 / 255., 196 / 255., 118 / 255.)
red5 = (251 / 255., 106 / 255., 74 / 255.)
gray8 = (37 / 255., 37 / 255., 37 / 255.)

# Standard NWS composite-reflectivity palette (MetPy-derived, 5-dBZ steps
# 20-70), copied verbatim (values only) from wofs_colortables.py's own
# nws_dz_cmap -- used with REFLECTIVITY_LEVELS via a BoundaryNorm, per the
# established convention in python_test_plot/plot_regrid_step1.py.
_dz20 = (0.0, 1.0, 0.0)
_dz25 = (0.0, 0.7843137254901961, 0.0)
_dz30 = (0.0, 0.5647058823529412, 0.0)
_dz35 = (1.0, 1.0, 0.0)
_dz40 = (0.9058823529411765, 0.7529411764705882, 0.0)
_dz45 = (1.0, 0.5647058823529412, 0.0)
_dz50 = (1.0, 0.0, 0.0)
_dz55 = (0.8392156862745098, 0.0, 0.0)
_dz60 = (0.7529411764705882, 0.0, 0.0)
_dz65 = (1.0, 0.0, 1.0)
_dz70 = (0.6, 0.3333333333333333, 0.788235294117647)

nws_dz_cmap = ListedColormap([_dz20, _dz25, _dz30, _dz35, _dz40, _dz45, _dz50, _dz55, _dz60, _dz65, _dz70])
REFLECTIVITY_LEVELS = np.arange(20.0, 75.0, 5.0)
