"""Validation of plots.performance_diagram.init_performance_diagram() --
confirms it faithfully reproduces python_base/wofs_verif_cbook.py's
perf_diagram_init()/perf_diagram_cbar_init() styling: axis limits/ticks,
CSI contour levels, bias radial-line levels, figure geometry -- checked
numerically (contour line collection levels, tick label text), never by
looking at the rendered image (this project's "no direct image analysis"
rule).

Run with: /opt/anaconda3/envs/pysteps_env/bin/python -m pytest py_obj_plotting/tests/test_performance_diagram.py -v -s
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from py_obj_plotting import cb_colors
from py_obj_plotting.plots.performance_diagram import init_performance_diagram


def test_axes_limits_and_labels():
    fig, ax = init_performance_diagram()
    print(f"\n[perfdiag-check1] xlim={ax.get_xlim()}, ylim={ax.get_ylim()}, "
          f"xlabel={ax.get_xlabel()!r}, ylabel={ax.get_ylabel()!r}")
    assert ax.get_xlim() == (0.0, 1.0)
    assert ax.get_ylim() == (0.0, 1.0)
    assert ax.get_xlabel() == "Success Ratio (1 - FAR)"
    assert ax.get_ylabel() == "Probability of Detection"
    plt.close(fig)


def test_tick_positions_and_blanked_endpoint_labels():
    fig, ax = init_performance_diagram()
    xticks = list(ax.get_xticks())
    xlabels = [t.get_text() for t in ax.get_xticklabels()]
    print(f"\n[perfdiag-check2] xticks={xticks}, xlabels={xlabels}")
    assert np.allclose(xticks, [0.1 * i for i in range(11)])
    assert xlabels[0] == "" and xlabels[-1] == ""  # 0 and 1.0 endpoints blanked
    assert xlabels[1:-1] == ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
    plt.close(fig)


def test_csi_and_bias_contour_levels_match_legacy_exactly():
    """Direct numeric proof this reproduces perf_diagram_init()'s exact
    CSI (np.arange(0.1,1.1,0.1)) and bias ([0.25,0.5,1.,1.5,2.,3.,5.])
    contour levels, not an approximation."""
    fig, ax = init_performance_diagram()
    csi_con, bias_con = ax.collections[0], ax.collections[1]
    csi_levels = np.array(csi_con.levels)
    bias_levels = np.array(bias_con.levels)
    print(f"\n[perfdiag-check3] csi_levels={csi_levels}, bias_levels={bias_levels}")
    assert np.allclose(csi_levels, np.arange(0.1, 1.1, 0.1))
    assert np.allclose(bias_levels, [0.25, 0.5, 1., 1.5, 2., 3., 5.])
    plt.close(fig)


def test_csi_contour_is_solid_purple_bias_is_gray_with_darker_unity_line():
    fig, ax = init_performance_diagram()
    csi_con, bias_con = ax.collections[0], ax.collections[1]
    csi_colors = [tuple(c[:3]) for c in csi_con.get_edgecolor()]
    bias_colors = [tuple(c[:3]) for c in bias_con.get_edgecolor()]
    print(f"\n[perfdiag-check4] csi_colors[0]={csi_colors[0]}, bias_colors={bias_colors}")
    assert all(np.allclose(c, cb_colors.purple5) for c in csi_colors)
    # bias levels are [0.25, 0.5, 1., 1.5, 2., 3., 5.] -- index 2 (bias=1) must be the darker gray7
    assert np.allclose(bias_colors[2], cb_colors.gray7)
    for i, c in enumerate(bias_colors):
        if i != 2:
            assert np.allclose(c, cb_colors.gray5)
    plt.close(fig)


def test_figsize_no_colorbar_vs_reserve_colorbar():
    """reserve_colorbar widens the figure (room for a colorbar) but must
    NOT change its height -- both variants reserve identical headroom
    above the (identically square) axes for a title."""
    fig1, ax1 = init_performance_diagram()
    fig2, ax2 = init_performance_diagram(reserve_colorbar=True)
    print(f"\n[perfdiag-check5] no_cbar={fig1.get_size_inches()}, reserve_cbar={fig2.get_size_inches()}")
    assert tuple(fig1.get_size_inches()) == (8.0, 8.44)
    assert tuple(fig2.get_size_inches()) == (9.4, 8.44)
    plt.close(fig1)
    plt.close(fig2)


def test_reserved_colorbar_width_fits_colorbar_and_label():
    """The colorbar's tick labels and axis label must fit inside the figure
    (regression: the old 0.75in reserve cut ~0.4in of them off)."""
    from matplotlib.colors import BoundaryNorm
    from py_obj_plotting.plots.performance_diagram import _add_colorbar_axes

    fig, ax = init_performance_diagram(reserve_colorbar=True)
    norm = BoundaryNorm(np.arange(-0.15, 0.1501, 0.025), 256)
    scat = ax.scatter([0.5], [0.5], c=[0.05], cmap="RdBu_r", norm=norm)
    cbar = fig.colorbar(scat, cax=_add_colorbar_axes(fig), extend="both")
    cbar.set_label("CSI difference (SystemA - SystemB)")
    fig.canvas.draw()
    right_in = cbar.ax.get_tightbbox(fig.canvas.get_renderer()).x1 / fig.dpi
    print(f"\n[perfdiag-cbar-fit] colorbar right edge {right_in:.3f}in, figure width {fig.get_size_inches()[0]}in")
    assert right_in < fig.get_size_inches()[0]
    plt.close(fig)


def test_axes_box_is_square_in_physical_inches_both_variants():
    """The actual SR-vs-POD plot area (not the whole figure) must be
    exactly square in physical inches, for both the plain and
    reserve_colorbar variants -- reserve_colorbar's extra figure width
    must land entirely outside the axes' own box, not stretch it."""
    for reserve_colorbar in (False, True):
        fig, ax = init_performance_diagram(reserve_colorbar=reserve_colorbar)
        fig_w_in, fig_h_in = fig.get_size_inches()
        bbox = ax.get_position()
        axes_w_in, axes_h_in = bbox.width * fig_w_in, bbox.height * fig_h_in
        print(f"\n[perfdiag-check-square] reserve_colorbar={reserve_colorbar}, "
              f"axes_size_in=({axes_w_in:.4f}, {axes_h_in:.4f})")
        assert axes_w_in == pytest.approx(axes_h_in, abs=1e-6)
        plt.close(fig)


def test_axes_stays_square_after_a_real_colorbar_is_attached():
    """Regression test for a real reported bug: fig.colorbar(scat, ax=ax)
    auto-shrinks `ax` to make room for the colorbar (confirmed directly:
    it cut the main axes' own width by ~20%, e.g. 7.04in -> 5.63in, while
    leaving height untouched) -- silently breaking the square aspect ratio
    that init_performance_diagram(reserve_colorbar=True) is supposed to
    guarantee. The fix (_add_colorbar_axes -> an explicit, separately-
    positioned cax) must keep the main axes square even after a real
    colorbar is drawn, not just immediately after init_performance_diagram()
    returns (which the check above alone would not have caught)."""
    from py_obj_plotting.plots.performance_diagram import _add_colorbar_axes

    fig, ax = init_performance_diagram(reserve_colorbar=True)
    fig_w_in, fig_h_in = fig.get_size_inches()
    before = ax.get_position()
    before_w_in, before_h_in = before.width * fig_w_in, before.height * fig_h_in

    scat = ax.scatter([0.5], [0.5], c=[0.5])
    cax = _add_colorbar_axes(fig)
    fig.colorbar(scat, cax=cax, extend="both")

    after = ax.get_position()
    after_w_in, after_h_in = after.width * fig_w_in, after.height * fig_h_in
    print(f"\n[perfdiag-check-cbar-regression] before=({before_w_in:.4f},{before_h_in:.4f}), "
          f"after=({after_w_in:.4f},{after_h_in:.4f})")
    assert after_w_in == pytest.approx(before_w_in, abs=1e-6)
    assert after_h_in == pytest.approx(before_h_in, abs=1e-6)
    assert after_w_in == pytest.approx(after_h_in, abs=1e-6)  # still square
    plt.close(fig)


def test_title_has_headroom_above_axes():
    """A supplied title must not be clipped -- confirmed by checking the
    axes' top edge sits well below the figure's top edge (nonzero
    headroom), not just that a title string was set."""
    fig, ax = init_performance_diagram()
    ax.set_title("A two-line title\nfor good measure", fontsize=13)
    bbox = ax.get_position()
    headroom_fraction = 1.0 - (bbox.y0 + bbox.height)
    fig_h_in = fig.get_size_inches()[1]
    headroom_in = headroom_fraction * fig_h_in
    print(f"\n[perfdiag-check-title] headroom_in={headroom_in:.3f}")
    assert headroom_in > 0.3  # comfortably more than the old ~2%-of-8in (~0.16in)
    plt.close(fig)


def test_explicit_figsize_override():
    fig, ax = init_performance_diagram(figsize=(6.0, 6.0))
    assert tuple(fig.get_size_inches()) == (6.0, 6.0)
    plt.close(fig)


def test_spines_styled_per_legacy():
    fig, ax = init_performance_diagram()
    for side in ("top", "bottom", "left", "right"):
        spine = ax.spines[side]
        assert spine.get_alpha() == 0.7
        assert spine.get_linewidth() == 0.5
    plt.close(fig)
