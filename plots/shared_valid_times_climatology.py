"""Percentile-table computation + matplotlib plotting over
climatology.aggregate_shared_valid_times's per-source PDF/CDF (each source's
own climatology, restricted to valid times where every source has data).

Reuses py_obj.histogram.value_at_percentile unchanged for the actual
CDF lookup -- this module only loops over sources/thresholds and calls it,
plus does the matplotlib presentation.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from py_obj.histogram import value_at_percentile

from .. import cb_colors
from ..climatology import SharedValidTimesClimatology

DEFAULT_SYSTEM_COLORS = {
    "MRMS": "black",
    "WoFS": cb_colors.blue5,
    "WoFSCast": cb_colors.orange5,
    "NowcastNet": cb_colors.green5,
}


def compute_percentile_table(
    clim: SharedValidTimesClimatology, thresholds: tuple = (40.0, 45.0), mrms_name: str = "MRMS",
) -> list:
    """One row per threshold: the percentile MRMS's own threshold value
    occurs at in MRMS's shared-valid-time climatology, plus every other
    source's dBZ value at that same percentile (python_base's own matched-
    percentile-threshold method, applied here across sources instead of
    across forecast lead hours)."""
    mrms_cdf = clim.cdf[mrms_name]
    rows = []
    for t in thresholds:
        idx = int(np.argmin(np.abs(clim.bins[:-1] - t)))
        mrms_percentile = float(mrms_cdf[idx])
        row = {"threshold_dbz": t, "mrms_percentile": mrms_percentile * 100.0}
        for name in clim.system_names:
            if name == mrms_name:
                continue
            row[f"{name}_matched_value_dbz"] = value_at_percentile(clim.bins, clim.cdf[name], mrms_percentile)
        rows.append(row)
    return rows


def plot_pdf_cdf_per_system(
    clim: SharedValidTimesClimatology, out_dir: str, system_colors: dict = None,
) -> list:
    """One figure per source, two side-by-side single-axis panels (PDF,
    CDF) -- never a dual-axis combination of the two, since they're
    different measures/scales."""
    colors = system_colors or DEFAULT_SYSTEM_COLORS
    x = clim.bins[:-1]
    n_shared = len(clim.shared_valid_times)
    written = []
    for name in clim.system_names:
        color = colors.get(name, "black")
        fig, (ax_pdf, ax_cdf) = plt.subplots(1, 2, figsize=(12, 5))

        ax_pdf.plot(x, clim.pdf[name], color=color, linewidth=1.5)
        ax_pdf.set_xlabel("Composite reflectivity (dBZ)")
        ax_pdf.set_ylabel("Probability density")
        ax_pdf.set_title("PDF")
        ax_pdf.set_xlim(clim.bins[0], clim.bins[-1])
        ax_pdf.grid(alpha=0.3)

        ax_cdf.plot(x, clim.cdf[name], color=color, linewidth=1.5)
        ax_cdf.set_xlabel("Composite reflectivity (dBZ)")
        ax_cdf.set_ylabel("Cumulative probability")
        ax_cdf.set_title("CDF")
        ax_cdf.set_xlim(clim.bins[0], clim.bins[-1])
        ax_cdf.set_ylim(0, 1)
        ax_cdf.grid(alpha=0.3)

        fig.suptitle(f"{name} reflectivity climatology -- {n_shared} shared valid times")
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"pdf_cdf_{name.replace(' ', '_')}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        written.append(out_path)
    return written


def plot_combined_pdf(clim: SharedValidTimesClimatology, out_path: str, system_colors: dict = None) -> None:
    colors = system_colors or DEFAULT_SYSTEM_COLORS
    x = clim.bins[:-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in clim.system_names:
        ax.plot(x, clim.pdf[name], color=colors.get(name, "black"), linewidth=1.5, label=name)
    ax.set_xlabel("Composite reflectivity (dBZ)")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Reflectivity climatology comparison -- {len(clim.shared_valid_times)} shared valid times")
    ax.set_xlim(clim.bins[0], clim.bins[-1])
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_combined_cdf(
    clim: SharedValidTimesClimatology, out_path: str, thresholds: tuple = (40.0, 45.0), system_colors: dict = None,
    xlim: tuple = None, ylim: tuple = (0.0, 1.0),
) -> None:
    """xlim: optional (min_dbz, max_dbz) to zoom the x-axis -- e.g. (30, 70)
    to inspect the threshold-relevant range without the clear-air-dominated
    rise near 0 dBZ compressing it. Defaults to the full bin range.
    ylim: (min, max) cumulative-probability range, default full (0, 1) --
    e.g. (0.9, 1.0) to zoom into the upper tail where the 40/45 dBZ
    thresholds actually sit."""
    colors = system_colors or DEFAULT_SYSTEM_COLORS
    x = clim.bins[:-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in clim.system_names:
        ax.plot(x, clim.cdf[name], color=colors.get(name, "black"), linewidth=1.5, label=name)
    for t in thresholds:
        ax.axvline(t, color=cb_colors.gray5, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Composite reflectivity (dBZ)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(f"Reflectivity climatology CDF comparison -- {len(clim.shared_valid_times)} shared valid times")
    ax.set_xlim(xlim if xlim is not None else (clim.bins[0], clim.bins[-1]))
    ax.set_ylim(ylim)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
