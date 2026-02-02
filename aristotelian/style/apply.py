"""Core utilities for applying PRH plotting style.

This module provides the base style application functions that set up
matplotlib with PRH aesthetics. Use these at the start of plotting scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt

from .palettes import LINE_COLORS
from .tokens import (
    ICML_DOUBLE_COL,
    ICML_SINGLE_COL,
    PRH_AX_FACE,
    PRH_BASE_FONT,
    PRH_DPI,
    PRH_GRID,
    PRH_GRID_ALPHA,
    PRH_GRID_WIDTH,
    PRH_LABEL,
    PRH_LABELSIZE,
    PRH_LEGEND_ALPHA,
    PRH_LEGEND_FACECOLOR,
    PRH_LEGEND_SHADOW,
    PRH_LEGENDSIZE,
    PRH_LINEWIDTH,
    PRH_MARKEREDGE,
    PRH_MARKERSIZE,
    PRH_PAD_INCHES,
    PRH_SANS,
    PRH_SAVE_DPI,
    PRH_SPINE,
    PRH_SPINE_WIDTH,
    PRH_TICK,
    PRH_TICK_DIRECTION,
    PRH_TICK_MAJOR_SIZE,
    PRH_TICK_MAJOR_WIDTH,
    PRH_TICKSIZE,
    PRH_TITLESIZE,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure


def use_prh_base(mplstyle_path: str | Path | None = None) -> None:
    """Apply PRH base style to matplotlib.

    This should be called once at the start of a plotting script or notebook.
    It sets up the global matplotlib rcParams with PRH aesthetics adapted
    for ICML paper format.

    Args:
        mplstyle_path: Optional path to .mplstyle file to load first.
                      If None, sets rcParams directly.

    Example:
        >>> from aristotelian.style.apply import use_prh_base
        >>> use_prh_base("styles/paper.mplstyle")
        >>> # Now all plots will use PRH style
    """
    # Load style file if provided
    if mplstyle_path is not None:
        plt.style.use(str(mplstyle_path))

    # Hard-enforce the critical PRH parameters
    # This ensures consistency even if other styles were loaded
    mpl.rcParams.update(
        {
            # Figure settings
            "figure.facecolor": "white",
            "figure.dpi": PRH_DPI,
            # Axes settings
            "axes.facecolor": PRH_AX_FACE,
            "axes.edgecolor": PRH_SPINE,
            "axes.linewidth": PRH_SPINE_WIDTH,
            "axes.grid": True,  # Grid on by default
            "axes.axisbelow": True,  # Grid behind data
            "axes.labelcolor": PRH_LABEL,
            "axes.labelsize": PRH_LABELSIZE,
            "axes.titlesize": PRH_TITLESIZE,
            "axes.titleweight": "normal",
            "axes.labelpad": 3,
            "axes.titlepad": 8,
            # Grid settings
            "grid.color": PRH_GRID,
            "grid.linewidth": PRH_GRID_WIDTH,
            "grid.alpha": PRH_GRID_ALPHA,
            # Font settings
            "font.family": "sans-serif",
            "font.sans-serif": PRH_SANS,
            "font.size": PRH_BASE_FONT,
            # Tick settings
            "xtick.color": PRH_TICK,
            "ytick.color": PRH_TICK,
            "xtick.labelsize": PRH_TICKSIZE,
            "ytick.labelsize": PRH_TICKSIZE,
            "xtick.direction": PRH_TICK_DIRECTION,
            "ytick.direction": PRH_TICK_DIRECTION,
            "xtick.major.size": PRH_TICK_MAJOR_SIZE,
            "ytick.major.size": PRH_TICK_MAJOR_SIZE,
            "xtick.major.width": PRH_TICK_MAJOR_WIDTH,
            "ytick.major.width": PRH_TICK_MAJOR_WIDTH,
            # Line and marker settings
            "lines.linewidth": PRH_LINEWIDTH,
            "lines.markersize": PRH_MARKERSIZE,
            "lines.markeredgewidth": PRH_MARKEREDGE,
            # Legend settings (base, can be overridden per plot)
            "legend.frameon": True,
            "legend.framealpha": PRH_LEGEND_ALPHA,
            "legend.facecolor": PRH_LEGEND_FACECOLOR,
            "legend.edgecolor": PRH_SPINE,
            "legend.fontsize": PRH_LEGENDSIZE,
            "legend.shadow": PRH_LEGEND_SHADOW,
            "legend.columnspacing": 1.0,
            "legend.labelspacing": 0.3,
            # Color cycle - use project palette
            "axes.prop_cycle": plt.cycler(color=LINE_COLORS),
            # Export settings
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": PRH_PAD_INCHES,
            "savefig.dpi": PRH_SAVE_DPI,
            "pdf.fonttype": 42,  # TrueType fonts in PDF
            "ps.fonttype": 42,  # TrueType fonts in PostScript
        }
    )


def style_axes_base(ax: Axes, *, grid: bool = True) -> None:
    """Apply base PRH styling to an axes object.

    This applies the fundamental PRH aesthetics: gray background, grid,
    and colored spines/ticks. Does NOT modify spine visibility - that's
    handled by preset-specific functions.

    Args:
        ax: Matplotlib axes to style
        grid: Whether to show grid (default True)

    Example:
        >>> fig, ax = plt.subplots()
        >>> style_axes_base(ax)
    """
    # Background and grid
    ax.set_facecolor(PRH_AX_FACE)
    ax.grid(grid, which="major", axis="both")
    ax.set_axisbelow(True)

    # Style all spines (visibility controlled separately)
    for spine in ax.spines.values():
        spine.set_color(PRH_SPINE)
        spine.set_linewidth(PRH_SPINE_WIDTH)

    # Tick colors
    ax.tick_params(axis="both", colors=PRH_TICK, which="major")


def hide_spines(ax: Axes, top: bool = True, right: bool = True) -> None:
    """Hide specific spines on an axes.

    Args:
        ax: Matplotlib axes
        top: Hide top spine (default True)
        right: Hide right spine (default True)

    Example:
        >>> hide_spines(ax, top=True, right=True)  # Fig. 9 style
    """
    if top:
        ax.spines["top"].set_visible(False)
    if right:
        ax.spines["right"].set_visible(False)


def prh_colorbar(
    fig: Figure,
    ax: Axes,
    mappable: mpl.cm.ScalarMappable,
    *,
    label: str | None = None,
    fraction: float = 0.046,
    pad: float = 0.04,
) -> Colorbar:
    """Create a PRH-styled colorbar.

    Args:
        fig: Figure containing the axes
        ax: Axes to attach colorbar to
        mappable: The image/contour/etc to create colorbar for
        label: Optional label for the colorbar
        fraction: Fraction of axes width for colorbar
        pad: Padding between axes and colorbar

    Returns:
        The created Colorbar object

    Example:
        >>> im = ax.imshow(data, cmap='viridis')
        >>> prh_colorbar(fig, ax, im, label="Value")
    """
    cbar = fig.colorbar(mappable, ax=ax, fraction=fraction, pad=pad, label=label)
    # Style the outline
    cbar.outline.set_linewidth(PRH_SPINE_WIDTH * 0.5)
    cbar.outline.set_edgecolor(PRH_SPINE)
    return cbar


def get_figure_size(single_column: bool = True) -> tuple[float, float]:
    """Get appropriate figure size for ICML format.

    Args:
        single_column: If True, return single column width (3.25")
                      If False, return double column width (6.75")

    Returns:
        Tuple of (width, height) in inches
    """
    return ICML_SINGLE_COL if single_column else ICML_DOUBLE_COL
