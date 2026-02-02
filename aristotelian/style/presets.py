"""Plot type presets for PRH style.

This module provides preset styling functions for different plot types:
- Heatmaps (imshow, contour)
- Line plots (with or without error bars)
- Scatter plots (Fig. 3 style with white edges)
- Bar plots
- Histograms

Each preset applies appropriate PRH aesthetics for that plot type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .apply import hide_spines, style_axes_base
from .tokens import (
    PRH_ERR_CAPSIZE,
    PRH_ERR_LW,
    PRH_LEGEND_FRAME_WIDTH,
    PRH_LEGENDSIZE,
    PRH_SCATTER_EDGE,
    PRH_SCATTER_EW,
    PRH_SCATTER_SIZE,
    PRH_SPINE,
)

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.container import ErrorbarContainer
    from matplotlib.legend import Legend
    from matplotlib.patches import Rectangle


def style_heatmap_axes(ax: Axes) -> None:
    """Style axes for heatmaps (imshow, pcolormesh, contour).

    Heatmaps use:
    - Gray background (PRH aesthetic)
    - Grid OFF (would interfere with continuous color)
    - All spines visible (boxed appearance)

    Args:
        ax: Axes to style

    Example:
        >>> fig, ax = plt.subplots()
        >>> style_heatmap_axes(ax)
        >>> ax.imshow(data, cmap='viridis')
    """
    style_axes_base(ax, grid=False)  # No grid for heatmaps
    # All spines visible (boxed)
    for spine in ax.spines.values():
        spine.set_visible(True)


def style_line_axes(ax: Axes) -> None:
    """Style axes for line plots (Fig. 9 style).

    Line plots use:
    - Gray background + grid
    - Left and bottom spines only
    - Clean, minimal appearance

    Args:
        ax: Axes to style

    Example:
        >>> fig, ax = plt.subplots()
        >>> style_line_axes(ax)
        >>> ax.plot(x, y, label='Data')
    """
    style_axes_base(ax, grid=True)
    hide_spines(ax, top=True, right=True)  # Fig. 9 style


def style_scatter_axes(ax: Axes) -> None:
    """Style axes for scatter plots (Fig. 3 style).

    Scatter plots use:
    - Gray background + grid
    - All spines visible (boxed appearance)
    - Suitable for correlation/comparison plots

    Args:
        ax: Axes to style

    Example:
        >>> fig, ax = plt.subplots()
        >>> style_scatter_axes(ax)
        >>> prh_scatter(ax, x, y, color='C0', label='Points')
    """
    style_axes_base(ax, grid=True)
    # All spines visible (Fig. 3 style, boxed)
    for spine in ax.spines.values():
        spine.set_visible(True)


def style_bar_axes(ax: Axes) -> None:
    """Style axes for bar plots.

    Bar plots use:
    - Gray background + grid
    - Left and bottom spines only
    - Grid helps read bar heights

    Args:
        ax: Axes to style

    Example:
        >>> fig, ax = plt.subplots()
        >>> style_bar_axes(ax)
        >>> prh_bar(ax, x, heights, label='Values')
    """
    style_axes_base(ax, grid=True)
    hide_spines(ax, top=True, right=True)


def style_hist_axes(ax: Axes) -> None:
    """Style axes for histograms.

    Histograms use:
    - Gray background + grid
    - Left and bottom spines only
    - Similar to line plot style

    Args:
        ax: Axes to style

    Example:
        >>> fig, ax = plt.subplots()
        >>> style_hist_axes(ax)
        >>> ax.hist(data, bins=30, alpha=0.7)
    """
    style_axes_base(ax, grid=True)
    hide_spines(ax, top=True, right=True)


def prh_scatter(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    label: str | None = None,
    s: float = PRH_SCATTER_SIZE,
    alpha: float = 0.7,
    zorder: int = 3,
    **kwargs,
) -> object:
    """Create scatter plot with PRH Fig. 3 style (white marker edges).

    Args:
        ax: Axes to plot on
        x: X coordinates
        y: Y coordinates
        color: Marker face color
        label: Optional label for legend
        s: Marker size in points^2
        alpha: Marker transparency
        zorder: Drawing order (higher = on top)
        **kwargs: Additional arguments passed to ax.scatter()

    Returns:
        PathCollection from scatter()

    Example:
        >>> style_scatter_axes(ax)
        >>> prh_scatter(ax, x, y, color='#2E7D99', label='Series A')
    """
    return ax.scatter(
        x,
        y,
        s=s,
        c=[color],
        label=label,
        alpha=alpha,
        edgecolors=PRH_SCATTER_EDGE,
        linewidths=PRH_SCATTER_EW,
        zorder=zorder,
        **kwargs,
    )


def prh_errorbar(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray | None = None,
    *,
    label: str | None = None,
    color: str | None = None,
    marker: str = "o",
    **kwargs,
) -> ErrorbarContainer:
    """Create line plot with error bars (PRH Fig. 9 style).

    Error bars use:
    - Thick lines (from rcParams)
    - Filled markers matching line color
    - Vertical error bars with no caps
    - PRH error bar styling

    Args:
        ax: Axes to plot on
        x: X coordinates
        y: Y coordinates
        yerr: Y error values (1D or 2D)
        label: Label for legend
        color: Line/marker color (None = use color cycle)
        marker: Marker style
        **kwargs: Additional arguments for errorbar()

    Returns:
        ErrorbarContainer from errorbar()

    Example:
        >>> style_line_axes(ax)
        >>> prh_errorbar(ax, x, y, yerr=stds, label='Data', color='#2E7D99')
    """
    # If no color specified, let matplotlib use the color cycle
    plot_kwargs = {
        "label": label,
        "marker": marker,
        "elinewidth": PRH_ERR_LW,
        "capsize": PRH_ERR_CAPSIZE,
        "zorder": 3,
    }

    if color is not None:
        plot_kwargs.update(
            {
                "color": color,
                "mfc": color,
                "mec": color,
                "mew": 0.0,
            }
        )

    plot_kwargs.update(kwargs)
    return ax.errorbar(x, y, yerr=yerr, **plot_kwargs)


def prh_bar(
    ax: Axes,
    x: np.ndarray | list,
    height: np.ndarray | list,
    *,
    width: float = 0.8,
    label: str | None = None,
    color: str | None = None,
    alpha: float = 0.8,
    edgecolor: str = "white",
    linewidth: float = 0.5,
    **kwargs,
) -> Rectangle:
    """Create bar plot with PRH styling.

    Bars use:
    - Clean edges (white or light)
    - Slightly transparent fill
    - PRH color palette

    Args:
        ax: Axes to plot on
        x: X positions
        height: Bar heights
        width: Bar width
        label: Label for legend
        color: Bar color (None = use color cycle)
        alpha: Bar transparency
        edgecolor: Bar edge color
        linewidth: Bar edge width
        **kwargs: Additional arguments for bar()

    Returns:
        BarContainer from bar()

    Example:
        >>> style_bar_axes(ax)
        >>> prh_bar(ax, [0, 1, 2], [1.5, 2.3, 1.8], label='Metric')
    """
    bar_kwargs = {
        "width": width,
        "label": label,
        "alpha": alpha,
        "edgecolor": edgecolor,
        "linewidth": linewidth,
    }

    if color is not None:
        bar_kwargs["color"] = color

    bar_kwargs.update(kwargs)
    return ax.bar(x, height, **bar_kwargs)


def prh_legend(
    ax: Axes,
    *,
    loc: str = "best",
    style: str = "square",
    fontsize: float | None = None,
    **kwargs,
) -> Legend:
    """Create PRH-styled legend.

    Args:
        ax: Axes to add legend to
        loc: Legend location
        style: "square" (Fig. 9) or "rounded" (Fig. 3)
        fontsize: Font size (None = use PRH_LEGENDSIZE)
        **kwargs: Additional arguments for legend()

    Returns:
        Legend object

    Example:
        >>> prh_legend(ax, style="square", loc="upper left")
    """
    fancybox = style == "rounded"

    if fontsize is None:
        fontsize = PRH_LEGENDSIZE

    leg = ax.legend(
        loc=loc,
        frameon=True,
        fancybox=fancybox,
        shadow=True,
        fontsize=fontsize,
        facecolor="white",
        edgecolor=PRH_SPINE,
        framealpha=1.0,
        borderpad=0.6,
        labelspacing=0.6,
        handlelength=2.2,
        handletextpad=0.8,
        **kwargs,
    )

    leg.get_frame().set_linewidth(PRH_LEGEND_FRAME_WIDTH)
    return leg
