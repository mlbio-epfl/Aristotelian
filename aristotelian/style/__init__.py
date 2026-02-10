"""PRH (Platonic Representation Hypothesis) plotting style for ICML papers.

This package provides a complete styling system based on the PRH paper's
visual aesthetic, adapted for ICML paper format (narrow columns).

Quick start:
    >>> from aristotelian.style import use_prh_base, style_line_axes, prh_legend
    >>> use_prh_base()  # Apply base style
    >>> fig, ax = plt.subplots(figsize=(3.25, 2.5))
    >>> style_line_axes(ax)
    >>> ax.plot(x, y, label='Data')
    >>> prh_legend(ax, style='square')

Modules:
    tokens: Design tokens (colors, sizes)
    apply: Base style application
    palettes: Color palettes
    presets: Plot type presets (line, scatter, bar, etc.)
"""

from __future__ import annotations

# Core functionality
from .apply import (
    get_figure_size,
    hide_spines,
    prh_colorbar,
    style_axes_base,
    use_prh_base,
)

# Color palettes
from .palettes import (
    DIVERGING,
    FIG9_COLORS,
    FIG9_PALETTE,
    LINE_COLORS,
    SEQUENTIAL,
    get_categorical_colors,
    get_color_cycle,
    get_fig3_colors,
)

# Plot presets
from .presets import (
    prh_bar,
    prh_errorbar,
    prh_legend,
    prh_scatter,
    style_bar_axes,
    style_heatmap_axes,
    style_hist_axes,
    style_line_axes,
    style_scatter_axes,
)

# Design tokens (for advanced users)
from .tokens import ICML_DOUBLE_COL, ICML_SINGLE_COL

__all__ = [
    # Core
    "use_prh_base",
    "style_axes_base",
    "hide_spines",
    "prh_colorbar",
    "get_figure_size",
    # Presets
    "style_heatmap_axes",
    "style_line_axes",
    "style_scatter_axes",
    "style_bar_axes",
    "style_hist_axes",
    "prh_scatter",
    "prh_errorbar",
    "prh_bar",
    "prh_legend",
    # Palettes
    "LINE_COLORS",
    "SEQUENTIAL",
    "DIVERGING",
    "FIG9_COLORS",
    "FIG9_PALETTE",
    "get_fig3_colors",
    "get_color_cycle",
    "get_categorical_colors",
    # Tokens
    "ICML_SINGLE_COL",
    "ICML_DOUBLE_COL",
]

__version__ = "1.0.0"
