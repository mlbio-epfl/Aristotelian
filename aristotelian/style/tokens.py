"""Design tokens for PRH (Platonic Representation Hypothesis) plotting style.

This module defines the core visual design tokens adapted from the PRH paper
for ICML paper format. All values are scaled appropriately for narrow column widths
(3.25" single column, 6.75" double column).
"""

from __future__ import annotations

# Core color palette - PRH neutrals
# These create the signature cool-gray, sophisticated aesthetic
PRH_AX_FACE = "#F8F9FA"  # Axes background (cool light gray)
PRH_GRID = "#ECEDEC"  # Major grid lines (very subtle)
PRH_SPINE = "#C8CDD4"  # Spine/edge color (blue-gray)
PRH_TICK = "#3B4F6D"  # Tick labels and tick marks (darker blue-gray)
PRH_LABEL = "#324766"  # Axis labels (darkest, highest contrast)

# Typography
# Font stack prioritizes common sans-serif fonts
PRH_SANS = ["Helvetica", "Arial", "DejaVu Sans"]

# Sizing tokens - ICML-adapted
# Original PRH uses larger sizes for full-page figures
# These are scaled by ~0.45× for ICML column format
PRH_BASE_FONT = 9  # Base font size
PRH_LABELSIZE = 10  # Axis labels (was 22 in PRH)
PRH_TICKSIZE = 8  # Tick labels (was 16 in PRH)
PRH_LEGENDSIZE = 8  # Legend text (was 16 in PRH)
PRH_TITLESIZE = 11  # Subplot titles (was not specified, scaled proportionally)

# Line and marker sizing
PRH_LINEWIDTH = 2.0  # Line plots (was 3.0 in PRH)
PRH_MARKERSIZE = 6.0  # Marker size (was 11.5 in PRH)
PRH_MARKEREDGE = 0.5  # Marker edge width for general use

# Fig. 3 style - scatter plots with white outlines
PRH_SCATTER_EDGE = "white"  # White outline for scatter markers
PRH_SCATTER_EW = 0.8  # Scatter edge width (was 1.4 in PRH)
PRH_SCATTER_SIZE = 110  # Scatter marker size in points^2

# Fig. 9 style - line plots with error bars
PRH_ERR_LW = 1.5  # Error bar line width (was 2.6 in PRH)
PRH_ERR_CAPSIZE = 0  # No caps on error bars (PRH style)

# Spine widths
PRH_SPINE_WIDTH = 1.2  # Width of axis spines

# Grid settings
PRH_GRID_WIDTH = 0.8  # Grid line width (was 1.1 in PRH, scaled for ICML)
PRH_GRID_ALPHA = 1.0  # Full opacity for grid (PRH uses subtle color, not alpha)

# Legend settings
PRH_LEGEND_FRAME_WIDTH = 1.2  # Legend frame line width
PRH_LEGEND_SHADOW = True  # PRH uses shadows on legends
PRH_LEGEND_FACECOLOR = "white"  # White background for legends
PRH_LEGEND_ALPHA = 1.0  # Fully opaque legend background

# Tick settings
PRH_TICK_DIRECTION = "out"  # Ticks point outward
PRH_TICK_MAJOR_SIZE = 4  # Major tick length (was 6 in PRH, scaled)
PRH_TICK_MAJOR_WIDTH = 1.0  # Major tick width (was 1.2, scaled)

# Colorbar settings
PRH_CBAR_OUTLINE_WIDTH = 0.5  # Colorbar outline width

# Export settings
PRH_DPI = 200  # Display DPI
PRH_SAVE_DPI = 300  # Saved figure DPI
PRH_PAD_INCHES = 0.02  # Padding around saved figures

# Figure sizes (ICML format - keep existing)
ICML_SINGLE_COL = (3.25, 2.5)  # Single column width
ICML_DOUBLE_COL = (6.75, 2.5)  # Double column width (full page width)

# ICML adjustments for multi-panel figures
ICML_WSPACE = 0.35  # Horizontal spacing between subplots
ICML_HSPACE = 0.55  # Vertical spacing between subplots
