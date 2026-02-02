"""Color palettes for PRH-style plotting.

Defines color schemes for different plot types, including:
- Current project palette (muted professional colors)
- PRH Fig. 3 palette (viridis-like gradient)
- PRH Fig. 9 palette (categorical, from the paper)
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np

# Current project palette - muted, professional colors
# This is the existing color scheme used in the codebase
LINE_COLORS = [
    "#2E7D99",  # Deep teal (primary)
    "#E8795A",  # Muted coral
    "#7A5C99",  # Muted purple
    "#65A765",  # Sage green
    "#D4A574",  # Warm tan
    "#C16C84",  # Dusty rose
]

# Colormaps - perceptually uniform and professional
SEQUENTIAL = "viridis"  # For heatmaps, single-direction data
DIVERGING = "RdBu_r"  # For data with meaningful zero/center


def get_fig3_colors(n: int = 4) -> np.ndarray:
    """Get colors from viridis colormap (PRH Fig. 3 style).

    PRH Fig. 3 uses a gradient palette: purple → blue → green → yellow.
    This is essentially the viridis colormap.

    Args:
        n: Number of colors to extract (default 4, as in PRH Fig. 3)

    Returns:
        Array of RGBA colors from viridis
    """
    vir = mpl.colormaps["viridis"]
    # Extract colors at specific points to match PRH Fig. 3
    positions = np.linspace(0.95, 0.0, n)  # Reverse to go purple→yellow
    return vir(positions)


# PRH Fig. 9 categorical palette
# These are render-matched approximations from the PRH paper
FIG9_COLORS = {
    "ImageNet21K": "#000000",  # Black
    "MAE": "#410967",  # Deep purple
    "DINOv2": "#932567",  # Magenta-purple
    "CLIP": "#DC5039",  # Red-orange
    "CLIP (l12K ft)": "#FBA40B",  # Orange
}

# Alternative: List version for iteration
FIG9_PALETTE = list(FIG9_COLORS.values())


def get_color_cycle(palette: str = "project") -> list[str]:
    """Get a color cycle for line plots.

    Args:
        palette: Which palette to use ("project", "fig9", or "viridis")

    Returns:
        List of color hex codes
    """
    if palette == "project":
        return LINE_COLORS
    elif palette == "fig9":
        return FIG9_PALETTE
    elif palette == "viridis":
        return [mpl.colors.rgb2hex(c) for c in get_fig3_colors(6)]
    else:
        raise ValueError(f"Unknown palette: {palette}")


def get_categorical_colors(n: int, palette: str = "project") -> list[str]:
    """Get n categorical colors from a palette.

    Args:
        n: Number of colors needed
        palette: Which palette to use

    Returns:
        List of n color hex codes
    """
    cycle = get_color_cycle(palette)
    # Repeat if we need more colors than available
    return [cycle[i % len(cycle)] for i in range(n)]
