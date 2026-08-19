"""Project-local plotting style helpers.

This module centralizes matplotlib rcParams used across CLI diagnostics.
"""

import matplotlib as mpl

PLOT_RC_PARAMS = {
    "font.size": 7,
    "xtick.major.size": 0,
    "xtick.minor.size": 0,
    "xtick.major.width": 0,
    "xtick.minor.width": 0,
    "ytick.major.size": 0,
    "ytick.minor.size": 0,
    "ytick.major.width": 0,
    "ytick.minor.width": 0,
    "figure.figsize": (3.5, 3.5),
    "axes.grid": True,
    "grid.color": "grey",
    "grid.linewidth": 0.3,
    "grid.linestyle": ":",
    "axes.grid.axis": "both",
    "axes.grid.which": "both",
    "axes.axisbelow": False,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8,
    "figure.dpi": 300,
    "figure.subplot.left": 0.195,
    "figure.subplot.right": 0.97,
    "figure.subplot.bottom": 0.145,
    "figure.subplot.top": 0.97,
    "figure.subplot.wspace": 0.2,
    "figure.subplot.hspace": 0.2,
}


def plot_style_context():
    """Return a matplotlib rc_context using the project plotting style."""
    return mpl.rc_context(PLOT_RC_PARAMS)
