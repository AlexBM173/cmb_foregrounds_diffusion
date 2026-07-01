"""
Matplotlib style baseline for publication-quality figures.

Usage (at the top of every paper-figure notebook):

    from foregrounds_diffusion.plot_style import apply, WONG
    apply(fig_width_pt=246.0)          # MNRAS single column
    apply(fig_width_pt=510.0, n_cols=2)  # MNRAS double column

Save convention (always PDF primary, PNG backup):

    fig.savefig(FIGURES_DIR / "fig01_cib_fullsky.pdf")
    fig.savefig(FIGURES_DIR / "fig01_cib_fullsky.png", dpi=300)
"""

import matplotlib as mpl
from cycler import cycler

# Wong (2011) 8-colour palette — distinguishable under deuteranopia,
# protanopia, and tritanopia.
WONG = [
    "#000000",  # 0 black
    "#E69F00",  # 1 orange
    "#56B4E9",  # 2 sky blue
    "#009E73",  # 3 bluish green
    "#F0E442",  # 4 yellow
    "#0072B2",  # 5 blue
    "#D55E00",  # 6 vermillion
    "#CC79A7",  # 7 reddish purple
]


def apply(fig_width_pt: float = 246.0, n_cols: int = 1) -> list[str]:
    """Set rcParams for publication-quality figures.

    Parameters
    ----------
    fig_width_pt:
        Journal text width in points.  246 pt ≈ MNRAS single column;
        510 pt ≈ MNRAS double column.  Pass the LaTeX \\textwidth value.
    n_cols:
        Figure column span (1 or 2).  Width is multiplied accordingly so
        ``fig_width_pt=246, n_cols=2`` gives a figure spanning both columns.

    Returns
    -------
    list[str]
        The WONG colour palette, for convenience (avoids a second import).
    """
    inches_per_pt = 1.0 / 72.27
    fig_width = fig_width_pt * inches_per_pt * n_cols
    golden = (1 + 5**0.5) / 2

    mpl.rcParams.update(
        {
            # Figure size and DPI
            "figure.figsize": (fig_width, fig_width / golden),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # Fonts (match LaTeX body font)
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            # Lines and markers
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            # Colour cycle
            "axes.prop_cycle": cycler(color=WONG),
            # Legend
            "legend.frameon": False,
            "legend.handlelength": 1.5,
            # Layout
            "figure.constrained_layout.use": True,
        }
    )
    return WONG
