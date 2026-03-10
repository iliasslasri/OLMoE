"""
Shared plotting configuration for MoE scaling law analysis.
All figures use consistent styling, colors, and formatting.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Regime aesthetics ─────────────────────────────────────────────────────────
REGIME_COLORS = {
    "D": "#1B3A6B",
    "M0": "#E67E22",
    "Malpha": "#27AE60",
    "Mlfb": "#C0392B",
}

REGIME_LABELS = {
    "D": "Dense",
    "M0": "MoE (no bal.)",
    "Malpha": r"MoE + $\mathcal{L}_{bal}$",
    "Mlfb": "MoE + LFB",
}

REGIME_MARKERS = {
    "D": "o",
    "M0": "s",
    "Malpha": "^",
    "Mlfb": "D",
}

# Map from config regime names to plot keys
REGIME_MAP = {
    "dense": "D",
    "M0": "M0",
    "M_alpha": "Malpha",
    "M_lfb": "Mlfb",
}

# ── Figure dimensions ─────────────────────────────────────────────────────────
SINGLE_COL_WIDTH = 3.5   # inches
DOUBLE_COL_WIDTH = 7.0   # inches
ASPECT_RATIO = 1.3        # width / height


def setup_style():
    """Apply publication-quality plot style."""
    mpl.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        # Spines
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        # Background
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": False,
        # Lines
        "lines.linewidth": 1.2,
        "lines.markersize": 5,
        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        # Save
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def get_fig(ncols=1, nrows=1, width="single", aspect=None):
    """Create a figure with standardised dimensions."""
    w = SINGLE_COL_WIDTH if width == "single" else DOUBLE_COL_WIDTH
    ar = aspect or ASPECT_RATIO
    h = w / ar
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows))
    return fig, axes


def save_fig(fig, path_stem):
    """Save figure as both PDF (vector) and PNG (raster)."""
    fig.savefig(f"{path_stem}.pdf", format="pdf")
    fig.savefig(f"{path_stem}.png", format="png", dpi=300)
    plt.close(fig)


def regime_color(regime):
    key = REGIME_MAP.get(regime, regime)
    return REGIME_COLORS.get(key, "#333333")


def regime_label(regime):
    key = REGIME_MAP.get(regime, regime)
    return REGIME_LABELS.get(key, regime)


def regime_marker(regime):
    key = REGIME_MAP.get(regime, regime)
    return REGIME_MARKERS.get(key, "o")
