"""
irr_plot.py — Minimal plotting helpers to reproduce thesis-like style.

This module intentionally mirrors a subset of the plotting conveniences used in
Francesco Mariottini's thesis codebase (e.g., `irradiance_plot.py`) so that
figures generated here keep a consistent aesthetic across the document.

The functions are self-contained and rely only on matplotlib / seaborn.
"""
from __future__ import annotations
from pathlib import Path   
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Iterable


# Optional: global style close to thesis figures (feel free to tweak)
mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlelocation": "left",
})



def thesis_scatter(
    ax: plt.Axes,
    x, y,
    label: Optional[str] = None,
    *,
    s: float = 10, alpha: float = 0.9,
    color: Optional[str] = None,
    zorder: int = 2,
    **kwargs
):
    """Scatter with dots only (no lines), thesis defaults."""
    return ax.scatter(x, y, s=s, alpha=alpha, label=label, color=color, zorder=zorder, **kwargs)

def put_bottom_legend(fig: plt.Figure, ax: plt.Axes, ncol: Optional[int] = None):
    """Place legend at the bottom, outside axes, spanning multiple columns."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if ncol is None:
        # heuristic: ~3 columns, or len(labels) if less than 3
        ncol = min(3, max(1, len(labels)))
    # figure-level legend
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=ncol, frameon=True)

def savefig(fig: plt.Figure, outpath: str | Path, tight: bool = True, dpi: int = 200):
    """Persist figure to disk creating parent folders when needed."""
    p = Path(outpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(p, dpi=dpi)


# Default style parameters approximating the thesis style
_THESIS_RC = {
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.35,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.frameon': False,
    'figure.dpi': 120,
}

_DEF_PALETTE = sns.color_palette('deep')


def use_thesis_style():
    """Activate a lightweight, thesis-like mpl style."""
    mpl.rcParams.update(_THESIS_RC)
    sns.set_palette(_DEF_PALETTE)


def savefig(fig: plt.Figure, outpath: str, tight: bool = True, dpi: int = 200):
    """Persist figure to disk creating parent folders when needed.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure instance.
    outpath : str
        Output path (png recommended). Parents will be created.
    tight : bool
        Whether to call tight_layout before saving.
    dpi : int
        Output raster DPI.
    """
    p = Path(outpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(p, dpi=dpi)


def plot_error_vs_aoi(df, aoi_col: str, series: dict, title: str = '',
                      ylim: Optional[Tuple[float, float]] = None,
                      ylabel: str = 'Error', xlabel: str = 'Angle of incidence (°)') -> plt.Figure:
    """Scatter/line plot of error metrics vs angle of incidence.

    Parameters
    ----------
    df : DataFrame
        Input data.
    aoi_col : str
        Column name with angle of incidence in degrees.
    series : dict
        Mapping display label -> column name to plot. For example,
        {'Absolute error (W/m²)': 'error_abs', 'Cosine error (%)': 'error_cos'}
    title : str
        Plot title.
    ylim : tuple
        Y limits.
    ylabel : str
        Y-axis label.
    xlabel : str
        X-axis label.
    """
    import pandas as pd
    import numpy as np
    use_thesis_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    # Sort by AOI to draw lines in order
    d = df.copy()
    d = d.sort_values(by=aoi_col)

    for label, col in series.items():
        if col in d.columns:
            ax.plot(d[aoi_col], d[col], marker='.', linestyle='-', alpha=0.8, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc='best', ncol=1)
    return fig


def plot_timeseries(df, t_col: str, series: dict, title: str = '',
                    ylabel: str = 'Error', xlabel: str = 'Time',
                    ylim=None) -> plt.Figure:
    """Time series plot for one or more metrics.

    Parameters
    ----------
    df : DataFrame
    t_col : str
        Datetime-like column.
    series : dict
        Mapping label -> column name.
    title : str
        Title.
    ylabel, xlabel : str
        Axis labels.
    ylim : tuple, optional
        Y axis limits.
    """
    import pandas as pd
    use_thesis_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    d = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(d[t_col]):
        d[t_col] = pd.to_datetime(d[t_col], errors='coerce')
    d = d.sort_values(by=t_col)
    for label, col in series.items():
        if col in d.columns:
            ax.plot(d[t_col], d[col], marker='.', linestyle='-', alpha=0.8, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc='best', ncol=1)
    fig.autofmt_xdate()
    return fig
