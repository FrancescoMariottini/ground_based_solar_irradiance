# --- Plotting helpers: update for (a) no titles; (b) legend below; (c,d) dual-axis rules ---
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import v_results_3_source_py.irr_plot as irplt
import os

# --- legend text measurement & wrapping helpers (FIXED) -----------------------


#loc giving freedom since change depending how many graphs
# Loc can also be a 2-tuple giving the coordinates of the lower-left corner of the legend in axes coordinates
# (in which case bbox_to_anchor will be ignored). @FM does not happen
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
_DEFAULT_KWARGS ={plt.legend: {'loc': (0, 0), #'best' 0, 'upper right' 1, 'upper left' 2, 'lower left' 3, 'lower right' 4,
                 #'right' 5, 'center left'	6, 'center right' 7, 'lower center'	8, 'upper center' 9, 'center' 10
                 #loc position change starting from bbox_to_anchor. l/r & u/l seems inverted when bbox_to_anchor
                 'bbox_to_anchor': (0, -1)  # box location (x,y) or (x, y, width, height)
                 }}


LGN_BBOX_TO_ANCHOR =  _DEFAULT_KWARGS[plt.legend]['bbox_to_anchor']
LGN_LOC =  _DEFAULT_KWARGS[plt.legend]['loc']

TEST = True

def _get_renderer(fig):
    """Ensure a renderer exists (needed to measure text width in px)."""
    renderer = None
    try:
        renderer = fig.canvas.get_renderer()
    except Exception:
        pass
    if renderer is None:
        try:
            fig.canvas.draw()  # force a draw to get a renderer
            renderer = fig.canvas.get_renderer()
        except Exception:
            renderer = None
    return renderer


def _text_width_px(fig, s: str, fp: FontProperties) -> float:
    """Return text width in pixels for string ``s`` with font properties ``fp``.

    Robust implementation that avoids AttributeError when a Text artist has no
    attached Figure. We prefer the renderer's fast width computation; if that is
    unavailable, we fall back to creating a Text, *attach the figure*, and ask
    for its window extent; as a last resort, we estimate width heuristically.
    """
    renderer = _get_renderer(fig)
    # Preferred path: use the renderer's direct method (fast, no artist needed)
    if renderer is not None:
        try:
            w, _, _ = renderer.get_text_width_height_descent(
                s, fp, ismath=False
            )
            return float(w)
        except Exception:
            pass
    # Fallback: create a Text artist and *attach* the figure before measuring
    try:
        t = mpl.text.Text(0, 0, s, fontproperties=fp)
        t.set_figure(fig)
        bb = t.get_window_extent(renderer=_get_renderer(fig))
        return float(bb.width)
    except Exception:
        # Final fallback: rough estimate (chars × 0.6 × fontsize × 1.333 px/pt)
        fs = fp.get_size_in_points() or mpl.rcParams.get('legend.fontsize', 10)
        return max(1, len(s)) * 0.6 * float(fs) * 1.333


# ----------------------------- FLEXIBLE LEGEND (figure-level) -----------------
# Adjust these two values when you need more/less space for the legend.
LEGEND_OPTS = {
    'bottom': 0.28,  # reserved bottom margin (0.22–0.34 works well)
    'y': 0.03,       # vertical anchor for the legend in figure coords (0.01–0.04)
    'ncol': 'auto',  # 'auto' or an integer (e.g., 2, 3, 4)
    'fontsize': None,
    'frameon': True,
    'columnspacing': 1.2,
    'handletextpad': 0.6,
}

def legend_below(fig=None, ax=None, **overrides):
    """
    Place a *figure-level* legend centered BELOW the plot, with reserved space
    and **automatic two-line wrapping** for long labels (no overlap, no clipping).

    Tunables (same keys as before + wrapping controls):
      bottom: reserved bottom margin (0.22–0.34 typical) [default from LEGEND_OPTS]
      y: legend anchor y in figure coords (0.01–0.04)
      ncol: 'auto' or int
      fontsize: legend font size (None => rcParams['legend.fontsize'])
      frameon, columnspacing, handletextpad
      max_cols: upper bound when auto-trying more columns to help long labels (default 6)
      wrap_pad_px: extra pixels reserved for marker/spacing when computing wrap width (default 50)
    """
    import math

    fig = fig or plt.gcf()
    ax = ax or plt.gca()

    # ------------- options & defaults -------------
    # pull from global LEGEND_OPTS if present; otherwise define sensible defaults
    base = {
        'bottom': 0.28,
        'y': 0.03,
        'ncol': 'auto',
        'fontsize': mpl.rcParams.get('legend.fontsize', 10),
        'frameon': True,
        'columnspacing': 1.2,
        'handletextpad': 0.6,
    }
    try:
        # If LEGEND_OPTS defined above, merge it
        base.update(LEGEND_OPTS)
    except NameError:
        pass

    opts = {
        **base,
        'max_cols': 6,        # will try up to this many columns if wrapping still too long
        'wrap_pad_px': 50.0,  # reserve px for marker + spacing inside each legend cell
    }
    opts.update(overrides)

    # Collect entries from the axis that should appear in the legend
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None

    # Ensure layout engines do not interfere
    try: fig.set_constrained_layout(False)
    except Exception: pass
    try: fig.set_tight_layout(False)  # mpl>=3.7
    except Exception: pass

    # Reserve bottom band for legend and shrink the axes upward
    fig.subplots_adjust(bottom=float(opts['bottom']))

    # Font properties used for width measurement & legend drawing
    fontsize = opts.get('fontsize', mpl.rcParams.get('legend.fontsize', 10))
    fp = FontProperties(size=fontsize)

    # Figure pixel width available for the legend row
    fig_w_px = fig.get_figwidth() * fig.dpi * 0.96  # small safety factor

    # Decide number of columns (auto → prefer ~2 rows)
    if opts['ncol'] == 'auto':
        ncol_guess = min(len(labels), max(2, int(math.ceil(len(labels)/2))))
    else:
        ncol_guess = int(opts['ncol'])

    max_cols = max(1, int(opts['max_cols']))
    wrap_pad = float(opts['wrap_pad_px'])

    # Utility to wrap one label into max two lines within a given cell width
    def wrap_two_lines(label: str, cell_px: float) -> str:
        # If it already fits, leave it
        if _text_width_px(fig, label, fp) <= cell_px:
            return label
        words = label.split()
        if not words:
            return label
        # Greedy first line
        line1 = words[0]
        cut = 1
        for i in range(1, len(words)):
            candidate = ' '.join(words[:i+1])
            if _text_width_px(fig, candidate, fp) <= cell_px:
                line1 = candidate
                cut = i+1
            else:
                break
        line2 = ' '.join(words[cut:]).strip()
        if not line2:
            return line1
        # If line2 still too wide, truncate and add ellipsis to keep two lines
        w2 = _text_width_px(fig, line2, fp)
        if w2 > cell_px:
            est_chars = max(1, int(len(line2) * (cell_px / max(1.0, w2))) - 2)
            line2 = (line2[:est_chars] + '…').rstrip()
        return f"{line1}\n{line2}"

    # Try increasing number of columns until each wrapped label fits its cell width
    final_ncol = max(1, ncol_guess)
    wrapped_labels = labels[:]
    for ncol_try in range(final_ncol, max_cols+1):
        cell_px = max(60.0, fig_w_px / ncol_try - wrap_pad)
        candidate = [wrap_two_lines(s, cell_px) for s in labels]
        ok = True
        for s in candidate:
            longest = max(_text_width_px(fig, part, fp) for part in s.split('\n'))
            if longest > cell_px:
                ok = False
                break
        wrapped_labels = candidate
        final_ncol = ncol_try
        if ok:
            break

    # Build a figure-level legend centered at the bottom
    leg = fig.legend(
        handles, wrapped_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, float(opts['y'])),
        # bbox_to_anchor= LGN_BBOX_TO_ANCHOR,
        bbox_transform=fig.transFigure,
        ncol=int(final_ncol),
        frameon=opts['frameon'],
        fontsize=fontsize,
        columnspacing=opts['columnspacing'],
        handletextpad=opts['handletextpad'],
        labelspacing=0.8,  # slightly tighter vertical spacing for 2-line labels
        borderaxespad=0.2,
    )
    try: leg.set_in_layout(False)
    except Exception: pass
    return leg

# ----------------------------------- UTILITIES --------------------------------
MONTH_NAMES = {
    1:"January", 2:"February", 3:"March", 4:"April", 5:"May", 6:"June",
    7:"July", 8:"August", 9:"September", 10:"October", 11:"November", 12:"December"
}

def ensure_month_name(df: pd.DataFrame, month_col: str = 'month') -> pd.DataFrame:
    d = df.copy()
    if month_col in d.columns:
        d['month_name'] = d[month_col].astype('Int64').map(MONTH_NAMES)
    return d

def split_by_hour(df: pd.DataFrame, hour_col: str = 'hour') -> Tuple[pd.DataFrame, pd.DataFrame]:
    before = df[df[hour_col] < 12].copy() if hour_col in df.columns else df.iloc[0:0].copy()
    after = df[df[hour_col] >= 12].copy() if hour_col in df.columns else df.iloc[0:0].copy()
    return before, after

def split_by_aoi(df: pd.DataFrame, aoi_col: str = 'aoi') -> Tuple[pd.DataFrame, pd.DataFrame]:
    if aoi_col not in df.columns:
        return df.copy(), df.iloc[0:0].copy()
    tilted = df[df[aoi_col].notna()].copy()
    horizontal = df[df[aoi_col].isna()].copy()
    return horizontal, tilted

def compute_absolute_from_cos_err(df: pd.DataFrame, cos_err_col: str = 'cos_err') -> pd.DataFrame:
    d = df.copy()
    if cos_err_col in d.columns:
        # 20/10/25 fix
        # d['abs_err_from_cos'] = 10.0 * d[cos_col].astype(float)  # Δ = 10 × δ_cos
        d['abs_err'] = 10.0 * d[cos_err_col].astype(float)  # Δ = 10 × δ_cos

    return d

def compute_cos_from_absolute(df: pd.DataFrame, aoi_col: str = 'aoi', abs_col: str = 'abs_err') -> pd.DataFrame:
    d = df.copy()
    if 'cos_err' not in d.columns and abs_col in d.columns and aoi_col in d.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            cos_th = np.cos(np.deg2rad(d[aoi_col].astype(float)))
            d['cos_err'] = 100.0 * d[abs_col].astype(float) / (1000.0 * np.clip(cos_th, 1e-6, None))
    return d

def find_columns(df: pd.DataFrame, contains: str) -> List[str]:
    c = contains.lower()
    return [col for col in df.columns if c in str(col).lower()]

def choose_aoi_column(df: pd.DataFrame) -> Optional[str]:
    aoi_cols = find_columns(df, 'aoi')
    for pref in ['aoi ventilated', 'aoi nonvent', 'aoi']:
        for c in aoi_cols:
            if c.lower() == pref:
                return c
    return aoi_cols[0] if aoi_cols else None

# 20/10/25 added
def choose_cos_err_column(df: pd.DataFrame) -> Optional[str]:
    aoi_cols = find_columns(df, 'cos_err')
    for pref in ['cos_err ventilated', 'cos_err nonvent', 'cos_err']:
        for c in aoi_cols:
            if c.lower() == pref:
                return c
    return aoi_cols[0] if aoi_cols else None

def scatter(ax: plt.Axes, x, y, label: Optional[str] = None, color: Optional[str] = None, **kw):
    """Dots only; reuse irr_plot's thesis scatter."""
    return irplt.thesis_scatter(ax, x, y, label=label, color=color, s=12, alpha=0.9, **kw)

def apply_title(ax: plt.Axes, name: str):
    """No titles anywhere (requirement a)."""
    return

def palette(n: int) -> List:
    return sns.color_palette('tab10', n_colors=max(10, n))

# ----------------------------------- CORE PLOTS -------------------------------






def plot_abs_vs_aoi(df: pd.DataFrame, name: str, outdir: Path,
                    aoi_col: Optional[str] = None,
                    y_col: str = 'abs_err',
                    series_label: str = 'absolute deviation [W/m2]',
                    suffix: str = '_abs') -> Optional[Path]:
    if aoi_col is None:
        aoi_col = choose_aoi_column(df)
    if aoi_col is None or y_col not in df.columns:
        return None
    d = df[[aoi_col, y_col]].dropna().sort_values(by=aoi_col)
    if d.empty:
        return None
    irplt.use_thesis_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    scatter(ax, d[aoi_col], d[y_col], label=series_label)
    ax.set_xlabel("angle of incidence [degree]")
    ax.set_ylabel("absolute deviation [W/m2]")
    # (a) no title

    legend_below(fig, ax, ncol='auto')  # (b) legend below
    out = Path(outdir) / f"{name}{suffix}.png"
    out = os.path.join(outdir,f"{name}{suffix}.png")
    irplt.savefig(fig, out, tight=False); plt.close(fig)  # avoid tight_layout
    return out


def plot_directional_dual_y(df: pd.DataFrame, name: str, outdir: Path,
                            aoi_col: Optional[str] = None,
                            cos_col: str = 'cos_err',
                            suffix: str = '_dir_dualy') -> Optional[Path]:
    """
    Dual-Y:
    LEFT = absolute error [W/m2] (legend entries here only) (c)
    RIGHT = cosine error [%] (no legend)
    """
    if aoi_col is None:
        aoi_col = choose_aoi_column(df)
    if aoi_col is None or cos_col not in df.columns:
        return None
    d = compute_absolute_from_cosine(df, cos_col=cos_col)
    if 'abs_err_from_cos' not in d.columns:
        return None
    d = d[[aoi_col, cos_col, 'abs_err_from_cos']].dropna().sort_values(by=aoi_col)
    if d.empty:
        return None
    irplt.use_thesis_style()
    fig, ax_left = plt.subplots(figsize=(6.6, 4.2))
    ax_right = ax_left.twinx()

    col = palette(1)[0]
    # LEFT: absolute error (with legend label)
    scatter(ax_left, d[aoi_col], d['abs_err_from_cos'], label='absolute error [W/m2]', color=col, zorder=3)
    # RIGHT: cosine error (no legend)
    scatter(ax_right, d[aoi_col], d[cos_col], label=None, color=col, zorder=2)

    ax_left.set_xlabel("angle of incidence [degree]")
    ax_left.set_ylabel("absolute error [W/m2]")  # (d) absolute on left
    ax_right.set_ylabel("cosine error [%]")      # (d) cosine on right
    # (a) no title
    
    legend_below(fig, ax_left, ncol=1)           # (b) legend below; (c) only abs series

    out = Path(outdir) / f"{name}{suffix}.png"
    out = os.path.join(outdir,f"{name}{suffix}.png")
    irplt.savefig(fig, out, tight=False); plt.close(fig)
    return out

# ------------------------------ FOLDER / FILE SPECIAL CASES -------------------

def crest_roof_four_series_abs(df: pd.DataFrame, name: str, outdir: Path) -> Optional[Path]:
    aoi_v_col = next((c for c in df.columns if c.lower().strip() == 'aoi ventilated'), None) or choose_aoi_column(df)
    if aoi_v_col is None or 'hour' not in df.columns:
        return None
    col_v = [c for c in df.columns if 'abs_err' in c.lower() and 'ventilated' in c.lower()]
    col_n = [c for c in df.columns if 'abs_err' in c.lower() and ('nonvent' in c.lower() or 'non-vent' in c.lower())]
    if not (col_v and col_n):
        return None
    y_v, y_n = col_v[0], col_n[0]

    before, after = split_by_hour(df, 'hour')

    irplt.use_thesis_style()

    locs = list(range(0,11))
    locs = ['lower center']
    bbox_to_anchors=[]
    for x in [0.5]:
        # for y in [0.05]: 
        for y in [0, 0.05, 0.1, 0.5, 1]:
            bbox_to_anchors.append((x,y))

    for l in locs:
        for b in bbox_to_anchors:
            if name in ["crest_roof_180825_abs_bi200cr20"]:
                # 31/8/25 too big affect label movement ?
                # fig, ax = plt.subplots(figsize=(15, 15))
                fig, ax = plt.subplots(figsize=(6.8, 4.4))
            else:
                fig, ax = plt.subplots(figsize=(6.8, 4.4))
            cols = palette(4)
            fig.subplots_adjust(bottom=0.42) #0.42) 

            series = [
                (before, y_v, r"ventilated horizontal pyranometer absolute deviation before 12:00", cols[0]),
                (after,  y_v, r"ventilated horizontal pyranometer absolute deviation after 12:00",  cols[1]),
                (before, y_n, r"horizontal pyranometer absolute deviation before 12:00",           cols[2]),
                (after,  y_n, r"horizontal pyranometer absolute deviation after 12:00",            cols[3]),
            ]
            for dsub, y, lbl, col in series:
                if aoi_v_col in dsub.columns and y in dsub.columns:
                    dd = dsub[[aoi_v_col, y]].dropna().sort_values(by=aoi_v_col)
                    if not dd.empty:
                        scatter(ax, dd[aoi_v_col], dd[y], label=lbl, color=col)

            ax.set_xlabel("angle of incidence [degree]")
            ax.set_ylabel("absolute deviation [W/m2]")
            if name in ["crest_roof_180825_abs_bi200cr20"]:
                # fig.legend(loc=(0,0), bbox_to_anchor=(0, -10), ncol=1, frameon=True, fontsize=9) #nocl=2 (0.5, 0.18)
                # fig.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5), ncol=1, frameon=True, fontsize=9) #nocl=2 (0.5, 0.18)
                try:
                    fig.legend(loc=l, bbox_to_anchor=b, ncol=1, frameon=False, fontsize=9) #nocl=2 (0.5, 0.18)
                except ValueError:
                    print(f"Not valid: l {str(l)} b {str(b)}")
                    plt.close(fig)
                    break
                # out = Path(outdir) / f"{name}_ba12_abs.png"
                out = Path(outdir) / f"crest_roof_180825_abs_bi200cr20_l{str(l)}_b{str(b)}.png"
                out = os.path.join(outdir,f"crest_roof_180825_abs_bi200cr20_l{str(l)}_b{str(b)}.png")
                irplt.savefig(fig, out, tight=False); plt.close(fig)                
            else:
                legend_below(fig, ax, ncol=1)   #nocl=2
                out = Path(outdir) / f"{name}_ba12_abs.png"
                out = os.path.join(outdir,f"{name}_ba12_abs.png")
                irplt.savefig(fig, out, tight=False); plt.close(fig)
                break
        if name not in ["crest_roof_180825_abs_bi200cr20"]: break

    return out


def crest_roof_four_series_dir(df: pd.DataFrame, name: str, outdir: Path) -> Optional[Path]:
    aoi_v_col = next((c for c in df.columns if c.lower().strip() == 'aoi ventilated'), None) or choose_aoi_column(df)
    if aoi_v_col is None or 'hour' not in df.columns:
        return None
    c_v = [c for c in df.columns if 'cos_err' in c.lower() and 'ventilated' in c.lower()]
    c_n = [c for c in df.columns if 'cos_err' in c.lower() and ('nonvent' in c.lower() or 'non-vent' in c.lower())]
    if not (c_v and c_n):
        return None
    yv, yn = c_v[0], c_n[0]

    before, after = split_by_hour(df, 'hour')
    groups = [
        ("ventilated horizontal pyranometer absolute error before 12:00", before, yv),
        ("ventilated horizontal pyranometer absolute error after 12:00",  after,  yv),
        ("horizontal pyranometer absolute error before 12:00",            before, yn),
        ("horizontal pyranometer absolute error after 12:00",             after,  yn),
    ]

    irplt.use_thesis_style()

    locs = list(range(0,11))
    locs = ['lower center']
    #locs = ['upper center']
    bbox_to_anchors=[]

    for x in [0.5]:
        # for y in [0.05]: 
        for y in [+1]: #[0, -0.05, -0.1, -0.5, -1, 0.05, 0.1, 0.5, 1]:
            bbox_to_anchors.append((x,y))

    for l in locs:
        for b in bbox_to_anchors:
            fig, ax_left = plt.subplots(figsize=(8.6, 5.6))
            ax_right = ax_left.twinx()
            cols = palette(4)

            for i, (lbl_abs, data, ycol) in enumerate(groups):
                if aoi_v_col in data.columns and ycol in data.columns:
                    dd = compute_absolute_from_cosine(data[[aoi_v_col, ycol]].rename(columns={ycol: 'cos_err'}))
                    dd = dd.dropna().sort_values(by=aoi_v_col)
                    if dd.empty:
                        continue
                    # LEFT: absolute (legend)
                    scatter(ax_left, dd[aoi_v_col], dd['abs_err_from_cos'], label=lbl_abs, color=cols[i], zorder=3)
                    # RIGHT: cosine (no legend)
                    scatter(ax_right, dd[aoi_v_col], dd['cos_err'], label=None, color=cols[i], zorder=2)

            ax_left.set_xlabel("angle of incidence [degree]")
            ax_left.set_ylabel("absolute error [W/m2]")
            ax_right.set_ylabel("cosine error [%]")
            # 19/10/25 not "crest_roof_180825_dir_dualy_bi200cr20
            if name in ["crest_roof_180825_cos_bi200cr20"]:
                # fig.legend(loc=(0,0), bbox_to_anchor=(0, -10), ncol=1, frameon=True, fontsize=9) #nocl=2 (0.5, 0.18)
                # fig.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5), ncol=1, frameon=True, fontsize=9) #nocl=2 (0.5, 0.18)
                try:
                    fig.legend(loc=l, bbox_to_anchor=b, ncol=1, frameon=True, fontsize=9) #nocl=2 (0.5, 0.18)
                except ValueError:
                    print(f"Not valid: l {str(l)} b {str(b)}")
                    plt.close(fig)
                    break
                # out = Path(outdir) / f"{name}_ba12_abs.png"
                out = Path(outdir) / f"crest_roof_180825_dir_dualy_bi200cr20_l{str(l)}_b{str(b)}.png"
                out = os.path.join(outdir,f"crest_roof_180825_dir_dualy_bi200cr20_l{str(l)}_b{str(b)}.png")
                irplt.savefig(fig, out, tight=False); plt.close(fig)                
            else:
                legend_below(fig, ax_left, ncol=1)   #nocl=2
                out = Path(outdir) / f"{name}_ba12_dir_dualy.png"
                out = os.path.join(outdir,f"{name}_ba12_dir_dualy.png")
                irplt.savefig(fig, out, tight=False); plt.close(fig)
                break

            if TEST == False: 
                legend_below(fig, ax_left, ncol=1) #nocl=2

                out = Path(outdir) / f"{name}_ba12_dir_dualy.png"
                out = os.path.join(outdir,f"{name}_ba12_dir_dualy.png")
                irplt.savefig(fig, out, tight=False); plt.close(fig)


    return out


def months_multi_series(df: pd.DataFrame, name: str, outdir: Path,
                        aoi_col: Optional[str],
                        y_col: str,
                        label_prefix: str,
                        suffix: str) -> Optional[Path]:
    if aoi_col is None or y_col not in df.columns or 'month' not in df.columns:
        return None

    d = ensure_month_name(df)
    irplt.use_thesis_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    cols = palette(12)

    plotted = 0
    for m in range(1, 13):
        sl = d[d['month'] == m][[aoi_col, y_col]].dropna()
        if sl.empty:
            continue
        sl = sl.sort_values(by=aoi_col)
        # 19/10/25 {label_prefix} removed
        scatter(ax, sl[aoi_col], sl[y_col], label=f"{MONTH_NAMES[m]}", color=cols[(m-1) % len(cols)],
        # 19/10/25 test empty
                facecolors='none')
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_xlabel("angle of incidence [degree]")
    ax.set_ylabel("cosine error [%]" if 'cos' in y_col.lower() else "absolute deviation [W/m2]")
    legend_below(fig, ax, ncol='auto')

    out = Path(outdir) / f"{name}{suffix}.png"
    out = os.path.join(outdir,f"{name}{suffix}.png")
    irplt.savefig(fig, out, tight=False); plt.close(fig)
    return out


def tilted_four_series_abs(df: pd.DataFrame, name: str, outdir: Path) -> Optional[Path]:
    aoi = choose_aoi_column(df)
    if 'hour' not in df.columns or 'abs_err' not in df.columns:
        return None

    before, after = split_by_hour(df)
    horiz_b, tilt_b = split_by_aoi(before, aoi_col=aoi if aoi else 'aoi')
    horiz_a, tilt_a = split_by_aoi(after,  aoi_col=aoi if aoi else 'aoi')

    irplt.use_thesis_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    cols = palette(4)

    series = [
        (horiz_b, "absolute deviation before 12:00 horizontal pyranometer", cols[0]),
        (horiz_a, "absolute deviation after 12:00 horizontal pyranometer",  cols[1]),
        (tilt_b,  "absolute deviation before 12:00 29° tilted pyranometer", cols[2]),
        (tilt_a,  "absolute deviation after 12:00 29° tilted pyranometer",  cols[3]),
    ]

    aoi_col = aoi if aoi else 'aoi'
    for dsub, lbl, col in series:
        if aoi_col in dsub.columns and 'abs_err' in dsub.columns:
            dd = dsub[[aoi_col, 'abs_err']].dropna().sort_values(by=aoi_col)
            if not dd.empty:
                scatter(ax, dd[aoi_col], dd['abs_err'], label=lbl, color=col)

    ax.set_xlabel("angle of incidence [degree]")
    ax.set_ylabel("absolute deviation [W/m2]")
    legend_below(fig, ax, ncol=1) #nocl=2

    out = Path(outdir) / f"{name}_ba12_abs.png"
    out = os.path.join(outdir,f"{name}_ba12_abs.png")
    irplt.savefig(fig, out, tight=False); plt.close(fig)
    return out


def tilted_four_series_dir(df: pd.DataFrame, name: str, outdir: Path) -> Optional[Path]:
    aoi = choose_aoi_column(df)
    if aoi is None or 'hour' not in df.columns or 'cos_err' not in df.columns:
        return None

    before, after = split_by_hour(df)

    def _split(d):
        tilted = d[d[aoi].notna()].copy()
        horiz = d[d[aoi].isna()].copy()
        return horiz, tilted

    horiz_b, tilt_b = _split(before)
    horiz_a, tilt_a = _split(after)

    irplt.use_thesis_style()
    fig, ax_left = plt.subplots(figsize=(7.0, 4.6))
    ax_right = ax_left.twinx()
    cols = palette(4)

    series = [
        (horiz_b, "absolute error before 12:00 horizontal pyranometer", cols[0]),
        (horiz_a, "absolute error after 12:00 horizontal pyranometer",  cols[1]),
        (tilt_b,  "absolute error before 12:00 29° tilted pyranometer",  cols[2]),
        (tilt_a,  "absolute error after 12:00 29° tilted pyranometer",   cols[3]),
    ]

    for dsub, lbl, col in series:
        if aoi in dsub.columns and 'cos_err' in dsub.columns:
            dd = compute_absolute_from_cosine(dsub[[aoi, 'cos_err']]).dropna().sort_values(by=aoi)
            if dd.empty:
                continue
            # LEFT: absolute (legend)
            scatter(ax_left, dd[aoi], dd['abs_err_from_cos'], label=lbl, color=col, zorder=3)
            # RIGHT: cosine (no legend)
            scatter(ax_right, dd[aoi], dd['cos_err'], label=None, color=col, zorder=2)

    ax_left.set_xlabel("angle of incidence [degree]")
    ax_left.set_ylabel("absolute error [W/m2]")
    ax_right.set_ylabel("cosine error [%]")
    legend_below(fig, ax_left, ncol=1) #nocl=2

    out = Path(outdir) / f"{name}_ba12_dir_dualy.png"
    out = os.path.join(outdir,f"{name}_ba12_dir_dualy.png")
    irplt.savefig(fig, out, tight=False); plt.close(fig)
    return out


def tilted_tcorrected_abs(df: pd.DataFrame, name: str, outdir: Path) -> Optional[Path]:
    if 'hour' not in df.columns:
        return None

    before, after = split_by_hour(df)
    tc_cols = [c for c in df.columns if 'abs_err' in c.lower() and ('tcorrect' in c.lower() or 't_corr' in c.lower())]
    base_col = 'abs_err' if 'abs_err' in df.columns else (tc_cols[0] if tc_cols else None)
    if base_col is None:
        return None
    tcol = tc_cols[0] if tc_cols else base_col
    aoi = choose_aoi_column(df)

    irplt.use_thesis_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    cols = palette(4)

    series = [
        (before, base_col, "absolute deviation before 12:00 horizontal pyranometer",         cols[0]),
        (after,  base_col, "absolute deviation after 12:00 horizontal pyranometer",          cols[1]),
        (before, tcol,     "absolute deviation before 12:00 horizontal pyranometer T corrected", cols[2]),
        (after,  tcol,     "absolute deviation after 12:00 horizontal pyranometer T corrected",  cols[3]),
    ]

    if aoi is None:
        return None

    for dsub, ycol, lbl, col in series:
        if aoi in dsub.columns and ycol in dsub.columns:
            dd = dsub[[aoi, ycol]].dropna().sort_values(by=aoi)
            if not dd.empty:
                scatter(ax, dd[aoi], dd[ycol], label=lbl, color=col)

    ax.set_xlabel("angle of incidence [degree]")
    ax.set_ylabel("absolute deviation [W/m2]")
    legend_below(fig, ax, ncol=1) #nocl=2

    out = Path(outdir) / f"{name}_ba12_abs.png"
    out = os.path.join(outdir,f"{name}_ba12_abs.png")
    irplt.savefig(fig, out, tight=False); plt.close(fig)
    return out


def tilted_tcorrected_dir(df: pd.DataFrame, name: str, outdir: Path) -> Optional[Path]:
    if 'hour' not in df.columns:
        return None

    before, after = split_by_hour(df)
    tc_cols = [c for c in df.columns if 'cos_err' in c.lower() and ('tcorrect' in c.lower() or 't_corr' in c.lower())]
    base_col = 'cos_err' if 'cos_err' in df.columns else (tc_cols[0] if tc_cols else None)
    if base_col is None:
        return None
    tcol = tc_cols[0] if tc_cols else base_col
    aoi = choose_aoi_column(df)
    if aoi is None:
        return None

    irplt.use_thesis_style()
    fig, ax_left = plt.subplots(figsize=(7.0, 4.6))
    ax_right = ax_left.twinx()
    cols = palette(4)

    series = [
        (before, base_col, "absolute error before 12:00 horizontal pyranometer",         cols[0]),
        (after,  base_col, "absolute error after 12:00 horizontal pyranometer",          cols[1]),
        (before, tcol,     "absolute error before 12:00 horizontal pyranometer T corrected", cols[2]),
        (after,  tcol,     "absolute error after 12:00 horizontal pyranometer T corrected",  cols[3]),
    ]

    for dsub, ycol, lbl, col in series:
        if aoi in dsub.columns and ycol in dsub.columns:
            # 20/10/25 fixed, do it for all at make_figures_for_df
            """dd = dsub[[aoi, ycol]].rename(columns={ycol: 'cos_err'})
            dd = compute_absolute_from_cos_err(dd).dropna().sort_values(by=aoi)
            # LEFT: absolute (legend)
            scatter(ax_left, dd[aoi], dd['abs_err_from_cos'], label=lbl, color=col, zorder=3)
            # RIGHT: cosine (no legend)
            scatter(ax_right, dd[aoi], dd['cos_err'], label=None, color=col, zorder=2)"""
            dd = dsub[[aoi, ycol]]
            dd = dd.dropna().sort_values(by=aoi)
            # LEFT: absolute (legend)
            scatter(ax_left, dd[aoi], dd['abs_err'], label=lbl, color=col, zorder=3)
            # RIGHT: cosine (no legend)
            scatter(ax_right, dd[aoi], dd['cos_err'], label=None, color=col, zorder=2)

    ax_left.set_xlabel("angle of incidence [degree]")
    ax_left.set_ylabel("absolute error [W/m2]")
    ax_right.set_ylabel("cosine error [%]")
    legend_below(fig, ax_left, ncol=1) # ncol=2

    out = Path(outdir) / f"{name}_ba12_dir_dualy.png"
    out = os.path.join(outdir,f"{name}_ba12_dir_dualy.png")
    irplt.savefig(fig, out, tight=False); plt.close(fig)
    return out

# ------------------------------------ DISPATCHER ------------------------------

def make_figures_for_df(df: pd.DataFrame, name: str, outdir: Path):
    d0 = df.copy()
    aoi_col = choose_aoi_column(d0)
    # 20/10/25 fix
    # if aoi_col:
    #    d0 = compute_cos_from_absolute(d0, aoi_col=aoi_col, abs_col='abs_err')
    cos_err_col = choose_cos_err_column(d0)
    if cos_err_col:
        d0 = compute_absolute_from_cos_err(d0, aoi_col=aoi_col, abs_col='abs_err')

    d0 = ensure_month_name(d0)

    # Base figures for all files
    plot_abs_vs_aoi(d0, name, outdir, aoi_col=aoi_col, y_col='abs_err',
                    series_label='absolute deviation [W/m2]', suffix='_abs')
    plot_directional_dual_y(d0, name, outdir, aoi_col=aoi_col, cos_col='cos_err', suffix='_dir_dualy')

    lname = name.lower()

    # crest_roof special cases
    if 'crest_roof' in lname:
        if lname == 'crest_roof_180825_abs_bi200cr20':
            crest_roof_four_series_abs(d0, name, outdir)
        if lname == 'crest_roof_180825_cos_bi200cr20':
            crest_roof_four_series_dir(d0, name, outdir)
        if 'almostclear' in lname and '_abs_' in lname:
            abs_v_cols = [c for c in d0.columns if 'abs_err' in c.lower() and 'ventilated' in c.lower()]
            ycol = abs_v_cols[0] if abs_v_cols else 'abs_err'
            months_multi_series(d0, name, outdir, aoi_col=aoi_col,
                                y_col=ycol,
                                label_prefix="ventilated horizontal pyranometer",
                                suffix='_months_abs')
            cos_v_cols = [c for c in d0.columns if 'cos_err' in c.lower() and 'ventilated' in c.lower()]
            ycos = cos_v_cols[0] if cos_v_cols else 'cos_err'
            months_multi_series(d0, name.replace('_abs_', '_cos_'), outdir, aoi_col=aoi_col,
                                y_col=ycos,
                                label_prefix="ventilated horizontal pyranometer",
                                suffix='_months_dir')
        if 'months_all' in lname and '_abs_' in lname:
            abs_v_cols = [c for c in d0.columns if 'abs_err' in c.lower() and 'ventilated' in c.lower()]
            ycol = abs_v_cols[0] if abs_v_cols else 'abs_err'
            months_multi_series(d0, name, outdir, aoi_col=aoi_col,
                                y_col=ycol,
                                label_prefix="ventilated horizontal pyranometer",
                                suffix='_months_abs')
            cos_v_cols = [c for c in d0.columns if 'cos_err' in c.lower() and 'ventilated' in c.lower()]
            ycos = cos_v_cols[0] if cos_v_cols else 'cos_err'
            months_multi_series(d0, name.replace('_abs_', '_cos_'), outdir, aoi_col=aoi_col,
                                y_col=ycos,
                                label_prefix="ventilated horizontal pyranometer",
                                suffix='_months_dir')

    # EURAC 2015 (12-series, horizontal)
    if lname == 'eurac_2015_error_abs_bi200cr20':
        months_multi_series(d0, name, outdir, aoi_col=aoi_col,
                            y_col='abs_err',
                            label_prefix="horizontal pyranometer",
                            suffix='_months_abs')
        # FIX: use outdir (outdirR was undefined and caused a NameError)
        months_multi_series(d0, 'eurac_2015_directional error_cos_bi200cr20', outdir, aoi_col=aoi_col,
                            y_col='cos_err',
                            label_prefix="horizontal pyranometer",
                            suffix='_months_dir')

    # Tilted set
    if lname == 'crest_ground_180824_29_error_abs_bi200cr20':
        tilted_four_series_abs(d0, name, outdir)
    if lname == 'crest_ground_180824_29_error_cos_bi200cr20':
        tilted_four_series_dir(d0, name, outdir)
    if lname == 'crest_ground_180824_29_error_tcorrected_abs_bi200cr20':
        tilted_tcorrected_abs(d0, name, outdir)
    if lname == 'crest_ground_180824_29_error_tcorrected_cos_bi200cr20':
        tilted_tcorrected_dir(d0, name, outdir)
