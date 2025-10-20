"""
irr_io.py — IO & wrangling helpers for the directional response CSVs.

The CSVs are expected to be produced previously and stored in the two folders:
- Directional_response_outdoor
- Directional_response_outdoor_tilted

This module provides resilient loaders that try to standardize column names
based on common patterns found in the thesis datasets (e.g., 'aoi', 'dt',
'error_abs', 'error_cos', etc.).
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Heuristic column maps: maps any of these aliases to our standard name
_ALIAS_MAP = {
    'dt': ['dt', 'datetime', 'time', 'timestamp', 'date_time', 'DateTime', 'Date', 'TmStamp'],
    'aoi': ['aoi', 'angle_of_incidence', 'angleofincidence', 'theta', 'zenith', 'zenith_angle', 'AOI'],
    'error_abs': ['error_abs', 'err_abs', 'abs_error', 'absolute_error', 'Delta1000', 'delta1000_Wm2'],
    'error_cos': ['error_cos', 'err_cos', 'cosine_error', 'delta_cos', 'dcos_percent'],
    'beam': ['beam', 'dni_on_plane', 'E_b_tilt', 'Eb_poa', 'Eb', 'beam_poa'],
    'diffuse': ['diffuse', 'Ed_tilt', 'Ed', 'diffuse_poa'],
    'global': ['gpoa', 'global', 'E_tilt', 'Epoa', 'EG', 'global_poa'],
    'azimuth': ['azimuth', 'phi', 'phi_s', 'solar_azimuth'],
}


def _std_name(col: str) -> Optional[str]:
    lc = col.strip().lower()
    for std, aliases in _ALIAS_MAP.items():
        for al in aliases:
            if lc == al.lower():
                return std
    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with standardized, lower-case column names where possible."""
    cmap = {}
    for c in df.columns:
        std = _std_name(c)
        cmap[c] = std if std is not None else c.strip().lower()
    out = df.rename(columns=cmap).copy()

    # Normalize datetime column name to 'dt' and ensure tz-naive pandas datetime
    if 'dt' in out.columns:
        out['dt'] = pd.to_datetime(out['dt'], errors='coerce', utc=False)

    # If cosine error missing but absolute error & aoi available, attempt compute
    if 'error_cos' not in out.columns and 'error_abs' in out.columns and 'aoi' in out.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            cos_th = np.cos(np.deg2rad(out['aoi'].astype(float)))
            out['error_cos'] = 100.0 * out['error_abs'].astype(float) / (1000.0 * np.clip(cos_th, 1e-6, None))

    return out


def find_csvs(root: Path, base_names: List[str]) -> Dict[str, Path]:
    """Return mapping base_name -> path if found under root (recursive).

    Parameters
    ----------
    root : Path
        Root directory to search.
    base_names : list of str
        Target file base names (no extension).
    """
    root = Path(root)
    found = {}
    all_files = list(root.rglob('*.csv'))
    idx = {p.stem: p for p in all_files}
    for name in base_names:
        if name in idx:
            found[name] = idx[name]
    return found


def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV with robust datetime parsing and return standardized columns."""
    df = pd.read_csv(path, engine='python')
    return standardize_columns(df)
