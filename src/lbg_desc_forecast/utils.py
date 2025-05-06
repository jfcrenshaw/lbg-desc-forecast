"""Simple utility functions"""

import contextlib
import os
from pathlib import Path

import healpy as hp
import numpy as np
import rubin_sim.maf as maf

data_dir = Path(__file__).parents[2] / "data"
fig_dir = Path(__file__).parents[2] / "figures"


def get_lensing_noise() -> tuple[np.ndarray, np.ndarray]:
    """Load the lensing noise.

    This is the minimum-variance baseline forecast for the Simons Obs.

    Returns
    -------
    np.ndarray
        A grid of multipoles (ell)
    np.ndarray
        Lensing noise as a function of ell (Nkk)
    """
    lensing_noise = np.genfromtxt(
        data_dir / "nlkk_v3_1_0_deproj0_SENS1_fsky0p4_it_lT30-3000_lP30-5000.dat"
    )
    ell = lensing_noise[:, 0]
    Nkk = lensing_noise[:, 7]

    return ell, Nkk


def load_m5_map(band: str, year: int) -> np.ma.MaskedArray:
    """Load m5 map.

    Parameters
    ----------
    band : str
        Name of the band.
    year : int
        Year at which m5 is calculated.

    Returns
    -------
    np.ma.MaskedArray
        Masked array of 5-sigma depths
    """
    try:
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            map = maf.MetricBundle.load(
                data_dir / "m5_maps" / f"baseline_v4_0_{year}yrs_ExgalM5_{band}.npz"
            )
        return map.metric_values
    except OSError:
        raise ValueError(
            "Could not find an m5 map for band {band} in year {year}. "
            "Perhaps you have not run bin/calc_m5.py? Or maybe that script "
            "does not produce the band/year combo you have requested?"
        )


def plot_map(
    values: np.ma.MaskedArray,
    title: str | None = None,
    cbar_label: str | None = None,
    n_dec: int = 2,
    symm_limits: bool = True,
    sub: int = 111,
) -> None:
    """Plot metric on a Mollweide map.

    Parameters
    ----------
    values: np.ma.MaskedArray
        Metric values
    title: str or None, default=None
        Title for map
    cbar_label: str or None, default=None
        Label for the color bar.
    n_dec: int, default=2
        Number of decimals to display in colorbar ticks
    symm_limits: bool, default=True
        Whether to enforce symmetric limits for the color bar
    sub: int, default=111
        Integer indicating with subplot to plot the map on. For example, if you
        do `fig, axes = plt.subplots(2, 2)`, and want to put this map on the
        lower right subplot, you set `sub=224`.
    """
    # Don't allow healpy to override font sizes
    fontsize = {
        "xlabel": None,
        "ylabel": None,
        "title": None,
        "xtick_label": None,
        "ytick_label": None,
        "cbar_label": None,
        "cbar_tick_label": None,
    }

    # Get value limits
    if symm_limits:
        limit = np.abs(values).max()
        cbar_min = -limit
        cbar_max = +limit
        cbar_ticks = [cbar_min, 0, cbar_max]
    else:
        cbar_min = None
        cbar_max = None
        cbar_ticks = None

    # Plot map
    hp.projview(
        values,
        title=title,
        sub=sub,
        cmap="coolwarm",
        min=cbar_min,
        max=cbar_max,
        cbar_ticks=cbar_ticks,
        unit=cbar_label,
        format=f"%.{n_dec}f",
        fontsize=fontsize,
        fig=1,
        hold=True,
    )
