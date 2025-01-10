from pathlib import Path

import numpy as np
import rubin_sim.maf as maf

data_dir = Path(__file__).parents[2] / "data"


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
