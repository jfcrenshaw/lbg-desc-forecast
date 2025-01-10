"""Function that interpolates tomographic LBG number densities"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .utils import data_dir

# Create 2D interpolators of number density
_file_cache_number_density = data_dir / "caches" / "cache_number_density.npz"
if not _file_cache_number_density.exists():
    raise RuntimeError(
        "Number density cache does not exist. Please run bin/create_caches.py"
    )
_cache_number_density = np.load(_file_cache_number_density, allow_pickle=True)
cache_number_density = {
    "m5": _cache_number_density["m5"],
    "n": _cache_number_density["n"].tolist(),
}
_number_density_interpolator = {
    band: RegularGridInterpolator(
        (cache_number_density["m5"], cache_number_density["m5"]),
        cache_number_density["n"][band],
        bounds_error=False,
        fill_value=0,
    )
    for band in cache_number_density["n"]
}


def interpolate_number_density(
    band: str,
    mag_cut: np.ndarray,
    m5_det: np.ndarray,
) -> np.ndarray:
    """Interpolate the number density using the cache.

    Cache must first be created using bin/create_caches.py

    Parameters
    ----------
    band : str
        Name of the dropout band
    mag_cut : np.ndarray
        Magnitude cuts in the detection band
    m5_det : np.ndarray
        5-sigma depths in the detection band

    Returns
    -------
    np.ndarray
        number densities, in deg^-2
    """
    interpolator = _number_density_interpolator[band]
    return interpolator((mag_cut, m5_det))
