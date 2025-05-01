"""Functions that generate default LBG params for forecasting."""

import numpy as np
from scipy.integrate import simpson

from .mapper import Mapper


def _calc_effective_bias(params: dict) -> float:
    """Calculate effective bias for the given mapper params.

    Parameters
    ----------
    params : dict
        Dictionary of mapper params

    Returns
    -------
    float
        Effective linear galaxy bias
    """
    mapper = Mapper(**params)
    z, pz = mapper.tomographic_bin.pz
    _, b = mapper.tomographic_bin.g_bias
    return simpson(b * pz, z)


def get_lbg_params(year: int, contamination: float) -> tuple[dict, dict, dict]:
    """Get default LBG params for the given year

    Parameters
    ----------
    year : int
        Year of LSST
    contamination : float
        Non-uniformity contamination level

    Returns
    -------
    dict
        Default param dict for u-dropouts
    dict
        Default param dict for g-dropouts
    dict
        Default param dict for r-dropouts
    """
    error = ValueError(
        "Default config not found for " f"year {year} and contamination {contamination}"
    )

    if year == 10:
        if np.isclose(contamination, 0.1):
            # Parameters from cut optimization
            u_params = dict(
                band="u",
                year=10,
                mag_cut=24.81,
                m5_min=25.75,
                contamination=contamination,
            )

            g_params = dict(
                band="g",
                year=10,
                mag_cut=25.90,
                m5_min=25.96,
                contamination=contamination,
            )

            r_params = dict(
                band="r",
                year=10,
                mag_cut=25.52,
                m5_min=25.52,
                contamination=contamination,
            )
        elif np.isclose(contamination, 0.0):
            # Parameters from cut optimization
            u_params = dict(
                band="u",
                year=10,
                mag_cut=24.81,
                m5_min=25.75,
                contamination=contamination,
            )

            g_params = dict(
                band="g",
                year=10,
                mag_cut=25.90,
                m5_min=25.91,
                contamination=contamination,
            )

            r_params = dict(
                band="r",
                year=10,
                mag_cut=25.52,
                m5_min=25.52,
                contamination=contamination,
            )
        else:
            raise error

        # Interloper fraction (taken from eye-balling DESI papers)
        u_params |= {"f_interlopers": 0.14}
        g_params |= {"f_interlopers": 0.02}
        r_params |= {"f_interlopers": 0.02}

    elif year == 1:
        if np.isclose(contamination, 0.1):
            # Parameters from cut optimization
            u_params = dict(
                band="u",
                year=10,
                mag_cut=23.88,
                m5_min=24.39,
            )

            g_params = dict(
                band="g",
                year=10,
                mag_cut=24.72,
                m5_min=24.76,
            )

            r_params = dict(
                band="r",
                year=10,
                mag_cut=24.35,
                m5_min=24.35,
            )
        elif np.isclose(contamination, 0.0):
            # Parameters from cut optimization
            u_params = dict(
                band="u",
                year=10,
                mag_cut=23.87,
                m5_min=24.39,
            )

            g_params = dict(
                band="g",
                year=10,
                mag_cut=24.74,
                m5_min=24.74,
            )

            r_params = dict(
                band="r",
                year=10,
                mag_cut=24.35,
                m5_min=24.35,
            )
        else:
            raise error

        # Interloper fraction (taken from eye-balling DESI papers)
        u_params |= {"f_interlopers": 0.26}
        g_params |= {"f_interlopers": 0.05}
        r_params |= {"f_interlopers": 0.05}

    else:
        raise ValueError(f"Year {year} not implemented")

    # Calculate effective bias of each LBG sample
    u_params |= {"g_bias": _calc_effective_bias(u_params)}
    g_params |= {"g_bias": _calc_effective_bias(g_params)}
    r_params |= {"g_bias": _calc_effective_bias(r_params)}

    # Interloper bias taken from lens bias of bins 1, 2, 4 of
    #   https://arxiv.org/abs/2105.13549 (bins chosen to be closest to
    #   interloper pop for each LBG sample; bias from Table V)
    u_params |= {"g_bias_inter": 1.49}  # +/- 0.10
    g_params |= {"g_bias_inter": 1.69}  # +/- 0.11
    r_params |= {"g_bias_inter": 1.79}  # +/- 0.11

    # Set mag bias to value calculated from the luminosity function
    u_params |= {"mag_bias": Mapper(**u_params).tomographic_bin.mag_bias}  # type:ignore
    g_params |= {"mag_bias": Mapper(**g_params).tomographic_bin.mag_bias}  # type:ignore
    r_params |= {"mag_bias": Mapper(**r_params).tomographic_bin.mag_bias}  # type:ignore

    return u_params, g_params, r_params


def get_lbg_mappers(year: int, contamination: float = 0.1) -> list[Mapper]:
    """Get default LBG mappers for the given year

    Parameters
    ----------
    year : int
        Year of LSST

    Returns
    -------
    list[Mapper]
        List of default mappers for u,g,r-dropouts
    """
    return [Mapper(**params) for params in get_lbg_params(year, contamination)]
