"""Functions that generate default LBG params for forecasting."""

from scipy.integrate import simpson

from .mapper import Mapper

# These cuts are taken from optimize_cuts.ipynb
_lbg_cuts = {
    1: {
        0: {
            "u": {"mag_cut": 23.68, "m5_min": 23.68},
            "g": {"mag_cut": 24.76, "m5_min": 24.76},
            "r": {"mag_cut": 24.34, "m5_min": 24.34},
        },
        0.1: {
            "u": {"mag_cut": 23.68, "m5_min": 23.68},
            "g": {"mag_cut": 24.74, "m5_min": 24.75},
            "r": {"mag_cut": 24.34, "m5_min": 24.34},
        },
        1: {
            "u": {"mag_cut": 23.68, "m5_min": 23.68},
            "g": {"mag_cut": 24.58, "m5_min": 24.65},
            "r": {"mag_cut": 24.34, "m5_min": 24.34},
        },
    },
    10: {
        0: {
            "u": {"mag_cut": 24.76, "m5_min": 24.85},
            "g": {"mag_cut": 25.85, "m5_min": 25.87},
            "r": {"mag_cut": 25.51, "m5_min": 25.51},
        },
        0.1: {
            "u": {"mag_cut": 24.76, "m5_min": 24.85},
            "g": {"mag_cut": 25.83, "m5_min": 25.89},
            "r": {"mag_cut": 25.50, "m5_min": 25.50},
        },
        1: {
            "u": {"mag_cut": 24.75, "m5_min": 24.85},
            "g": {"mag_cut": 25.71, "m5_min": 25.88},
            "r": {"mag_cut": 25.50, "m5_min": 25.50},
        },
    },
}


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
    # Check parameters
    if year not in [1, 10] or contamination not in [0, 0.1, 1]:
        raise ValueError(
            f"Default config not found for year {year} "
            f"and contamination {contamination}."
        )

    # Create dict for each band
    u_params = dict(
        band="u",
        year=year,
        mag_cut=_lbg_cuts[year][contamination]["u"]["mag_cut"],
        m5_min=_lbg_cuts[year][contamination]["u"]["m5_min"],
        contamination=contamination,
    )
    g_params = dict(
        band="g",
        year=year,
        mag_cut=_lbg_cuts[year][contamination]["g"]["mag_cut"],
        m5_min=_lbg_cuts[year][contamination]["g"]["m5_min"],
        contamination=contamination,
    )
    r_params = dict(
        band="r",
        year=year,
        mag_cut=_lbg_cuts[year][contamination]["r"]["mag_cut"],
        m5_min=_lbg_cuts[year][contamination]["r"]["m5_min"],
        contamination=contamination,
    )

    # Interloper fraction (taken from eye-balling DESI papers)
    if year == 1:
        u_params |= {"f_interlopers": 0.26}
        g_params |= {"f_interlopers": 0.05}
        r_params |= {"f_interlopers": 0.05}
    elif year == 10:
        u_params |= {"f_interlopers": 0.14}
        g_params |= {"f_interlopers": 0.02}
        r_params |= {"f_interlopers": 0.02}

    # Calculate effective bias of each LBG sample
    u_params |= {"g_bias": _calc_effective_bias(u_params)}
    g_params |= {"g_bias": _calc_effective_bias(g_params)}
    r_params |= {"g_bias": _calc_effective_bias(r_params)}

    # Interloper bias taken from lens bias of bins 1, 2, 4 of
    # https://arxiv.org/abs/2105.13549 (bins chosen to be closest
    # to interloper pop for each LBG sample; bias from Table V)
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
