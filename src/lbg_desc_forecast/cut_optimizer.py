"""Function to optimize the LBG magnitude cuts for LSS SNR."""

import numpy as np
import pyccl as ccl
from scipy.optimize import minimize_scalar

from .mapper import Mapper


def get_cut_for_fsky(
    mapper: Mapper,
    f_sky: float,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    """Calculate cut that generates the given f_sky.

    Note this assumes mag_cut = m5_min = cut.

    Parameters
    ----------
    mapper : Mapper
        The Mapper for which the cut is optimized.
    f_sky : float or None, optional
        The minimum f_sky allowed. Default is None.
    lower : float or None, optional
        Lower bound for minimizer. Default is None.
    upper : float or None, optional
        Upper bound for minimizer. Default is None.

    Returns
    -------
    float
        The cut that yields the requested f_sky
    """
    # Determine optimizer bounds
    if lower is None and upper is None:
        bounds = None
    elif lower is not None and upper is not None:
        bounds = (lower, upper)
    else:
        raise ValueError("If you provide either upper or lower you must provide both.")

    # Define the loss function
    def square_resid(cut: float) -> float:
        mapper.mag_cut = cut
        mapper.m5_min = cut
        return (mapper.f_sky - f_sky) ** 2

    # Minimize loss
    res = minimize_scalar(
        fun=square_resid,
        bounds=bounds,
        tol=1e-3,
    )
    if not res.success:
        raise RuntimeError("get_cut_for_fsky failed.")
    if lower is not None and np.isclose(res.x, lower):
        raise RuntimeError("get_cut_for_fsky hit lower bound.")
    if upper is not None and np.isclose(res.x, upper):
        raise RuntimeError("get_cut_for_fsky hit upper bound.")

    return res.x


def _snr(
    mapper: Mapper,
    mag_cut: float,
    m5_min: float,
    ell_min: float,
    ell_max: float,
    cosmology: ccl.Cosmology,
) -> tuple[float, float, float]:
    """Calculate SNRs for the optimization function below.

    Parameters
    ----------
    mapper : Mapper
        The Mapper for which the cuts are optimized.
    mag_cut : float, optional
        Magnitude cut on LBGs in the detection band.
    m5_min : float, optional
        The minimum depth in the detection band.
    contamination : float or None, optional
        Fraction of non-uniformity map that contaminates LSS signal.
    dz : float, optional
        Amount by which to shift the distribution of true LBGs (i.e.
        interlopers are not shifted). This corresponds to the DES delta z
        nuisance parameters.
    f_interlopers : float, optional
        Fraction of low-redshift interlopers. Same p(z) shape is used
        for interlopers, but shifted to the redshift corresponding to
        Lyman-/Balmer-break confusion.
    mag_bias : float or None, optional
        Magnification bias alpha value.
    ell_min : float
        Minimum ell value. Default is 50.
    ell_max : float
        Minimum ell value. Default is 2000.
    cosmology : ccl.Cosmology or None
        CCL cosmology object.

    Returns
    -------
    float
        Total SNR of clustering and CMB Lensing x-corr
    float
        SNR of clustering
    float
        SNR of CMB Lensing x-corr
    """
    mapper.mag_cut = mag_cut
    mapper.m5_min = m5_min
    return mapper.calc_lss_snr(ell_min=ell_min, ell_max=ell_max, cosmology=cosmology)


def optimize_cuts(
    mapper: Mapper,
    f_sky_min: float | None = 0.25,
    contamination: float | None = 0.1,
    dz: float | None = 0.0,
    f_interlopers: float | None = 0.0,
    mag_bias: float | None = 1.0,
    metric: str = "tot",
    ell_min: float = 50,
    ell_max: float = 2000,
    cosmology: ccl.Cosmology | None = None,
):
    """Optimize mag_cut and m5_min by maximizing SNR of LSS signal.

    Parameters
    ----------
    mapper : Mapper
        The Mapper for which the cuts are optimized.
    f_sky_min : float or None, optional
        The minimum f_sky allowed. Default 0.25.
    contamination : float or None, optional
        Fraction of non-uniformity map that contaminates LSS signal.
        If provided, overrides value from mapper object.
        Default is 0.1.
    dz : float, optional
        Amount by which to shift the distribution of true LBGs (i.e.
        interlopers are not shifted). This corresponds to the DES delta z
        nuisance parameters. If provided, overrides value from mapper object.
        Default is zero.
    f_interlopers : float, optional
        Fraction of low-redshift interlopers. Same p(z) shape is used
        for interlopers, but shifted to the redshift corresponding to
        Lyman-/Balmer-break confusion. If provided, overrides value from mapper
        object. Default is zero.
    mag_bias : float or None, optional
        Magnification bias alpha value. If provided, overrides value from mapper
        object. Default is 1.0, which corresponds to no magnification bias.
    metric : str, optional
        Which SNR to maximize. "tot" maximizes the total SNR, "gg" maximizes
        the clustering signal, and "kg" maximizes the cross-correlation with
        CMB lensing. Default is "tot"
    ell_min : float
        Minimum ell value. Default is 50.
    ell_max : float
        Minimum ell value. Default is 2000.
    cosmology : ccl.Cosmology or None
        CCL cosmology object. If None, vanilla LCDM is used. Default is None.

    Returns
    -------
    float
        optimized mag_cut
    float
        optimized m5_min
    """
    # Handle logic with overrides
    contamination = mapper.contamination if contamination is None else contamination
    dz = mapper.dz if dz is None else dz
    f_interlopers = mapper.f_interlopers if f_interlopers is None else f_interlopers
    mag_bias = mapper.mag_bias if mag_bias is None else mag_bias

    mapper = mapper.copy()
    mapper.contamination = contamination
    mapper.dz = dz
    mapper.f_interlopers = f_interlopers
    mapper.mag_bias = mag_bias

    # Determine the metric index
    if metric == "tot":
        idx = 0
    elif metric == "gg":
        idx = 1
    elif metric == "kg":
        idx = 2
    else:
        raise ValueError(f"Metric {metric} not supported.")

    # Prep the cosmology object
    if cosmology is None:
        cosmology = ccl.CosmologyVanillaLCDM()

    # First determine the max mag_cut we will allow
    drop_shift = -mapper.dropout + 2.5 * np.log10(5 / mapper.snr_min)
    lower = max(23, np.ma.min(mapper.drop_map) + drop_shift)
    upper = min(np.ma.max(mapper.det_map), np.ma.max(mapper.drop_map) + drop_shift)
    if f_sky_min is not None:
        upper = get_cut_for_fsky(mapper, f_sky_min, lower, upper)

    # Now optimize mag_cut
    res = minimize_scalar(
        fun=lambda mag_cut: -_snr(
            mapper=mapper,
            mag_cut=mag_cut,
            m5_min=mag_cut,
            ell_min=ell_min,
            ell_max=ell_max,
            cosmology=cosmology,
        )[idx],
        bounds=(lower, upper),
        options=dict(xatol=1e-3),
    )
    if not res.success:
        raise RuntimeError("Optimizing mag_cut failed.")
    if np.isclose(res.x, lower):
        raise RuntimeError("mag_cut hit lower bound.")
    mag_cut = res.x

    # Now optimize m5_min
    lower = mag_cut
    upper = min(np.ma.max(mapper.det_map), upper)
    res = minimize_scalar(
        fun=lambda m5_min: -_snr(
            mapper=mapper,
            mag_cut=mag_cut,
            m5_min=m5_min,
            ell_min=ell_min,
            ell_max=ell_max,
            cosmology=cosmology,
        )[idx],
        bounds=(lower, upper),
        options=dict(xatol=1e-3),
    )
    if not res.success:
        raise RuntimeError("Optimizing m5_min failed.")
    m5_min = res.x

    return mag_cut, m5_min
