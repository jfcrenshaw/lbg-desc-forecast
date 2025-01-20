"""Function to optimize the LBG magnitude cuts for LSS SNR."""

import numpy as np
import pyccl as ccl
from scipy.optimize import minimize_scalar

from .mapper import Mapper


def _snr(
    mapper: Mapper,
    mag_cut: float,
    m5_min: float,
    cosmology,
):
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
    return mapper.calc_lss_snr(cosmology=cosmology)


def optimize_cuts(
    mapper: Mapper,
    contamination: float | None = 0.1,
    dz: float | None = 0.0,
    f_interlopers: float | None = 0.0,
    mag_bias: float | None = 1.0,
    metric: str = "tot",
    cosmology: ccl.Cosmology | None = None,
):
    """Optimize mag_cut and m5_min by maximizing SNR of LSS signal.

    Parameters
    ----------
    mapper : Mapper
        The Mapper for which the cuts are optimized.
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

    # First optimize mag_cut
    drop_shift = -mapper.dropout + 2.5 * np.log10(5 / mapper.snr_min)
    lower = max(23, np.ma.min(mapper.drop_map) + drop_shift)
    upper = min(np.ma.max(mapper.det_map), np.ma.max(mapper.drop_map) + drop_shift)
    res = minimize_scalar(
        fun=lambda mag_cut: -_snr(
            mapper=mapper,
            mag_cut=mag_cut,
            m5_min=mag_cut,
            cosmology=cosmology,
        )[idx],
        bounds=(lower, upper),
        options=dict(xatol=1e-3),
    )
    if not res.success:
        raise RuntimeError("Optimizing mag_cut failed.")
    if np.isclose(res.x, lower):
        raise RuntimeError("mag_cut hit lower bound.")
    if np.isclose(res.x, upper):
        raise RuntimeError("mag_cut hit upper bound.")
    mag_cut = res.x

    # Now optimize m5_min
    lower = mag_cut
    upper = np.ma.max(mapper.det_map)
    res = minimize_scalar(
        fun=lambda m5_min: -_snr(
            mapper=mapper,
            mag_cut=mag_cut,
            m5_min=m5_min,
            cosmology=cosmology,
        )[idx],
        bounds=(lower, upper),
        options=dict(xatol=1e-3),
    )
    if not res.success:
        raise RuntimeError("Optimizing m5_min failed.")
    if np.isclose(res.x, upper):
        raise RuntimeError("m5_min hit upper bound.")
    m5_min = res.x

    return mag_cut, m5_min
