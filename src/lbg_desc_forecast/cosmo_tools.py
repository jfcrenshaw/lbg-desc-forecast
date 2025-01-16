"""Tools for cosmology analyses"""

import numpy as np
import pyccl as ccl
from lbg_tools import TomographicBin

from lbg_desc_forecast.utils import get_lensing_noise

# Load lensing noise
ell, Nkk = get_lensing_noise()

# Contants for sky area
A_SKY = 41_253  # deg^2
DEG2_PER_STER = A_SKY / (4 * 3.14159)


def create_lbg_tracer(
    band: str,
    mag_cut: float,
    m5_det: float,
    dz: float = 0.0,
    f_interlopers: float = 0.0,
    mag_bias: float | None = None,
    cosmology: ccl.Cosmology | None = None,
) -> ccl.tracers.NzTracer:
    """Create number density tracer for LBG dropouts.

    Parameters
    ----------
    band : str
        Name of dropout band
    mag_cut : floa
        Magnitude cut in the detection band for LBGs
    m5_det : float or array or None
        5-sigma depth in the detection band. If None, mag_cut is used.
    dz : float, optional
        Amount by which to shift the distribution of true LBGs (i.e.
        interlopers are not shifted). This corresponds to the DES delta z
        nuisance parameters. Default is zero.
    f_interlopers : float, optional
        Fraction of low-redshift interlopers. Same p(z) shape is used
        for interlopers, but shifted to the redshift corresponding to
        Lyman-/Balmer-break confusion. Default is zero.
    mag_bias : float or None, optional
        Magnification bias alpha value. If None, value is pulled from the
        lbg_tools TomographicBin. Default is None.
    cosmology : ccl.Cosmology
        CCL cosmology object. If None, vanilla LCDM is used. Default is None.

    Returns
    -------
    ccl.tracers.NzTracer
        Number counts tracer for LBGs
    """
    # Create the tomographic bin
    tb = TomographicBin(
        band=band,
        mag_cut=mag_cut,
        m5_det=m5_det,
        dz=dz,
        f_interlopers=f_interlopers,
    )

    # Sample p(z) and bias on dense grid
    z = np.arange(0, 8.01, 0.01)
    pz = np.interp(z, *tb.pz)
    b = np.interp(z, *tb.g_bias)

    # Determine magnification bias
    # alpha = 2.5 * d log10(N) / d mag
    # s = alpha / 2.5
    z_interlopers, z_lbg = tb._get_z_grids()
    alpha = tb.mag_bias if mag_bias is None else mag_bias
    alpha = np.array([0.0] * len(z_interlopers) + [alpha] * len(z_lbg))
    sz = alpha / 2.5
    sz = np.interp(  # Resample on denser grid
        z, np.concatenate((z_interlopers, z_lbg)), sz
    )

    # Prep the cosmology object
    if cosmology is None:
        cosmology = ccl.CosmologyVanillaLCDM()

    # Create the tracer
    tracer = ccl.NumberCountsTracer(
        cosmology,
        has_rsd=False,
        dndz=(z, pz),
        bias=(z, b),
        mag_bias=(z, sz),
    )

    return tracer


def calc_lbg_spectra(
    band: str,
    mag_cut: float,
    m5_det: float,
    dz: float = 0.0,
    f_interlopers: float = 0.0,
    mag_bias: float | None = None,
    cosmology: ccl.Cosmology | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate angular cross-spectra of LBGs and CMB Lensing.

    Parameters
    ----------
    band : str
        Name of dropout band
    mag_cut : floa
        Magnitude cut in the detection band for LBGs
    m5_det : float or array or None, default=None
        5-sigma depth in the detection band. If None, mag_cut is used
    dz : float, optional
        Amount by which to shift the distribution of true LBGs (i.e.
        interlopers are not shifted). This corresponds to the DES delta z
        nuisance parameters. Default is zero.
    f_interlopers : float, optional
        Fraction of low-redshift interlopers. Same p(z) shape is used
        for interlopers, but shifted to the redshift corresponding to
        Lyman-/Balmer-break confusion. Default is zero.
    mag_bias : float or None, optional
        Magnification bias alpha value. If None, value is pulled from the
        lbg_tools TomographicBin. Default is None.
    cosmology : ccl.Cosmology
        CCL cosmology object. If None, vanilla LCDM is used.

    Returns
    -------
    np.ndarray
        Set of multipoles, ell
    np.ndarray
        Cgg -- LBG autospectrum
    np.ndarray
        Ckg -- LBG x CMB Lensing spectrum
    np.ndarray
        Ckk -- CMB Lensing autospectrum
    """
    # Prep the cosmology object
    if cosmology is None:
        cosmology = ccl.CosmologyVanillaLCDM()

    # Create tracers
    lbg_tracer = create_lbg_tracer(
        band=band,
        mag_cut=mag_cut,
        m5_det=m5_det,
        dz=dz,
        f_interlopers=f_interlopers,
        mag_bias=mag_bias,
        cosmology=cosmology,
    )
    cmb_lensing = ccl.CMBLensingTracer(cosmology, z_source=1100)

    # Calculate cross-spectra
    Cgg = ccl.angular_cl(cosmology, lbg_tracer, lbg_tracer, ell)
    Ckg = ccl.angular_cl(cosmology, cmb_lensing, lbg_tracer, ell)
    Ckk = ccl.angular_cl(cosmology, cmb_lensing, cmb_lensing, ell)

    return ell, Cgg, Ckg, Ckk


def calc_snr_from_products(
    Cgg: np.ndarray,
    Ckg: np.ndarray,
    Ckk: np.ndarray,
    Cff: np.ndarray,
    Ngg: float,
    f_sky: float,
) -> tuple[float, float, float]:
    """Calculate the LSS SNRs from the given products.

    Parameters
    ----------
    Cgg : np.ndarray
        Galaxy auto-spectrum
    Ckg : np.ndarray
        CMB Lensing x galaxy cross-spectrum
    Ckk : np.ndarray
        CMB Lensing auto-spectrum
    Cff : np.ndarray
        Auto-spectrum of non-uniformity
    Ngg : float
        Galaxy shot noise
    f_sky : float
        Fraction of the sky covered by analysis

    Returns
    -------
    float
        Total SNR of clustering and CMB Lensing x-corr
    float
        SNR of clustering
    float
        SNR of CMB Lensing x-corr
    """
    # Assemble signal vector
    mu = np.concatenate((Cgg, Ckg)).reshape(-1, 1)

    # Calculate covariances
    norm = 1 / (2 * ell + 1) / f_sky
    Cov_gggg = 2 * norm * (Cgg + Ngg) ** 2 + Cff**2
    Cov_ggkg = 2 * norm * Ckg * (Cgg + Ngg)
    Cov_kgkg = norm * ((Ckk + Nkk) * (Cgg + Ngg) + Ckg**2)

    # Inverse covariance matrix
    det = Cov_gggg * Cov_kgkg - Cov_ggkg**2  # Determinant
    Cov_inv = np.block(
        [
            [np.diag(Cov_kgkg / det), -np.diag(Cov_ggkg / det)],
            [-np.diag(Cov_ggkg / det), np.diag(Cov_gggg / det)],
        ]
    )

    # Calculate weighted SNRs
    snr_tot = np.sqrt(mu.T @ Cov_inv @ mu)[0, 0]
    snr_gg = np.sqrt(np.sum(Cgg**2 / Cov_gggg))
    snr_kg = np.sqrt(np.sum(Ckg**2 / Cov_kgkg))

    return snr_tot, snr_gg, snr_kg
