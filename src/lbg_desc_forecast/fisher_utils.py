"""Utils to load Fisher matrices."""

import numpy as np

from .forecaster import FisherMatrix
from .utils import data_dir

fisher_dir = data_dir / "fisher_matrices"


def load_srd_forecast(year: int) -> FisherMatrix:
    """Load Fisher Matrix for SRD forecast.

    Parameters
    ----------
    year : int
        The year of LSST. Can be year 1 or 10.

    Returns
    -------
    FisherMatrix
        The forecast Fisher matrix
    """
    # Load Fisher matrix
    matrix = np.load(data_dir / f"srd_y{year}_3x2pt_fisher_matrix_all.npy")

    # Create center and key vectors
    # Starting with shared cosmo params
    center = [
        0.3156,  # Omega_M
        0.831,  # sigma8
        0.9645,  # ns
        -1.0,  # w0
        0.0,  # wa
        0.0491685,  # Omega_b
        0.6727,  # h
    ]
    keys = [
        "Omega_M",
        "sigma8",
        "n_s",
        "w0",
        "wa",
        "Omega_b",
        "h",
    ]

    # Now galaxy bias for tomographic bins
    if year == 1:
        center += [  # galaxy bias for 5 nz bins
            1.250951,
            1.365806,
            1.500621,
            1.648023,
            1.799559,
        ]
        keys += [f"b{i}" for i in range(1, 6)]
    elif year == 10:
        center += [  # galaxy bias for 10 nz bins
            1.090801,
            1.143735,
            1.210781,
            1.267298,
            1.3251,
            1.397247,
            1.45733,
            1.531859,
            1.593606,
            1.669875,
        ]
        keys += [f"b{i}" for i in range(1, 11)]
    else:
        raise ValueError(f"year {year} not implemented")

    # Now IA params
    center += [
        5.92,  # a_0, IA amplitude
        1.1,  # beta, IA power law index
        -0.47,  # eta_low_z, exponent for low-z (eq. 7 and 8 in KEB)
        0.0,  # eta_high_z, exponent for high-z (eq. 7 and 8 in KEB)
    ]
    keys += [
        "a_0",
        "beta",
        "eta_low_z",
        "eta_high_z",
    ]

    srd_fmatrix = FisherMatrix(
        matrix=matrix,
        center=center,
        keys=keys,
    )

    # Marginalize over nuisance parameters
    srd_fmatrix = srd_fmatrix.marginalize(srd_fmatrix.keys[7:])

    return srd_fmatrix


def load_lbg_forecast(
    year: int,
    clean: bool = True,
    fix: list[str] = [],
    clustering: bool = True,
    xcorr: bool = True,
    lensing: bool = True,
    set_prior: bool = True,
) -> FisherMatrix:
    """Load Fisher matrix for LBG forecast

    Parameters
    ----------
    year : int
        Year of LSST
    clean : bool, optional
        Whether to load the forecast for the clean map (with 0% contamination),
        or the forecast for the fiducial map (10% contamination).
        Note it turns out this doesn't really matter.
        Default is True.
    fix : list[str], optional
        List of parameters to fix. Note that any non-fixed nuisance parameters
        are marginalized. Default is an empty list.
    clustering : bool, optional
        Whether to include LBG clustering in the forecast.
        Default is True.
    xcorr : bool, optional
        Whether to include LBGxCMB lensing in the forecast.
        Default is True.
    lensing : bool, optional
        Whether to include CMB lensing autocorrelation in the forecast.
        Default is True.
    set_prior : bool, optional
        Whether to set the fiducial prior for nuisance parameters.
        Default is True.

    Returns
    -------
    FisherMatrix
        Fisher matrix for LBG forecast
    """
    # Load the LBG forecast matrix
    if clean:
        if clustering and xcorr and lensing:
            lbg = FisherMatrix.load(fisher_dir / f"y{year}_clean.fisher_matrix.npz")
        elif clustering and xcorr:
            lbg = FisherMatrix.load(
                fisher_dir / f"y{year}_clean_clustering+xcorr.fisher_matrix.npz"
            )
        elif xcorr:
            lbg = FisherMatrix.load(
                fisher_dir / f"y{year}_clean_xcorr.fisher_matrix.npz"
            )
        else:
            lbg = FisherMatrix.load(
                fisher_dir / f"y{year}_clean_clustering.fisher_matrix.npz"
            )
    else:
        if clustering and xcorr and lensing:
            lbg = FisherMatrix.load(fisher_dir / f"y{year}_fiducial.fisher_matrix.npz")
        elif clustering and xcorr:
            lbg = FisherMatrix.load(
                fisher_dir / f"y{year}_fiducial_clustering+xcorr.fisher_matrix.npz"
            )
        elif xcorr:
            lbg = FisherMatrix.load(
                fisher_dir / f"y{year}_fiducial_xcorr.fisher_matrix.npz"
            )
        else:
            lbg = FisherMatrix.load(
                fisher_dir / f"y{year}_fiducial_clustering.fisher_matrix.npz"
            )

    # Fix neutrino mass for now
    lbg = lbg.fix("m_nu")

    # Set prior on nuisance parameters
    # - Interloper fraction priors are just vibes
    # - Interloper bias priors are described in default_lbg
    # - Mag bias are faint-end slope uncertainties from GOLDRUSH IV (Table 6)
    if set_prior:
        lbg.set_prior(
            u_dz=0.004 if year == 10 else np.inf,
            u_f_interlopers=0.05,
            g_f_interlopers=0.05,
            r_f_interlopers=0.05,
            u_g_bias_inter=0.10,
            g_g_bias_inter=0.11,
            r_g_bias_inter=0.11,
            u_mag_bias=0.02,
            g_mag_bias=0.03,
            r_mag_bias=0.04,
        )

    # Fix parameters
    lbg = lbg.fix(fix)

    # Set prior on cosmological parameters
    lbg.set_prior(
        Omega_M=0.15,
        Omega_b=5e-3,
        h=0.125,
        n_s=0.1,
        sigma8=0.2,
        w0=0.5,
        wa=1.3,
    )

    # Marginalize nuisance parameters
    for key in lbg.keys:
        if "dz" in key or "interlopers" in key or "bias" in key:
            lbg = lbg.marginalize(key)

    return lbg


def load_joint_forecast(
    year: int,
    clean: bool,
    fix: list[str] = [],
    clustering: bool = True,
    xcorr: bool = True,
    lensing: bool = True,
    set_prior: bool = True,
) -> tuple[FisherMatrix, FisherMatrix, FisherMatrix]:
    """Load Fisher matrices for joint low-z/high-z forecast.

    Parameters
    ----------
    year : int
        Year of LSST
    clean : bool, optional
        Whether to load the forecast for the clean map (with 0% contamination),
        or the forecast for the fiducial map (10% contamination).
        Note it turns out this doesn't really matter.
        Default is True.
    fix : list[str], optional
        List of parameters to fix. Note that any non-fixed nuisance parameters
        are marginalized. Default is an empty list.
    clustering : bool, optional
        Whether to include LBG clustering in the forecast.
        Default is True.
    xcorr : bool, optional
        Whether to include LBGxCMB lensing in the forecast.
        Default is True.
    lensing : bool, optional
        Whether to include CMB lensing autocorrelation in the forecast.
        Default is True.
    set_prior : bool, optional
        Whether to set the fiducial prior for nuisance parameters.
        Default is True.

    Returns
    -------
    FisherMatrix
        Fisher matrix for SRD forecast
    FisherMatrix
        Fisher matrix for LBG forecast
    FisherMatrix
        Fisher matrix for joint forecast
    """
    # Load component forecasts
    srd = load_srd_forecast(year)
    lbg_solo = load_lbg_forecast(
        year=year,
        clean=clean,
        fix=fix,
        clustering=clustering,
        xcorr=xcorr,
        lensing=lensing,
        set_prior=set_prior,
    )

    # To combine LBG forecast with SRD, we need to remove the prior on
    # cosmological parameters, as the SRD has already had the prior added in
    lbg = lbg_solo.copy()
    lbg.priors = np.full_like(lbg.priors, np.inf)

    # Create joint forecast
    joint = lbg + srd

    return srd, lbg_solo, joint
