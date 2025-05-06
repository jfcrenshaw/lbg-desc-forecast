"""Run Fisher forecasts for the main cosmology and 10% contamination"""

import numpy as np

from lbg_desc_forecast import (
    Forecaster,
    MainCosmology,
    data_dir,
    get_lbg_mappers,
    fisher_dir,
)

# Directory in which to save results
fisher_dir.mkdir(parents=True, exist_ok=True)

# Create and save 1-year fiducial signal covariance
forecaster = Forecaster(get_lbg_mappers(1), MainCosmology())
forecaster.create_cov()
np.save(data_dir / "signal_covariance_y1.npy", forecaster.cov)

# Year 1 fiducial forecast
forecaster.create_fisher_matrix()
forecaster.fisher_matrix.save(fisher_dir / "y1_fiducial.fisher_matrix.npz")

# Create and save 10-year fiducial signal covariance
forecaster = Forecaster(get_lbg_mappers(10), MainCosmology())
forecaster.create_cov()
np.save(data_dir / "signal_covariance_y10.npy", forecaster.cov)

# Year 10 fiducial forecast
forecaster.create_fisher_matrix()
forecaster.fisher_matrix.save(fisher_dir / "y10_fiducial.fisher_matrix.npz")

# Y10 with clustering only
forecaster = Forecaster(
    get_lbg_mappers(10),
    MainCosmology(),
    clustering=True,
    xcorr=False,
    lensing=False,
)
forecaster.create_cov()
forecaster.create_fisher_matrix()
forecaster.fisher_matrix.save(fisher_dir / "y10_fiducial_clustering.fisher_matrix.npz")

# Y10 with xcorr only
forecaster = Forecaster(
    get_lbg_mappers(10),
    MainCosmology(),
    clustering=False,
    xcorr=True,
    lensing=False,
)
forecaster.create_cov()
forecaster.create_fisher_matrix()
forecaster.fisher_matrix.save(fisher_dir / "y10_fiducial_xcorr.fisher_matrix.npz")

# Y10 with clustering and xcorr
forecaster = Forecaster(
    get_lbg_mappers(10),
    MainCosmology(),
    clustering=True,
    xcorr=True,
    lensing=False,
)
forecaster.create_cov()
forecaster.create_fisher_matrix()
forecaster.fisher_matrix.save(
    fisher_dir / "y10_fiducial_clustering+xcorr.fisher_matrix.npz"
)
