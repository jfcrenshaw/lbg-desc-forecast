"""Create and save Fisher forecasts"""

import numpy as np

from lbg_desc_forecast import Forecaster, MainCosmology, data_dir, get_lbg_mappers

# TODO: Y1 and Y10 with varying sigma8 cosmology
# TODO: think about whether running y4 forecasts would be interesting at all

# Create and save 10-year fiducial signal covariance
forecaster = Forecaster(get_lbg_mappers(10), MainCosmology())
forecaster.create_cov()
np.save(data_dir / "signal_covariance.npy", forecaster.cov)

# Year 10 fiducial forecast
fisher_dir = data_dir / "fisher_matrices"
fisher_dir.mkdir(parents=True, exist_ok=True)
forecaster.create_fisher_matrix()
forecaster.fisher_matrix.save(fisher_dir / "y10_main.fisher_matrix.npz")

"""
# Year 1 fiducial forecast
forecaster = Forecaster(get_lbg_mappers(1), MainCosmology())
forecaster.create_cov()
forecaster.create_fisher_matrix()
forecaster.fisher_matrix.save(fisher_dir / "y1_main.fisher_matrix.npz")
"""

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
forecaster.fisher_matrix.save(fisher_dir / "y10_main_clustering.fisher_matrix.npz")

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
forecaster.fisher_matrix.save(fisher_dir / "y10_main_xcorr.fisher_matrix.npz")

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
    fisher_dir / "y10_main_clustering+xcorr.fisher_matrix.npz"
)
