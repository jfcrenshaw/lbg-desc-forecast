"""Run Fisher forecasts for the main cosmology and 0% contamination"""

from lbg_desc_forecast import Forecaster, MainCosmology, get_lbg_mappers, fisher_dir

# Directory in which to save results
fisher_dir.mkdir(parents=True, exist_ok=True)

# Year 1 forecast
forecaster = Forecaster(get_lbg_mappers(1, contamination=0), MainCosmology())
forecaster.create_cov()
forecaster.create_fisher_matrix()
forecaster.fisher_matrix.save(fisher_dir / "y1_clean.fisher_matrix.npz")

# Year 10 forecast
forecaster = Forecaster(get_lbg_mappers(10, contamination=0), MainCosmology())
forecaster.create_cov()
forecaster.create_fisher_matrix()
forecaster.fisher_matrix.save(fisher_dir / "y10_clean.fisher_matrix.npz")

# Y10 with clustering only
forecaster = Forecaster(
    get_lbg_mappers(10, contamination=0),
    MainCosmology(),
    clustering=True,
    xcorr=False,
    lensing=False,
)
forecaster.create_cov()
forecaster.create_fisher_matrix()
forecaster.fisher_matrix.save(fisher_dir / "y10_clean_clustering.fisher_matrix.npz")

# Y10 with xcorr only
forecaster = Forecaster(
    get_lbg_mappers(10, contamination=0),
    MainCosmology(),
    clustering=False,
    xcorr=True,
    lensing=False,
)
forecaster.create_cov()
forecaster.create_fisher_matrix()
forecaster.fisher_matrix.save(fisher_dir / "y10_clean_xcorr.fisher_matrix.npz")

# Y10 with clustering and xcorr
forecaster = Forecaster(
    get_lbg_mappers(10, contamination=0),
    MainCosmology(),
    clustering=True,
    xcorr=True,
    lensing=False,
)
forecaster.create_cov()
forecaster.create_fisher_matrix()
forecaster.fisher_matrix.save(
    fisher_dir / "y10_clean_clustering+xcorr.fisher_matrix.npz"
)
