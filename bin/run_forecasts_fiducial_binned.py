"""Run binned Fisher forecasts for the main cosmology and 10% contamination

This runs for each tomographic bin one-at-a-time
"""

from lbg_desc_forecast import Forecaster, MainCosmology, get_lbg_mappers, fisher_dir

# Directory in which to save results
fisher_dir.mkdir(parents=True, exist_ok=True)

# Year 1 forecasts
u_mapper, g_mapper, r_mapper = get_lbg_mappers(1)

forecaster = Forecaster([u_mapper], MainCosmology())
forecaster.create_cov()
forecaster.create_fisher_matrix()
forecaster.fisher_matrix.save(fisher_dir / "y1_fiducial_ubin.fisher_matrix.npz")

forecaster = Forecaster([g_mapper], MainCosmology())
forecaster.create_cov()
forecaster.create_fisher_matrix()
forecaster.fisher_matrix.save(fisher_dir / "y1_fiducial_gbin.fisher_matrix.npz")

forecaster = Forecaster([r_mapper], MainCosmology())
forecaster.create_cov()
forecaster.create_fisher_matrix()
forecaster.fisher_matrix.save(fisher_dir / "y1_fiducial_rbin.fisher_matrix.npz")

# Year 10 forecasts
u_mapper, g_mapper, r_mapper = get_lbg_mappers(10)

forecaster = Forecaster([u_mapper], MainCosmology())
forecaster.create_cov()
forecaster.create_fisher_matrix()
forecaster.fisher_matrix.save(fisher_dir / "y10_fiducial_ubin.fisher_matrix.npz")

forecaster = Forecaster([g_mapper], MainCosmology())
forecaster.create_cov()
forecaster.create_fisher_matrix()
forecaster.fisher_matrix.save(fisher_dir / "y10_fiducial_gbin.fisher_matrix.npz")

forecaster = Forecaster([r_mapper], MainCosmology())
forecaster.create_cov()
forecaster.create_fisher_matrix()
forecaster.fisher_matrix.save(fisher_dir / "y10_fiducial_rbin.fisher_matrix.npz")
