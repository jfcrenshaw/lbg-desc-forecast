# LBG Cosmology Forecast for LSST DESC

Installation (from the root directory):

```bash
mamba env create -f environment.yaml
mamba activate lbg-desc-forecast
python -m ipykernel install --user --name lbg-desc-forecast --display-name "LBG DESC Forecast"
pip install -e .
```

or equivalent.

Steps:

1. Download OpSim run baseline v4.0 if you don't already have it:

```bash
wget -P data/ https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_featureScheduler_runs4.0/baseline/baseline_v4.0_10yrs.db
```

1. Source `bin/setup_rubin_sim.sh`. This will set `RUBIN_SIM_DATA_DIR` and download the required data if not already present. If you already have the rubin sim data downloaded somewhere other than the default (`data/rubin_sim_data`), you can set this location by calling the setup script with the path as an argument.

2. Source `bin/setup_baseline_sim.sh`. This will set `RUBIN_SIM_RUN_DIR` and, if not already present, will download the survey simulation run we are analyzing. If you already have the rubin sim runs downloaded somewhere other than the default (`data/`), you can set this location by calling the setup script with the path as an argument.

3. Calculate 5-sigma depths by running `bin/calc_m5.py`.
