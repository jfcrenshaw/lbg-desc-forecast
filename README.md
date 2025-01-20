# LBG Cosmology Forecast for LSST DESC

Installation (from the root directory):

```bash
mamba env create -f environment.yaml
mamba activate lbg-desc-forecast
python -m ipykernel install --user --name lbg-desc-forecast --display-name "LBG DESC Forecast"
pip install -e .
```

or equivalent.

Then you must run the following one time:

1. Source `bin/setup_rubin_sim.sh`. This will set `RUBIN_SIM_DATA_DIR` and download the required data if not already present. If you already have the rubin sim data downloaded somewhere other than the default (`data/rubin_sim_data`), you can set this location by calling the setup script with the path as an argument.

2. Source `bin/setup_baseline_sim.sh`. This will set `RUBIN_SIM_RUN_DIR` and, if not already present, will download the survey simulation run we are analyzing. If you already have the rubin sim runs downloaded somewhere other than the default (`data/`), you can set this location by calling the setup script with the path as an argument.

3. Calculate 5-sigma depths by running `python bin/calc_m5.py`.

4. Generate cached values for interpolating across 5-sigma depth maps: `python bin/create_caches.py`

5. Run forecasts using the script `bin/run_forecasts.py` (note this takes > 12h on my Mac)

The notebooks in the `notebooks/` directory plot the results and can be run in any order.