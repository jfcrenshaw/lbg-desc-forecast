#!/bin/bash

# This script downloads the runs we are analyzing. If the data already
# exists, it won't be downloaded again. By default,
# RUBIN_SIM_RUNS_DIR=data/rubin_sim_runs
# If you want to set a different directory (e.g. because you have already
# downloaded the data somewhere) you can call this script with an argument.
# E.g., `source download_runs_3p4.sh /path/to/rubin/sim/runs`

# Export path to the baseline v4.0 sim
if [ -z "$1" ]; then
    DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
    export RUBIN_SIM_RUNS_DIR="${DIR}/data/"
else
    export RUBIN_SIM_RUNS_DIR=$1
fi

# URL to the baseline sim
URL="https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_featureScheduler_runs4.0/baseline/baseline_v4.0_10yrs.db"

# Download the file if it doesn't already exist
if ! [ -f "${RUBIN_SIM_RUNS_DIR}/baseline_v4.0_10yrs.db" ]; then
    dir=$(dirname "${run}")
    wget -P "${RUBIN_SIM_RUNS_DIR}/" "${URL}"
fi