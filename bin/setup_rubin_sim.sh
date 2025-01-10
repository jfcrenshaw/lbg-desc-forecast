#!/bin/bash

# This script sets up the RUBIN_SIM_DATA_DIR and downloads the required
# Rubin sim data. If the data already exists, it won't be downloaded again.
# By default, RUBIN_SIM_DIR=data/rubin_sim_data. If you want to set a different
# directory (e.g. because you have already downloaded the data somewhere) you can
# call this script with an argument.
# E.g., `source setup_rubin_sim.sh /path/to/rubin/sim/data`

# Export the rubin sim data directory
if [ -z "$1" ]; then
    DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
    export RUBIN_SIM_DATA_DIR="${DIR}/data/rubin_sim_data"
else
    export RUBIN_SIM_DATA_DIR=$1
fi

# Download rubin_scheduler data
# 2> >(grep -v 'already exists') pipes stderr to a subprocess that supresses
# warning messages about skipping files that already exist
echo "\nrubin_scheduler"
scheduler_download_data --dirs scheduler,utils 2> >(grep -v 'already exists')

# Download rubin_sim data
echo "\nrubin_sim"
rs_download_data --dirs maps,throughputs 2> >(grep -v 'already exists')