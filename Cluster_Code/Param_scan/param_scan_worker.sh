#!/bin/bash

# Load environment
module purge
module load calcua/all
module load Julia/1.12.4

export JULIA_DEPOT_PATH="/data/antwerpen/210/vsc21017/jul_inst"

echo "Running Julia Worker on Node: $SLURMD_NODENAME"
echo "Parameters: Sites=$1 | Regime=$2 | Param=$3 | Points=$4"

# Execute the Julia script with the passed arguments
julia --project=@. Param_scan/param_scan_worker.jl "$1" "$2" "$3" "$4"