#!/bin/bash

# Load environment
module purge
module load calcua/all
module load Julia/1.12.4

# Keep your custom package installation directory
export JULIA_DEPOT_PATH="/data/antwerpen/210/vsc21017/jul_inst"

TASK_ID=$1
FILE_NAME=$2
PARAMS=$3

echo "Running Julia Worker on Node: $SLURMD_NODENAME"
echo "Executing Sweep Task ID: $TASK_ID for $FILE_NAME"

julia --project=@. Sweep/sweep_worker.jl "$TASK_ID" "$FILE_NAME" "$PARAMS"