#!/bin/bash

# Define the number of steps
NUM_STEPS=100
FILE_NAME="sites_15_det_long"

PARAMS="sites = 15,
        J = 1e0, 
        γ = 1.0e-3, 
        noise_strength = 0.0, 
        n_avg = 1.0e4, 
        M = 1.0e8, 
        B_21 = 1.0e-7, 
        Δ = -60, 
        T = 25.0, 
        saveDuration_factor = 1e8,
        t_begin_factor = 1e7,
        δ = 0.0,
        N = $NUM_STEPS"

echo "Clearing previous simulation data..."
rm -rf Sweep/Data/$FILE_NAME
rm -rf Sweep/logs/out/$FILE_NAME
rm -rf Sweep/logs/err/$FILE_NAME

# Create directories for the output data and logs
mkdir -p Sweep/Data/$FILE_NAME
mkdir -p Sweep/logs/out/$FILE_NAME
mkdir -p Sweep/logs/err/$FILE_NAME

echo "Submitting unified sweep jobs to the cluster..."

for task_id in $(seq 1 $NUM_STEPS); do
    
    JOB_NAME="${FILE_NAME}_step_${task_id}"
    
    sbatch -A ap_tqc \
           --job-name=$JOB_NAME \
           --output=Sweep/logs/out/${FILE_NAME}/step_${task_id}.out \
           --error=Sweep/logs/err/${FILE_NAME}/step_${task_id}.err \
           --nodes=1 \
           --ntasks=1 \
           --cpus-per-task=1 \
           --time=3-00:00:00 \
           --mem=8G \
           Sweep/sweep_worker.sh $task_id "$FILE_NAME" "$PARAMS"
           
done

echo "All $NUM_STEPS calculations submitted to Slurm!"