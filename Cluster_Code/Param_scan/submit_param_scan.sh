#!/bin/bash

# Define the arrays for the scan (Includes 2 sites natively)
SITES_LIST=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
REGIMES=("s" "l") # 's' for small thermalisation, 'l' for large thermalisation
PARAMS=("J" "gamma" "n_avg" "M" "B_21" "Delta" "T")

# Define the resolution of the scan
N_POINTS=1000

# Create temporary directory for the parallel outputs and logs
mkdir -p Param_scan/Data
mkdir -p Param_scan/out
mkdir -p Param_scan/err

echo "Submitting unified parameter scan jobs to the cluster..."

for sites in "${SITES_LIST[@]}"; do
    for regime in "${REGIMES[@]}"; do
        for param in "${PARAMS[@]}"; do
            
            JOB_NAME="${sites}_${regime}_${param}"
            
            # Use 'sbatch' to submit the job to the cluster.
            sbatch -A ap_tqc \
                   --job-name=$JOB_NAME \
                   --output=Param_scan/out/${JOB_NAME}.out \
                   --error=Param_scan/err/${JOB_NAME}.err \
                   --nodes=1 \
                   --ntasks=1 \
                   --cpus-per-task=1 \
                   --time=3-00:00:00 \
                   --mem=4G \
                   Param_scan/param_scan_worker.sh $sites $regime $param $N_POINTS
                   
        done
    done
done

echo "All calculations have been submitted to the Slurm queue!"