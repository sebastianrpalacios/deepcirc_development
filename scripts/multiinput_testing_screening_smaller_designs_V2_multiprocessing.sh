#!/bin/bash
#SBATCH --partition=xeon-p8
#SBATCH --cpus-per-task=30
#SBATCH --job-name=multiinput_testing_screening_smaller_designs_V2_multiprocessing.sh

OUTPUT_FOLDER="runs/multiinput_testing_screening_smaller_designs_V2_multiprocessing"
mkdir -p "${OUTPUT_FOLDER}"

exec > >(tee -a "$OUTPUT_FOLDER/slurm-${SLURM_JOB_ID}.log") 2>&1

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1 

source /etc/profile
module load anaconda/2023a-pytorch

python multiinput_testing_screening_smaller_designs_V2_multiprocessing.py \


