#!/bin/bash
#SBATCH --partition=xeon-p8
#SBATCH --cpus-per-task=20
#SBATCH --job-name=reg_zs_sf4

OUTPUT_FOLDER="runs/Shared registry ploting and analysis zero shot"
mkdir -p "${OUTPUT_FOLDER}"

exec > >(tee -a "$OUTPUT_FOLDER/slurm-${SLURM_JOB_ID}.log") 2>&1

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1 

source /etc/profile
module load anaconda/2023a-pytorch

python "Shared registry ploting and analysis zero shot.py" --folder "trained_masked"


