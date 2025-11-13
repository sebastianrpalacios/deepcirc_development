#!/bin/bash
#SBATCH --partition=xeon-p8
#SBATCH --cpus-per-task=20
#SBATCH --job-name=exp_scratch_reg_analysis

OUTPUT_FOLDER="/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/manuscript/shared_registry_analysis"
mkdir -p "${OUTPUT_FOLDER}"

exec > >(tee -a "$OUTPUT_FOLDER/slurm-${SLURM_JOB_ID}.log") 2>&1

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1 
""
source /etc/profile
module load anaconda/2023a-pytorch

python "Shared registry ploting and analysis v2.py"

