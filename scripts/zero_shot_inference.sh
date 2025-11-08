#!/bin/bash
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=0x13CE_inference

OUTPUT_FOLDER="runs/Fig3_zero_shot_0x13CE_test/seed_1"
mkdir -p "${OUTPUT_FOLDER}"

exec > >(tee -a "$OUTPUT_FOLDER/slurm-${SLURM_JOB_ID}.log") 2>&1

export OPENBLAS_NUM_THREADS=1 
export OMP_NUM_THREADS=1 

source /etc/profile
module load anaconda/2023a-pytorch

python zero_shot_inference.py \
  --model_path "/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/scripts/runs/Fig3_4input_100_logic_functions/seed_1/trained_model.zip" \
  --seed_files "/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/dgd/data/NIGs_4_inputs/0x13CE_NIG_unoptimized.pkl" \
  --output_folder_name "${OUTPUT_FOLDER}" \
  --episodes 100 \
  --max_steps 10


