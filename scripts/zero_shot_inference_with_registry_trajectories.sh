#!/bin/bash
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=0x1048_zero_shot_with_registry_test

OUTPUT_FOLDER="/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/manuscript/zero-shot/0x1048_test/seed_1"
mkdir -p "${OUTPUT_FOLDER}"

exec > >(tee -a "$OUTPUT_FOLDER/slurm-${SLURM_JOB_ID}.log") 2>&1

export OPENBLAS_NUM_THREADS=1 
export OMP_NUM_THREADS=1 

source /etc/profile
module load anaconda/2023a-pytorch

python zero_shot_inference_with_registry_trajectories.py \
  --model_path "/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/manuscript/trained_agents/GAT_without_scalars_4000_logic_functions/trained_model.zip" \
  --seed_files "/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/dgd/data/NIGs_4_inputs/0x1048_NIG_unoptimized.pkl" \
  --output_folder_name "${OUTPUT_FOLDER}" \
  --episodes 50 \
  --initial_state_sampling_factor 3 \
  --max_steps 10


