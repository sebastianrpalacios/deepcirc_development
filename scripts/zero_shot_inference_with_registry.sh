#!/bin/bash
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=0x96F7_zero_shot_with_registry_test

OUTPUT_FOLDER="runs/Fig3_zero_shot_with_registry_0x96F7_samplingfactor_0_20250903_4000nn_test_1/seed_1"
mkdir -p "${OUTPUT_FOLDER}"

exec > >(tee -a "$OUTPUT_FOLDER/slurm-${SLURM_JOB_ID}.log") 2>&1

export OPENBLAS_NUM_THREADS=1 
export OMP_NUM_THREADS=1 

source /etc/profile
module load anaconda/2023a-pytorch

python zero_shot_inference_with_registry.py \
  --model_path "/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/scripts/runs/Fig3_4input_4000_logic_functions_registry_sampling_drl3env_loader5_v2/seed_1/trained_model.zip" \
  --seed_files "/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/dgd/data/NIGs_4_inputs/0x96F7_NIG_unoptimized.pkl" \
  --output_folder_name "${OUTPUT_FOLDER}" \
  --episodes 100 \
  --initial_state_sampling_factor 0 \
  --max_steps 10

