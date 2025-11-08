#!/bin/bash
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=zero_shot_0x53D7_trajectories

IDS=("0x53D7")

export OPENBLAS_NUM_THREADS=1 
export OMP_NUM_THREADS=1 

source /etc/profile
module load anaconda/2023a-pytorch

for ID in "${IDS[@]}"; do
  OUTPUT_FOLDER="/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/manuscript/zero-shot/one_core_with_trajectories/1000_episodes/4in/sf0/${ID}/seed_1"
  mkdir -p "${OUTPUT_FOLDER}"

  (
    # Per-ID logging
    exec > >(tee -a "$OUTPUT_FOLDER/slurm-${SLURM_JOB_ID}.log") 2>&1

    python zero_shot_inference_with_registry_trajectories.py \
      --model_path "/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/manuscript/trained_agents/GAT_MLP_with_scalars_4000_logic_functions/trained_model.zip" \
      --seed_files "/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/dgd/data/NIGs_4_inputs/${ID}_NIG_unoptimized.pkl" \
      --output_folder_name "${OUTPUT_FOLDER}" \
      --episodes 1000 \
      --initial_state_sampling_factor 0 \
      --max_steps 10
  )
done







