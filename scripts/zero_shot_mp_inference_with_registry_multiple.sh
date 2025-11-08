#!/bin/bash
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --job-name=zs_mp_sf4_seed3

IDS=("0x0239" "0x040B" "0x0575" "0x0643" "0x0760" "0x09AF" "0x0F42" "0x1048"
     "0x10C9" "0x1284" "0x1714" "0x1858" "0x1AC6" "0x22C6" "0x23A7" "0x240F"
     "0x2A38" "0x2A56" "0x2FC7" "0x3060" "0x35C3" "0x3812" "0x3B68" "0x409B"
     "0x41B2" "0x429B" "0x4724" "0x48C1" "0x4A32" "0x4BF8" "0x53AF" "0x53D7"
     "0x5B30" "0x5DA9" "0x5F01" "0x616A" "0x850E" "0x8F63" "0x93AC"
     "0x9591" "0x96F7" "0x9BF5" "0xA2DA" "0xA7B2" "0xA960" "0xB744" "0xB8AD"
     "0xBC16" "0xBCA3" "0xBDF1" "0xBF36" "0xC248" "0xC4B2" "0xC766"
     "0xCBD6" "0xCE97" "0xD326" "0xDA80" "0xE605" "0xF43F" "0xF5A4")

export OPENBLAS_NUM_THREADS=1 
export OMP_NUM_THREADS=1 

source /etc/profile
module load anaconda/2023a-pytorch

for ID in "${IDS[@]}"; do
  OUTPUT_FOLDER="/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/manuscript/zero-shot/mp/5120_episodes/sf4/${ID}/seed_3"
  mkdir -p "${OUTPUT_FOLDER}"

  (
    # Per-ID logging
    exec > >(tee -a "$OUTPUT_FOLDER/slurm-${SLURM_JOB_ID}.log") 2>&1

    python zero_shot_mp_inference_with_registry.py \
      --model_path "/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/manuscript/trained_agents/GAT_MLP_with_scalars_4000_logic_functions/trained_model.zip" \
      --seed_files "/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/dgd/data/NIGs_4_inputs/${ID}_NIG_unoptimized.pkl" \
      --output_folder_name "${OUTPUT_FOLDER}" \
      --episodes 5120 \
      --initial_state_sampling_factor 4 \
      --max_steps 10 \
      --no_ablation \
      --n_envs 80
  )
done

