#!/bin/bash
#SBATCH --partition=xeon-p8
#SBATCH --cpus-per-task=20
#SBATCH --job-name=SA_1000
#SBATCH --array=0-60%10   # 61 IDs -> indices 0..60; run up to 7 at once

IDS=("0x1714")

# Safety check in case array range doesn't match IDS length
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#IDS[@]} )); then
  echo "Array index ${SLURM_ARRAY_TASK_ID} out of range"; exit 1
fi

ID="${IDS[$SLURM_ARRAY_TASK_ID]}"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

source /etc/profile
module load anaconda/2023a-pytorch

OUTPUT_FOLDER="/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/manuscript/logic_synthesis_SA/11400_steps/4_inputs/fw_preference/${ID}/seed_1"
mkdir -p "${OUTPUT_FOLDER}"

# Per-task logging
exec > >(tee -a "$OUTPUT_FOLDER/slurm-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log") 2>&1

python logic_synthesis_SA.py \
  --seed_files "/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/dgd/data/NIGs_4_inputs/${ID}_NIG_unoptimized.pkl" \
  --output_folder_name "${OUTPUT_FOLDER}" \
  --initial_state_sampling_factor 0 \
  --log_every_steps 10 \
  --steps 11400
