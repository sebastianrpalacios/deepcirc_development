#!/bin/bash
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=40
#SBATCH --job-name=designs_circuits_single_circuit.sh
#SBATCH --output=./test_run7.sh.log

OUTPUT_FOLDER="runs/20250610_run7_test"

source /etc/profile
module load anaconda/2023a-pytorch

mkdir -p "${OUTPUT_FOLDER}"

python design_circuits.py \
    --n_envs 80 \
    --learning_rate 0.0003 \
    --max_steps 10 \
    --n_steps 20 \
    --n_epochs 4 \
    --batch_size 160 \
    --output_folder_name "${OUTPUT_FOLDER}" \
    --total_timesteps 50000 \
    --use_registry \
    --initial_state_sampling_factor 3 \
    --target_hex "0x0FD5"

