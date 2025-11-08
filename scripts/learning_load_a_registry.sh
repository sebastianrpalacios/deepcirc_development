#!/bin/bash
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=40
#SBATCH --job-name=load_registry_test.sh

OUTPUT_FOLDER="runs/20250622_load_registry_test_2"
mkdir -p "${OUTPUT_FOLDER}"

exec > >(tee -a "$OUTPUT_FOLDER/slurm-${SLURM_JOB_ID}.log") 2>&1

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1 

source /etc/profile
module load anaconda/2023a-pytorch

python design_circuits.py \
    --n_envs 80 \
    --learning_rate 0.0003 \
    --max_steps 10 \
    --n_steps 20 \
    --n_epochs 4 \
    --batch_size 160 \
    --output_folder_name "${OUTPUT_FOLDER}" \
    --total_timesteps 100000 \
    --use_registry \
    --registry_read_only \
    --registry_sampling \
    --initial_state_sampling_factor 0 \
    --load_registry "/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/scripts/runs/20250612_run1_100logicfunctions_3input/shared_registry_400_000.pkl" \
    --global_seed 42

