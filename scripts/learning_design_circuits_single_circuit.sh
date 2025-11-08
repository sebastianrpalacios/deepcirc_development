#!/bin/bash
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --job-name=0x822B_seed1

OUTPUT_FOLDER="/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/manuscript/scratch_training/4in/0x822B/seed_1"
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
    --total_timesteps 50000 \
    --use_registry \
    --store_every_new_graph \
    --registry_sampling \
    --initial_state_sampling_factor 0 \
    --target_hex "0x822B" \

    