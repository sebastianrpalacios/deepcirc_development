#!/bin/bash
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --job-name=finetune_redo_s3

IDS=("0x1048" "0xD326")

export OPENBLAS_NUM_THREADS=1 
export OMP_NUM_THREADS=1 

source /etc/profile
module load anaconda/2023a-pytorch

for ID in "${IDS[@]}"; do
    OUTPUT_FOLDER="/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/manuscript/fine_tune/GAT_MLP_with_scalars/4000nn/initial_state_sampling_factor_0/4_inputs/50_000_steps/${ID}/seed_3"
    mkdir -p "${OUTPUT_FOLDER}"

    (
    exec > >(tee -a "$OUTPUT_FOLDER/slurm-${SLURM_JOB_ID}.log") 2>&1
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
        --target_hex "${ID}" \
        --fine_tuning "/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/manuscript/trained_agents/GAT_MLP_with_scalars_4000_logic_functions/trained_model.zip"
    )
done  
