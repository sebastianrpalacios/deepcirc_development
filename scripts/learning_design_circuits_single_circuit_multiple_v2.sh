#!/bin/bash
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=40
#SBATCH --job-name=4in_scratchnotindata_s1

IDS=("0x000D" "0x0304" "0x057A" "0x1038" "0x1323" "0x13CE" "0x1A60" "0x1CBF"
     "0x1D95" "0x1FDE" "0x226B" "0x30CE" "0x32AA" "0x36DC" "0x3A17" "0x3B31"
     "0x3B60" "0x41A2" "0x47FD" "0x5215" "0x599A" "0x5AAD" "0x5FE2" "0x648B"
     "0x6572" "0x680A" "0x6847" "0x699D" "0x6F2A" "0x7096" "0x70EC" "0x7176"
     "0x822B" "0x914C" "0x918A" "0x9917" "0x9F8A" "0xBEE9" "0xCB82" "0xD319"
     "0xD477" "0xD4E4" "0xD550" "0xDBFA" "0xE677" "0xE93A" "0xECF1" "0xEFEB"
     "0xF4E7" "0xFC79")
     
export OPENBLAS_NUM_THREADS=1 
export OMP_NUM_THREADS=1 

source /etc/profile
module load anaconda/2023a-pytorch

for ID in "${IDS[@]}"; do
    OUTPUT_FOLDER="/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/manuscript_v2/scratch_training/4in/${ID}/seed_1"
    mkdir -p "${OUTPUT_FOLDER}"

    (
    # Per-ID logging
    exec > >(tee -a "$OUTPUT_FOLDER/slurm-${SLURM_JOB_ID}.log") 2>&1

    python design_circuits_v2.py \
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
        --target_hex "${ID}"
    )
done        

