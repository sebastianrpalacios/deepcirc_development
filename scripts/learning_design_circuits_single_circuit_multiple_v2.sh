#!/bin/bash
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=40
#SBATCH --job-name=3in_scrat_s1

IDS=("0x03" "0x06" "0x09" "0x0A" "0x13" "0x18" "0x21" "0x23" "0x24" "0x2A"
 "0x2C" "0x2F" "0x35" "0x38" "0x3A" "0x3B" "0x42" "0x4C" "0x52" "0x55"
 "0x56" "0x58" "0x60" "0x61" "0x63" "0x68" "0x6D" "0x6F" "0x76" "0x83"
"0x85" "0x8B" "0x8C" "0x90" "0x91" "0x99" "0x9E" "0xA2" "0xA4" "0xA5"
 "0xA6" "0xAB" "0xB6" "0xB7" "0xBA" "0xBC" "0xC2" "0xC4" "0xC7" "0xD0"
     "0xD2" "0xDA" "0xDC" "0xE0" "0xE6" "0xEF" "0xF0" "0xF1" "0xFD")

   
export OPENBLAS_NUM_THREADS=1 
export OMP_NUM_THREADS=1 

source /etc/profile
module load anaconda/2023a-pytorch

for ID in "${IDS[@]}"; do
    OUTPUT_FOLDER="/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/manuscript_v2/scratch_training/3in/${ID}/seed_1"
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


