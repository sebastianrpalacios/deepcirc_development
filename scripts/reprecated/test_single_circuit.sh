#!/bin/bash
#SBATCH --output=runs/20250608_0x0FD5_2_search_designs_singlecircuit_yes_shared_solutions_tb_argparse/slurm_output.log
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=40
#SBATCH --job-name=search_designs_multicircuit_no_shared_solutions.sh

OUTPUT_FOLDER="runs/20250609_run1_0x0FD5_search_designs_singlecircuit_yes_shared_solutions_tb_argparse"

source /etc/profile
module load anaconda/2023a-pytorch

mkdir -p "${OUTPUT_FOLDER}"

python search_designs_singlecircuit_yes_shared_solutions_tb_argparse.py \
    --n 1 \
    --n_envs 80 \
    --learning_rate 0.0003 \
    --max_steps 10 \
    --n_steps 20 \
    --n_epochs 4 \
    --batch_size 160 \
    --folder_name "${OUTPUT_FOLDER}" \
    --total_timesteps 100000 \
    --target_hex 0x0FD5

