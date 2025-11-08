#!/bin/bash
#SBATCH --job-name=multiinput_drl
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=40
#SBATCH --job-name=search_designs_multicircuit_no_shared_solutions.sh
#SBATCH --output=runs/20250608_1_search_1000_designs_multicircuit_no_shared_solutions_tb_argparse/slurm_output.log

OUTPUT_FOLDER="runs/20250608_1_search_1000_designs_multicircuit_no_shared_solutions_tb_argparse"

source /etc/profile
module load anaconda/2023a-pytorch

mkdir -p "${OUTPUT_FOLDER}"

python search_designs_multicircuit_no_shared_solutions_tb_argparse.py \
    --n 1000 \
    --n_envs 80 \
    --learning_rate 0.0003 \
    --max_steps 10 \
    --n_steps 20 \
    --n_epochs 4 \
    --batch_size 160 \
    --folder_name "${OUTPUT_FOLDER}" \
    --total_timesteps 200000

