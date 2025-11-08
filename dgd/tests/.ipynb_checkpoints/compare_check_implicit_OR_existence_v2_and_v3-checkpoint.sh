#!/bin/bash
#SBATCH --output=./slurm_output.log
#SBATCH --cpus-per-task=1
#SBATCH --job-name=Update check_implicit_OR_existence_v2.sh

source /etc/profile
module load anaconda/2023a-pytorch

python compare_or_functions.py \
    --directory /home/gridsan/spalacios/DRL1/supercloud-testing/ABC-and-PPO-testing1/Verilog_files_for_all_4_input_1_output_truth_tables_as_NIGs \
    --num 500 \
    --inputs 2