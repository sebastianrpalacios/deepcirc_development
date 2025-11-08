#!/bin/bash
#SBATCH --output=./results_compare_check_implicit_OR_existence_v2_and_v3.log
#SBATCH --cpus-per-task=1
#SBATCH --partition=xeon-p8
#SBATCH --job-name=compare_check_implicit_OR_existence_v2_and_v3.sh

source /etc/profile
module load anaconda/2023a-pytorch

python compare_check_implicit_OR_existence_v2_and_v3.py \
    --directory /home/gridsan/spalacios/DRL1/supercloud-testing/ABC-and-PPO-testing1/Verilog_files_for_all_4_input_1_output_truth_tables_as_NIGs \
    --num 100 \
    --inputs 2