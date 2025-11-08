#!/bin/bash

source /etc/profile

module load anaconda/2023a-pytorch

python search_designs_multicircuit_no_shared_solutions_tb_argparse.py \
    --n 100 \
    --n_envs 80 \
    --learning_rate 0.0003 \
    --max_steps 10 \
    --n_steps 20 \
    --n_epochs 4 \
    --batch_size 160 \
    --folder_name runs/20250602_1_multiinput_not_pretrained_no_registry_env4_10_boolean_functions \
    --total_timesteps 100000

