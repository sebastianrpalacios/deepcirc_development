#!/bin/bash

source /etc/profile

module load anaconda/2023a-pytorch

python search_designs_multicircuit_no_shared_solutions_tb.py
