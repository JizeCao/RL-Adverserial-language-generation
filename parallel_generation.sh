#!/bin/bash

echo 'Generate'
screen -d -m python UCT_generation.py --gpu_id 0 --RL_index 321 --begin 0 --val_begin 0 --batch_size 10000 --val_batch_size 1000 --cuda
screen -d -m python UCT_generation.py --gpu_id 5 --RL_index 321 --begin 10000 --val_begin 1000 --batch_size 10000 --val_batch_size 1000 --cuda
screen -d -m python UCT_generation.py --gpu_id 7 --RL_index 321 --begin 20000 --val_begin 2000 --batch_size 10000 --val_batch_size 1000 --cuda
