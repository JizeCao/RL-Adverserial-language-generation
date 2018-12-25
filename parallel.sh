#!/bin/bash

for _ in $(seq 10); do
    screen -d -m python automatic_search.py --gpu_id 4 --dis_model best_dis_model_copy/ --threshold 100 --num_iter 1500 --seq_len 20 --unk_panalty 0.15 --reward_panalty 0.7 --generation --gen_model best_gen_model_copy/
    screen -d -m python automatic_search.py --gpu_id 5 --dis_model best_dis_model_copy/ --threshold 100 --num_iter 1500 --seq_len 20 --unk_panalty 0.15 --reward_panalty 0.7 --generation --partition 5000 --gen_model best_gen_model/
    screen -d -m python automatic_search.py --gpu_id 6 --dis_model best_dis_model_copy/ --threshold 100 --num_iter 1500 --seq_len 20 --unk_panalty 0.15 --reward_panalty 0.7 --generation --partition 10000 --gen_model best_gen_model_copy/
    echo start generation
    python concatenate.py --partition 5000
    CUDA_VISIBLE_DEVICES=6 python automatic_search.py --unit_test  --dis_model best_dis_model_copy/ --seq_len 20 --gen_model best_gen_model_copy/ --neg_data stored_text_300 --prev_dis_model best_dis_model_copy/ --epochs 2 
done

