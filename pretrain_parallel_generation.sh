#!/usr/bin/env bash

for i in `seq 1 10`;
        do
                CUDA_VISIBLE_DEVICES=$i
        done