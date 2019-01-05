import torch
import numpy as np
import argparse
import pickle
from random import randint
import torch.nn as nn
import sys
import time
import gc
import copy
import os
from search_utils import create_exp_dir, dis_retrain, Voc, trim_dummy_sen, load_model_dictionary_pairs, dis_evaluate_sen, logging, evaluateD
from gen_utils import gen_iter_train
from collections import Counter
from model import EncoderRNN, LuongAttnDecoderRNN, hierEncoder
from MCTS_generation import generation
import math


def generate_sens_uct(encoder, decoder, dis_model, num_loop, args, pos_train_sen, dis_reward, num_dis,
                      ix_to_word, dis_reward_list):
    # Randomizing the initial word
    start_index = np.random.randint(0, len(pos_train_sen) - args.batch_size)
    print('Generate sentences start at', start_index)
    # First, initialize MCTS search
    print("Start MCTS search")

    gen_sen_list, dis_reward, num_dis, num_iter_list, dis_panalty_list = generation(encoder, decoder, dis_model, num_loop, args,
                                                                  pos_train_sen, start_index, dis_reward, num_dis,
                                                                  ix_to_word, dis_reward_list)

    return gen_sen_list, dis_reward, num_dis, num_iter_list, start_index, dis_panalty_list


