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
from generate import evaluate
from tqdm import tqdm


def beam_generation(pairs, encoder, decoder, voc, args):
    generated_pairs = []
    counter = 0
    print('Start beam generation, generating {} pairs'.format(str(len(pairs))))
    for i in tqdm(range(len(pairs))):

        counter += 1

        temp = [pairs[i][0]]
        ai_response = evaluate(encoder, decoder, voc, sentence=pairs[i][0], beam=args.beam_size)
        temp.append(ai_response)

        generated_pairs.append(temp)
        if args.print:
            if counter % 50 == 0:
                print('Now generated {} sentence pairs, remaining {} sentence pairs'.format
                    (str(counter), str(len(pairs) - counter)))

    return generated_pairs


# If mix, half data generated by beam, the other half is uct
def generate_sens_uct(encoder, decoder, dis_model, num_loop, args, pos_train_sen, dis_reward, num_dis,
                      ix_to_word, dis_reward_list, voc=None, ensemble=None):

    # Randomizing the initial word
    start_index = np.random.randint(0, len(pos_train_sen) - args.batch_size)
    print('Generate sentences start at', start_index)
    # First, initialize MCTS search
    print("Start MCTS search")

    if args.mix:
        args.batch_size = args.batch_size // 2

    gen_sen_list, dis_reward, num_dis, num_iter_list, dis_panalty_list = generation(encoder, decoder, dis_model, num_loop, args,
                                                                  pos_train_sen, start_index, dis_reward, num_dis,
                                                                  ix_to_word, dis_reward_list, ensemble=ensemble)

    # Involve beam search in the retraining
    if args.mix:
        beam_sen_list = beam_generation(pos_train_sen[start_index + args.batch_size: start_index + args.batch_size * 2],
                                        encoder, decoder, voc, args)
        gen_sen_list += beam_sen_list

        if not args.frozen_gen:
            beam_panalty_list = []
            # TODO: Temporary solution, use batch-wise later
            for pair in beam_sen_list:
                reward = dis_evaluate_sen(pair, dis_model, args)
                beam_panalty_list.append(reward)

            dis_panalty_list += beam_panalty_list

        args.batch_size = args.batch_size * 2

    return gen_sen_list, dis_reward, num_dis, num_iter_list, start_index, dis_panalty_list


