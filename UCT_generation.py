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
from search_utils import create_exp_dir, dis_retrain, Voc, trim_dummy_sen, load_model_dictionary_pairs, dis_evaluate_sen
from gen_utils import gen_iter_train
from collections import Counter
from model import EncoderRNN, LuongAttnDecoderRNN, hierEncoder
from MCTS_generation import generation

import math

parser = argparse.ArgumentParser(description='UCT based on the language model')
parser.add_argument('--save_dir', type=str, default='../data/save',
                    help="directory of generative model")
parser.add_argument('--mapping', type=str, default='',
                    help="directory of the generative and discriminative mapping")
parser.add_argument('--max_seq_len', type=int, default=20,
                    help="length of the sequence")
parser.add_argument('--dis_iter', type=int, default=5,
                    help='number of dis_iter')
parser.add_argument('--gen_iter', type=int, default=10,
                    help='number of gen_iter')
parser.add_argument('--warm_words', type=str, default='I have been to somewhere',
                    help='warm-up words list')
parser.add_argument('--exp_cont', type=float, default=3 / np.sqrt(2),
                    help='warm-up words list')
parser.add_argument('--num_iter', type=int, default=10000000,
                    help='number of iterations done for one MCTS search')
parser.add_argument('--cuda', action='store_true',
                    help='use cuda')
parser.add_argument('--words', type=int, default=-1,
                    help='total number of words generated')
parser.add_argument('--log_interval', type=int, default=20,
                    help='the interval for each log')
parser.add_argument('--outf', type=str, default="test1_generated_tf.txt",
                    help='the directory of the output file')
parser.add_argument('--unk_id', type=int, default=-1,
                    help='unk id')
parser.add_argument('--threshold', type=int, default=15,
                    help='total number of output words')
parser.add_argument('--automatic', action='store_false',
                    help='automatic discrimination')
parser.add_argument('--num_loop', type=int, default=500,
                    help='number of whole loops processing')
parser.add_argument('--neg_data', type=str, default='stored_text_200000',
                    help="Data used to feed the generator")
parser.add_argument('--unit_test', action='store_true',
                    help="Do unit testing on test data")
parser.add_argument('--absolute_save', type=str, default='test',
                    help='the absolute save path of the models')
parser.add_argument('--unk_panalty', type=float, default=0.15,
                    help='the panalty on unk')
parser.add_argument('--reward_panalty', type=float, default=0.4,
                    help='the panalty on the generator reward')
# parser.add_argument('--prev_dis_model', type=str, default='best_dis_model_copy_train/',
#                    help='The dis_model used from retraining the generator')
parser.add_argument('--partition', type=int, default=0,
                    help='partition of the headers')
parser.add_argument('--generation', action='store_true',
                    help='Only generate sentences')
parser.add_argument('--print_log', action='store_false',
                    help='Print the log of generator for every batch')
parser.add_argument('--gpu_id', type=int, default=3,
                    help='The using gpu')
parser.add_argument('--checking_words', type=int, default=5,
                    help='The number of headers used to check the performance')
parser.add_argument('--iter_decay', action='store_false',
                    help='Start iteration decays')
parser.add_argument('--SOS_id', type=int, default=0,
                    help="the index of the start of the sentence")
parser.add_argument('--EOS_id', type=int, default=25001,
                    help="the index of the end of the sentence")
parser.add_argument('--RL_index', type=int, default=60,
                    help="the index of the RL network")
parser.add_argument('--batch_size', type=int, default=20000, metavar='N',
                    help='number of generated sentences')
parser.add_argument('--val_batch_size', type=int, default=2000, metavar='N',
                    help='number of generated valid sentences')

# Discriminator retraining parameters:
parser.add_argument('--epochs', type=int, default=1,
                    help='upper epoch limit')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')

args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args.device = device
if device == 'cuda:0':
    args.cuda = True



# def logging(s, print_=True, log_=True):
#     if print_:
#         print(s)
#     if log_:
#         with open(os.path.join(args.save, 'log.txt'), 'a+') as f_log:
#             f_log.write(s + '\n')


def generate_randint_list(args, sentence_list_len):
    np.random.seed(args.seed)
    return [np.random.randint(0, sentence_list_len) for _ in range(args.checking_words)]


# Data are in pairs
if __name__ == "__main__":

    hidden_size = 512
    encoder_n_layers = 3
    decoder_n_layers = 3
    dropout = 0.2
    attn_model = 'dot'

    voc = pickle.load(open('../data/save/whole_data_voc.p', 'rb'))
    if args.cuda:
        checkpoint = torch.load(open('../data/save/' + str(args.RL_index) + '_Reinforce_checkpoint.pt', 'rb'))
    else:
        checkpoint = torch.load(open('../data/save/' + str(args.RL_index) + '_Reinforce_checkpoint.pt', 'rb'), map_location=lambda storage, loc: storage)
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    embedding_sd = checkpoint['embedding']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    voc.__dict__ = checkpoint['voc_dict']
    pos_train_sen = pickle.load(open('../data/save/small_train_2000000.p', 'rb'))[:args.batch_size]
    pos_valid_sen = pickle.load(open('../data/save/small_valid_2000000.p', 'rb'))[:args.val_batch_size]


    embedding = nn.Embedding(voc.num_words, hidden_size)
    embedding.load_state_dict(embedding_sd)
    embedding.to(args.device)
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    encoder.to(args.device)
    decoder.to(args.device)
    dis_model = hierEncoder(len(voc.index2word), 500)
    dis_model.load_state_dict(checkpoint['dis'])
    dis_model.to(args.device)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    dis_reward = 0
    num_dis = 0
    dis_reward_sample = 0
    num_dis_sample = 0

    ix_to_word = voc.index2word
    counter = 0

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    encoder.eval()
    decoder.eval()
    dis_model.eval()
    # First, initialize MCTS search
    print("Start MCTS search")
    if args.threshold == -1:
        args.words = args.seq_len * len(pos_train_sen)
    else:
        args.words = args.threshold
    print("Start generating sentences")
    # TODO: Use the num_iter_list for generator's training
    dis_reward_list = []
    gen_sen_list, dis_reward, num_dis, num_iter_list = generation(encoder, decoder, dis_model, 0, args,
                                                                  pos_train_sen, 0, dis_reward, num_dis,
                                                                  ix_to_word, dis_reward_list)

    args.batch_size = args.val_batch_size
    gen_sen_list_val, dis_reward, num_dis, num_iter_list_val = generation(encoder, decoder, dis_model, 0, args,
                                                                  pos_valid_sen, 0, dis_reward, num_dis,
                                                                  ix_to_word, dis_reward_list)
    pickle.dump(gen_sen_list, open('Generated_sentences_' + str(args.RL_index), 'wb'))
    pickle.dump(gen_sen_list_val, open('Generated_sentences_valid_' + str(args.RL_index), 'wb'))




