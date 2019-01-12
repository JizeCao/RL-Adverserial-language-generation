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
from MCTS import generate_sens_uct
import math


parser = argparse.ArgumentParser(description='UCT based on the language model')
parser.add_argument('--save_dir', type=str, default='./data/save',
                    help="directory of generative model")
parser.add_argument('--mapping', type=str, default='',
                    help="directory of the generative and discriminative mapping")
parser.add_argument('--max_seq_len', type=int, default=20,
                    help="length of the sequence")
parser.add_argument('--dis_iter', type=int, default=1,
                    help='number of dis_iter')
parser.add_argument('--gen_iter', type=int, default=10,
                    help='number of gen_iter')
parser.add_argument('--warm_words', type=str, default='I have been to somewhere',
                    help='warm-up words list')
parser.add_argument('--exp_cont', type=float, default=2 / np.sqrt(2),
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
parser.add_argument('--num_loop', type=int, default=500000,
                    help='number of whole loops processing')
parser.add_argument('--neg_data', type=str, default='stored_text_200000',
                    help="Data used to feed the generator")
parser.add_argument('--unit_test', action='store_true',
                    help="Do unit testing on test data")
parser.add_argument('--absolute_save', type=str, default='test',
                    help='the absolute save path of the models')
parser.add_argument('--reward_panalty', type=float, default=1,
                    help='the panalty on the generator reward')
#parser.add_argument('--prev_dis_model', type=str, default='best_dis_model_copy_train/',
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
parser.add_argument('--tf_ratio', type=int, default=0,
                    help="ratio of sentence level teacher forcing (not the ratio between generated/true sens")
parser.add_argument('--retrain', action='store_true', help="Retrain the RL checkpoint")
parser.add_argument('--RL_index', type=int, default=0, help="RL index")

# Discriminator retraining parameters:
parser.add_argument('--gen_lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--dis_lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=1,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--adam', action='store_true',
                    help='Adam optimizer')
parser.add_argument('--embedding', action='store_true',
                    help='use embedding to generate sentences')
parser.add_argument('--frozen_dis', action='store_true',
                    help='freeze the discriminator')
parser.add_argument('--frozen_gen', action='store_true',
                    help='freeze the generator')
parser.add_argument('--prune', action='store_true',
                    help='Initialize the UCT with m sentences')
parser.add_argument('--num_prune', default=100, type=int,
                    help='number of sentences to prune the UCT root')
parser.add_argument('--mix', action='store_false',
                    help='mix generation during training')
parser.add_argument('--beam_size', type=int, default=8,
                    help='batch size')
parser.add_argument('--replay', action='store_false',
                    help='mix generation during training')



args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args.device = device


def training_data_permutation(pos_train_sen, start_index, dis_panalty_list=None):

    pos_data = copy.deepcopy(pos_train_sen[start_index: start_index + len(gen_sen_list)])
    neg_data = gen_sen_list
    labels = np.append(np.zeros(len(pos_data)), np.ones(len(neg_data))).reshape((-1, 1))
    permute = np.random.permutation(len(pos_data) + len(neg_data))
    # add_labels(pos_data, neg_data)
    data = pos_data + neg_data
    data = [data[i] for i in permute]
    labels = labels[permute]

    # Create [pos_panalty * len(pos_data), dis_panalties]
    if dis_panalty_list is not None:
        panalty_sum = 0

        # Append 1 as the reward of the positive sentences
        panalty_list = [1 for _ in range(len(pos_data))]

        for panalty in dis_panalty_list:
            panalty_sum += panalty

        baseline = panalty_sum / len(neg_data)

        for panalty in dis_panalty_list:
            dis_panalty = panalty - baseline
            panalty_list.append(dis_panalty)

        panalty_list = [panalty_list[i] for i in permute]

        return data, labels, panalty_list

    return data, labels


# Data are in pairs
if __name__ == "__main__":

    if args.replay:
        replay_list = []
        replay_dis_list = []

    encoder, decoder, dis_model, encoder_optimizer, decoder_optimizer, dis_model_optimizer, voc, pos_train_sen, pos_valid_sen, neg_train_sen, neg_valid_sen, embedding = load_model_dictionary_pairs(args)
    args.EOS_id = voc.word2index['<EOS>']
    args.SOS_id = voc.word2index['<SOS>']
    args.vocabulary_size = len(voc.index2word)

    if args.retrain:
        rl_checkpoint = torch.load(open('./data/save/' + str(args.RL_index) + '_freq_decay_Reinforce_checkpoint_with_dis_val_loss.pt', 'rb'))
        encoder.load_state_dict(rl_checkpoint['en'])
        decoder.load_state_dict(rl_checkpoint['de'])
        dis_model.load_state_dict(rl_checkpoint['dis'])
        if args.cuda:
            encoder.cuda()
            decoder.cuda()
            dis_model.cuda()
        dis_val_loss = rl_checkpoint['dis_val_loss']
        if args.embedding:
            embedded_dis = torch.load(open('AdverSuc_checkpoint_' + str(args.RL_index) + '_.pt', 'rb'))
        else:
            embedded_dis = None
    
    else:
        dis_val_loss = 0.12
        #dis_val_loss = evaluateD(dis_model, pos_valid=pos_valid_sen[:20000], neg_valid=neg_valid_sen[:20000], EOS_token=args.EOS_id)
        logging('initial dis_val loss is {:.2f}'.format(dis_val_loss), 'new_dis_val_loss.txt') 
    


    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    dis_reward = 0
    num_dis = 0
    dis_reward_sample = 0
    num_dis_sample = 0

    ix_to_word = voc.index2word
    counter = 0

    checking_list = []
    for _ in range(args.checking_words):
        random_int = randint(0, len(pos_train_sen))
        checking_list.append(pos_train_sen[random_int])
        pos_train_sen.pop(random_int)
    pickle.dump(checking_list, open('check_sentence_list_dist', 'wb'))
    
    neg_batch_size = args.batch_size
    dis_reward_list_sample = []
    dis_reward_list = []
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # The discriminator's loss, lr
    dis_lr = args.dis_lr

    # Start iterating
    for num_loop in range(args.RL_index + 1, args.RL_index + args.num_loop + 1):
        args.batch_size = neg_batch_size
        pickle.dump(num_loop, open('num_loop_tbooks', 'wb'))
        encoder.eval()
        decoder.eval()
        dis_model.eval()

        # # Generate sampling data
        if num_loop % 40 == 1:
            if args.prune:
                sample_file_name = 'sample_prune.txt'
            else:
                sample_file_name = 'sample.txt'
            if args.frozen_dis:
                sample_file_name += '_frozen_dis'
            if args.frozen_gen:
                sample_file_name += '_frozen_gen'
            sample_file_name += '.txt'

            with open(sample_file_name, 'a') as outf:
                outf.write('| The sampling data after training ' + str(num_loop) + ' loops|')
                outf.write('')
            # 0 because no need to random initialization
            _, dis_reward_sample, num_dis_sample, useless_list, dis_panalty_list = generation(encoder, decoder, dis_model, num_loop,
                                                                                              args, checking_list, 0, dis_reward_sample, num_dis_sample,
                                                                                              ix_to_word, dis_reward_list_sample, True, sample_file_name,
                                                                                              batch_size=len(checking_list))


        # Discriminating time!

        # Getting the X, Y_hat using MCTS
        gen_sen_list, dis_reward, num_dis, num_iter_list, start_index, dis_panalty_list = generate_sens_uct(encoder, decoder, dis_model,
                                                                                                            num_loop, args, pos_train_sen,
                                                                                                            dis_reward, num_dis,
                                                                                                            ix_to_word, dis_reward_list, voc=voc)


        # Get the positive/negative data and permute it such that its randomized
        data, labels, dis_panalty = training_data_permutation(pos_train_sen, start_index, dis_panalty_list)

        if args.replay:
            replay_list += data
            replay_dis_list += dis_panalty

        print(num_iter_list)
        print("Start discriminating")
        # Previous set-up
        if torch.cuda.is_available():
            if not args.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.manual_seed_all(args.seed)
        # args.save = '{}-{}'.format(args.absolute_save + "_dis", time.strftime("%Y%m%d-%H%M%S"))
        # create_exp_dir(args.save, scripts_to_save=['automatic_search.py', 'model.py'])
        print('The start index for MCTS training is', start_index)

        # Enlarge the batch size for correcting the number of batches
        
        if not args.frozen_dis:
        
            dis_model.train()
            valid = True
            if (num_loop) % 10 != 0:
                valid = False


            for i in range(args.dis_iter):
                # Start validation at first moment
                # Use the top 20000 sentences into sentences evaluation
                if valid and i == 0:
                    train_loss, dis_lr, dis_val_loss, curr_val_loss = dis_retrain(dis_model, args=args, train_data=data, labels=labels,
                         ix_to_word=ix_to_word, validation=valid, pos_valid_pairs=pos_valid_sen[:20000], neg_valid_pairs=neg_valid_sen[:20000], current_val_loss=dis_val_loss)

                    logging('current dis_val loss is {:.2f} at iteration {}'.format(curr_val_loss, num_loop), 'new_dis_val_loss.txt') 
                else:
                    dis_retrain(dis_model, args=args, train_data=data, labels=labels,
                                ix_to_word=ix_to_word, dis_lr=dis_lr)


        ### Generating time!

        if not args.frozen_gen:

            dis_model.eval()

            # Getting the X, Y_hat using MCTS
            gen_sen_list, dis_reward, num_dis, num_iter_list, start_index, dis_panalty_list = generate_sens_uct(encoder, decoder,
                                                                                                                dis_model,
                                                                                                                num_loop, args,
                                                                                                                pos_train_sen,
                                                                                                                dis_reward, num_dis,
                                                                                                                ix_to_word,
                                                                                                                dis_reward_list,
                                                                                                                voc=voc)

            # Get the positive/negative data and permute it such that its randomized
            data, labels, dis_panalty = training_data_permutation(pos_train_sen, start_index, dis_panalty_list)

            if args.replay:
                replay_list += data
                replay_dis_list += dis_panalty

            # Retrain the generator
            print("Start retraining generator")
            
            if torch.cuda.is_available():
                if not args.cuda:
                    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
                else:
                    torch.cuda.manual_seed_all(args.seed)

            # At any point you can hit Ctrl + C to break out of training early.
            for i in range(args.gen_iter):
                gen_iter_train(voc, data, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, 50, dis_panalty,
                               batch_size=args.batch_size * 2, teacher_forcing_ratio=args.tf_ratio)

        if args.replay:
            if num_loop % args.replay_iter == 1:
                dis_model.train()
                dis_retrain(dis_model, args=args, train_data=replay_list, labels=replay_dis_list,
                            ix_to_word=ix_to_word, dis_lr=dis_lr)


        filename = 'freq_decay_Reinforce_checkpoint_with_dis_val_loss'
        if args.frozen_dis:
            filename += '_no_retrain_discriminator'
        if args.frozen_gen:
            filename += '_no_retrain_generator'
        
        if (num_loop) % args.log_interval == 0:
            torch.save({
                'iteration': num_loop,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'dis': dis_model.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'dis_opt': dis_model_optimizer.state_dict(),
                'voc_dict': voc.__dict__,
                'dis_val_loss': dis_val_loss,
                'embedding': embedding.state_dict(),
                'num_loop': num_loop,
                'dis_lr': dis_lr
            }, os.path.join(args.save_dir, '{}_{}.pt'.format(num_loop + 1, filename)))
