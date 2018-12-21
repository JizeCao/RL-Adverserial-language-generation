import numpy as np
import argparse
import pickle
import torch
import torch.nn as nn
import sys
import time
import gc
import copy
import os
from search_utils import evaluate_sen, create_exp_dir, evaluate_word, tensorFromPair
from collections import Counter
import math
from UCT_search import UCTSearch



def warm_up(encoder, decoder, source, ix_to_word, device, SOS_inde=1):
    with torch.no_grad():
        source = torch.Tensor(source).long()
        # source is the source sentence
        encoder_output, init_reward, decoder_hidden = evaluate_sen(encoder, decoder, source, device)
    # Get the dimension of the output
    return encoder_output, torch.exp(init_reward.view(-1)), decoder_hidden, source



def logging(s, print_=True, log_=True, args=None):
    if print_:
        print(s)
    if log_:
        with open(os.path.join(args.save, 'log.txt'), 'a+') as f_log:
            f_log.write(s + '\n')


# Generate the reward for a specific tensor
def generate_reward(decoder, encoder_output, decoder_hidden, input_word):
    return evaluate_word(decoder, encoder_output, decoder_hidden, input_word)


def generation(encoder, decoder, dis_model, num_loop, args, warm_up_words_list, start_index, dis_reward, num_dis, ix_to_word, dis_reward_list, evaluation=False, output_file=None, batch_size=None):
    if batch_size is None:
        batch_size = args.batch_size
    if output_file is None:
        output_file = args.outf
    if evaluation:
        writing_statu = 'a'
    else:
        writing_statu = 'w'
    sen_iter_list = []
    with open(output_file, writing_statu) as outf:
        count = 0
        start_time = time.time()
        gen_sen_pairs = []
        # Warmup the generator
        for i in range(batch_size):
            if count >= batch_size:
                break
            source = warm_up_words_list[start_index + i][0]
            encoder_output, init_reward, hidden, source = warm_up(encoder, decoder, source, ix_to_word, args.device)
            # Want a deep copy for the hidden state
            #current_hidden = copy.deepcopy(hidden)
            # Do UCT search
                # Words_level mcts
                # for i in range(args.seq_len - len(source)):
                #     count += 1
                #     # Use caches to accelerate
                #     gen_cache = {}
                #     dis_cache = {}
                #     # Sentence level MCTS
                #     word = UCTSearch(init_reward, action_space=len(ix_to_word), gen_model=decoder,
                #               dis_model=dis_model, init_hidden=hidden, warm_up_word=source,
                #               gen_cache=gen_cache, dis_cache=dis_cache, args=args)
                #     source.append(word)
                #     # Get the reward for the next word's generation
                #     encoder_output, init_reward, current_hidden = generate_reward(decoder, word, current_hidden)
                #     hidden = copy.deepcopy(current_hidden)
                #     outf.write(ix_to_word[word] + ('\n' if i == args.seq_len - len(source) - 1 else ' '))
                #     if count % args.log_interval == 0:
                #         #sys.exit()
                #         search_time = time.time() - start_time
                #         print('| Generated {}/{} words | ave_time {:5.2f}s'.format(count, args.words, (time.time() - start_time) / count))

                # Sentence level MCTS
            count += 1
            # Use caches to accelerate
            gen_cache = {}
            dis_cache = {}
            # Sentence level MCTS
            gen_pair, dis_reward, num_dis, num_iter_sen= UCTSearch(init_reward, action_space=len(ix_to_word), gen_model=decoder,
                                                                   dis_model=dis_model, init_hidden=hidden, source=source,
                                                                   gen_cache = gen_cache, dis_cache=dis_cache, sentence=True, num_dis=num_dis, dis_reward = dis_reward, encoder_output=encoder_output, args=args,
                                                                   ix_to_word=ix_to_word)
            for i in range(len(gen_pair)):
                if i == 0:
                    print("Source:", end=' ')
                else:
                    print("Answer:", end=' ')
                for word in gen_pair[i]:
                    #print(ix_to_word[word.item()], end=' ')
                    outf.write(ix_to_word[word.item()] + ' ')
                print()
            sen_iter_list.append(num_iter_sen)
            outf.write('\n')
            print()
            # Label this pair as 0
            # 4th entry for iteration decay
            # gen_pair.append(num_iter_sen)
            if count % args.log_interval == 0:
                print('| Generated {}/{} sentences | ave_time {:5.2f}s'.format(count, args.words, (time.time() - start_time) / count))
                #sys.exit()
            gen_sen_pairs.append(gen_pair)
    if num_loop % 20 == 0 and not evaluation:
        dis_reward_list.append(dis_reward / (num_dis + 1))
        pickle.dump(dis_reward_list, open('dis_reward_list_tf', 'wb'))
    return gen_sen_pairs, dis_reward, num_dis, sen_iter_list


