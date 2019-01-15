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
from tqdm import tqdm



def warm_up(encoder, decoder, source, ix_to_word, args, SOS_index=1):
    with torch.no_grad():
        source = torch.Tensor(source).long()
        # source is the source sentence
        encoder_output, init_reward, decoder_hidden = evaluate_sen(encoder, decoder, source, args.device,
                                                                   SOS_token=args.SOS_id, EOS_token=args.EOS_id)
    # Get the dimension of the output
    return encoder_output, torch.exp(init_reward.view(-1)), decoder_hidden, source

def load_initialization(args):
    save_dir = args.save_dir
    targets = pickle.load(open(os.path.join(save_dir, 'heuristic_sentences'), 'rb'))
    vocab = pickle.load(open(os.path.join(save_dir, 'whole_data_voc.p'), 'rb'))
    hiddens = torch.load(os.path.join(save_dir, 'heuristic_normalized_sentences_hiddens'))

    return targets, vocab, hiddens

def logging(s, print_=True, log_=True, args=None):
    if print_:
        print(s)
    if log_:
        with open(os.path.join(args.save, 'log.txt'), 'a+') as f_log:
            f_log.write(s + '\n')


# Generate the reward for a specific tensor
def generate_reward(decoder, encoder_output, decoder_hidden, input_word):
    return evaluate_word(decoder, encoder_output, decoder_hidden, input_word)


def generation(encoder, decoder, dis_model, num_loop, args, warm_up_words_list, start_index, dis_reward, num_dis,
               ix_to_word, dis_reward_list, evaluation=False, output_file=None, batch_size=None, ensemble=None):
    if batch_size is None:
        batch_size = args.batch_size
    if output_file is None:
        output_file = args.outf
    if evaluation:
        writing_statu = 'a'
    else:
        writing_statu = 'w'
    if args.prune:
        targets, vocab, hiddens = load_initialization(args)
    else:
        targets = None
        vocab = None
        hiddens = None

    sen_iter_list = []
    dis_panalty_list = []
    with open(output_file, writing_statu) as outf:
        count = 0
        start_time = time.time()
        gen_sen_pairs = []
        # Warmup the generator
        for i in tqdm(range(batch_size)):
            if count >= batch_size:
                break
            source = warm_up_words_list[start_index + i][0]
            encoder_output, init_reward, hidden, source = warm_up(encoder, decoder, source, ix_to_word, args)

            count += 1
            # Use caches to accelerate
            gen_cache = {}
            dis_cache = {}
            # Sentence level MCTS
            gen_pair, dis_reward, num_dis, num_iter_sen, dis_panalty = UCTSearch(init_reward, action_space=len(ix_to_word), decoder=decoder,
                                                                                 dis_model=dis_model, init_hidden=hidden, source=source,
                                                                                 gen_cache = gen_cache, dis_cache=dis_cache, sentence=True,
                                                                                 num_dis=num_dis, dis_reward=dis_reward, encoder_output=encoder_output,
                                                                                 args=args,
                                                                                 ix_to_word=ix_to_word,
                                                                                 pruning=args.prune,
                                                                                 targets=targets,
                                                                                 voc=vocab,
                                                                                 hiddens=hiddens,
                                                                                 encoder=encoder,
                                                                                 ensemble=ensemble)
            for i in range(len(gen_pair)):
                if args.print:
                    if i == 0:
                        print("Source:", end=' ')
                    else:
                        print("Answer:", end=' ')
                    print()
                for word in gen_pair[i]:
                    #print(ix_to_word[word.item()], end=' ')
                    if type(word) is torch.Tensor:
                        outf.write(ix_to_word[word.item()] + ' ')
                    else:
                        outf.write(ix_to_word[word] + ' ')

            sen_iter_list.append(num_iter_sen)
            outf.write('\n')

            # Label this pair as 0
            # 4th entry for iteration decay
            # gen_pair.append(num_iter_sen)
            if args.print:
                if count % args.log_interval == 0:
                    print('| Generated {}/{} sentences | ave_time {:5.2f}s'.format(count, args.words, (time.time() - start_time) / count))
                    #sys.exit()
            gen_sen_pairs.append(gen_pair)
            dis_panalty_list.append(dis_panalty)

    if num_loop % 20 == 0 and not evaluation:
        dis_reward_list.append(dis_reward / (num_dis + 1))
        pickle.dump(dis_reward_list, open('dis_reward_list_tf', 'wb'))

    return gen_sen_pairs, dis_reward, num_dis, sen_iter_list, dis_panalty_list

# Want a deep copy for the hidden state
# current_hidden = copy.deepcopy(hidden)
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


