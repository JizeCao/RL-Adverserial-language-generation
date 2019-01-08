import torch
import numpy as np
import argparse
import pickle
import torch.nn as nn
import sys
import time
import gc
import copy
import os
from search_utils import create_exp_dir, evaluate_word, tensorFromPairEval
from collections import Counter
from UCT_initialization import UCT_initialization
import math


class Node(object):
    # Construct the node with the given initial reward(provided by generator) and the action space
    # (#words in dictionary) 
    def __init__(self, reward=None, num_visit=0,action_space=None):
        self.reward = reward
        self.action_space = action_space
        self.next = None
        self.children = Counter()
        self.num_visit = num_visit
        self.num_visited_children = np.zeros(action_space)
        self.hidden = None
        '''
        if self.num_visit != 0:
            # Uninitialized Nodes to that node
            self.num_visited_children = np.zeros(action_space)
            self.children = [Node(None, 0, self.action_space) for _ in action_space]
        else:
            self.num_visited_children = None
            self.children = None
        '''
    # Change the reward value
    def add_reward_value(self, reward):
        if self.reward is not None:
            self.reward = self.reward + reward
        else:
            self.reward = reward

    # Add 1 to the number of all visited children because of the prior dist
    # If the node is not initialized, intialized it to be zero
    def add_num_value(self):
        # self.num_visit = self.num_visit + 1
        if self.num_visited_children is not None:
            self.num_visited_children = self.num_visited_children + 1
        else:
            self.num_visited_children = np.zeros(self.action_space)

    # Initialize the node if it's empty, otherwise do nothing
    def initialize_Node(self, word, action_space, reward, hidden, prune=False):
        node = self.children[word]
        # the comming node is not initialized
        if node == 0:
            if not prune:
                self.children[word] = Node(reward, 1, action_space)
                self.children[word].hidden = hidden
            else:
                zero_reward = torch.zeros(action_space)
                if torch.cuda.is_available():
                    zero_reward = zero_reward.cuda()
                self.children[word] = Node(zero_reward, 1, action_space)
                self.children[word].hidden = None


    # Add the reward to the current node and increase all the possible childern's count by one. 
    # Then select the word based on the UCT policy
    def select(self, args):

        self.add_num_value()
        selected_word = np.argmax(self.reward / self.num_visited_children + args.exp_cont * np.sqrt(2 * np.log(self.num_visit) /
                                                                                                        self.num_visited_children))
        return selected_word

# state: the time step
# Now the reward is finished
def UCTSearch(init_reward, action_space, decoder, encoder_output, init_hidden, dis_model, source, gen_cache, dis_cache,
              num_dis, dis_reward, args, ix_to_word, sentence=False, pruning=False, targets=None, voc=None, hiddens=None,
              encoder=None):

    result = 0
    eos = False
    root = Node(None, 1, action_space)
    root.add_reward_value(init_reward.cpu().numpy())
    root.hidden = init_hidden

    if pruning:

        # Use pruning through the UCT
        root, return_pair, score = UCT_initialization(root, source, targets, encoder, dis_model, voc, hiddens, args)
        if return_pair is not None:
            print(score)
            for sen in return_pair:
                for word in sen:
                    print(ix_to_word[word], end=' ')
                print()
            final_result = result

            return return_pair, dis_reward, num_dis, 0, final_result

    final_result = 0
    # sys.exit()
    pair = [source, None]
    rep_count = 0
    for i in range(args.num_iter):
        # Refresh the hidden reward, the current node and the depth of the search
        wordlist = []
        depth = 0
        current = root
        # Get a deep copy of the hidden state so that the initial one can be reused
        hidden = copy.deepcopy(init_hidden)
        # the implementation of treePolicy
        while depth < args.max_seq_len and not eos:

            word_selected = current.select(args)
            word = torch.LongTensor([word_selected])
            wordlist.append(word_selected)
            wordtuple = tuple(wordlist)
            # if wordtuple not in gen_cache:
            #     reward, hidden = evaluate_word(decoder, encoder_output, word, hidden, args)
            #     reward = reward.cpu().numpy() * args.reward_panalty
            #     gen_cache[wordtuple] = [copy.deepcopy(reward), copy.deepcopy(hidden)]
            # else:
            #     reward = gen_cache[wordtuple][0]
            #     hidden = gen_cache[wordtuple][1]

            word = word.item()
            # Node is not initialized
            if current.children[word] == 0:
                reward, hidden = evaluate_word(decoder, encoder_output, word, hidden, args)
                current.initialize_Node(word, action_space, reward, hidden)

            # The node is initialized during pruning
            elif current.children[word].hidden is None:
                reward, hidden = evaluate_word(decoder, encoder_output, word, hidden, args)
                current.children[word].reward += reward
                current.children[word].hidden = hidden
            else:
                # The hidden of the children
                hidden = current.children[word].hidden

            # store the trace of the selected actions
            current.next = current.children[word]
            current.next.num_visit = current.num_visited_children[word]
            # Add 1 to the #visit of the coming evaluated node after having assigned
            # the value because of the backup
            current.num_visited_children[word] = current.num_visited_children[word] + 1
            # if it has been initialized, do nothing; otherwise creates a new node
            current = current.next
            depth = depth + 1
            if word == args.EOS_id:
                eos = True

        # the implementation of default policy
        # As it's impossible for a non-terminating state, only get the reward
        # from the discriminator
        if wordtuple not in dis_cache:
            pair[1] = torch.LongTensor(wordlist)
            result = evaluate_sen(pair, dis_model, args)
            dis_cache[wordtuple] = copy.deepcopy(result)
            rep_count = 0
        else:
            rep_count += 1
            result = dis_cache[wordtuple]
        dis_reward += result
        #print(result)
        #print([ix_to_word[word] for word in wordlist])

        # Early stopping
        if result >= 0.5 or rep_count == 15:
            print(result)
            for sen in pair:
                for word in sen:
                    print(ix_to_word[word.item()], end=' ')
                print()
            final_result = result
            break
        current = root
        # the implementation of backup
        # Following the trace of the front node, add the discriminator's reward to the
        # corresponding position. P.S: no need to increase #visit because it has been added
        # to the parent node during tree Policy
        depth = 0

        while current.next is not None:
            #print('check update')
            current.reward[wordlist[depth].item()] += result
            # Increase the number of visit during the backup rather than tree policy
            current.num_visit += 1
            current = current.next
            depth += 1
        eos = False
    num_dis += i
    # sys.exit()
    if sentence:
        # i: the used #iterations
        return pair, dis_reward, num_dis, i, final_result
    else:
        return np.argmax(root.reward), dis_reward, num_dis


# Data source is a pair
def evaluate_sen(data_source, model, args):

    with torch.no_grad():
        # data = get_batch(data_source, 0, args, batch_size=batch_size, evaluation=True)
        data = data_source
        #if args.cuda:
        #    data[0].cuda()
        #    data[1].cuda()
        # print(type(data), data.shape)
        data = tensorFromPairEval(data, EOS_Token=args.EOS_id)
        log_prob = model(data, to_device=True)
        # Only evaluate last element is important
        prob = torch.exp(log_prob)[0][0].item()
        # sys.exit()
    return prob

