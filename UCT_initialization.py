from similarity_pruning import get_optimal_batches
import numpy as np
import torch
from collections import Counter
from search_utils import tensorFromPairEval


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


### Initialize the UCT tree to help pruning

# def descriminator_reward(modelD, source, targets):
#     results = evaluate_sen(source, targets)


# Manually shaping the UCT reward for now, may use generator to do prune first
# Targets and source require to add EOS
def UCT_initialization(root, source, targets, encoder, discriminator, voc, hiddens, args):

    if type(source) is torch.Tensor:
        source = source.tolist()
    chosen_targets = get_optimal_batches(source, encoder, hiddens, targets, args)
    for sen in chosen_targets:
        # Use the last sentence as a tool to prune
        root, early_stopping, score = initialize_path(root, source, sen[1], args.vocabulary_size, discriminator, args)
        if early_stopping:
            return root, [source, sen[1]], score

    return root, None, None

# Data source is a pair
def evaluate_sen(data_source, model, args):

    with torch.no_grad():
        # data = get_batch(data_source, 0, args, batch_size=batch_size, evaluation=True)
        data = data_source

        data = tensorFromPairEval(data, EOS_Token=args.EOS_id)
        log_prob = model(data, to_device=True)
        # Only evaluate last element is important
        prob = torch.exp(log_prob)[0][0].item()
        # sys.exit()
    return prob


def initialize_path(root, source, target, action_space, dis_model, args):
    current = root
    depth = 0

    early_stopping = False

    for word in target:
        current.initialize_Node(word, action_space, reward=None, hidden=None, prune=True)
        current.next = current.children[word]
        current.next.num_visit = current.num_visited_children[word]
        # Add 1 to the #visit of the coming evaluated node after having assigned
        # the value because of the backup
        current.num_visited_children[word] = current.num_visited_children[word] + 1
        # if it has been initialized, do nothing; otherwise creates a new node
        current = current.next
        depth = depth + 1

    result = evaluate_sen([source, target], dis_model, args=args)
    depth = 0

    if result > 0.5:
        early_stopping = True
    else:
        current = root
        while current.next is not None:
            #print('check update')
            if type(target[depth]) is torch.Tensor:
                current.reward[target[depth].item()] += result
            else:
                current.reward[target[depth]] += result
            # Increase the number of visit during the backup rather than tree policy
            current.num_visit += 1
            current = current.next
            depth += 1

    return root, early_stopping, result





