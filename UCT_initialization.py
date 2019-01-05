from UCT_search import Node, evaluate_sen


### Initialize the UCT tree to help pruning

# Manually shaping teh UCT reward for now, may use generator to do prune first
def UCT_initialization(root, source, targets, warmup_hiddens, args):

    reward = args.reward

    for sen in targets:
        root, early_stopping = initialize_path(root, reward, source, sen, args.vocabulary_size, args)
        if early_stopping:
            return root, [source, sen]

    return root, None

def initialize_path(root, reward, source, target, action_space, dis_model, args):
    current = root
    eos = False
    early_stopping = False

    for word in target:
        current.initialize_Node(word, reward, action_space)
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
            current.reward[target[depth].item()] += result
            # Increase the number of visit during the backup rather than tree policy
            current.num_visit += 1
            current = current.next
            depth += 1

    return root, early_stopping





