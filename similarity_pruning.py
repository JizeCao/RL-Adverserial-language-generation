import pickle
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from model import EncoderRNN
import itertools
from search_utils import Voc, tensorFromPairEval


### Choose sentences

# Do zero padding given a list of either source or target sens
# Return: a list contains #max_length of lists, each list contain #batch_size elements
def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    # indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    indexes_batch = [sen + [EOS_token] for sen in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch, only_source=False):
    # pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    if only_source:
        input_batch = [pair_batch]
        inp, lengths = inputVar(input_batch, voc)
        return inp, lengths
    pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    return inp, lengths, input_batch, output_batch

def get_hidden(input_variable, lengths, encoder, args):
    device = args.device
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)

    with torch.no_grad():
        # Forward pass through encoder
        encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

        forward_hidden = torch.sum(encoder_hidden[:len(encoder_hidden) // 2], dim=0)
        backward_hidden = torch.sum(encoder_hidden[len(encoder_hidden) // 2:], dim=0)
        concatenated_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)

    return concatenated_hidden.cpu()

# Get one sentence's hidden
def get_sen_hidden(sen, encoder, voc, args):
    # 0 for dummy sentence
    inp, lengths = batch2TrainData(voc, sen, only_source=True)
    hidden = get_hidden(inp, lengths, encoder, args)
    hidden = hidden / torch.sqrt(torch.sum(torch.pow(hidden, 2), dim=1).unsqueeze(dim=1))

    return hidden


def get_optimal_batches(sen, encoder, hiddens, sen_list, args):

    sen_hidden = get_sen_hidden(sen, encoder, voc, args)
    sen_hidden = sen_hidden.unsqueeze(0)
    similarity = torch.mm(sen_hidden.squeeze(0), torch.transpose(hiddens, 0, 1))
    _, sens_indexes = torch.topk(similarity, args.num_prune)
    chosen_sens = []
    sens_indexes = sens_indexes.squeeze(0)
    for index in sens_indexes:
        chosen_sens.append(sen_list[index.item()])

    return chosen_sens

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




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='UCT pruning')

    parser.add_argument('--save_dir', type=str, default='./data/save',
                        help="directory of data")
    parser.add_argument('--num_warm_up', type=int, default=10000,
                        help="number of sentences to warm up UCT")
    parser.add_argument('--batch_size', type=int, default=1000,
                        help="batch size")

    args = parser.parse_args()

    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    ### Load the encoder and data
    hidden_size = 512
    encoder_n_layers = 3
    dropout = 0.2

    model_save_dir = os.path.join(args.save_dir, 'cb_model/Open_subtitles/3-3_512')

    train_data = pickle.load(open(os.path.join(args.save_dir, 'small_train_2000000.p'), 'rb'))
    voc = pickle.load(open(os.path.join(args.save_dir, 'Vocabulary'), 'rb'))

    loadFilename = os.path.join(model_save_dir, 'best_model_checkpoint_original_setting_no_valid.pt')

    if torch.cuda.is_available():
        cp = torch.load(open(loadFilename, 'rb'))
    else:
        cp = torch.load(open(loadFilename, 'rb'), map_location=lambda storage, loc: storage)

    voc.__dict__ = cp['voc_dict']
    embedding_sd = cp['embedding']
    encoder_sd = cp['en']

    PAD_token = voc.word2index['<PAD>']
    EOS_token = voc.word2index['<EOS>']

    embedding = nn.Embedding(voc.num_words, hidden_size)
    embedding.load_state_dict(embedding_sd)
    embedding.to(args.device)
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    encoder.load_state_dict(encoder_sd)
    encoder.to(args.device)
    encoder.eval()

    permutation = np.random.permutation(len(train_data))
    picked_pairs = []
    train_pairs = []

    for i in range(len(permutation)):
        if i < args.num_warm_up:
            picked_pairs.append(train_data[i])
        else:
            train_pairs.append(train_data[i])

    n_iteration = len(picked_pairs) // args.batch_size

    training_batches = [batch2TrainData(voc, [picked_pairs[i + j * args.batch_size] for i in range(args.batch_size)])
                                for j in range(n_iteration)]

    # Data will be sorted by length by length, reformat it
    input_hiddens = None
    input_sens = None
    output_sens = None

    for iteration in range(1, n_iteration + 1):

        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, input_batch, output_batch = training_batch
        if input_sens is None:
            input_sens = input_batch
            output_sens = output_batch
        else:
            input_sens += input_batch
            output_sens += output_batch

        if input_hiddens is None:
            # Run a training iteration with batch
            input_hiddens = get_hidden(input_variable, lengths, encoder, args)
        else:
            input_hidden = get_hidden(input_variable, lengths, encoder, args)
            input_hiddens = torch.cat((input_hiddens, input_hidden), dim=0)
        print('Generated {} pairs hidden'.format(str(iteration * args.batch_size)))

    # Reformat the sentences' list
    rearranged_sens = [[input_sens[i], output_sens[i]] for i in range(len(input_sens))]

    normalized_hiddens = input_hiddens / torch.sqrt(torch.sum(torch.pow(input_hiddens, 2), dim=1).unsqueeze(dim=1))

    torch.save(normalized_hiddens, 'heuristic_normalized_sentences_hiddens')
    pickle.dump(rearranged_sens, open('heuristic_sentences', 'wb'))
    pickle.dump(train_pairs, open('train_remain_pairs', 'wb'))






