import pickle
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from model import EncoderRNN
import itertools
from search_utils import Voc

parser = argparse.ArgumentParser(description='UCT pruning')

parser.add_argument('--save_dir', type=str, default='./data/save',
                    help="directory of data")
parser.add_argument('--num_warm_up', type=int, default=10000,
                    help="number of sentences to warm up UCT")
parser.add_argument('--batch_size', type=int, default=1024,
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
def batch2TrainData(voc, pair_batch):
    # pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    return inp, lengths

def get_hidden(input_variable, lengths, encoder, args):
    device = args.device
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    forward_hidden = torch.sum(encoder_hidden[:len(encoder_hidden) // 2], dim=0)
    backward_hidden = torch.sum(encoder_hidden[len(encoder_hidden) // 2:], dim=0)
    concatenated_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)


    return concatenated_hidden


permutation = np.random.permutation(len(train_data))
picked_pairs = []
for i in permutation:
    picked_pairs.append(train_data[i])

n_iteration = len(picked_pairs) // args.batch_size

training_batches = [batch2TrainData(voc, [picked_pairs[i + j * args.batch_size] for i in range(args.batch_size)])
                            for j in range(n_iteration)]

input_hiddens = None

for iteration in range(1, n_iteration + 1):

    training_batch = training_batches[iteration - 1]
    # Extract fields from batch
    input_variable, lengths = training_batch

    if input_hiddens is None:
        # Run a training iteration with batch
        input_hiddens = get_hidden(input_variable, lengths, encoder, args)
    else:
        input_hidden = get_hidden(input_variable, lengths, encoder, args)
        input_hiddens = torch.cat((input_hiddens, input_hidden), dim=0)

torch.save(input_hiddens, 'heuristic_sentences_hiddens')
pickle.dump(picked_pairs, open('heuristic_sentences', 'wb'))






