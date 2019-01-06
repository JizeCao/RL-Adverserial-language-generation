import os
import pickle
import torch
import random
import torch.nn as nn
import numpy as np
from torch import optim
import time
from search_utils import tensorFromPair, logging
from model import hierEncoder_frequency_batchwise
import argparse
import itertools

parser = argparse.ArgumentParser(description='UCT based on the language model')
parser.add_argument('--save_dir', type=str, default='./data/save',
                    help='directory of the save place')
parser.add_argument('--retrain', action='store_true', 
                    help='retrain from an existing checkpoint')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

# Do zero padding given a list of either source or target sens
# Return: a list contains #max_length of lists, each list contain #batch_size elements
def zeroPadding(l, fillvalue):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

# Returns padded input sequence tensor and lengths
def inputVar(l, EOS_token, PAD_token):
    # indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]

    #indexes_batch = [sen + [EOS_token] if sen[-1] != EOS_token else sen for sen in l]
    indexes_batch = l
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch, PAD_token)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

def indexing(pair_batch, EOS_token):

    new_batch = []
    for pair in pair_batch:
        temp = []
        for sen in pair:
            if sen[-1] != EOS_token:
                if type(sen) is list:
                    sen.append(EOS_token)
                else:
                    sen = torch.cat((sen, torch.LongTensor([EOS_token])), 0) 
        
            temp.append(sen)
        
        new_batch.append(temp)

    return new_batch

# Returns all items for a given batch of pairs
def batch2TrainData(vocab, pair_batch):
    # pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    EOS_token = vocab.word2index['<EOS>']
    PAD_token = vocab.word2index['<PAD>']
    pair_batch = indexing(pair_batch, EOS_token)
    input_lengths = [len(pair[0]) for pair in pair_batch]
    input_order = np.flip(np.argsort(np.asarray(input_lengths)))
    pair_batch = [pair_batch[i] for i in input_order]
    input_batch, output_batch = [], []
    output_lengths = []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
        output_lengths.append(len(pair[1]))
    output_order = np.flip(np.argsort(np.asarray(output_lengths)))
    output_batch_in_order = [output_batch[i] for i in output_order]
    inp, lengths = inputVar(input_batch, EOS_token, PAD_token)
    output, output_lengths = inputVar(output_batch_in_order, EOS_token, PAD_token)
    return inp, lengths, output, output_lengths, output_order, input_order




class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)


def pretrainD(modelD, TrainSet, GenSet, EOS_token, vocabulary, learning_rate=0.001, batch_size=128, to_device=True):

    modelD.train()
    # prepare data
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # half positive, half negative
    sub_batch_size = batch_size // 2
    #pos_data = [tensorFromPair(random.choice(TrainSet), EOS_Token=EOS_token, to_device=to_device) for _ in range(batch_size)]
    #neg_data = [tensorFromPair(random.choice(GenSet), EOS_Token=EOS_token, to_device=to_device) for _ in range(batch_size)]
    pos_data = [random.choice(TrainSet) for _ in range(sub_batch_size)]
    neg_data = [random.choice(GenSet) for _ in range(sub_batch_size)]
    data = pos_data + neg_data
    input_data, input_length, output_data, output_length, output_order, input_order = batch2TrainData(vocabulary, data)

    #  Get the order that is used to retrive the original order
    output_order = np.argsort(output_order)

    # define optimizer & criterion
    discOptimizer = optim.SGD(modelD.parameters(), lr=learning_rate, momentum=0.8)
    criterion = nn.NLLLoss()
    discOptimizer.zero_grad()
    # some predefined variable
    # 注意：规定 Discriminator 的输出概率的含义为 [positive_probability, negative_probability]

    labels = np.asarray([0] * sub_batch_size + [1] * sub_batch_size)


    labels = labels[input_order]
    labels = torch.from_numpy(labels).to(device)


    start_time = time.time()
    misclassification = 0

    output = modelD(sources=input_data, targets=output_data, sources_length=input_length, targets_length=output_length,
                    targets_order=output_order, to_device=to_device)
    outputTag = torch.argmax(output, dim=1)
    for i in range(len(labels)):
        if labels[i] != outputTag[i]:
            misclassification += 1

    loss = criterion(output, labels)


    # BPTT & params updating
    loss.backward()
    discOptimizer.step()

    print("Time consumed: {} Batch loss: {:.2f}, Misclassification rate {:.2f} ".format((time.time()-start_time),
                                                          loss.item(), misclassification / batch_size))


def evaluateD(modelD, pos_valid, neg_valid, EOS_token, vocab, log_name):
    # prepare data
    batch_size = 512
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    pos_data_batches = [batch2TrainData(vocab, pos_valid[i * batch_size: i * batch_size + batch_size]) for i in range(len(pos_valid) // batch_size)]
    neg_data_batches = [batch2TrainData(vocab, neg_valid[i * batch_size: i * batch_size + batch_size]) for i in range(len(neg_valid) // batch_size)]

    posTags = torch.tensor([0] * batch_size).to(device)
    negTags = torch.tensor([1] * batch_size).to(device)
    loss = torch.Tensor([0]).to(device)
    start_time = time.time()
    criterion = nn.NLLLoss()
    missclassification = 0
    num_sen = 0


    modelD.eval()
    with torch.no_grad():
        for batch in pos_data_batches:

            input_data, input_length, output_data, output_length, output_order, input_order = batch

            #  Get the order that is used to retrive the original order
            retrive_order = np.argsort(output_order)

            output = modelD(sources=input_data, targets=output_data, sources_length=input_length,
                            targets_length=output_length, targets_order=retrive_order)

            predictions = torch.argmax(output, dim=1)

            for prediction in predictions:
                if prediction != 0:
                    missclassification += 1
                num_sen += 1
            loss += criterion(output, posTags)

        for batch in neg_data_batches:

            input_data, input_length, output_data, output_length, output_order, input_order = batch

            #  Get the order that is used to retrive the original order
            retrive_order = np.argsort(output_order)

            output = modelD(sources=input_data, targets=output_data, sources_length=input_length,
                            targets_length=output_length, targets_order=retrive_order)

            predictions = torch.argmax(output, dim=1)

            for prediction in predictions:
                if prediction != 1:
                    missclassification += 1
                num_sen += 1
            loss += criterion(output, negTags)

    logging("Time consumed: {}, Batch loss: {:.2f}, AdverSuc {:.2f}".format((time.time()-start_time),
                                                         loss.item() / (len(pos_data_batches) + len(neg_data_batches)),
                                                        missclassification / num_sen),
                                                        log_name=log_name)
    return loss.item() / (len(pos_data_batches) + len(neg_data_batches)), missclassification / num_sen


def load_data(args):
    #voc = pickle.load(open(os.path.join(args.save_dir, 'processed_voc.p'), 'rb'))
    #train_pos_pairs = pickle.load(open(os.path.join(args.save_dir, 'processed_train_sen_2000000.p'), 'rb'))
    #valid_pos_pairs = pickle.load(open(os.path.join(args.save_dir, 'processed_valid_sen_2000000.p'), 'rb'))
    #neg_train_pairs = pickle.load(open(os.path.join(args.save_dir, 'Generated_data_beam_search_processed_train.p'), 'rb'))
    #neg_valid_pairs = pickle.load(open(os.path.join(args.save_dir, 'Generated_data_beam_search_processed_valid.p'), 'rb'))

    voc = pickle.load(open(os.path.join(args.save_dir, 'whole_data_voc.p'), 'rb'))
    train_pos_pairs = pickle.load(open(os.path.join(args.save_dir, 'small_train_sen_2000000.p'), 'rb'))
    valid_pos_pairs = pickle.load(open(os.path.join(args.save_dir, 'small_valid_sen_2000000.p'), 'rb'))
    neg_train_pairs = pickle.load(open(os.path.join(args.save_dir, 'Generated_data_beam_search_train.p'), 'rb'))
    neg_valid_pairs = pickle.load(open(os.path.join(args.save_dir, 'Generated_data_beam_search_valid.p'), 'rb'))



    return voc, train_pos_pairs, valid_pos_pairs, neg_train_pairs, neg_valid_pairs

if __name__ == '__main__':

    args = parser.parse_args()
    args.gen_lr = 0
    args.dis_lr = 0.001
    args.cuda = True if torch.cuda.is_available() else False
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #np.random.seed(args.seed)
    #torch.manual_seed(args.seed)
    #random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab, pos_train, pos_valid, neg_train, neg_valid = load_data(args)

    PAD_token = vocab.word2index['<PAD>']  # Used for padding short sentences
    SOS_token = vocab.word2index['<SOS>']  # Start-of-sentence token
    EOS_token = vocab.word2index['<EOS>']  # End-of-sentence token

    log_name = 'discriminator_with_freq_decay' + '.txt'

    counter = 0

    embedding_size = 500

    Discriminator = hierEncoder_frequency_batchwise(len(vocab.index2word), 500)
    save_point_name = 'dist_freq_batch.pt'
    if args.retrain:
        if args.cuda:
            cp = torch.load('dist_freq_batch.pt')
        else:
            cp = torch.load('dist_freq_batch.pt', map_location=lambda storage, loc: storage)
        start_iteration = cp['iteration']
        val_loss = cp['val_loss']
        AdverSuc = cp['AdverSuc']
        Discriminator.load_state_dict(cp['disc'])
        save_point_name = 'dist_finetune_freq_batch.pt'
    else:
        val_loss = 100000000
        AdverSuc = 10000
        start_iteration = 0

    n_iterations = 2000000
    Discriminator.to(device)
    for i in range(n_iterations):
        try:
            pretrainD(Discriminator, pos_train, neg_train, EOS_token, vocab, batch_size=128)
            if (i + 1) % 500 == 0:
                print('Start validation check')
                current_val_loss, curr_AdverSuc = evaluateD(Discriminator, pos_valid[:20000], neg_valid[:20000], EOS_token, vocab,
                                                            log_name)

                if curr_AdverSuc < AdverSuc:
                    val_loss = current_val_loss
                    AdverSuc = curr_AdverSuc
                    torch.save({
                        'iteration': i + start_iteration,
                        'AdverSuc': AdverSuc,
                        'val_loss': val_loss,
                        'disc': Discriminator.state_dict()
                    },  save_point_name)

        except KeyboardInterrupt:
            logging('The final AdverSuc for the baseline is {:.2f}'.format(AdverSuc), log_name)
            exit()



