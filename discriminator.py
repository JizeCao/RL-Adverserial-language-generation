import os
import pickle
import torch
import random
import torch.nn as nn
from torch import optim
import time
from search_utils import tensorFromPair, trim_dummy_sen, load_model_dictionary_pairs, logging
from model import hierEncoder_frequency
import argparse

parser = argparse.ArgumentParser(description='UCT based on the language model')
parser.add_argument('--save_dir', type=str, default='./data/save',
                    help='directory of the save place')
parser.add_argument('--retrain', action='store_true', 
                    help='retrain from an existing checkpoint')


args = parser.parse_args()
args.gen_lr = 0
args.dis_lr = 0.001
args.cuda = True if torch.cuda.is_available() else False
args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    pos_data = [tensorFromPair(random.choice(TrainSet), EOS_Token=EOS_token, to_device=to_device) for _ in range(batch_size)]
    neg_data = [tensorFromPair(random.choice(GenSet), EOS_Token=EOS_token, to_device=to_device) for _ in range(batch_size)]
    # define optimizer & criterion
    discOptimizer = optim.SGD(modelD.parameters(), lr=learning_rate, momentum=0.8)
    criterion = nn.NLLLoss()
    discOptimizer.zero_grad()
    # some predefined variable
    # 注意：规定 Discriminator 的输出概率的含义为 [positive_probability, negative_probability]
    if to_device:
        posTag = torch.tensor([0]).to(device)
        negTag = torch.tensor([1]).to(device)
    else:
        posTag = torch.tensor([0])
        negTag = torch.tensor([1])

    loss = 0
    start_time = time.time()
    misclassification = 0

    for iter in range(batch_size):
        # choose positive or negative pair randomly
        pick_positive_data = True if random.random() < 0.5 else False

        if pick_positive_data:
            output = modelD(pos_data[iter], to_device=to_device)
            outputTag = torch.argmax(output)
            if outputTag == posTag:
                misclassification += 1
            loss += criterion(output, posTag)
        else:
            output = modelD(neg_data[iter], to_device=to_device)
            outputTag = torch.argmax(output)
            if outputTag == negTag:
                misclassification += 1
            loss += criterion(output, negTag)

    # BPTT & params updating
    loss.backward()
    discOptimizer.step()

    print("Time consumed: {} Batch loss: {:.2f}, Misclassification rate {:.2f} ".format((time.time()-start_time),
                                                          loss.item(), 1 - misclassification / batch_size))

def evaluateD(modelD, pos_valid, neg_valid, EOS_token, vocab, log_name):
    # prepare data
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    to_device = True
    pos_data = [tensorFromPair(pos_valid[i], EOS_Token=EOS_token, to_device=to_device) for i in range(10000)]
    neg_data = [tensorFromPair(neg_valid[i], EOS_Token=EOS_token, to_device=to_device) for i in range(10000)]

    posTag = torch.tensor([0]).to(device)
    negTag = torch.tensor([1]).to(device)
    loss = 0
    start_time = time.time()
    criterion = nn.NLLLoss()
    missclassification = 0

    modelD.eval()
    with torch.no_grad():
        for pos_pair in pos_data:
            output = modelD(pos_pair, to_device=True)
            outputTag = torch.argmax(torch.exp(output))
            if outputTag != posTag.long():
                missclassification += 1
            loss += criterion(output, posTag)

        for neg_pair in neg_data:
            output = modelD(neg_pair, to_device=True)
            outputTag = torch.argmax(torch.exp(output))
            if outputTag != negTag.long():
                missclassification += 1
            loss += criterion(output, negTag)

    logging("Time consumed: {}, Batch loss: {:.2f}, AdverSuc {:.2f}".format((time.time()-start_time),
                                                         loss.item() / (len(pos_data) + len(neg_data)),
                                                        missclassification / (len(pos_data) + len(neg_data))),
                                                        log_name=log_name)
    return loss.item() / (len(pos_data) + len(neg_data)), missclassification / (len(pos_data) + len(neg_data))

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    PAD_token = 25002  # Used for padding short sentences
    SOS_token = 0  # Start-of-sentence token
    EOS_token = 25001  # End-of-sentence token

    log_name = 'discriminator_with_freq_decay' + '.txt'

    vocab, pos_train, pos_valid, neg_train, neg_valid = load_model_dictionary_pairs(args, dis_model=False)

    counter = 0

    # for Senset in [pos_train, pos_valid, neg_train, neg_valid]:
    #     counter += 1
    #     for pair in Senset:
    #         try:
    #             source = len(pair[0])
    #             target = len(pair[1])
    #         except IndexError:
    #             print(counter)
    #             exit()

    embedding_size = 500
    if args.retrain:
        if args.cuda:
            Discriminator = torch.load('dist_freq.pt')
        else:
            Discriminator = torch.load('dist_freq.pt', map_location=lambda storage, loc: storage)
    else:
        Discriminator = hierEncoder_frequency(len(vocab.index2word), 500)

    Discriminator.to(device)

    n_iterations = 4000
    val_loss = 100000000
    AdverSuc = 10000

    for i in range(n_iterations):
        try:
            pretrainD(Discriminator, pos_train, neg_train, EOS_token, vocab, batch_size=256)
            if (i + 1) % 30 == 0:
                print('Start validation check')
                current_val_loss, curr_AdverSuc = evaluateD(Discriminator, pos_valid, neg_valid, EOS_token, vocab,
                                                            log_name)

                if curr_AdverSuc < AdverSuc:
                    val_loss = current_val_loss
                    AdverSuc = curr_AdverSuc
                    torch.save({
                        'iteration': i,
                        'AdverSuc': AdverSuc,
                        'val_loss': val_loss,
                        'disc': Discriminator.state_dict()
                    },  'dist_freq' + '.pt')

        except KeyboardInterrupt:
            logging('The final AdverSuc for the baseline is {:.2f}'.format(AdverSuc), log_name)
            exit()



