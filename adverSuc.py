from model import hierEncoder
from discriminator import pretrainD
import pickle
import torch
import random
import torch.nn as nn
from torch import optim
import time
from search_utils import tensorFromPair, logging, Voc
import argparse

def evaluateD(modelD, pos_valid, neg_valid, EOS_token, vocab, log_name):
    # prepare data
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    to_device = True
    pos_data = [tensorFromPair(pos_valid[i], EOS_Token=EOS_token, to_device=to_device) for i in range(len(pos_valid))]
    neg_data = [tensorFromPair(neg_valid[i], EOS_Token=EOS_token, to_device=to_device) for i in range(len(neg_valid))]
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


parser = argparse.ArgumentParser(description='AdverSuc metric')
parser.add_argument('--RL_index', type=int, default=60,
                    help="index of the RL save checkpoint")

args = parser.parse_args()


log_name = str(args.RL_index) + '.txt'


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PAD_token = 25002  # Used for padding short sentences
SOS_token = 0  # Start-of-sentence token
EOS_token = 25001  # End-of-sentence token

data_dir = '../data/save/'
vocab = pickle.load(open(data_dir + 'whole_data_voc.p', 'rb'))
pos_train = pickle.load(open(data_dir + 'small_train_2000000.p', 'rb'))[:20000]
pos_valid = pickle.load(open(data_dir + 'small_valid_2000000.p', 'rb'))[:2000]
neg_train = pickle.load(open(data_dir + 'Generated_sentences_' + str(args.RL_index), 'rb'))
neg_valid = pickle.load(open(data_dir + 'Generated_sentences_valid_' + str(args.RL_index), 'rb'))


counter = 0
for Senset in [pos_train, pos_valid, neg_train, neg_valid]:
    counter += 1
    for pair in Senset:
        try:
            source = len(pair[0])
            target = len(pair[1])
        except IndexError:
            print(counter)
            exit()

embedding_size = 500
Discriminator = hierEncoder(len(vocab.index2word), 500)
Discriminator.to(device)

n_iterations = 40000000000
val_loss = 100000000
AdverSuc = 10000

for i in range(n_iterations):
    try:
        pretrainD(Discriminator, pos_train, neg_train, EOS_token, vocab, batch_size=256)
        if (i + 1) % 30 == 0:
            print('Start validation check')
            current_val_loss, curr_AdverSuc = evaluateD(Discriminator, pos_valid, neg_valid, EOS_token, vocab, log_name)

            if curr_AdverSuc < AdverSuc:
                val_loss = current_val_loss
                AdverSuc = curr_AdverSuc
                torch.save({
                    'iteration': i,
                    'AdverSuc': AdverSuc,
                    'val_loss': val_loss,
                    'disc': Discriminator.state_dict()
                }, 'AdverSuc_dist' + str(args.RL_index) + '.pt')

    except KeyboardInterrupt:
        logging('The final AdverSuc for RL checkpoint {} is {:.2f}'.format(str(args.RL_index), AdverSuc), log_name)
        exit()








