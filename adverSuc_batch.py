import os
import pickle
import torch
import random
import torch.nn as nn
import numpy as np
from torch import optim
import time
from discriminator import pretrainD, batch2TrainData
from search_utils import tensorFromPair, logging, Voc
from model import hierEncoder_frequency_batchwise
import argparse
import itertools

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
    train_pos_pairs = pickle.load(open(os.path.join(args.save_dir, 'small_train_2000000.p'), 'rb'))
    valid_pos_pairs = pickle.load(open(os.path.join(args.save_dir, 'small_valid_2000000.p'), 'rb'))
    neg_train_pairs = pickle.load(open(os.path.join(args.save_dir, 'Generated_sentences_' + str(args.RL_index)), 'rb'))
    neg_valid_pairs = pickle.load(open(os.path.join(args.save_dir, 'Generated_sentences_valid_'+ str(args.RL_index)), 'rb'))
    return voc, train_pos_pairs, valid_pos_pairs, neg_train_pairs, neg_valid_pairs

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='UCT based on the language model')
    parser.add_argument('--save_dir', type=str, default='./data/save',
                        help='directory of the save place')
    parser.add_argument('--RL_index', type=int, default=60,
                        help="index of the RL save checkpoint")
    parser.add_argument('--retrain', action='store_true',
                        help='retrain from an existing checkpoint')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')

    args = parser.parse_args()
    args.gen_lr = 0
    args.dis_lr = 0.001
    args.cuda = True if torch.cuda.is_available() else False
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab, pos_train, pos_valid, neg_train, neg_valid = load_data(args)

    PAD_token = vocab.word2index['<PAD>']  # Used for padding short sentences
    SOS_token = vocab.word2index['<SOS>']  # Start-of-sentence token
    EOS_token = vocab.word2index['<EOS>']  # End-of-sentence token

    log_name = 'AdverSuc_' + str(args.RL_index) + '_.txt'

    counter = 0

    embedding_size = 500

    Discriminator = hierEncoder_frequency_batchwise(len(vocab.index2word), 500)
    save_point_name = 'AdverSuc_checkpoint_' + str(args.RL_index) + '_.pt'
    if args.retrain:
        if args.cuda:
            cp = torch.load('AdverSuc_checkpoint_' + str(args.RL_index) + '_.pt')
        else:
            cp = torch.load('AdverSuc_checkpoint_' + str(args.RL_index) + '_.pt', map_location=lambda storage, loc: storage)
        start_iteration = cp['iteration']
        val_loss = cp['val_loss']
        AdverSuc = cp['AdverSuc']
        Discriminator.load_state_dict(cp['disc'])
        save_point_name = 'AdverSuc_finetune_checkpoint_' + str(args.RL_index) + '_.pt'
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
                current_val_loss, curr_AdverSuc = evaluateD(Discriminator, pos_valid[:len(neg_valid)], neg_valid, EOS_token, vocab,
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
            logging('The final AdverSuc for the RL_checkpoint {} is {:.2f}'.format(args.RL_index, AdverSuc), log_name)
            exit()
